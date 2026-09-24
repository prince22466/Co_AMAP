#!/usr/bin/env python3
"""Agents-SDK runtime, durable research memory, and observability for v23."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings, RunConfig, RunContextWrapper, RunHooks, Runner, SQLiteSession, SessionSettings
from agents.decorators import tool
from openai.types.shared import Reasoning

from research_agent_v1 import (
    DEFAULT_MODEL, DEFAULT_SESSION_BUDGET_USD, DEFAULT_TOTAL_BUDGET_USD,
    LEDGER_PATH, MODEL_PRICING_USD_PER_M, SYSTEM_PROMPT as V1_SYSTEM_PROMPT,
    WORKSPACE, BudgetLedger, LocalTools, RunLog, Usage as LegacyUsage,
)

STATE_DB = WORKSPACE / "experiments.sqlite3"
SESSION_DB = WORKSPACE / "agent_sessions.sqlite3"
DEFAULT_SESSION_ID = "v23-research"

SYSTEM_PROMPT = V1_SYSTEM_PROMPT + """

v2 research-memory contract:
- Before candidate evaluation, call start_experiment with one falsifiable hypothesis.
- Pass experiment_id to static_replay_candidate.
- After analysis, call finish_experiment with SUPPORTED, REJECTED, UNRESOLVED, or ERROR.
- Call project_status before repeating an idea. Previous negative results are durable evidence.
- Call set_goal when the task has an explicit numeric target.
- Goal completion comes only from structured replay metrics, never prose.
- Model-written candidate policies may execute only through static_replay_candidate.
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS runs(
          run_id TEXT PRIMARY KEY, session_id TEXT, task TEXT, model TEXT,
          started_at TEXT, ended_at TEXT, elapsed_seconds REAL, status TEXT,
          api_requests INTEGER DEFAULT 0, input_tokens INTEGER DEFAULT 0,
          cached_tokens INTEGER DEFAULT 0, cache_write_tokens INTEGER DEFAULT 0,
          output_tokens INTEGER DEFAULT 0, reasoning_tokens INTEGER DEFAULT 0,
          total_tokens INTEGER DEFAULT 0, conservative_cost_usd REAL DEFAULT 0,
          experiments_started INTEGER DEFAULT 0, replay_calls INTEGER DEFAULT 0,
          replay_cases INTEGER DEFAULT 0, final_output TEXT);
        CREATE TABLE IF NOT EXISTS experiments(
          experiment_id TEXT PRIMARY KEY, run_id TEXT, hypothesis TEXT,
          candidate TEXT, parent_candidate TEXT, notes TEXT, started_at TEXT,
          ended_at TEXT, elapsed_seconds REAL, status TEXT, conclusion TEXT,
          replay_calls INTEGER DEFAULT 0, replay_cases INTEGER DEFAULT 0,
          best_margin_improvement REAL, mean_margin_improvement REAL,
          wins INTEGER, losses INTEGER, regressions INTEGER);
        CREATE TABLE IF NOT EXISTS replays(
          replay_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
          experiment_id TEXT, replay_call_id TEXT, candidate TEXT, episode TEXT,
          started_at TEXT, elapsed_seconds REAL, valid INTEGER,
          original_v20_margin REAL, candidate_margin REAL, margin_improvement REAL,
          result TEXT, action_divergences INTEGER, error TEXT);
        CREATE TABLE IF NOT EXISTS goals(
          goal_id TEXT PRIMARY KEY, metric TEXT, operator TEXT, target REAL,
          max_regressions INTEGER, created_at TEXT, reached_at TEXT,
          reached_run_id TEXT, reached_experiment_id TEXT, reached_value REAL,
          reached_observability_json TEXT, active INTEGER DEFAULT 1);
        """)
        goal_columns = {row[1] for row in self.db.execute("PRAGMA table_info(goals)").fetchall()}
        if "reached_observability_json" not in goal_columns:
            self.db.execute("ALTER TABLE goals ADD COLUMN reached_observability_json TEXT")
        self.db.commit()

    def start_run(self, run_id, session_id, task, model):
        self.db.execute(
            "INSERT INTO runs(run_id,session_id,task,model,started_at,status) VALUES(?,?,?,?,?,'RUNNING')",
            (run_id, session_id, task, model, utcnow()))
        self.db.commit()

    def finish_run(self, run_id, status, usage, cost, output, elapsed):
        self.db.execute("""UPDATE runs SET ended_at=?,elapsed_seconds=?,status=?,
          api_requests=?,input_tokens=?,cached_tokens=?,cache_write_tokens=?,
          output_tokens=?,reasoning_tokens=?,total_tokens=?,
          conservative_cost_usd=?,final_output=? WHERE run_id=?""",
          (utcnow(), elapsed, status, usage["requests"], usage["input_tokens"],
           usage["cached_tokens"], usage["cache_write_tokens"], usage["output_tokens"],
           usage["reasoning_tokens"], usage["total_tokens"], cost, output, run_id))
        self.db.commit()

    def start_experiment(self, run_id, hypothesis, candidate="", parent_candidate="", notes=""):
        eid = "exp_" + uuid.uuid4().hex[:10]
        self.db.execute("""INSERT INTO experiments(
          experiment_id,run_id,hypothesis,candidate,parent_candidate,notes,started_at,status)
          VALUES(?,?,?,?,?,?,?,'RUNNING')""",
          (eid, run_id, hypothesis, candidate or None, parent_candidate or None,
           notes or None, utcnow()))
        self.db.execute("UPDATE runs SET experiments_started=experiments_started+1 WHERE run_id=?", (run_id,))
        self.db.commit()
        return eid

    def finish_experiment(self, eid, status, conclusion):
        row = self.db.execute("SELECT started_at FROM experiments WHERE experiment_id=?", (eid,)).fetchone()
        if row is None:
            return {"error": "unknown experiment_id: " + eid}
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(row["started_at"])).total_seconds()
        self.db.execute("""UPDATE experiments SET ended_at=?,elapsed_seconds=?,status=?,conclusion=?
                           WHERE experiment_id=?""",
                        (utcnow(), elapsed, status, conclusion, eid))
        self.db.commit()
        return {"experiment_id": eid, "status": status, "elapsed_seconds": round(elapsed, 3)}

    def set_goal(self, metric, operator, target, max_regressions):
        self.db.execute("UPDATE goals SET active=0 WHERE active=1")
        gid = "goal_" + uuid.uuid4().hex[:10]
        self.db.execute("""INSERT INTO goals(goal_id,metric,operator,target,max_regressions,created_at,active)
                           VALUES(?,?,?,?,?,?,1)""",
                        (gid, metric, operator, float(target), max_regressions, utcnow()))
        self.db.commit()
        return gid

    def maybe_reach_goal(self, run_id, eid, summary, usage_snapshot):
        goal = self.db.execute("SELECT * FROM goals WHERE active=1 ORDER BY created_at DESC LIMIT 1").fetchone()
        if goal is None or goal["metric"] not in summary:
            return None
        value = summary.get(goal["metric"])
        if not isinstance(value, (int, float)):
            return None
        regressions = int(summary.get("margin_worsened_cases", 0) or 0)
        if goal["max_regressions"] is not None and regressions > int(goal["max_regressions"]):
            return None
        target, op, value = float(goal["target"]), goal["operator"], float(value)
        passed = {">=": value >= target, ">": value > target, "<=": value <= target,
                  "<": value < target, "==": value == target}[op]
        if not passed:
            return None

        prior = self.db.execute("""SELECT
          COALESCE(SUM(api_requests),0) requests,
          COALESCE(SUM(input_tokens),0) input_tokens,
          COALESCE(SUM(cached_tokens),0) cached_tokens,
          COALESCE(SUM(output_tokens),0) output_tokens,
          COALESCE(SUM(reasoning_tokens),0) reasoning_tokens,
          COALESCE(SUM(total_tokens),0) total_tokens,
          COALESCE(SUM(conservative_cost_usd),0) cost
          FROM runs WHERE status!='RUNNING'""").fetchone()
        now = datetime.now(timezone.utc)
        created = datetime.fromisoformat(goal["created_at"])
        exp = self.db.execute("SELECT started_at FROM experiments WHERE experiment_id=?", (eid,)).fetchone()
        exp_elapsed = None
        if exp:
            exp_elapsed = (now - datetime.fromisoformat(exp["started_at"])).total_seconds()
        snapshot = {
          "goal_elapsed_seconds": (now - created).total_seconds(),
          "experiment_elapsed_seconds": exp_elapsed,
          "experiments_started": int(self.db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]),
          "replay_calls": int(self.db.execute("SELECT COALESCE(SUM(replay_calls),0) FROM experiments").fetchone()[0]),
          "replay_cases": int(self.db.execute("SELECT COUNT(*) FROM replays").fetchone()[0]),
          "api_requests": int(prior["requests"]) + int(usage_snapshot.get("requests",0)),
          "input_tokens": int(prior["input_tokens"]) + int(usage_snapshot.get("input_tokens",0)),
          "cached_tokens": int(prior["cached_tokens"]) + int(usage_snapshot.get("cached_tokens",0)),
          "output_tokens": int(prior["output_tokens"]) + int(usage_snapshot.get("output_tokens",0)),
          "reasoning_tokens": int(prior["reasoning_tokens"]) + int(usage_snapshot.get("reasoning_tokens",0)),
          "total_tokens": int(prior["total_tokens"]) + int(usage_snapshot.get("total_tokens",0)),
          "prior_completed_run_cost_usd": float(prior["cost"]),
        }
        self.db.execute("""UPDATE goals SET reached_at=?,reached_run_id=?,
          reached_experiment_id=?,reached_value=?,reached_observability_json=?,active=0
          WHERE goal_id=?""",
          (utcnow(), run_id, eid, value, json.dumps(snapshot, sort_keys=True), goal["goal_id"]))
        self.db.commit()
        return {"goal_id": goal["goal_id"], "metric": goal["metric"], "value": value,
                "operator": op, "target": target, "observability": snapshot}

    def record_replay_call(self, run_id, eid, call_id, candidate, result, started_at, elapsed):
        rows = result.get("matches", []) if isinstance(result, dict) else []
        each = elapsed / max(1, len(rows))
        for row in rows:
            self.db.execute("""INSERT INTO replays(
              run_id,experiment_id,replay_call_id,candidate,episode,started_at,
              elapsed_seconds,valid,original_v20_margin,candidate_margin,
              margin_improvement,result,action_divergences,error)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (run_id, eid, call_id, candidate, row.get("episode",""), started_at, each,
               int(bool(row.get("valid"))), row.get("original_v20_margin"),
               row.get("candidate_margin"), row.get("margin_improvement"), row.get("result"),
               row.get("action_divergences"), row.get("error","")))
        summary = result.get("summary", {}) if isinstance(result, dict) else {}
        cases = int(summary.get("games_total", len(rows)) or 0)
        self.db.execute("UPDATE runs SET replay_calls=replay_calls+1,replay_cases=replay_cases+? WHERE run_id=?",
                        (cases, run_id))
        self.db.execute("""UPDATE experiments SET replay_calls=replay_calls+1,replay_cases=replay_cases+?,
          best_margin_improvement=?,mean_margin_improvement=?,wins=?,losses=?,regressions=?
          WHERE experiment_id=?""",
          (cases, summary.get("best_margin_improvement"), summary.get("mean_margin_improvement"),
           summary.get("wins"), summary.get("losses"), summary.get("margin_worsened_cases"), eid))
        self.db.commit()
        return summary

    def project_status(self):
        r = self.db.execute("""SELECT COUNT(*) n,COALESCE(SUM(elapsed_seconds),0) elapsed,
          COALESCE(SUM(api_requests),0) requests,COALESCE(SUM(input_tokens),0) input_tokens,
          COALESCE(SUM(cached_tokens),0) cached_tokens,COALESCE(SUM(output_tokens),0) output_tokens,
          COALESCE(SUM(reasoning_tokens),0) reasoning_tokens,COALESCE(SUM(total_tokens),0) total_tokens,
          COALESCE(SUM(conservative_cost_usd),0) cost FROM runs WHERE status!='RUNNING'""").fetchone()
        e = self.db.execute("""SELECT COUNT(*) n,
          COALESCE(SUM(status='SUPPORTED'),0) supported,
          COALESCE(SUM(status='REJECTED'),0) rejected FROM experiments""").fetchone()
        goal = self.db.execute("SELECT * FROM goals ORDER BY created_at DESC LIMIT 1").fetchone()
        recent = [dict(x) for x in self.db.execute("""SELECT experiment_id,hypothesis,status,
          elapsed_seconds,replay_cases,mean_margin_improvement,regressions,conclusion
          FROM experiments ORDER BY started_at DESC LIMIT 8""").fetchall()]
        return {
          "runs_completed": int(r["n"]), "wall_clock_seconds_sum": float(r["elapsed"]),
          "api_requests": int(r["requests"]), "input_tokens": int(r["input_tokens"]),
          "cached_tokens": int(r["cached_tokens"]), "output_tokens": int(r["output_tokens"]),
          "reasoning_tokens": int(r["reasoning_tokens"]), "total_tokens": int(r["total_tokens"]),
          "conservative_cost_usd": float(r["cost"]), "experiments_total": int(e["n"]),
          "experiments_supported": int(e["supported"]), "experiments_rejected": int(e["rejected"]),
          "replay_cases_total": int(self.db.execute("SELECT COUNT(*) FROM replays").fetchone()[0]),
          "goal": ({**dict(goal),
                    "reached_observability": json.loads(goal["reached_observability_json"])
                     if goal and goal["reached_observability_json"] else None}
                   if goal else None),
          "recent_experiments": recent,
        }


@dataclass
class AppContext:
    run_id: str
    local: LocalTools
    db: ResearchDB
    log: RunLog


def j(x):
    return json.dumps(x, sort_keys=True, default=str)


class BudgetHooks(RunHooks[AppContext]):
    """Per-model-call usage logging and conservative hard-stop before the next call."""

    def __init__(self, log, starting_project_cost, session_limit, total_limit, input_price, output_price):
        self.log = log
        self.starting_project_cost = float(starting_project_cost)
        self.session_limit = float(session_limit)
        self.total_limit = float(total_limit)
        self.input_price = float(input_price)
        self.output_price = float(output_price)
        self.last_usage = {"requests":0,"input_tokens":0,"cached_tokens":0,"cache_write_tokens":0,
                           "output_tokens":0,"reasoning_tokens":0,"total_tokens":0}

    def _cost(self, usage):
        return (usage["input_tokens"] * self.input_price +
                usage["output_tokens"] * self.output_price) / 1_000_000

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        usage = usage_dict(context.usage)
        session_cost = self._cost(usage)
        if session_cost >= self.session_limit:
            raise RuntimeError("session budget ceiling reached before next model call")
        if self.starting_project_cost + session_cost >= self.total_limit:
            raise RuntimeError("project budget ceiling reached before next model call")
        self.log.event("llm_start", {"usage_before_call": usage,
                                     "session_conservative_cost_usd": session_cost})

    async def on_llm_end(self, context, agent, response):
        usage = usage_dict(context.usage)
        self.last_usage = usage
        self.log.event("llm_end", {"usage_after_call": usage,
                                   "session_conservative_cost_usd": self._cost(usage)})


@tool
def list_tree(ctx: RunContextWrapper[AppContext], path: str, max_depth: int = 2, max_entries: int = 200) -> str:
    """List a bounded v23 subtree."""
    return j(ctx.context.local.list_tree(path, max_depth, max_entries))

@tool
def read_text(ctx: RunContextWrapper[AppContext], path: str, start_line: int = 1, max_lines: int = 200) -> str:
    """Read a narrow UTF-8 range from a v23 file."""
    return j(ctx.context.local.read_text(path, start_line, max_lines))

@tool
def search_text(ctx: RunContextWrapper[AppContext], query: str, path: str = ".", max_matches: int = 40) -> str:
    """Search bounded v23 text files."""
    return j(ctx.context.local.search_text(query, path, max_matches))

@tool
def summarize_jsonl(ctx: RunContextWrapper[AppContext], path: str, tail_rows: int = 20) -> str:
    """Summarize local JSONL without sending the full log."""
    return j(ctx.context.local.summarize_jsonl(path, tail_rows))

@tool
def write_workspace_file(ctx: RunContextWrapper[AppContext], path: str, content: str, overwrite: bool = False) -> str:
    """Write only under v23/workspace."""
    return j(ctx.context.local.write_workspace_file(path, content, overwrite))

@tool
def run_python(ctx: RunContextWrapper[AppContext], script: str, args: list[str] | None = None, timeout_seconds: int = 120) -> str:
    """Run an existing non-workspace v23 Python file with a bounded timeout."""
    return j(ctx.context.local.run_python(script, args, timeout_seconds))

@tool
def start_experiment(ctx: RunContextWrapper[AppContext], hypothesis: str, candidate: str = "",
                     parent_candidate: str = "", notes: str = "") -> str:
    """Create a durable experiment record before candidate evaluation."""
    eid = ctx.context.db.start_experiment(ctx.context.run_id, hypothesis, candidate, parent_candidate, notes)
    ctx.context.log.event("experiment_start", {"experiment_id": eid, "hypothesis": hypothesis})
    return j({"experiment_id": eid})

@tool
def finish_experiment(ctx: RunContextWrapper[AppContext], experiment_id: str, status: str, conclusion: str) -> str:
    """Finish an experiment: SUPPORTED, REJECTED, UNRESOLVED, or ERROR."""
    status = status.upper()
    if status not in {"SUPPORTED","REJECTED","UNRESOLVED","ERROR"}:
        return j({"error":"invalid status"})
    out = ctx.context.db.finish_experiment(experiment_id, status, conclusion)
    ctx.context.log.event("experiment_finish", out)
    return j(out)

@tool
def set_goal(ctx: RunContextWrapper[AppContext], metric: str, operator: str, target: float,
             max_regressions: int | None = None) -> str:
    """Set a numeric goal for automatic time/tests/tokens-to-goal accounting."""
    if operator not in {">=",">","<=","<","=="}:
        return j({"error":"invalid operator"})
    gid = ctx.context.db.set_goal(metric, operator, target, max_regressions)
    return j({"goal_id":gid,"metric":metric,"operator":operator,"target":target})

@tool
def project_status(ctx: RunContextWrapper[AppContext]) -> str:
    """Return durable project usage, experiment, replay, and goal status."""
    return j(ctx.context.db.project_status())

@tool
def static_replay_candidate(ctx: RunContextWrapper[AppContext], candidate: str, experiment_id: str,
                            episodes: list[str] | None = None, max_episodes: int = 25) -> str:
    """Run static replay and persist timing and per-case metrics for an experiment."""
    if ctx.context.db.db.execute("SELECT 1 FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone() is None:
        return j({"error":"unknown experiment_id: " + experiment_id})
    call_id = "replay_" + uuid.uuid4().hex[:10]
    started_at, started = utcnow(), time.monotonic()
    result = ctx.context.local.static_replay_candidate(candidate, episodes, max_episodes)
    elapsed = time.monotonic() - started
    summary = ctx.context.db.record_replay_call(
        ctx.context.run_id, experiment_id, call_id, candidate, result, started_at, elapsed)
    reached = ctx.context.db.maybe_reach_goal(
        ctx.context.run_id, experiment_id, summary, usage_dict(ctx.usage))
    result["replay_call_id"] = call_id
    result["elapsed_seconds"] = round(elapsed, 3)
    if reached:
        result["goal_reached"] = reached
    ctx.context.log.event("static_replay", {"experiment_id":experiment_id,"replay_call_id":call_id,
                          "elapsed_seconds":round(elapsed,3),"summary":summary,"goal_reached":reached})
    return j(result)


def usage_dict(u):
    cached = cache_write = reasoning = 0
    for e in getattr(u, "request_usage_entries", []) or []:
        inp, out = getattr(e, "input_tokens_details", None), getattr(e, "output_tokens_details", None)
        cached += int(getattr(inp, "cached_tokens", 0) or 0)
        cache_write += int(getattr(inp, "cache_write_tokens", 0) or 0)
        reasoning += int(getattr(out, "reasoning_tokens", 0) or 0)
    return {"requests":int(getattr(u,"requests",0) or 0),
            "input_tokens":int(getattr(u,"input_tokens",0) or 0),
            "cached_tokens":cached,"cache_write_tokens":cache_write,
            "output_tokens":int(getattr(u,"output_tokens",0) or 0),
            "reasoning_tokens":reasoning,"total_tokens":int(getattr(u,"total_tokens",0) or 0)}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",required=True); p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--reasoning-effort",default="low",choices=["none","low","medium","high","xhigh","max"])
    p.add_argument("--max-turns",type=int,default=12); p.add_argument("--max-output-tokens",type=int,default=2500)
    p.add_argument("--allow-exec",action="store_true"); p.add_argument("--session-id",default=DEFAULT_SESSION_ID)
    p.add_argument("--session-history-limit",type=int,default=80)
    p.add_argument("--total-budget-usd",type=float,default=DEFAULT_TOTAL_BUDGET_USD)
    p.add_argument("--session-budget-usd",type=float,default=DEFAULT_SESSION_BUDGET_USD)
    p.add_argument("--input-usd-per-m",type=float); p.add_argument("--output-usd-per-m",type=float)
    p.add_argument("--disable-tracing",action="store_true")
    return p.parse_args()


def pricing_for(args):
    if (args.input_usd_per_m is None)!=(args.output_usd_per_m is None):
        raise SystemExit("pass both explicit token prices")
    if args.input_usd_per_m is not None:
        return float(args.input_usd_per_m),float(args.output_usd_per_m)
    if args.model not in MODEL_PRICING_USD_PER_M:
        raise SystemExit("unknown model pricing; pass explicit token prices")
    return MODEL_PRICING_USD_PER_M[args.model]


def main():
    args=parse_args()
    if not os.getenv("OPENAI_API_KEY"): raise SystemExit("OPENAI_API_KEY is not set")
    if not 1<=args.max_turns<=50: raise SystemExit("--max-turns must be 1..50")
    if not 1<=args.session_history_limit<=500: raise SystemExit("--session-history-limit must be 1..500")
    if not 0<args.session_budget_usd<=args.total_budget_usd: raise SystemExit("invalid budget ceilings")
    inp_price,out_price=pricing_for(args)
    WORKSPACE.mkdir(parents=True,exist_ok=True)
    run_id="run_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    config={"version":2,"run_id":run_id,"task":args.task,"model":args.model,
            "session_id":args.session_id,"max_turns":args.max_turns,
            "tracing_enabled":not args.disable_tracing,"trace_sensitive_data":False}
    log=RunLog(WORKSPACE,config); local=LocalTools(allow_exec=args.allow_exec,log=log)
    db=ResearchDB(STATE_DB); db.start_run(run_id,args.session_id,args.task,args.model)
    app=AppContext(run_id,local,db,log)
    budget=BudgetLedger(LEDGER_PATH,model=args.model,input_usd_per_m=inp_price,
        output_usd_per_m=out_price,total_budget_usd=args.total_budget_usd,
        session_budget_usd=args.session_budget_usd)
    if not budget.can_call(): raise SystemExit("budget ceiling already reached")

    agent=Agent[AppContext](name="v23 Kaggriculture Research Agent",instructions=SYSTEM_PROMPT,
      model=args.model,model_settings=ModelSettings(reasoning=Reasoning(effort=args.reasoning_effort),
      max_tokens=args.max_output_tokens,verbosity="low",parallel_tool_calls=False),
      tools=[list_tree,read_text,search_text,summarize_jsonl,write_workspace_file,run_python,
             start_experiment,finish_experiment,set_goal,project_status,static_replay_candidate])
    session=SQLiteSession(args.session_id,str(SESSION_DB))
    prompt=("Research task:\n"+args.task+"\n\nExecution enabled: "+str(args.allow_exec)+
            ". Conservative per-run ceiling: USD "+format(args.session_budget_usd,".2f")+
            "; remaining project ledger: USD "+format(budget.remaining_total(),".4f")+
            ". Use durable experiment records and minimize model turns.")
    hooks=BudgetHooks(log,budget.total.estimated_cost_usd,args.session_budget_usd,
                      args.total_budget_usd,inp_price,out_price)
    started=time.monotonic(); status="DONE"; output=""
    try:
        result=Runner.run_sync(agent,prompt,context=app,session=session,max_turns=args.max_turns,
          hooks=hooks,
          run_config=RunConfig(workflow_name="v23 autonomous research",group_id=args.session_id,
          trace_include_sensitive_data=False,tracing_disabled=args.disable_tracing,
          trace_metadata={"run_id":run_id,"model":args.model},
          session_settings=SessionSettings(limit=args.session_history_limit)))
        output=str(result.final_output or ""); usage=usage_dict(result.context_wrapper.usage)
    except Exception as exc:
        status="ERROR"; output=type(exc).__name__+": "+str(exc)
        usage=dict(hooks.last_usage)
    elapsed=time.monotonic()-started
    cost=(usage["input_tokens"]*inp_price+usage["output_tokens"]*out_price)/1_000_000
    delta=LegacyUsage(usage["input_tokens"],usage["output_tokens"],usage["requests"],cost)
    budget.session.add(delta); budget.total.add(delta)
    LEDGER_PATH.write_text(json.dumps({**vars(budget.total),"model_last_used":args.model,
      "updated_at_utc":utcnow(),"cached_tokens_observed_last_run":usage["cached_tokens"],
      "cache_write_tokens_observed_last_run":usage["cache_write_tokens"],
      "reasoning_tokens_observed_last_run":usage["reasoning_tokens"],
      "pricing_assumption":{"input_usd_per_m":inp_price,"output_usd_per_m":out_price,
                            "cached_tokens_priced_as_uncached":True}},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    db.finish_run(run_id,status,usage,cost,output,elapsed)
    log.event("run_summary",{"run_id":run_id,"elapsed_seconds":round(elapsed,3),
              "usage":usage,"conservative_cost_usd":cost,"project_status":db.project_status()})
    log.final(output)
    print(output); print("\n[v23 observability]")
    print(j({"run_id":run_id,"elapsed_seconds":round(elapsed,3),"usage":usage,
             "conservative_cost_usd":round(cost,8),"session_id":args.session_id,
             "experiments_db":str(STATE_DB),"session_db":str(SESSION_DB),
             "trace_enabled":not args.disable_tracing}))
    print("[v23 run] "+str(log.dir))
    return 0 if status=="DONE" else 2


if __name__=="__main__":
    raise SystemExit(main())
