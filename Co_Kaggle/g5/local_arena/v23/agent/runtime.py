#!/usr/bin/env python3
"""Agents-SDK runtime, durable research memory, and observability for v23."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings, RunConfig, RunContextWrapper, RunHooks, Runner, SQLiteSession, SessionSettings, function_tool, set_default_openai_key
from openai.types.shared import Reasoning

from .support import (
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
- Call set_goal when the task has an explicit numeric target. If the target is defined over a fixed corpus (for example wins across all 25 histories), set min_games_total to that corpus size so a subset cannot satisfy the final goal.
- Goal completion comes only from structured replay metrics, never prose.
- Model-written candidate policies may execute only through static_replay_candidate.
- When execution is enabled, do not stop at analysis or planning. You must create/select a concrete candidate, start an experiment, execute at least one static replay, analyze the measured result, and finish the experiment unless a concrete runtime error or budget stop blocks execution.
- A rejected candidate is evidence, not a blocker. If the durable goal is unmet, continue with the next justified experiment.
- After four consecutive rejected experiments with no wins and no positive mean margin improvement, treat the search as stagnant: abandon the current tweak family, re-inspect raw loss evidence, and move to a different causal layer (for example worker actions/task ranking/logistics/planning/inventory/market). Do not keep making parameter variants of the same idea.
- Do not use one fixed 5-game screen forever. Rotate or stratify screening histories when a screen repeatedly rejects candidates, and periodically run the strongest candidate family on all 25 histories because local replay is cheaper than additional model reasoning.
- Only call report_blocker for a concrete runtime/environment failure that makes further research impossible without external intervention.
- FP16 is mandatory for candidate floating-point model weights, activations, tensors, and learned numeric compute. Integer/boolean/schema-mandated types are exempt. Do not emit float32, float64, double, or bfloat16 candidate model/tensor code unless a backend operation is provably unsupported in FP16; any such exception must be narrowly scoped, documented, and converted back to FP16 immediately.
"""

PERFORMANCE_ANALYST_PROMPT = """You are the independent v23 Performance Analyst.

You are read-only. You do not write candidate code and you do not execute replay.
Your job is to diagnose why research is or is not improving and give the Experiment
Engineer a higher-information direction.

Use the available read-only tools selectively:
- call project_status first;
- inspect recent experiment/replay evidence;
- inspect v20 model structure and raw loss histories only where needed;
- distinguish worker/task-ranking, movement/logistics, crop/animal planning,
  inventory/capacity, market/selling, hiring/purchases, and other causal layers;
- treat rejected hypotheses as negative evidence;
- if recent search is stagnant, explicitly move away from the repeated hypothesis
  family instead of proposing another parameter tweak.

Return a concise review with:
1. PERFORMANCE EVIDENCE: what the measurements actually show.
2. FAILURE MECHANISMS: 1-3 likely causal bottlenecks, with evidence.
3. DO-NOT-REPEAT: recent idea families that evidence argues against.
4. IMPROVEMENT OPTIONS: up to 3 structurally distinct ideas.
5. NEXT EXPERIMENT: select exactly one idea, state a falsifiable prediction, the
   smallest useful replay screen, and what result would justify expansion to 5/25
   and then all 25 histories.

Do not claim improvement without measured replay evidence. Keep the output compact.
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
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
          max_regressions INTEGER, min_games_total INTEGER,
          created_at TEXT, reached_at TEXT,
          reached_run_id TEXT, reached_experiment_id TEXT, reached_value REAL,
          reached_observability_json TEXT, active INTEGER DEFAULT 1);
        CREATE TABLE IF NOT EXISTS strategy_reviews(
          review_id TEXT PRIMARY KEY, run_id TEXT, created_at TEXT, trigger TEXT,
          experiments_seen INTEGER, replay_cases_seen INTEGER,
          analyst_output TEXT, usage_json TEXT, conservative_cost_usd REAL);
        """)
        goal_columns = {row[1] for row in self.db.execute("PRAGMA table_info(goals)").fetchall()}
        if "reached_observability_json" not in goal_columns:
            self.db.execute("ALTER TABLE goals ADD COLUMN reached_observability_json TEXT")
        if "min_games_total" not in goal_columns:
            self.db.execute("ALTER TABLE goals ADD COLUMN min_games_total INTEGER")
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

    def set_goal(self, metric, operator, target, max_regressions, min_games_total=None):
        self.db.execute("UPDATE goals SET active=0 WHERE active=1")
        gid = "goal_" + uuid.uuid4().hex[:10]
        self.db.execute("""INSERT INTO goals(
                           goal_id,metric,operator,target,max_regressions,min_games_total,
                           created_at,active)
                           VALUES(?,?,?,?,?,?,?,1)""",
                        (gid, metric, operator, float(target), max_regressions,
                         min_games_total, utcnow()))
        self.db.commit()
        return gid

    def maybe_reach_goal(self, run_id, eid, summary, usage_snapshot, input_price, output_price):
        goal = self.db.execute("SELECT * FROM goals WHERE active=1 ORDER BY created_at DESC LIMIT 1").fetchone()
        if goal is None or goal["metric"] not in summary:
            return None
        value = summary.get(goal["metric"])
        if not isinstance(value, (int, float)):
            return None
        regressions = int(summary.get("margin_worsened_cases", 0) or 0)
        if goal["max_regressions"] is not None and regressions > int(goal["max_regressions"]):
            return None
        games_total = int(summary.get("games_total", 0) or 0)
        if goal["min_games_total"] is not None and games_total < int(goal["min_games_total"]):
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
          "conservative_cost_usd_to_goal": (
              float(prior["cost"])
              + conservative_cost_usd(usage_snapshot, input_price, output_price)
          ),
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
              (run_id, eid, call_id, candidate, row.get("episode",""), started_at,
               float(row.get("elapsed_seconds", each)),
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

    def record_strategy_review(self, run_id, trigger, analyst_output,
                               usage=None, conservative_cost_usd=0.0):
        rid = "review_" + uuid.uuid4().hex[:10]
        experiments_seen = int(
            self.db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        )
        replay_cases_seen = int(
            self.db.execute("SELECT COUNT(*) FROM replays").fetchone()[0]
        )
        self.db.execute(
            """INSERT INTO strategy_reviews(
                 review_id,run_id,created_at,trigger,experiments_seen,
                 replay_cases_seen,analyst_output,usage_json,conservative_cost_usd)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                rid, run_id, utcnow(), trigger, experiments_seen,
                replay_cases_seen, analyst_output,
                json.dumps(usage or {}, sort_keys=True),
                float(conservative_cost_usd),
            ),
        )
        self.db.commit()
        return rid

    def latest_strategy_review(self):
        row = self.db.execute(
            """SELECT * FROM strategy_reviews
               ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
        return dict(row) if row is not None else None

    def strategy_review_trigger(self):
        latest = self.latest_strategy_review()
        signal = self.research_signal()
        experiments_total = int(
            self.db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        )
        if latest is None:
            return "initial_diagnosis"
        if signal["stagnating"]:
            return "stagnation"
        if experiments_total - int(latest["experiments_seen"] or 0) >= 3:
            return "periodic_after_3_experiments"
        return None

    def research_signal(self, recent_limit=6):
        recent_limit = max(4, min(int(recent_limit), 20))
        rows = [dict(x) for x in self.db.execute(
            """SELECT experiment_id,hypothesis,status,wins,losses,replay_cases,
                      mean_margin_improvement,best_margin_improvement,started_at
               FROM experiments
               WHERE status!='RUNNING'
               ORDER BY started_at DESC LIMIT ?""",
            (recent_limit,),
        ).fetchall()]

        consecutive_rejected = 0
        for row in rows:
            if row.get("status") == "REJECTED":
                consecutive_rejected += 1
            else:
                break

        positive_recent = any(
            int(row.get("wins") or 0) > 0
            or float(row.get("mean_margin_improvement") or 0.0) > 0.0
            for row in rows
        )
        no_effect_recent = sum(
            1 for row in rows
            if row.get("mean_margin_improvement") is not None
            and abs(float(row["mean_margin_improvement"])) < 1e-9
        )
        best = self.db.execute(
            """SELECT COALESCE(MAX(wins),0) best_wins,
                      COALESCE(MAX(mean_margin_improvement),0) best_mean_margin
               FROM experiments WHERE status!='RUNNING'"""
        ).fetchone()

        return {
            "recent_count": len(rows),
            "consecutive_rejected": consecutive_rejected,
            "positive_recent": positive_recent,
            "no_effect_recent": no_effect_recent,
            "best_wins": int(best["best_wins"] or 0),
            "best_mean_margin_improvement": float(best["best_mean_margin"] or 0.0),
            "stagnating": (
                len(rows) >= 4
                and consecutive_rejected >= 4
                and not positive_recent
            ),
            "recent_hypotheses": [
                row.get("hypothesis", "") for row in rows[:4]
            ],
        }

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
          "research_signal": self.research_signal(),
          "latest_strategy_review": (
              {
                  "review_id": self.latest_strategy_review()["review_id"],
                  "created_at": self.latest_strategy_review()["created_at"],
                  "trigger": self.latest_strategy_review()["trigger"],
                  "experiments_seen": self.latest_strategy_review()["experiments_seen"],
                  "replay_cases_seen": self.latest_strategy_review()["replay_cases_seen"],
                  "analyst_output": (
                      self.latest_strategy_review()["analyst_output"][:3000]
                      if self.latest_strategy_review()["analyst_output"] else ""
                  ),
              }
              if self.latest_strategy_review() else None
          ),
        }


@dataclass
class AppContext:
    run_id: str
    local: LocalTools
    db: ResearchDB
    log: RunLog
    input_price: float
    output_price: float
    replay_python: str
    blocker_reason: str = ""


def j(x):
    return json.dumps(x, sort_keys=True, default=str)


class BudgetStopError(RuntimeError):
    """Normal autonomous stop when a configured API budget ceiling is reached."""


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
        return conservative_cost_usd(usage, self.input_price, self.output_price)

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        usage = usage_dict(context.usage)
        session_cost = self._cost(usage)
        if session_cost >= self.session_limit:
            raise BudgetStopError("session budget ceiling reached before next model call")
        if self.starting_project_cost + session_cost >= self.total_limit:
            raise BudgetStopError("project budget ceiling reached before next model call")
        self.log.event("llm_start", {"usage_before_call": usage,
                                     "session_conservative_cost_usd": session_cost})

    async def on_llm_end(self, context, agent, response):
        usage = usage_dict(context.usage)
        self.last_usage = usage
        self.log.event("llm_end", {"usage_after_call": usage,
                                   "session_conservative_cost_usd": self._cost(usage)})


@function_tool
def list_tree(ctx: RunContextWrapper[AppContext], path: str, max_depth: int = 2, max_entries: int = 200) -> str:
    """List a bounded v23 subtree."""
    return j(ctx.context.local.list_tree(path, max_depth, max_entries))

@function_tool
def read_text(ctx: RunContextWrapper[AppContext], path: str, start_line: int = 1, max_lines: int = 200) -> str:
    """Read a narrow UTF-8 range from a v23 file."""
    return j(ctx.context.local.read_text(path, start_line, max_lines))

@function_tool
def search_text(ctx: RunContextWrapper[AppContext], query: str, path: str = ".", max_matches: int = 40) -> str:
    """Search bounded v23 text files."""
    return j(ctx.context.local.search_text(query, path, max_matches))

@function_tool
def summarize_jsonl(ctx: RunContextWrapper[AppContext], path: str, tail_rows: int = 20) -> str:
    """Summarize local JSONL without sending the full log."""
    return j(ctx.context.local.summarize_jsonl(path, tail_rows))

@function_tool
def write_workspace_file(ctx: RunContextWrapper[AppContext], path: str, content: str, overwrite: bool = False) -> str:
    """Write only under v23/workspace."""
    return j(ctx.context.local.write_workspace_file(path, content, overwrite))

@function_tool
def run_python(ctx: RunContextWrapper[AppContext], script: str, args: list[str] | None = None, timeout_seconds: int = 120) -> str:
    """Run an existing non-workspace v23 Python file with a bounded timeout."""
    return j(ctx.context.local.run_python(script, args, timeout_seconds))

@function_tool
def start_experiment(ctx: RunContextWrapper[AppContext], hypothesis: str, candidate: str = "",
                     parent_candidate: str = "", notes: str = "") -> str:
    """Create a durable experiment record before candidate evaluation."""
    eid = ctx.context.db.start_experiment(ctx.context.run_id, hypothesis, candidate, parent_candidate, notes)
    ctx.context.log.event("experiment_start", {"experiment_id": eid, "hypothesis": hypothesis})
    return j({"experiment_id": eid})

@function_tool
def finish_experiment(ctx: RunContextWrapper[AppContext], experiment_id: str, status: str, conclusion: str) -> str:
    """Finish an experiment: SUPPORTED, REJECTED, UNRESOLVED, or ERROR."""
    status = status.upper()
    if status not in {"SUPPORTED","REJECTED","UNRESOLVED","ERROR"}:
        return j({"error":"invalid status"})
    out = ctx.context.db.finish_experiment(experiment_id, status, conclusion)
    ctx.context.log.event("experiment_finish", out)
    return j(out)

@function_tool
def set_goal(ctx: RunContextWrapper[AppContext], metric: str, operator: str, target: float,
             max_regressions: int | None = None,
             min_games_total: int | None = None) -> str:
    """Set a numeric goal plus optional regression and evaluation-size guards."""
    if operator not in {">=",">","<=","<","=="}:
        return j({"error":"invalid operator"})
    if min_games_total is not None and min_games_total < 1:
        return j({"error":"min_games_total must be >= 1"})
    gid = ctx.context.db.set_goal(
        metric, operator, target, max_regressions, min_games_total
    )
    return j({
        "goal_id":gid,"metric":metric,"operator":operator,"target":target,
        "max_regressions":max_regressions,"min_games_total":min_games_total
    })

@function_tool
def project_status(ctx: RunContextWrapper[AppContext]) -> str:
    """Return durable project usage, experiment, replay, and goal status."""
    return j(ctx.context.db.project_status())

@function_tool
def report_blocker(ctx: RunContextWrapper[AppContext], failed_step: str, reason: str) -> str:
    """Report a concrete runtime/environment blocker that makes further research impossible."""
    failed_step = failed_step.strip()
    reason = reason.strip()
    if not failed_step or not reason:
        return j({"error":"failed_step and reason are required"})
    ctx.context.blocker_reason = failed_step + ": " + reason
    ctx.context.log.event("research_blocker", {
        "failed_step": failed_step,
        "reason": reason,
    })
    return j({"blocker_recorded": True, "failed_step": failed_step, "reason": reason})


@function_tool
def static_replay_candidate(ctx: RunContextWrapper[AppContext], candidate: str, experiment_id: str,
                            episodes: list[str] | None = None, max_episodes: int = 25) -> str:
    """Run static replay in an isolated child process and persist per-case metrics."""
    if not ctx.context.local.allow_exec:
        return j({"error":"execution disabled; rerun with --allow-exec"})
    if ctx.context.db.db.execute(
        "SELECT 1 FROM experiments WHERE experiment_id=?", (experiment_id,)
    ).fetchone() is None:
        return j({"error":"unknown experiment_id: " + experiment_id})

    try:
        candidate_path = ctx.context.local._read_path(candidate)
    except Exception as exc:
        return j({"error":f"{type(exc).__name__}: {exc}"})
    if not candidate_path.is_file() or candidate_path.suffix not in {".py",".ipynb"}:
        return j({"error":"candidate must be an existing .py or .ipynb under v23"})

    call_id = "replay_" + uuid.uuid4().hex[:10]
    started_at, started = utcnow(), time.monotonic()
    runner = ctx.context.local.root / "replay" / "runner.py"
    child_env = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(secret in upper for secret in ("OPENAI_API_KEY","TOKEN","SECRET","PASSWORD","CREDENTIAL")):
            continue
        child_env[key] = value
    child_env["PYTHONUNBUFFERED"] = "1"

    argv = [
        ctx.context.replay_python, str(runner),
        "--candidate", str(candidate_path.relative_to(ctx.context.local.root)),
        "--episodes-json", json.dumps(episodes or []),
        "--max-episodes", str(max(1, min(int(max_episodes), 50))),
    ]
    try:
        completed = subprocess.run(
            argv,
            cwd=ctx.context.local.root,
            capture_output=True,
            text=True,
            timeout=300,
            env=child_env,
        )
        marker = "__V23_RESULT__"
        payload_line = next(
            (line[len(marker):] for line in reversed(completed.stdout.splitlines())
             if line.startswith(marker)),
            None,
        )
        if payload_line is None:
            result = {
                "error":"static replay child did not emit a result marker",
                "returncode":completed.returncode,
                "stdout":completed.stdout[-4000:],
                "stderr":completed.stderr[-4000:],
            }
        else:
            result = json.loads(payload_line)
            if completed.returncode != 0 and "error" not in result:
                result["error"] = f"static replay child exited {completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        result = {
            "error":"static replay timeout",
            "timeout_seconds":300,
            "stdout":exc.stdout[-4000:] if isinstance(exc.stdout,str) else "",
            "stderr":exc.stderr[-4000:] if isinstance(exc.stderr,str) else "",
        }
    except Exception as exc:
        result = {"error":f"{type(exc).__name__}: {exc}"}

    elapsed = time.monotonic() - started
    summary = ctx.context.db.record_replay_call(
        ctx.context.run_id, experiment_id, call_id, candidate, result, started_at, elapsed)
    reached = ctx.context.db.maybe_reach_goal(
        ctx.context.run_id, experiment_id, summary, usage_dict(ctx.usage),
        ctx.context.input_price, ctx.context.output_price)
    result["replay_call_id"] = call_id
    result["elapsed_seconds"] = round(elapsed, 3)
    if reached:
        result["goal_reached"] = reached
    ctx.context.log.event("static_replay", {
        "experiment_id":experiment_id,
        "replay_call_id":call_id,
        "candidate":candidate,
        "elapsed_seconds":round(elapsed,3),
        "summary":summary,
        "goal_reached":reached,
        "child_returncode": result.get("returncode"),
        "replay_python": ctx.context.replay_python,
    })
    return j(result)


def conservative_cost_usd(usage, input_price, output_price):
    """Estimate Standard-tier cost including prompt-cache reads/writes, then add 10% headroom."""
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached_tokens = int(usage.get("cached_tokens", 0) or 0)
    cache_write_tokens = int(usage.get("cache_write_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    uncached_tokens = max(0, input_tokens - cached_tokens - cache_write_tokens)
    raw = (
        uncached_tokens * float(input_price)
        + cached_tokens * float(input_price) * 0.10
        + cache_write_tokens * float(input_price) * 1.25
        + output_tokens * float(output_price)
    ) / 1_000_000.0
    return raw * 1.10


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


def add_usage(total: dict[str, int], delta: dict[str, int]) -> None:
    for key in total:
        total[key] += int(delta.get(key, 0) or 0)


def latest_goal_state(db: ResearchDB) -> dict[str, Any] | None:
    row = db.db.execute(
        "SELECT * FROM goals ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row is not None else None


def autonomous_stop_reason(goal: dict[str, Any] | None,
                           blocker_reason: str,
                           allow_exec: bool) -> str | None:
    if goal and goal.get("reached_at"):
        return "goal_reached"
    if blocker_reason:
        return "reported_blocker"
    if not allow_exec:
        return "execution_disabled"
    return None


def persist_budget_ledger(budget: BudgetLedger, model: str, usage: dict[str, int],
                          input_price: float, output_price: float) -> None:
    LEDGER_PATH.write_text(
        json.dumps({
            **vars(budget.total),
            "model_last_used": model,
            "updated_at_utc": utcnow(),
            "cached_tokens_observed_last_run": usage["cached_tokens"],
            "cache_write_tokens_observed_last_run": usage["cache_write_tokens"],
            "reasoning_tokens_observed_last_run": usage["reasoning_tokens"],
            "pricing_assumption": {
                "input_usd_per_m": input_price,
                "output_usd_per_m": output_price,
                "cached_input_multiplier": 0.10,
                "cache_write_multiplier": 1.25,
                "safety_multiplier": 1.10,
            },
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",required=True); p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key for this agent run. Falls back to OPENAI_API_KEY.",
    )
    p.add_argument("--reasoning-effort",default="low",choices=["none","low","medium","high","xhigh","max"])
    p.add_argument("--max-turns",type=int,default=12); p.add_argument("--max-output-tokens",type=int,default=2500)
    p.add_argument("--allow-exec",action="store_true"); p.add_argument("--session-id",default=DEFAULT_SESSION_ID)
    p.add_argument("--session-history-limit",type=int,default=80)
    p.add_argument("--total-budget-usd",type=float,default=DEFAULT_TOTAL_BUDGET_USD)
    p.add_argument("--session-budget-usd",type=float,default=DEFAULT_SESSION_BUDGET_USD)
    p.add_argument("--input-usd-per-m",type=float); p.add_argument("--output-usd-per-m",type=float)
    p.add_argument("--disable-tracing",action="store_true")
    p.add_argument(
        "--replay-python",
        default=os.getenv("V23_REPLAY_PYTHON"),
        help="Python interpreter from the isolated replay environment. "
             "Defaults to V23_REPLAY_PYTHON or v23/.venv-replay.",
    )
    return p.parse_args()


def resolve_replay_python(value: str | None) -> str:
    """Resolve the isolated replay interpreter without importing Kaggle into the agent env."""
    candidates = []
    if value:
        candidates.append(Path(value).expanduser())
    root = Path(__file__).resolve().parents[1]
    if os.name == "nt":
        candidates.append(root / ".venv-replay" / "Scripts" / "python.exe")
    else:
        candidates.append(root / ".venv-replay" / "bin" / "python")

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            if candidate == Path(sys.executable).resolve():
                raise SystemExit(
                    "replay interpreter must be separate from the agent interpreter; "
                    "create .venv-replay or pass --replay-python"
                )
            return str(candidate)

    raise SystemExit(
        "isolated replay Python not found. Create v23/.venv-replay from "
        "requirements-replay.txt or pass --replay-python / V23_REPLAY_PYTHON."
    )


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
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OpenAI API key is required: pass --api-key or set OPENAI_API_KEY"
        )
    api_key_source = "cli" if args.api_key else "environment"
    set_default_openai_key(api_key)
    args.api_key = None
    if not 1<=args.max_turns<=50: raise SystemExit("--max-turns must be 1..50")
    if not 256<=args.max_output_tokens<=20000: raise SystemExit("--max-output-tokens must be 256..20000")
    if not 1<=args.session_history_limit<=500: raise SystemExit("--session-history-limit must be 1..500")
    if not 0<args.session_budget_usd<=args.total_budget_usd: raise SystemExit("invalid budget ceilings")
    inp_price,out_price=pricing_for(args)
    replay_python=resolve_replay_python(args.replay_python) if args.allow_exec else ""
    WORKSPACE.mkdir(parents=True,exist_ok=True)
    run_id="run_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    config={"version":2,"run_id":run_id,"task":args.task,"model":args.model,
            "api_key_source":api_key_source,
            "session_id":args.session_id,"max_turns":args.max_turns,
            "replay_python":replay_python or None,
            "tracing_enabled":not args.disable_tracing,"trace_sensitive_data":False}
    log=RunLog(WORKSPACE,config); local=LocalTools(allow_exec=args.allow_exec,log=log)
    db=ResearchDB(STATE_DB); db.start_run(run_id,args.session_id,args.task,args.model)
    app=AppContext(run_id,local,db,log,inp_price,out_price,replay_python)
    budget=BudgetLedger(LEDGER_PATH,model=args.model,input_usd_per_m=inp_price,
        output_usd_per_m=out_price,total_budget_usd=args.total_budget_usd,
        session_budget_usd=args.session_budget_usd)
    if not budget.can_call(): raise SystemExit("budget ceiling already reached")

    agent=Agent[AppContext](name="v23 Kaggriculture Research Agent",instructions=SYSTEM_PROMPT,
      model=args.model,model_settings=ModelSettings(reasoning=Reasoning(effort=args.reasoning_effort),
      max_tokens=args.max_output_tokens,verbosity="low",parallel_tool_calls=False,
      store=False,prompt_cache_options={"mode":"implicit","ttl":"30m"}),
      tools=[list_tree,read_text,search_text,summarize_jsonl,write_workspace_file,run_python,
             start_experiment,finish_experiment,set_goal,project_status,report_blocker,
             static_replay_candidate])
    session=SQLiteSession(args.session_id,str(SESSION_DB))
    prompt=("Research task:\n"+args.task+"\n\nExecution enabled: "+str(args.allow_exec)+
            ". Conservative per-run ceiling: USD "+format(args.session_budget_usd,".2f")+
            "; remaining project ledger: USD "+format(budget.remaining_total(),".4f")+
            ". Use durable experiment records and minimize model turns.")
    started=time.monotonic(); status="DONE"; output=""
    usage={"requests":0,"input_tokens":0,"cached_tokens":0,"cache_write_tokens":0,
           "output_tokens":0,"reasoning_tokens":0,"total_tokens":0}
    cycle = 0
    continuation = prompt

    while True:
        if not budget.can_call():
            status = "BUDGET_STOP"
            output = (
                output.rstrip()
                + "\n\nAutonomous loop stopped because the configured API budget "
                  "ceiling was reached before the durable goal."
            ).strip()
            log.event("autonomous_stop", {"reason":"budget","cycle":cycle})
            break

        remaining_session_budget = max(
            0.0, args.session_budget_usd - budget.session.estimated_cost_usd
        )
        if remaining_session_budget <= 0:
            status = "BUDGET_STOP"
            output = (
                output.rstrip()
                + "\n\nAutonomous loop stopped because the per-run API budget "
                  "ceiling was reached before the durable goal."
            ).strip()
            log.event("autonomous_stop", {"reason":"session_budget","cycle":cycle})
            break

        cycle += 1
        hooks=BudgetHooks(
            log,
            budget.total.estimated_cost_usd,
            remaining_session_budget,
            args.total_budget_usd,
            inp_price,
            out_price,
        )
        cycle_output = ""
        cycle_usage = {
            "requests":0,"input_tokens":0,"cached_tokens":0,"cache_write_tokens":0,
            "output_tokens":0,"reasoning_tokens":0,"total_tokens":0
        }
        try:
            result=Runner.run_sync(
              agent, continuation, context=app, session=session,
              max_turns=args.max_turns, hooks=hooks,
              run_config=RunConfig(
                workflow_name="v23 autonomous research",
                group_id=args.session_id,
                trace_include_sensitive_data=False,
                tracing_disabled=args.disable_tracing,
                trace_metadata={"run_id":run_id,"model":args.model,"cycle":cycle},
                session_settings=SessionSettings(limit=args.session_history_limit),
              ),
            )
            cycle_output=str(result.final_output or "")
            cycle_usage=usage_dict(result.context_wrapper.usage)
        except Exception as exc:
            cycle_output=type(exc).__name__+": "+str(exc)
            cycle_usage=dict(hooks.last_usage)
            exc_name=type(exc).__name__
            if isinstance(exc, BudgetStopError):
                add_usage(usage, cycle_usage)
                cycle_cost=conservative_cost_usd(cycle_usage,inp_price,out_price)
                delta=LegacyUsage(
                    cycle_usage["input_tokens"],cycle_usage["output_tokens"],
                    cycle_usage["requests"],cycle_cost
                )
                budget.session.add(delta); budget.total.add(delta)
                persist_budget_ledger(budget,args.model,usage,inp_price,out_price)
                status="BUDGET_STOP"
                output=cycle_output
                log.event("autonomous_stop", {
                    "reason":"budget_hook","cycle":cycle,"detail":cycle_output
                })
                break
            if exc_name not in {"MaxTurnsExceeded"}:
                status="ERROR"
                output=cycle_output
                add_usage(usage, cycle_usage)
                cycle_cost=conservative_cost_usd(cycle_usage,inp_price,out_price)
                delta=LegacyUsage(
                    cycle_usage["input_tokens"],cycle_usage["output_tokens"],
                    cycle_usage["requests"],cycle_cost
                )
                budget.session.add(delta); budget.total.add(delta)
                persist_budget_ledger(budget,args.model,usage,inp_price,out_price)
                log.event("autonomous_stop", {
                    "reason":"runtime_error","cycle":cycle,"error":cycle_output
                })
                break

        add_usage(usage, cycle_usage)
        cycle_cost=conservative_cost_usd(cycle_usage,inp_price,out_price)
        delta=LegacyUsage(
            cycle_usage["input_tokens"],cycle_usage["output_tokens"],
            cycle_usage["requests"],cycle_cost
        )
        budget.session.add(delta); budget.total.add(delta)
        persist_budget_ledger(budget,args.model,usage,inp_price,out_price)
        output=cycle_output or output

        goal=latest_goal_state(db)
        goal_reached=bool(goal and goal.get("reached_at"))
        signal=db.research_signal()
        log.event("autonomous_cycle", {
            "cycle":cycle,
            "cycle_output":cycle_output,
            "cycle_usage":cycle_usage,
            "cycle_conservative_cost_usd":cycle_cost,
            "goal":goal,
            "goal_reached":goal_reached,
            "research_signal":signal,
        })

        stop_reason=autonomous_stop_reason(goal, app.blocker_reason, args.allow_exec)
        if stop_reason == "goal_reached":
            status="DONE"
            output = (
                cycle_output.rstrip()
                + "\n\nDurable goal reached; autonomous research loop stopped."
            ).strip()
            break

        if stop_reason == "reported_blocker":
            status="BLOCKED"
            output = (
                cycle_output.rstrip()
                + "\n\nAutonomous research loop stopped on recorded blocker: "
                + app.blocker_reason
            ).strip()
            log.event("autonomous_stop", {
                "reason":"reported_blocker","cycle":cycle,
                "blocker":app.blocker_reason,
            })
            break

        if stop_reason == "execution_disabled":
            status="DONE"
            break

        if signal["stagnating"]:
            continuation=(
                "STRATEGY RESET REQUIRED. The durable goal is still unmet and the recent "
                "search is stagnant: at least four consecutive rejected experiments have "
                "produced neither a win nor positive mean margin improvement. Do NOT make "
                "another small variant of the recent hypotheses. Re-inspect raw v20 loss "
                "evidence and v20 policy logic, identify a different causal layer, and run "
                "a structurally different experiment. Consider worker actions/task ranking/"
                "logistics/planning/inventory/market rather than staying in one family. "
                "Rotate or stratify the screening histories if the same small screen has "
                "been reused. Use project_status and the recent hypotheses as negative "
                "evidence, then execute the next experiment. Recent hypotheses: "
                + j(signal["recent_hypotheses"])
            )
        else:
            continuation=(
                "Continue the SAME research task and session. The durable goal is still "
                "unmet. Do not stop merely because a candidate was rejected, one experiment "
                "finished, or you have a recommendation for the next step. Inspect "
                "project_status, use prior negative results, create the next justified "
                "candidate/experiment, execute static replay, and continue making measured "
                "progress. Only a reached durable goal, exhausted configured API budget, or "
                "a concrete runtime blocker may terminate the autonomous loop."
            )

    elapsed=time.monotonic()-started
    if status=="DONE" and args.allow_exec:
        progress = db.db.execute(
            "SELECT replay_calls,replay_cases,experiments_started FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        valid_replays = int(db.db.execute(
            "SELECT COUNT(*) FROM replays WHERE run_id=? AND valid=1",
            (run_id,),
        ).fetchone()[0])
        open_experiments = int(db.db.execute(
            "SELECT COUNT(*) FROM experiments WHERE run_id=? AND status='RUNNING'",
            (run_id,),
        ).fetchone()[0])
        contract_ok = (
            progress is not None
            and int(progress["replay_calls"] or 0) > 0
            and valid_replays > 0
            and open_experiments == 0
        )
        if not contract_ok:
            status="INCOMPLETE"
            output = (
                output.rstrip()
                + "\n\nExecution contract violation: --allow-exec was enabled, "
                  "but the run did not complete a valid static-replay experiment. "
                  "A successful run requires at least one valid replay case and no "
                  "unfinished experiment. The run is marked INCOMPLETE."
            )
            log.event("execution_contract_violation", {
                "run_id": run_id,
                "experiments_started": int(progress["experiments_started"] or 0) if progress else 0,
                "replay_calls": int(progress["replay_calls"] or 0) if progress else 0,
                "replay_cases": int(progress["replay_cases"] or 0) if progress else 0,
                "valid_replay_cases": valid_replays,
                "open_experiments": open_experiments,
            })
    cost=conservative_cost_usd(usage,inp_price,out_price)
    db.finish_run(run_id,status,usage,cost,output,elapsed)
    log.event("run_summary",{"run_id":run_id,"elapsed_seconds":round(elapsed,3),
              "usage":usage,"conservative_cost_usd":cost,"project_status":db.project_status()})
    log.final(output)
    print(output); print("\n[v23 observability]")
    print(j({"run_id":run_id,"elapsed_seconds":round(elapsed,3),"usage":usage,
             "conservative_cost_usd":round(cost,8),"session_id":args.session_id,
             "autonomous_cycles":cycle,"goal":latest_goal_state(db),
             "experiments_db":str(STATE_DB),"session_db":str(SESSION_DB),
             "trace_enabled":not args.disable_tracing}))
    print("[v23 run] "+str(log.dir))
    return 0 if status=="DONE" else 2


if __name__=="__main__":
    raise SystemExit(main())
