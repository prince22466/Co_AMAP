#!/usr/bin/env python3
"""Static replay of a Python agent against recorded v20 opponents.

    python local_arena/static_reply/reply_template.py v20_bench.py
    python local_arena/static_reply/reply_template.py v20_bench.py --limit 1

Opponent commands are fixed; candidate actions and game states are recomputed.
Outputs use official env.toJSON() format. No training or model code is included.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_HISTORY_DIR = ROOT / "game_history" / "v20"


# --- Engine and source histories ---------------------------------------------
def load_engine():
    # Optional OpenSpiel imports emit native stderr noise unrelated to this game.
    with open(os.devnull, "w") as quiet:
        saved_fd = os.dup(2)
        try:
            os.dup2(quiet.fileno(), 2)
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                import kaggle_environments
            return kaggle_environments
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


def make_environment(engine, history):
    if history.get("name") != "kaggriculture":
        raise ValueError("expected a kaggriculture history")
    configuration = copy.deepcopy(history["configuration"])
    info = copy.deepcopy(history.get("info", {}))
    if configuration.get("seed") is None and info.get("seed") is None:
        raise ValueError("history has no seed; cannot reproduce its initial state")
    return engine.make("kaggriculture", configuration=configuration, info=info, debug=False)


def load_history(path):
    history = json.loads(path.read_text(encoding="utf-8"))
    steps = history.get("steps", [])
    if len(steps) < 2 or any(not isinstance(s, list) or len(s) != 2 for s in steps):
        raise ValueError("history must contain at least two steps with two seats each")
    if [s.get("status") for s in steps[-1]] != ["DONE", "DONE"]:
        raise ValueError("source history is not a completed two-player game")
    final_rewards(steps[-1])
    return history


def final_rewards(states):
    rewards = [float(s["reward"]) for s in states]
    if len(rewards) != 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("missing/non-finite final rewards")
    return rewards


def candidate_seat(history, seat):
    if seat != "auto":
        return int(seat)
    # game_history/v20 is documented as a corpus of v20 losses.
    rewards = final_rewards(history["steps"][-1])
    if rewards[0] == rewards[1]:
        raise ValueError("cannot infer v20 seat from a tie; specify --seat 0 or 1")
    return 0 if rewards[0] < rewards[1] else 1


def recorded_actions(history, step):
    actions = [copy.deepcopy(s.get("action")) for s in history["steps"][step]]
    if any(a is None for a in actions):
        raise ValueError(f"missing recorded action at step {step}")
    return actions


def state_core(states):
    return [{k: s.get(k) for k in ("observation", "reward", "status")} for s in states]


def verify_recorded_replay(engine, history):
    """Verify every observation/reward/status, not just terminal money."""
    env = make_environment(engine, history)
    if state_core(env.state) != state_core(history["steps"][0]):
        raise ValueError("recorded-action parity failed at initial state")
    for step in range(1, len(history["steps"])):
        env.step(recorded_actions(history, step))
        if state_core(env.state) != state_core(history["steps"][step]):
            raise ValueError(f"recorded-action parity failed at step {step}")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


# --- Candidate loading: supplied file owns all agent/model logic --------------
@contextlib.contextmanager
def load_agent(path):
    """Fresh main-module globals each game; allow imports beside the agent file."""
    name = "_static_replay_candidate"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        agent = getattr(module, "agent", None)
        if not callable(agent):
            raise ValueError(f"{path}: expected callable agent(obs) or agent(obs, config)")
        signature = inspect.signature(agent)
        try:
            signature.bind({}, {})
            with_config = True
        except TypeError:
            signature.bind({})
            with_config = False
        yield agent, with_config
    finally:
        sys.path[:] = old_path
        sys.modules.pop(name, None)


# --- Static replay: live candidate + fixed opponent -> official game history --
def run_replay(engine, history, agent_path, seat):
    env = make_environment(engine, history)
    getter = getattr(env, "_Environment__get_shared_state", None)
    if getter is None:
        raise RuntimeError("installed engine does not expose shared agent observations")
    max_action_seconds = 0.0
    with load_agent(agent_path) as (agent, with_config):
        for step in range(1, len(history["steps"])):
            if env.done:
                raise RuntimeError(f"candidate game ended early at step {step - 1}")
            # Public state plus only this player's private state, for either seat.
            obs = getter(seat).observation
            started = time.perf_counter()
            action = agent(obs, copy.deepcopy(env.configuration)) if with_config else agent(obs)
            max_action_seconds = max(max_action_seconds, time.perf_counter() - started)
            if not isinstance(action, dict):
                raise ValueError(f"agent returned {type(action).__name__}, not dict, at step {step}")
            actions = recorded_actions(history, step)
            actions[seat] = copy.deepcopy(action)
            env.step(actions)
    if not env.done or [s.status for s in env.state] != ["DONE", "DONE"]:
        raise RuntimeError("candidate replay did not finish with both seats DONE")
    return env.toJSON(), max_action_seconds


# --- CLI and batch results ----------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent", help="Python file defining agent(obs); bare names also resolve beside this script")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output", type=Path, help="new output directory; default: runs/<agent>/<UTC timestamp>")
    parser.add_argument("--episodes", nargs="+", help="source episode IDs or JSON filenames (optional subset)")
    parser.add_argument("--limit", type=int, help="run only the first N selected histories")
    parser.add_argument("--seat", choices=("auto", "0", "1"), default="auto", help="auto replaces the losing seat in this v20 loss corpus")
    parser.add_argument("--skip-parity", action="store_true", help="skip the full recorded-action fidelity check")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    agent_path = Path(args.agent).expanduser()
    if not agent_path.is_file() and not agent_path.is_absolute():
        agent_path = HERE / agent_path
    agent_path = agent_path.resolve()
    if not agent_path.is_file() or agent_path.suffix != ".py":
        parser.error(f"agent must be an existing .py file: {agent_path}")
    history_dir = args.history_dir.expanduser().resolve()
    paths = sorted(history_dir.glob("*.json"))
    if args.episodes:
        wanted = {Path(p).stem for p in args.episodes}
        missing = wanted - {p.stem for p in paths}
        if missing:
            parser.error(f"unknown episodes: {sorted(missing)}")
        paths = [p for p in paths if p.stem in wanted]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        paths = paths[:args.limit]
    if not paths:
        parser.error(f"no matching histories in {history_dir}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or HERE / "runs" / agent_path.stem / stamp).expanduser().resolve()
    if output == history_dir or output in history_dir.parents or history_dir in output.parents:
        parser.error("output must be separate from the source history directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output is not an empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    histories_out = output / "histories"
    histories_out.mkdir(exist_ok=True)
    engine = load_engine()
    digest = hashlib.sha256(agent_path.read_bytes()).hexdigest()
    rows = []
    summary = {"protocol": "static_recorded_opponent", "agent": str(agent_path),
               "agent_sha256": digest, "history_dir": str(history_dir),
               "engine_version": engine.__version__, "seat_mode": args.seat,
               "timeouts_enforced": False, "parity_required": not args.skip_parity,
               "games_requested": len(paths), "games_completed": 0,
               "wins": 0, "ties": 0, "losses": 0, "errors": 0, "games": rows}
    print(f"Agent: {agent_path}\nHistories: {len(paths)}\nOutput: {output}", flush=True)
    for index, path in enumerate(paths, 1):
        started = time.perf_counter()
        row = {"episode": path.stem, "ok": False, "parity": "skipped" if args.skip_parity else "pending"}
        print(f"[{index}/{len(paths)}] {path.stem}: replaying...", flush=True)
        try:
            history = load_history(path)
            seat = candidate_seat(history, args.seat)
            row["candidate_seat"] = seat
            if not args.skip_parity:
                verify_recorded_replay(engine, history)
                row["parity"] = "exact"
            replay, max_seconds = run_replay(engine, history, agent_path, seat)
            original = final_rewards(history["steps"][-1])
            rewards = final_rewards(replay["steps"][-1])
            margin = rewards[seat] - rewards[1 - seat]
            result = "WIN" if margin > 0 else "LOSS" if margin < 0 else "TIE"
            # Preserve original identities as provenance, not as new-run labels.
            replay["info"] = {"seed": replay["info"].get("seed"),
                "TeamNames": [agent_path.stem if i == seat else "Recorded opponent" for i in range(2)],
                "static_replay": {"source_episode": path.stem, "source_info": history.get("info", {}),
                                  "candidate_seat": seat, "agent_sha256": digest}}
            destination = histories_out / path.name
            write_json(destination, replay)
            row.update(ok=True, result=result, rewards=rewards, original_rewards=original,
                       margin=margin, margin_improvement=margin - (original[seat] - original[1 - seat]),
                       steps=len(replay["steps"]), history=str(destination), max_action_seconds=max_seconds)
            summary["games_completed"] += 1
            summary[{"WIN": "wins", "LOSS": "losses", "TIE": "ties"}[result]] += 1
            print(f"  {result}: seat={seat}, rewards={rewards}, margin={margin:+.2f}", flush=True)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            summary["errors"] += 1
            print(f"  ERROR: {row['error']}", flush=True)
        row["elapsed_seconds"] = time.perf_counter() - started
        rows.append(row)
        write_json(output / "summary.json", summary)
    print(f"Finished: {summary['wins']} wins, {summary['losses']} losses, "
          f"{summary['ties']} ties, {summary['errors']} errors. See {output / 'summary.json'}", flush=True)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
