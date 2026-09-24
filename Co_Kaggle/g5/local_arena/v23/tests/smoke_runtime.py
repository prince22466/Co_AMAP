#!/usr/bin/env python3
"""No-API smoke test for v23 agent infrastructure v2."""
from __future__ import annotations

import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

V23_ROOT = Path(__file__).resolve().parents[1]
if str(V23_ROOT) not in sys.path:
    sys.path.insert(0, str(V23_ROOT))

from agent import runtime


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="v23_smoke_") as tmp:
        db = runtime.ResearchDB(Path(tmp) / "experiments.sqlite3")
        run_id = "run_smoke"
        db.start_run(run_id, "smoke-session", "smoke task", "smoke-model")

        experiment_id = db.start_experiment(
            run_id,
            "A structured experiment can be persisted.",
            "workspace/candidates/smoke.py",
            "",
            "no API call",
        )
        summary = {
            "games_total": 1,
            "games_valid": 1,
            "wins": 1,
            "losses": 0,
            "margin_worsened_cases": 0,
            "mean_margin_improvement": 1.0,
            "best_margin_improvement": 1.0,
            "repair_rate": 1.0,
        }
        db.record_replay_call(
            run_id,
            experiment_id,
            "replay_smoke",
            "workspace/candidates/smoke.py",
            {
                "summary": summary,
                "matches": [{
                    "episode": "smoke",
                    "valid": True,
                    "original_v20_margin": -1.0,
                    "candidate_margin": 1.0,
                    "margin_improvement": 2.0,
                    "result": "WIN",
                    "action_divergences": 1,
                    "elapsed_seconds": 0.001,
                    "error": "",
                }],
            },
            runtime.utcnow(),
            0.001,
        )

        goal_id = db.set_goal("repair_rate", ">=", 1.0, 0)
        active_goal = runtime.latest_goal_state(db)
        assert active_goal is not None
        assert active_goal["goal_id"] == goal_id
        assert active_goal["reached_at"] is None

        reached = db.maybe_reach_goal(
            run_id,
            experiment_id,
            summary,
            {
                "requests": 1,
                "input_tokens": 100,
                "cached_tokens": 20,
                "cache_write_tokens": 10,
                "output_tokens": 25,
                "reasoning_tokens": 5,
                "total_tokens": 125,
            },
            0.10,
            0.50,
        )
        assert reached and reached["goal_id"] == goal_id
        reached_goal = runtime.latest_goal_state(db)
        assert reached_goal is not None
        assert reached_goal["goal_id"] == goal_id
        assert reached_goal["reached_at"] is not None
        assert runtime.autonomous_stop_reason(reached_goal, "", True) == "goal_reached"
        assert runtime.autonomous_stop_reason({"reached_at": None}, "", True) is None
        assert runtime.autonomous_stop_reason({"reached_at": None}, "replay failed", True) == "reported_blocker"
        assert runtime.autonomous_stop_reason({"reached_at": None}, "", False) == "execution_disabled"

        finished = db.finish_experiment(
            experiment_id, "SUPPORTED", "smoke experiment completed"
        )
        assert finished["status"] == "SUPPORTED"

        usage = {
            "requests": 1,
            "input_tokens": 100,
            "cached_tokens": 20,
            "cache_write_tokens": 10,
            "output_tokens": 25,
            "reasoning_tokens": 5,
            "total_tokens": 125,
        }
        cost = runtime.conservative_cost_usd(usage, 0.10, 0.50)
        assert cost > 0
        db.finish_run(run_id, "DONE", usage, cost, "smoke complete", 0.01)

        # Agents SDK function tools may run on worker threads. Reproduce that
        # boundary explicitly using a separate DB so the main lifecycle assertions
        # remain unchanged.
        thread_db = runtime.ResearchDB(Path(tmp) / "threaded.sqlite3")
        thread_run_id = "run_thread_smoke"
        thread_db.start_run(
            thread_run_id, "thread-smoke-session", "thread smoke task", "smoke-model"
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            threaded_status = pool.submit(thread_db.project_status).result()
            threaded_goal = pool.submit(
                thread_db.set_goal, "wins", ">=", 1.0, 0
            ).result()
            threaded_experiment = pool.submit(
                thread_db.start_experiment,
                thread_run_id,
                "Cross-thread DB tool call works.",
                "",
                "",
                "thread smoke",
            ).result()
            threaded_finished = pool.submit(
                thread_db.finish_experiment,
                threaded_experiment,
                "UNRESOLVED",
                "thread smoke complete",
            ).result()
        assert threaded_status["runs_completed"] == 0
        assert threaded_goal.startswith("goal_")
        assert threaded_finished["status"] == "UNRESOLVED"

        status = db.project_status()
        assert status["runs_completed"] == 1
        assert status["experiments_total"] == 1
        assert status["experiments_supported"] == 1
        assert status["replay_cases_total"] == 1
        assert status["goal"]["reached_at"] is not None
        assert status["goal"]["reached_observability"]["replay_cases"] == 1

    print("v23 infrastructure smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
