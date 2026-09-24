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

        goal_id = db.set_goal("repair_rate", ">=", 1.0, 0, 1)
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

        subset_goal_id = db.set_goal("wins", ">=", 1.0, 0, 2)
        subset_not_reached = db.maybe_reach_goal(
            run_id,
            experiment_id,
            summary,
            {
                "requests": 0,
                "input_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            },
            0.10,
            0.50,
        )
        assert subset_not_reached is None
        subset_goal = runtime.latest_goal_state(db)
        assert subset_goal is not None
        assert subset_goal["goal_id"] == subset_goal_id
        assert subset_goal["reached_at"] is None
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

        # Stagnation detection: four consecutive rejected, non-improving
        # experiments must force a strategy reset.
        stagnation_db = runtime.ResearchDB(Path(tmp) / "stagnation.sqlite3")
        stagnation_run = "run_stagnation"
        stagnation_db.start_run(
            stagnation_run, "stagnation-session", "stagnation smoke", "smoke-model"
        )
        for i in range(4):
            eid = stagnation_db.start_experiment(
                stagnation_run,
                f"Repeated local tweak {i}",
                "",
                "",
                "stagnation smoke",
            )
            stagnation_db.db.execute(
                """UPDATE experiments
                   SET wins=0, losses=5, replay_cases=5,
                       mean_margin_improvement=0.0,
                       best_margin_improvement=0.0
                   WHERE experiment_id=?""",
                (eid,),
            )
            stagnation_db.db.commit()
            stagnation_db.finish_experiment(
                eid, "REJECTED", "no measurable improvement"
            )
        signal = stagnation_db.research_signal()
        assert signal["stagnating"] is True
        assert signal["consecutive_rejected"] == 4
        assert signal["positive_recent"] is False

        progress_eid = stagnation_db.start_experiment(
            stagnation_run,
            "Structurally different candidate",
            "",
            "",
            "stagnation recovery smoke",
        )
        stagnation_db.db.execute(
            """UPDATE experiments
               SET wins=1, losses=4, replay_cases=5,
                   mean_margin_improvement=10.0,
                   best_margin_improvement=20.0
               WHERE experiment_id=?""",
            (progress_eid,),
        )
        stagnation_db.db.commit()
        stagnation_db.finish_experiment(
            progress_eid, "SUPPORTED", "positive signal"
        )
        recovered = stagnation_db.research_signal()
        assert recovered["stagnating"] is False
        assert recovered["positive_recent"] is True

        # Independent strategy review trigger is evidence-driven, not every cycle.
        review_db = runtime.ResearchDB(Path(tmp) / "reviews.sqlite3")
        review_run = "run_reviews"
        review_db.start_run(
            review_run, "review-session", "review smoke", "smoke-model"
        )
        assert review_db.strategy_review_trigger() == "initial_diagnosis"
        review_id = review_db.record_strategy_review(
            review_run, "initial_diagnosis", "initial analyst review", {}, 0.0
        )
        assert review_id.startswith("review_")
        assert review_db.strategy_review_trigger() is None

        for i in range(2):
            eid = review_db.start_experiment(
                review_run, f"Rejected idea {i}", "", "", "review trigger smoke"
            )
            review_db.db.execute(
                """UPDATE experiments
                   SET wins=0, losses=5, replay_cases=5,
                       mean_margin_improvement=0.0,
                       best_margin_improvement=0.0
                   WHERE experiment_id=?""",
                (eid,),
            )
            review_db.db.commit()
            review_db.finish_experiment(eid, "REJECTED", "no progress")
        # Only two experiments are not enough for periodic review unless the
        # global stagnation detector is active.
        assert review_db.strategy_review_trigger() is None

        eid = review_db.start_experiment(
            review_run, "Rejected idea 2", "", "", "review trigger smoke"
        )
        review_db.db.execute(
            """UPDATE experiments
               SET wins=0, losses=5, replay_cases=5,
                   mean_margin_improvement=0.0,
                   best_margin_improvement=0.0
               WHERE experiment_id=?""",
            (eid,),
        )
        review_db.db.commit()
        review_db.finish_experiment(eid, "REJECTED", "no progress")
        assert review_db.strategy_review_trigger() == "periodic_after_3_experiments"

        status = db.project_status()
        assert status["runs_completed"] == 1
        assert status["experiments_total"] == 1
        assert status["experiments_supported"] == 1
        assert status["replay_cases_total"] == 1

        # project_status reports the latest goal, which is intentionally the
        # unreached full-corpus guard created above.
        assert status["goal"]["goal_id"] == subset_goal_id
        assert status["goal"]["reached_at"] is None

        # The earlier goal did reach and retained its observability snapshot.
        reached_goal_row = db.db.execute(
            "SELECT * FROM goals WHERE goal_id=?", (goal_id,)
        ).fetchone()
        assert reached_goal_row is not None
        assert reached_goal_row["reached_at"] is not None
        reached_observability = runtime.json.loads(
            reached_goal_row["reached_observability_json"]
        )
        assert reached_observability["replay_cases"] == 1

    print("v23 infrastructure smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
