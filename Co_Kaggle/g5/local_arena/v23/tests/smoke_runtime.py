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
from agent.context import (
    append_progress_files,
    build_analyst_context,
    build_engineer_context,
    build_progress_snapshot,
)
from agent.analysis import (
    analyze_cash_flow,
    analyze_experiment_records,
    analyze_inventory_flow,
    analyze_loss_history,
    analyze_loss_window,
    analyze_worker_utilization,
    cluster_loss_games,
    compare_candidate_to_v20,
    component_effect_matrix,
    hypothesis_evidence,
)


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

        # Real history deterministic analysis smoke.
        history_summary = analyze_loss_history(
            V23_ROOT, "111548564", window_size=24, top_windows=3
        )
        assert history_summary["episode"] == "111548564"
        assert history_summary["turns"] > 0
        assert history_summary["v20_final_margin"] < 0
        assert len(history_summary["critical_windows"]) <= 3
        if history_summary["critical_windows"]:
            first_window = history_summary["critical_windows"][0]
            detail = analyze_loss_window(
                V23_ROOT,
                "111548564",
                first_window["start_turn"],
                min(first_window["start_turn"] + 2, first_window["end_turn"]),
            )
            assert detail["rows"]
            assert detail["episode"] == "111548564"

        cash = analyze_cash_flow(V23_ROOT, "111548564")
        assert cash["episode"] == "111548564"
        assert cash["final_margin"] < 0

        inventory = analyze_inventory_flow(V23_ROOT, "111548564")
        assert inventory["episode"] == "111548564"
        assert inventory["final_margin"] < 0

        utilization = analyze_worker_utilization(V23_ROOT, "111548564")
        assert utilization["episode"] == "111548564"
        assert utilization["action_category_counts"]

        clusters = cluster_loss_games(V23_ROOT)
        assert clusters["games"]
        assert clusters["clusters"]
        assert any(
            row["episode"] == "111548564" for row in clusters["games"]
        )

        # Analyst idea batch queue smoke: exactly 10 ideas, deterministic order,
        # idea -> experiment linkage, then batch exhaustion -> analyst review.
        batch_db = runtime.ResearchDB(Path(tmp) / "idea_batch.sqlite3")
        batch_run = "run_idea_batch"
        batch_db.start_run(
            batch_run, "idea-batch-session", "idea batch smoke", "smoke-model"
        )
        analyst_json = {
            "performance_evidence": "smoke evidence",
            "failure_mechanisms": ["smoke failure"],
            "do_not_repeat": ["old smoke idea"],
            "ideas": [
                {
                    "title": f"Idea {i}",
                    "hypothesis": f"Hypothesis {i}",
                    "causal_layer": (
                        "multi_component" if i <= 3
                        else ("worker" if i % 2 else "market")
                    ),
                    "components": (
                        ["animal_plan", "market_orders"]
                        if i == 1 else
                        ["crop_plan", "task_ranking"]
                        if i == 2 else
                        ["logistics", "inventory_capacity"]
                        if i == 3 else
                        ["worker_policy"]
                    ),
                    "interaction_hypothesis": (
                        f"Interaction hypothesis {i}"
                        if i <= 3 else "single-component"
                    ),
                    "system_prediction": f"System prediction {i}",
                    "rationale": f"Rationale {i}",
                    "smallest_test": "1 case",
                    "promotion_rule": "positive margin -> expand",
                }
                for i in range(1, 11)
            ],
        }
        parsed_batch = runtime.parse_analyst_batch(
            runtime.json.dumps(analyst_json)
        )

        bounded = runtime.j_bounded({"x": "a" * 20000}, max_chars=3000)
        bounded_obj = runtime.json.loads(bounded)
        assert bounded_obj["truncated"] is True
        assert bounded_obj["original_chars"] > 3000
        assert len(bounded_obj["preview"]) == 3000
        # Regression: a huge single-line JSON/history artifact must never be
        # echoed back as a tool result. The local file may be large; the model
        # receives only bounded previews/metadata.
        large_root = Path(tmp) / "large_tool_root"
        large_root.mkdir(parents=True, exist_ok=True)
        large_jsonl = large_root / "history.jsonl"
        large_jsonl.write_text(
            runtime.json.dumps({"history": "x" * 12_000_000}) + "\n",
            encoding="utf-8",
        )
        local = object.__new__(runtime.LocalTools)
        local.root = large_root.resolve()
        local.workspace = (large_root / "workspace").resolve()
        local.allow_exec = False
        local.log = None
        large_summary = local.summarize_jsonl("history.jsonl", tail_rows=1)
        large_summary_raw = runtime.json.dumps(large_summary)
        assert large_summary["file_bytes"] > 10_000_000
        assert large_summary["raw_tail_omitted"] is True
        assert large_summary["tail_previews"][0]["truncated"] is True
        assert len(large_summary_raw) < 20_000

        api_cap_regression = runtime.j_bounded(
            {"payload": "z" * 12_000_000}, max_chars=24_000
        )
        assert len(api_cap_regression) < 30_000

        # Malformed-but-valid replay JSON must normalize instead of crashing DB/runtime code.
        normalized = runtime.normalize_replay_result([])
        assert normalized["error"] == "static replay child returned non-object JSON"
        assert normalized["matches"] == []
        assert normalized["summary"] == {}

        normalized = runtime.normalize_replay_result({"matches": None, "summary": []})
        assert normalized["matches"] == []
        assert normalized["summary"] == {}
        assert "error" in normalized

        normalized = runtime.normalize_replay_result({
            "matches": [None, "bad", 7, {"episode": "ok", "valid": True}],
            "summary": {"games_total": 1},
        })
        assert normalized["matches"] == [{"episode": "ok", "valid": True}]
        assert normalized["malformed_matches_dropped"] == 3


        retry = runtime.model_retry_settings()
        assert retry.max_retries == 5
        assert retry.backoff.initial_delay == 0.5
        assert retry.backoff.max_delay == 16.0

        pacer = runtime.ModelCallPacer(0.0)
        assert pacer.min_interval_seconds == 0.0


        # Replay runner must have the serialization helper it uses for step traces.
        import replay.runner as replay_runner
        assert callable(replay_runner._plain)

        # Candidate descriptions must never poison immutable experiment binding.
        tool_local = object.__new__(runtime.LocalTools)
        tool_local.root = Path(tmp).resolve()
        tool_local.workspace = (Path(tmp) / "workspace").resolve()
        tool_local.allow_exec = False
        tool_local.log = None
        candidate_file = Path(tmp) / "candidate.py"
        candidate_file.write_text("def agent(obs):\n    return []\n", encoding="utf-8")

        candidate, error = runtime.validate_candidate_path(
            tool_local, "descriptive candidate text"
        )
        assert candidate == ""
        assert error and "existing .py/.ipynb path" in error

        candidate, error = runtime.validate_candidate_path(tool_local, "candidate.py")
        assert candidate == "candidate.py"
        assert error is None
        assert len(parsed_batch["ideas"]) == 10

        invalid_attempt_id = batch_db.record_analyst_attempt(
            batch_run,
            "initial_diagnosis",
            "INVALID",
            "{truncated-json",
            "invalid analyst batch: JSONDecodeError",
            {"requests": 2, "input_tokens": 1000, "output_tokens": 1200},
            0.05,
        )
        assert invalid_attempt_id.startswith("attempt_")
        valid_attempt_id = batch_db.record_analyst_attempt(
            batch_run,
            "initial_diagnosis",
            "VALID",
            runtime.json.dumps(analyst_json),
            "",
            {"requests": 1, "input_tokens": 500, "output_tokens": 2500},
            0.04,
        )
        assert valid_attempt_id.startswith("attempt_")
        attempt_rows = batch_db.db.execute(
            "SELECT status,error,analyst_output FROM analyst_attempts ORDER BY created_at"
        ).fetchall()
        assert len(attempt_rows) == 2
        assert attempt_rows[0]["status"] == "INVALID"
        assert "JSONDecodeError" in attempt_rows[0]["error"]
        assert attempt_rows[1]["status"] == "VALID"
        assert batch_db.project_status()["analyst_attempts_total"] == 2
        try:
            runtime.parse_analyst_batch(
                runtime.json.dumps({**analyst_json, "ideas": analyst_json["ideas"][:9]})
            )
            raise AssertionError("9-idea analyst batch should fail")
        except ValueError:
            pass

        too_few_multi = runtime.json.loads(runtime.json.dumps(analyst_json))
        for idea in too_few_multi["ideas"]:
            idea["components"] = ["worker_policy"]
        try:
            runtime.parse_analyst_batch(runtime.json.dumps(too_few_multi))
            raise AssertionError("batch with <3 multi-component ideas should fail")
        except ValueError:
            pass

        review_id = batch_db.record_strategy_review(
            batch_run, "initial_diagnosis",
            runtime.json.dumps(analyst_json), {}, 0.0
        )
        queued = batch_db.add_idea_batch(review_id, parsed_batch["ideas"])
        assert queued["count"] == 10
        queue_status = batch_db.idea_batch_status()
        assert queue_status["pending"] == 10
        assert batch_db.strategy_review_trigger() is None

        pending_for_context = batch_db.current_work_idea()
        engineer_context = build_engineer_context(
            batch_db, pending_for_context, max_chars=8000
        )
        assert engineer_context["role"] == "experiment_engineer"
        assert engineer_context["assigned_idea"]["idea_id"] == (
            pending_for_context["idea_id"]
        )
        assert engineer_context["assigned_idea"]["promotion_rule"]

        analyst_context = build_analyst_context(
            batch_db, V23_ROOT, max_chars=12000
        )
        assert analyst_context["role"] == "performance_analyst"
        assert analyst_context["progress"]["batch"]["pending"] == 10
        assert "selection_policy" in analyst_context

        progress_snapshot = build_progress_snapshot(batch_db)
        assert progress_snapshot["batch"]["pending"] == 10
        progress_root = Path(tmp) / "progress_root"
        append_progress_files(
            progress_root, progress_snapshot, runtime.utcnow()
        )
        latest_progress = progress_root / "workspace" / "progress_latest.json"
        history_progress = progress_root / "workspace" / "progress.jsonl"
        assert latest_progress.is_file()
        assert history_progress.is_file()
        latest_data = runtime.json.loads(
            latest_progress.read_text(encoding="utf-8")
        )
        assert latest_data["batch"]["pending"] == 10
        assert history_progress.read_text(encoding="utf-8").count("\n") == 1

        # A partially completed engineer cycle must resume the same RUNNING idea
        # instead of silently advancing to the next PENDING idea.
        resume_db = runtime.ResearchDB(Path(tmp) / "resume.sqlite3")
        resume_run = "run_resume"
        resume_db.start_run(
            resume_run, "resume-session", "resume smoke", "smoke-model"
        )
        resume_review = resume_db.record_strategy_review(
            resume_run, "initial_diagnosis",
            runtime.json.dumps(analyst_json), {}, 0.0
        )
        resume_db.add_idea_batch(resume_review, parsed_batch["ideas"])
        first_work = resume_db.current_work_idea()
        assert first_work is not None
        assert first_work["batch_index"] == 1
        assert first_work["resume_existing"] is False
        partial_eid = resume_db.start_experiment(
            resume_run,
            first_work["hypothesis"],
            "",
            "",
            "partial cycle smoke",
            first_work["idea_id"],
        )
        resumed_work = resume_db.current_work_idea()
        assert resumed_work is not None
        assert resumed_work["idea_id"] == first_work["idea_id"]
        assert resumed_work["experiment_id"] == partial_eid
        assert resumed_work["resume_existing"] is True
        same_eid = resume_db.start_experiment(
            resume_run,
            first_work["hypothesis"],
            "",
            "",
            "resume smoke",
            first_work["idea_id"],
        )
        assert same_eid == partial_eid

        for expected_index in range(1, 11):
            idea = batch_db.next_pending_idea()
            assert idea is not None
            assert idea["batch_index"] == expected_index
            eid = batch_db.start_experiment(
                batch_run,
                idea["hypothesis"],
                "",
                "",
                "idea batch smoke",
                idea["idea_id"],
            )
            assert isinstance(eid, str)
            exp_row = batch_db.db.execute(
                "SELECT idea_id FROM experiments WHERE experiment_id=?",
                (eid,),
            ).fetchone()
            assert exp_row["idea_id"] == idea["idea_id"]

            candidate_path = f"workspace/candidates/idea_{expected_index}.py"
            candidate_hash = f"{expected_index:064x}"
            bound = batch_db.bind_candidate(eid, candidate_path, candidate_hash)
            assert bound["idea_id"] == idea["idea_id"]
            assert bound["candidate"] == candidate_path
            assert bound["candidate_sha256"] == candidate_hash
            same = batch_db.bind_candidate(eid, candidate_path, candidate_hash)
            assert same["candidate_sha256"] == candidate_hash
            changed = batch_db.bind_candidate(
                eid, candidate_path, f"{expected_index + 100:064x}"
            )
            assert changed["error"] == "candidate content changed within experiment"

            batch_db.record_replay_call(
                batch_run,
                eid,
                f"replay_batch_{expected_index}",
                candidate_path,
                {
                    "summary": {
                        "games_total": 1,
                        "games_valid": 1,
                        "wins": 0,
                        "losses": 1,
                        "margin_worsened_cases": 0,
                        "mean_margin_improvement": 0.0,
                        "best_margin_improvement": 0.0,
                    },
                    "matches": [{
                        "episode": f"episode_{expected_index}",
                        "valid": True,
                        "original_v20_margin": -10.0,
                        "candidate_margin": -10.0,
                        "margin_improvement": 0.0,
                        "result": "LOSS",
                        "action_divergences": 1,
                        "game_record_path": (
                            f"workspace/replay_records/{idea['idea_id']}/"
                            f"{eid}/replay_batch_{expected_index}/"
                            f"episode_{expected_index}.json"
                        ),
                        "elapsed_seconds": 0.001,
                        "error": "",
                    }],
                },
                runtime.utcnow(),
                0.001,
            )

            dossier = batch_db.idea_dossier(idea["idea_id"])
            assert dossier is not None
            assert dossier["idea"]["idea_id"] == idea["idea_id"]
            assert dossier["experiments"][0]["candidate"] == candidate_path
            assert dossier["experiments"][0]["candidate_sha256"] == candidate_hash
            assert dossier["replay_calls"][0]["replay_call_id"] == (
                f"replay_batch_{expected_index}"
            )
            assert dossier["replay_calls"][0]["games"][0]["game_record_path"].endswith(
                f"episode_{expected_index}.json"
            )

            if expected_index <= 3:
                stored = batch_db.db.execute(
                    """SELECT components_json,interaction_hypothesis,system_prediction
                       FROM research_ideas WHERE idea_id=?""",
                    (idea["idea_id"],),
                ).fetchone()
                assert len(runtime.json.loads(stored["components_json"])) >= 2
                assert stored["interaction_hypothesis"]
                assert stored["system_prediction"]
            batch_db.finish_experiment(
                eid, "REJECTED", "smoke idea rejected"
            )

        exhausted = batch_db.idea_batch_status()
        assert exhausted["pending"] == 0
        assert exhausted["running"] == 0
        assert exhausted["completed"] == 10
        assert batch_db.next_pending_idea() is None
        assert batch_db.strategy_review_trigger() == "idea_batch_exhausted"
        results = batch_db.recent_idea_results(review_id)
        assert len(results) == 10
        assert all(row["experiment_id"] for row in results)
        assert all(row["candidate"] for row in results)
        assert all(row["candidate_sha256"] for row in results)
        lineage = batch_db.batch_lineage_summary(review_id)
        assert len(lineage) == 10
        assert lineage[0]["idea_id"]
        assert lineage[0]["experiments"][0]["replay_call_ids"]
        assert lineage[0]["experiments"][0]["game_record_paths"]

        experiment_analysis = analyze_experiment_records(batch_db, review_id)
        assert experiment_analysis["scope"] == "idea_batch"
        assert len(experiment_analysis["ideas"]) == 10
        assert experiment_analysis["by_causal_layer"]
        assert experiment_analysis["by_component_combination"]

        effect_matrix = component_effect_matrix(batch_db, review_id)
        assert effect_matrix["component_matrix"]
        assert effect_matrix["component_combinations"]

        first_idea_id = queued["idea_ids"][0]
        evidence = hypothesis_evidence(V23_ROOT, batch_db, first_idea_id)
        assert evidence["idea_id"] == first_idea_id
        assert evidence["replay_cases"] == 1
        assert evidence["verdict"] in {
            "supported_by_current_replay",
            "contradicted_by_current_replay",
            "mixed_or_insufficient",
        }

        # Staged 1->5->25 replay must not double-weight the same episode.
        first_exp_for_stage = batch_db.db.execute(
            "SELECT experiment_id,candidate FROM experiments WHERE idea_id=?",
            (first_idea_id,),
        ).fetchone()
        batch_db.record_replay_call(
            batch_run,
            first_exp_for_stage["experiment_id"],
            "replay_stage_later",
            first_exp_for_stage["candidate"],
            {
                "summary": {
                    "games_total": 1,
                    "games_valid": 1,
                    "wins": 1,
                    "losses": 0,
                    "margin_worsened_cases": 0,
                    "mean_margin_improvement": 5.0,
                    "best_margin_improvement": 5.0,
                },
                "matches": [{
                    "episode": "episode_1",
                    "valid": True,
                    "original_v20_margin": -10.0,
                    "candidate_margin": 5.0,
                    "margin_improvement": 15.0,
                    "result": "WIN",
                    "action_divergences": 2,
                    "game_record_path": None,
                    "elapsed_seconds": 0.001,
                    "error": "",
                }],
            },
            runtime.utcnow(),
            0.001,
        )
        staged_evidence = hypothesis_evidence(
            V23_ROOT, batch_db, first_idea_id
        )
        assert staged_evidence["replay_cases"] == 1
        assert staged_evidence["wins"] == 1
        assert staged_evidence["losses"] == 0
        assert staged_evidence["mean_margin_improvement"] == 15.0
        staged_analysis = analyze_experiment_records(batch_db, review_id)
        staged_row = next(
            row for row in staged_analysis["ideas"]
            if row["idea_id"] == first_idea_id
        )
        assert staged_row["replay_cases"] == 1
        assert staged_row["wins"] == 1
        assert staged_row["losses"] == 0

        # Candidate-v20 comparison follows the exact stored game_record_path.
        compare_root = Path(tmp) / "compare_root"
        record_rel = (
            "workspace/replay_records/"
            + first_idea_id
            + "/synthetic/replay/episode_1.json"
        )
        record_path = compare_root / record_rel
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            runtime.json.dumps({
                "steps": [
                    {
                        "replay_step": 1,
                        "day": 0,
                        "hour": 1,
                        "candidate_action": {"farmer": ["PASS"]},
                        "recorded_v20_action": {"farmer": ["MOVE"]},
                        "recorded_opponent_action": {"farmer": ["PASS"]},
                        "diverged_from_v20": True,
                    }
                ]
            }),
            encoding="utf-8",
        )
        first_exp = batch_db.db.execute(
            "SELECT experiment_id FROM experiments WHERE idea_id=?",
            (first_idea_id,),
        ).fetchone()["experiment_id"]
        batch_db.db.execute(
            """UPDATE replays SET game_record_path=?
               WHERE experiment_id=? AND episode='episode_1'""",
            (record_rel, first_exp),
        )
        batch_db.db.commit()
        compared = compare_candidate_to_v20(
            compare_root, batch_db, first_idea_id, "episode_1"
        )
        assert compared["idea_id"] == first_idea_id
        assert compared["first_divergence"]["replay_step"] == 1
        assert compared["trace_steps"] == 1

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
