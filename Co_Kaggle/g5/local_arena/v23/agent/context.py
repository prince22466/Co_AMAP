"""Role-specific context packs and compact research progress snapshots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import analyze_experiment_records, component_effect_matrix


def _clip(value: Any, max_chars: int) -> Any:
    raw = json.dumps(value, sort_keys=True, default=str)
    if len(raw) <= max_chars:
        return value
    return {
        "truncated": True,
        "original_chars": len(raw),
        "preview": raw[:max_chars],
    }


def build_progress_snapshot(db: Any) -> dict[str, Any]:
    status = db.project_status()
    latest_review = db.latest_strategy_review()
    batch = db.idea_batch_status()
    current = db.current_work_idea()
    experiments = analyze_experiment_records(db)
    best = None
    if experiments.get("scope") == "idea_batch":
        candidates = [
            row for row in experiments.get("ideas", [])
            if int(row.get("replay_cases") or 0) > 0
        ]
        if candidates:
            best = max(
                candidates,
                key=lambda row: (
                    int(row.get("wins") or 0),
                    float(row.get("mean_margin_improvement") or 0.0),
                ),
            )
    goal = status.get("goal")
    return {
        "goal": {
            "metric": goal.get("metric") if goal else None,
            "operator": goal.get("operator") if goal else None,
            "target": goal.get("target") if goal else None,
            "min_games_total": goal.get("min_games_total") if goal else None,
            "reached_at": goal.get("reached_at") if goal else None,
        },
        "runs_completed": status.get("runs_completed"),
        "experiments_total": status.get("experiments_total"),
        "experiments_supported": status.get("experiments_supported"),
        "experiments_rejected": status.get("experiments_rejected"),
        "replay_cases_total": status.get("replay_cases_total"),
        "conservative_cost_usd": status.get("conservative_cost_usd"),
        "research_signal": status.get("research_signal"),
        "batch": batch,
        "current_work": (
            {
                "idea_id": current.get("idea_id"),
                "batch_index": current.get("batch_index"),
                "title": current.get("title"),
                "status": current.get("status"),
                "experiment_id": current.get("experiment_id"),
                "resume_existing": current.get("resume_existing"),
            }
            if current else None
        ),
        "latest_review": (
            {
                "review_id": latest_review.get("review_id"),
                "created_at": latest_review.get("created_at"),
                "trigger": latest_review.get("trigger"),
            }
            if latest_review else None
        ),
        "best_current_batch_result": best,
    }


def append_progress_files(root: Path, snapshot: dict[str, Any], timestamp: str) -> None:
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": timestamp, **snapshot}
    with (workspace / "progress.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    latest = workspace / "progress_latest.json"
    tmp = workspace / "progress_latest.json.tmp"
    tmp.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(latest)


def build_analyst_context(
    db: Any,
    root: Path,
    max_chars: int = 12000,
) -> dict[str, Any]:
    progress = build_progress_snapshot(db)
    experiments = analyze_experiment_records(db)
    effects = component_effect_matrix(db)
    lineage = db.batch_lineage_summary()
    pack = {
        "role": "performance_analyst",
        "context_budget_chars": int(max_chars),
        "progress": _clip(progress, 2500),
        "current_batch_evidence": _clip(experiments, 4200),
        "component_effects": _clip(effects, 2200),
        "batch_lineage": _clip(lineage, 2200),
        "selection_policy": {
            "purpose": "Reason from durable measured evidence, not conversation history.",
            "included": [
                "durable goal and progress",
                "current batch canonical experiment evidence",
                "component-level descriptive effects",
                "idea/experiment/candidate/replay lineage",
            ],
            "excluded_by_default": [
                "full raw history JSON",
                "full replay traces",
                "old conversational chatter",
                "duplicate staged replay rows",
            ],
            "drill_down": (
                "Use deterministic analysis tools and idea_dossier only when a "
                "specific question requires more detail."
            ),
        },
    }
    return pack


def build_engineer_context(
    db: Any,
    idea: dict[str, Any] | None,
    max_chars: int = 8000,
) -> dict[str, Any]:
    progress = build_progress_snapshot(db)
    if idea is None:
        return _clip(
            {
                "role": "experiment_engineer",
                "progress": progress,
                "assigned_idea": None,
            },
            max_chars,
        )
    dossier = db.idea_dossier(str(idea["idea_id"]))
    compact_lineage = None
    if dossier is not None:
        compact_lineage = {
            "experiments": dossier.get("experiments", [])[-2:],
            "replay_calls": [
                {
                    "replay_call_id": call.get("replay_call_id"),
                    "candidate": call.get("candidate"),
                    "games": [
                        {
                            "episode": game.get("episode"),
                            "valid": game.get("valid"),
                            "result": game.get("result"),
                            "margin_improvement": game.get("margin_improvement"),
                            "game_record_path": game.get("game_record_path"),
                        }
                        for game in call.get("games", [])
                    ],
                }
                for call in dossier.get("replay_calls", [])[-3:]
            ],
        }
    pack = {
        "role": "experiment_engineer",
        "context_budget_chars": int(max_chars),
        "progress": _clip(progress, 2200),
        "assigned_idea": {
            "idea_id": idea.get("idea_id"),
            "batch_index": idea.get("batch_index"),
            "title": idea.get("title"),
            "hypothesis": idea.get("hypothesis"),
            "causal_layer": idea.get("causal_layer"),
            "components": json.loads(idea.get("components_json") or "[]"),
            "interaction_hypothesis": idea.get("interaction_hypothesis"),
            "system_prediction": idea.get("system_prediction"),
            "rationale": idea.get("rationale"),
            "smallest_test": idea.get("smallest_test"),
            "promotion_rule": idea.get("promotion_rule"),
            "resume_existing": bool(idea.get("resume_existing")),
            "existing_experiment_id": idea.get("experiment_id"),
        },
        "existing_lineage": _clip(compact_lineage, 3200),
        "execution_policy": {
            "scope": "Implement only the assigned idea.",
            "resume": (
                "If resume_existing is true, continue the existing RUNNING experiment_id. "
                "If repair_after_error is true, the previous experiment is closed and immutable: "
                "diagnose its replay error, call start_experiment again for the SAME idea, "
                "write a NEW corrected candidate artifact, and replay again."
            ),
            "self_correction": (
                "Generated candidate-code failures are repair tasks, not blockers. Repair only "
                "agent-generated artifacts under workspace/candidates/. Treat working_files/ "
                "and every repository/reference input outside workspace/candidates/ as read-only. "
                "Use the previous experiment/replay error in existing_lineage to create a NEW "
                "corrected candidate artifact, and repeat fresh experiment attempts until a "
                "valid measured replay result exists."
            ),
            "evidence": (
                "After replay, verify idea_dossier before finishing the experiment."
            ),
        },
    }
    return pack
