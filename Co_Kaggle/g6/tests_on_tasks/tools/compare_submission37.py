"""Compare paired S37 research control vs deterministic ADK guard.

Use official per-task JSONL outcomes, not number of blocked duplicate calls,
as the primary performance measure. No Kaggle public-score claims.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from compare_submission24 import PILOT_IDS, S1_IDS, load_rows, normalize


def load_experiment(folder: Path, variant: str, stage: str) -> dict:
    if not folder.is_dir():
        raise ValueError("Missing run directory: " + str(folder))
    manifest = json.loads((folder / "run_manifest.json").read_text())
    telemetry = json.loads(
        (folder / "submission37_research_telemetry.json").read_text())
    if (telemetry.get("variant") != variant or
            bool(telemetry.get("guard_installed")) != (variant == "guard") or
            telemetry.get("archive_sha256") != manifest.get("archive_sha256") or
            not telemetry.get("research_only")):
        raise ValueError("Invalid run/guard/archive provenance: " + str(folder))
    task_ids = PILOT_IDS if stage == "pilot" else S1_IDS
    if manifest.get("task_ids") != list(task_ids):
        raise ValueError("Mismatched task set or ordering")
    if variant == "guard":
        expected = len(task_ids) * int(manifest["repeats"])
        counts = telemetry.get("guard_counts") or {}
        if (telemetry.get("app_instances") != expected
                or telemetry.get("expected_app_instances") != expected
                or not counts.get("callback_tool_calls")
                or counts.get("missing_session", 0)):
            raise ValueError("Guard coverage incomplete or missing session identity")
    elif telemetry.get("app_instances") != 0:
        raise ValueError("Control unexpectedly has guarded App instances")
    candidates = {}
    for number in range(1, int(manifest["repeats"]) + 1):
        name = f"candidate_{number}"
        rows = normalize(
            load_rows(folder / name), expected=task_ids)
        durations = []
        for task_id, row in rows.items():
            duration = row.get("duration_seconds")
            if (not isinstance(duration, (int, float)) or
                    isinstance(duration, bool) or
                    not math.isfinite(duration) or duration < 0):
                raise ValueError("Missing/invalid duration: " + task_id)
            durations.append(float(duration))
        candidates[name] = {
            "resolved": sorted(i for i, row in rows.items() if row["resolved"]),
            "empty_patch_ids": sorted(
                i for i, row in rows.items() if not row["agent_patch"].strip()),
            "agent_error_ids": sorted(
                i for i, row in rows.items()
                if row.get("error") or row.get("error_message")),
            "duration_seconds": round(sum(durations), 2),
        }
    return {"manifest": manifest, "telemetry": telemetry,
            "candidates": candidates}


def compare(control: dict, guarded: dict) -> dict:
    a, b = control["manifest"], guarded["manifest"]
    checked = (
        "task_ids", "archive_sha256", "tasks_sha256", "model_alias",
        "served_model", "sandbox", "context_cache", "compaction",
        "runtime_versions", "effective_limits", "repeats",
    )
    if any(a.get(key) != b.get(key) for key in checked):
        raise ValueError("Control and guard were not run under matched settings")
    if set(control["candidates"]) != set(guarded["candidates"]):
        raise ValueError("Repeat count mismatch")
    metrics = guarded["telemetry"].get("guard_counts") or {}
    blocked = int(metrics.get("duplicate_failures_blocked", 0))
    rows = []
    for repeat in sorted(control["candidates"]):
        old, new = control["candidates"][repeat], guarded["candidates"][repeat]
        before, after = set(old["resolved"]), set(new["resolved"])
        rows.append({
            "repeat": repeat,
            "control_resolved": len(before),
            "guard_resolved": len(after),
            "delta_resolved": len(after) - len(before),
            "gained_ids": sorted(after - before),
            "lost_ids": sorted(before - after),
            "retained_ids": sorted(after & before),
            "control_empty_patches": len(old["empty_patch_ids"]),
            "guard_empty_patches": len(new["empty_patch_ids"]),
            "control_errors": old["agent_error_ids"],
            "guard_errors": new["agent_error_ids"],
            "control_duration_seconds": old["duration_seconds"],
            "guard_duration_seconds": new["duration_seconds"],
        })
    return {
        "scope": "S37 RESEARCH ONLY; same Kaggle package as S36",
        "settings_matched": True,
        "interceptor_installed": guarded["telemetry"]["guard_installed"],
        "duplicate_failed_calls_blocked_total": blocked,
        "interceptor_observed_edit_attempts": metrics.get("edit_attempts", 0),
        "runs": rows,
        "intervention_confirmed": blocked > 0,
        "promotion_approved": False,
        "caveats": [
            "The guard exists only in the local harness and is absent from the Kaggle ZIP.",
            "A blocked call does not block E4B generation, reasoning or model tokens.",
            "Skipped native tool invocations alter the effective charged-tool budget.",
            "One repeated S1 screen is insufficient to prove generalization.",
            "Verify raw traces and the actual patches before claiming improvements.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "s1"), default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    report = compare(
        load_experiment(args.control, "control", args.stage),
        load_experiment(args.guard, "guard", args.stage))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
