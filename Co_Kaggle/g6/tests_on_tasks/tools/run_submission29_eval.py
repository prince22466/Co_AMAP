"""Official-runtime E4B S29 pilot/S1 evaluation; never auto-promote."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from build_submission29 import ARTIFACT, MODEL, build
from run_submission24_eval import evaluate as evaluate_official, nonnegative_minutes
from compare_submission28 import objective_pilot_pass


async def evaluate(args) -> dict:
    """Run the established preflight/evaluator without replacing benchmark tests."""
    if args.archive is None:
        build()
        args.archive = ARTIFACT
    report = await evaluate_official(args)
    report["candidate"] = "submission_29"
    report["runtime_tool_interface"] = "native S27/S28 six-tool baseline"
    report["promotion"] = "NOT_APPROVED"
    report["external_trace_gate"] = (
        "After E4B execution, run gate_submission29.py against S27 and "
        "S29 matched printouts and manually inspect final diffs/ATIF.")
    if report.get("stage") == "evaluation_complete" and args.stage == "pilot":
        report["score_gate_per_repeat"] = (
            bool(report.get("runs")) and all(
                objective_pilot_pass(run) for run in report["runs"]))
        report["ready_for_s1"] = False
        report["manual_gate"] = (
            "Require >=3/5 wins including rich_3905 and rich_3454 on "
            "EVERY repeat, <=1 empty, zero agent errors; then inspect "
            "unsafe shell mutations, patch size, failed edits, tool "
            "arguments, actual diffs, and task-specific checks.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "submission29_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--served-model", default=MODEL)
    parser.add_argument("--sandbox", choices=["docker", "subprocess"],
                        default="subprocess")
    parser.add_argument("--stage", choices=["pilot", "s1"], default="pilot")
    parser.add_argument("--repeats", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--time-limit", type=nonnegative_minutes, default=4)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    print(json.dumps(report, indent=2))
    if (report.get("stage") == "baseline_blocked" or
        (args.stage == "pilot" and
         report.get("score_gate_per_repeat") is False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
