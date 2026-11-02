"""Matched E4B evaluation for S32 guarded edit vs native-only ablation.

Uses the existing official swegemma evaluator; no changes to task scoring.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from build_submission32 import ARTIFACT_DIR, MODEL, VARIANTS, build_variant
from gate_submission32 import PILOT_IDS, S29_WINS, S30_GAIN_WINS
import run_submission24_eval as official


def score_passed(run: dict, stage: str) -> bool:
    ids = set(run.get("solved_ids") or ())
    minimum = 4 if stage == "pilot" else 5
    return (run.get("resolved", -1) >= minimum
            and S29_WINS <= ids
            and bool(S30_GAIN_WINS & ids)
            and run.get("empty_patches", 99) <= 1
            and not run.get("agent_error_ids"))


async def evaluate(args) -> dict:
    if args.archive is None:
        build_variant(args.variant)
        args.archive = ARTIFACT_DIR / VARIANTS[args.variant][1]
    # Only the pilot task set differs; verification remains the stock harness.
    official.PILOT_IDS = PILOT_IDS
    report = await official.evaluate(args)
    report["candidate"] = "submission_32"
    report["variant"] = args.variant
    report["promotion_approved"] = False
    if report.get("stage") == "evaluation_complete":
        runs = report.get("runs") or ()
        report["score_gate_each_repeat"] = (
            args.repeats >= 2 and len(runs) == args.repeats
            and all(score_passed(run, args.stage) for run in runs))
    else:
        report["score_gate_each_repeat"] = False
    report["note"] = (
        "Official resolution is the only score. Printed trace gate and "
        "manual ATIF/patch review are required; any shell edit needs "
        "independent exact-script inspection. Historical traces used "
        "unlimited per-task time; run 4-minute package limit separately.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "submission32_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS),
                        default="primary")
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
    parser.add_argument("--time-limit", type=official.nonnegative_minutes,
                        default=0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(evaluate(args))
    print(json.dumps(result, indent=2))
    if (result.get("stage") == "baseline_blocked"
        or (result.get("stage") == "evaluation_complete"
            and not result["score_gate_each_repeat"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
