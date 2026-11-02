"""S31 official-runtime E4B preflight and matched pilot/S1 runner.

Requires official harness, Kaggle snapshots and an already served E4B model.
Does not alter grading, tests, benchmark snapshots or reference solutions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from build_submission31 import ARCHIVE, MODEL, build
from gate_submission31 import PILOT_IDS, S29_WINS, S30_NEW_WINS
import run_submission24_eval as official


def score_passed(run: dict) -> bool:
    solved = set(run.get("solved_ids") or ())
    return (run.get("resolved", -1) >= 5
            and S29_WINS <= solved
            and bool(S30_NEW_WINS & solved)
            and run.get("empty_patches", 99) <= 1
            and not run.get("agent_error_ids"))


async def evaluate(args) -> dict:
    if args.archive is None:
        build()
        args.archive = ARCHIVE
    # The existing official evaluator module only allows predeclared pilots;
    # replace its PILOT_IDS selection, not its grading or patch execution.
    official.PILOT_IDS = PILOT_IDS
    report = await official.evaluate(args)
    report["candidate"] = "submission_31"
    report["stage_design"] = "graph navigator -> implementer -> independent verifier"
    report["promotion_approved"] = False
    if report.get("stage") == "evaluation_complete":
        runs = report.get("runs") or []
        report["score_gate_each_repeat"] = (
            len(runs) == args.repeats and
            all(score_passed(run) for run in runs))
    else:
        report["score_gate_each_repeat"] = False
    report["note"] = (
        "Score gate does not imply safety or actual graph use. "
        "Apply gate_submission31.py to each printed trace and inspect ATIF. "
        "A 0-minute setting reproduces historic no-time-limit S1; "
        "4 minutes matches submission config. Do not compare unlike limits.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "submission31_summary.json").write_text(
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
    parser.add_argument("--time-limit", type=official.nonnegative_minutes,
                        default=0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    print(json.dumps(report, indent=2))
    if (report.get("stage") == "baseline_blocked" or
        (report.get("stage") == "evaluation_complete"
         and not report["score_gate_each_repeat"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
