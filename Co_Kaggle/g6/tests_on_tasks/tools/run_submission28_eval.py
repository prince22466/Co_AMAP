"""Real E4B S28 pilot/S1 using the existing official baseline preflight."""
import argparse
import asyncio
import json
from pathlib import Path

from build_submission28 import ARCHIVE, MODEL, build
from run_submission24_eval import evaluate as evaluate_official, nonnegative_minutes
from run_submission26_eval import objective_pilot_pass


async def evaluate(args):
    """Build the candidate and evaluate; release promotion is NEVER automatic."""
    if args.archive is None:
        build()
        args.archive = ARCHIVE
    report = await evaluate_official(args)
    report["candidate"] = "submission_28"
    if report["stage"] == "evaluation_complete" and args.stage == "pilot":
        report["objective_pilot_pass_each_repeat"] = bool(report["runs"]) and all(
            objective_pilot_pass(run) for run in report["runs"])
        report["ready_for_full_s1"] = False
        report["manual_gate"] = (
            "Require >=3/5 solved, both S27 successes retained, <=1 empty, "
            "zero agent errors per repeat; inspect full ATIF tool arguments, "
            "real patch diffs, assertions and explicit submit_patch. Compare "
            "with S27 on the same environment before full S1.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "submission28_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    """Parse CLI arguments and exit nonzero on blocked or failed pilot."""
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
    if (report["stage"] == "baseline_blocked" or
            (args.stage == "pilot" and
             report.get("objective_pilot_pass_each_repeat") is False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
