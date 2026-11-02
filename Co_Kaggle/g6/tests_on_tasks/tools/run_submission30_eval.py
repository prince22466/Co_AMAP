"""Official-runtime E4B S30 pilot/S1 runner; no implicit promotion.

Requires Kaggle snapshots/wheels, installed official harness and an E4B
inference server; ordinary packaging or synthetic tests cannot replace it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from build_submission30 import ARTIFACT, MODEL, build
from gate_submission30 import PILOT_IDS, S29_WINS
from run_submission24_eval import nonnegative_minutes
import run_submission24_eval as official


def run_passed(row: dict, minimum: int) -> bool:
    wins = set(row.get("solved_ids") or [])
    return (row.get("resolved", 0) >= minimum
            and S29_WINS <= wins
            and row.get("empty_patches", 99) <= 1
            and not row.get("agent_error_ids"))


async def evaluate(args) -> dict:
    if args.archive is None:
        build()
        args.archive = ARTIFACT
    # The official runner uses its module-level PILOT_IDS. Change only the
    # predeclared task selection, never its verifier, budget, test or scoring.
    official.PILOT_IDS = PILOT_IDS
    report = await official.evaluate(args)
    report["candidate"] = "submission_30"
    report["graph_tools"] = [
        "get_code_neighbors", "search_similar_code", "get_code_subgraph"
    ]
    report["promotion_approved"] = False
    report["minimum_correct"] = 5
    report["S29_retained_ids"] = sorted(S29_WINS)
    if report.get("stage") == "evaluation_complete":
        runs = report.get("runs") or []
        report["score_gate_all_repeats"] = (
            len(runs) == args.repeats and
            all(run_passed(row, 5) for row in runs))
        report["promotion_note"] = (
            "Even a passed score gate requires manual graph/edit trace, "
            "behavioral test and diff inspection. Never auto-promote.")
    else:
        report["score_gate_all_repeats"] = False
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "submission30_summary.json").write_text(
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
    if report.get("stage") == "baseline_blocked" or (
        report.get("stage") == "evaluation_complete" and
        not report["score_gate_all_repeats"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
