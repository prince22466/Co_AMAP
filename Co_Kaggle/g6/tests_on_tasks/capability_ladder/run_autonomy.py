"""L3: frozen S29 autonomy, using the existing official evaluator unmodified.

Supports explicit held-out task IDs, reusing the same E4B proxy, compaction,
submission package, baseline preflight, and verification path as prior S29 runs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from probe import MODEL, PILOT, S1
from build_submission29 import ARTIFACT, build
import run_submission24_eval as official


async def evaluate(args: argparse.Namespace) -> dict:
    task_ids = tuple(args.task_ids or (PILOT if args.stage == "pilot" else S1))
    if len(set(task_ids)) != len(task_ids) or not task_ids:
        raise ValueError("Duplicate/empty task list")
    if official.MODEL != MODEL:
        raise ValueError("Official evaluator model drift")
    # S29 package is rebuilt deterministically and kept byte-for-byte unchanged.
    build()
    args.archive = ARTIFACT
    old_pilot, old_s1 = official.PILOT_IDS, official.S1_IDS
    try:
        if args.stage == "pilot":
            official.PILOT_IDS = task_ids
        else:
            official.S1_IDS = task_ids
        result = await official.evaluate(args)
    finally:
        official.PILOT_IDS, official.S1_IDS = old_pilot, old_s1
    # Record source bytes, not just package version: organizer wheel builds
    # have changed without version bumps in previous local evaluations.
    from swegemma.harness.verification import verify_task
    verifier_source = Path(verify_task.__code__.co_filename)
    result["verifier_source_sha256"] = hashlib.sha256(
        verifier_source.read_bytes()).hexdigest()
    result["diagnostic_level"] = "autonomous"
    result["submission"] = "submission_29_unmodified"
    result["task_ids"] = list(task_ids)
    result["promotion_approved"] = False
    (args.output / "capability_l3_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--served-model", default=MODEL)
    parser.add_argument("--sandbox", choices=("docker", "subprocess"),
                        default="docker")
    parser.add_argument("--stage", choices=("pilot", "s1"), default="pilot")
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--time-limit", type=official.nonnegative_minutes,
                        default=4.0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    print(json.dumps(report, indent=2))
    if report.get("stage") == "baseline_blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
