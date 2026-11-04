"""Paired S35 31B official-harness evaluation; no new benchmark or scorer.

Uses the same official evaluator as earlier g6 runs. The only variant
difference is the optional instruction-only recovery skill. The global
model override below is confined to this single-process CLI runner.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from build_submission35 import MODEL, VARIANTS, archive_path, build_variant
from gate_submission35 import PILOT_IDS
import run_submission24_eval as official


def same_bytes(a: Path, b: Path) -> bool:
    return hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(
        b.read_bytes()).digest()


def verified_archive(variant: str, supplied: Path | None) -> Path:
    if variant not in VARIANTS:
        raise ValueError("Unknown S35 variant: " + variant)
    expected = archive_path(variant)
    if not expected.exists():
        build_variant(variant)
    selected = supplied or expected
    if not selected.is_file() or not same_bytes(selected, expected):
        raise ValueError("Archive mismatch: expected reviewed S35 " + variant)
    return selected


async def evaluate(args) -> dict:
    args.archive = verified_archive(args.variant, args.archive)
    old_model, old_pilot = official.MODEL, official.PILOT_IDS
    official.MODEL, official.PILOT_IDS = MODEL, PILOT_IDS
    try:
        report = await official.evaluate(args)
    finally:
        official.MODEL, official.PILOT_IDS = old_model, old_pilot
    report["submission"] = 35
    report["variant"] = args.variant
    report["model"] = MODEL
    report["archive_sha256"] = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    report["promotion_approved"] = False
    report["caveat"] = (
        "Official issue-resolution score is in per-task results, not in CPU "
        "packaging checks. This runner does not enforce model behavior, edit "
        "retry limits, or skill activation. Review full traces and patches."
    )
    (args.output / "submission35_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="primary")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--served-model", default=MODEL)
    parser.add_argument("--sandbox", choices=("docker", "subprocess"),
                        default="subprocess")
    parser.add_argument("--stage", choices=("pilot", "s1"), default="pilot")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--time-limit", type=official.nonnegative_minutes,
                        default=0.0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    print(json.dumps(report, indent=2))
    if report.get("stage") == "baseline_blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
