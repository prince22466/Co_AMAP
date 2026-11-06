#!/usr/bin/env python3
"""Read-only numeric gate for original BI-V150 4K/16K half-run CSVs.

Half-run RUNS=2 SKIP_FIRST=0; no MATH-500 check (user runs later).
This does not prove repeatability, correctness, kernel invocation, or provenance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CASES = {4096: (1024, 64, 256), 16384: (1024, 64, 128)}
TARGETS = {"4K": 2450.0, "16K": 1450.0}
RUNS = 2
SKIP_FIRST = 0


def positive_float(value: str, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive and finite: {value!r}")
    return number


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        return list(reader)


def case_key(row: dict[str, str]) -> int:
    size = int(row["Prefill"])
    if size not in CASES:
        raise ValueError(f"Unexpected workload length: {size}")
    decode, concurrency, prompts = CASES[size]
    actual = (int(row["Decode"]), int(row["Conc"]), int(row["Num Prompts"]))
    if actual != (decode, concurrency, prompts):
        raise ValueError(f"Invalid benchmark case: {size}, {actual}")
    return size


def summary(path: Path) -> dict[int, dict[str, float]]:
    result = {}
    for row in rows(path):
        size = case_key(row)
        if size in result:
            raise ValueError(f"Duplicate summary case {size} in {path}")
        result[size] = {
            "tokens_per_s": positive_float(row["Total tok/s"], "Total tok/s"),
            "ttft_ms": positive_float(row["Mean TTFT (ms)"], "Mean TTFT (ms)"),
        }
    if set(result) != set(CASES):
        raise ValueError(f"Summary must contain BOTH half-run cases: {path}")
    return result


def validate_raw(path: Path) -> None:
    grouped = {size: [] for size in CASES}
    for row in rows(path):
        grouped[case_key(row)].append(row)
    for size, case_rows in grouped.items():
        if len(case_rows) != RUNS:
            raise ValueError(f"{path}: {size} requires exactly {RUNS} raw runs, got {len(case_rows)}")
        for row in case_rows:
            if row["Run Status"] != "SUCCESS":
                raise ValueError(f"{path}: {size} contains a failed run")
            successes = positive_float(row["Successful Requests"], "Successful Requests")
            if successes != CASES[size][2]:
                raise ValueError(f"{path}: {size} has missing requests: {successes}")
            positive_float(row["Total tok/s"], "raw Total tok/s")


def evaluate(args: argparse.Namespace) -> dict:
    baseline = summary(args.baseline_summary)
    candidate = summary(args.candidate_summary)
    validate_raw(args.baseline_raw)
    validate_raw(args.candidate_raw)

    comparisons = {}
    for size in CASES:
        name = f"{size // 1024}K"
        original = baseline[size]
        new = candidate[size]
        gain = 100 * (new["tokens_per_s"] / original["tokens_per_s"] - 1)
        comparisons[name] = {
            "baseline_tokens_per_s": original["tokens_per_s"],
            "candidate_tokens_per_s": new["tokens_per_s"],
            "throughput_change_pct": round(gain, 4),
            "baseline_ttft_ms": original["ttft_ms"],
            "candidate_ttft_ms": new["ttft_ms"],
            "ttft_change_pct": round(100 * (new["ttft_ms"] / original["ttft_ms"] - 1), 4),
            "numeric_gain": gain > 0,
            "milestone_met": new["tokens_per_s"] >= TARGETS[name],
        }

    improved_numerically = any(c["numeric_gain"] for c in comparisons.values())
    target_met = any(c["milestone_met"] for c in comparisons.values())
    return {
        "passed": improved_numerically,
        "target_met": target_met,
        "any_numeric_gain": improved_numerically,
        "all_half_raw_runs_valid": True,
        "benchmark_type": "development_half_run",
        "runs_per_case": RUNS,
        "skip_first": SKIP_FIRST,
        "accuracy_evaluation": "deferred_to_user",
        "comparison": comparisons,
        "source_paths": {
            "baseline_summary": str(args.baseline_summary),
            "baseline_raw": str(args.baseline_raw),
            "candidate_summary": str(args.candidate_summary),
            "candidate_raw": str(args.candidate_raw),
        },
        "provenance_warning": (
            "Numeric gate only. Confirm code SHA, benchmark hash, same serving args "
            "and GPU conditions, production kernel invocation, functional tests and "
            "repeatable paired A/B before committing a performance PR or stopping."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline-summary", "baseline-raw", "candidate-summary", "candidate-raw"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        report = {"passed": False, "target_met": False, "error": str(exc)}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
