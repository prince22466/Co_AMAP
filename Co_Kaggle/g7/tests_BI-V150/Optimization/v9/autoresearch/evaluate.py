#!/usr/bin/env python3
"""Strict numeric acceptance gate for untouched FlagOS serving CSV output.

No benchmark or accuracy code is modified. Provenance and the MATH-500
accuracy log must also be reviewed by a human/agent before any PR claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CASES = {4096: (1024, 64, 256), 16384: (1024, 64, 128)}
GAIN_FLOOR = 0.01
REGRESSION_FLOOR = -0.01
TTFT_MAX_RATIO = 1.01
ACCURACY_FLOOR = 0.95
RUNS = 4
SKIP_FIRST = 1


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
        return list(reader)


def key(row: dict[str, str]) -> int:
    size = int(row["Prefill"])
    if size not in CASES:
        raise ValueError(f"Unexpected workload length: {size}")
    decode, concurrency, num_prompts = CASES[size]
    actual = (int(row["Decode"]), int(row["Conc"]), int(row["Num Prompts"]))
    if actual != (decode, concurrency, num_prompts):
        raise ValueError(f"Invalid official case: {size}, {actual}")
    return size


def summary(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for row in rows(path):
        size = key(row)
        if size in result:
            raise ValueError(f"Duplicate summary case {size} in {path}")
        result[size] = {
            "tokens_per_s": positive_float(row["Total tok/s"], "Total tok/s"),
            "ttft_ms": positive_float(row["Mean TTFT (ms)"], "Mean TTFT (ms)"),
        }
    if set(result) != set(CASES):
        raise ValueError(f"Summary must contain BOTH official cases: {path}")
    return result


def validate_raw(path: Path) -> None:
    grouped: dict[int, list[dict[str, str]]] = {size: [] for size in CASES}
    for row in rows(path):
        grouped[key(row)].append(row)
    for size, case_rows in grouped.items():
        if len(case_rows) != RUNS:
            raise ValueError(f"{path}: {size} requires exactly {RUNS} raw runs, got {len(case_rows)}")
        for row in case_rows:
            if row["Run Status"] != "SUCCESS":
                raise ValueError(f"{path}: {size} has a failed run")
            success = positive_float(row["Successful Requests"], "Successful Requests")
            if success != CASES[size][2]:
                raise ValueError(f"{path}: {size} has missing requests: {success}")


def evaluate(args: argparse.Namespace) -> dict:
    baseline = summary(args.baseline_summary)
    candidate = summary(args.candidate_summary)
    validate_raw(args.baseline_raw)
    validate_raw(args.candidate_raw)

    math_log = args.math500_log
    if not math_log.is_file() or not math_log.stat().st_size:
        raise ValueError(f"Missing/nonempty MATH-500 output: {math_log}")
    accuracy = args.math500_score
    if not math.isfinite(accuracy) or accuracy < 0 or accuracy > 1:
        raise ValueError("MATH-500 score must be a finite fraction in [0,1]")

    comparisons = {}
    for size in CASES:
        original, new = baseline[size], candidate[size]
        gain = new["tokens_per_s"] / original["tokens_per_s"] - 1
        ttft_ratio = new["ttft_ms"] / original["ttft_ms"]
        comparisons[f"{size // 1024}K"] = {
            "baseline_tokens_per_s": original["tokens_per_s"],
            "candidate_tokens_per_s": new["tokens_per_s"],
            "throughput_change_pct": round(100 * gain, 4),
            "baseline_ttft_ms": original["ttft_ms"],
            "candidate_ttft_ms": new["ttft_ms"],
            "ttft_change_pct": round(100 * (ttft_ratio - 1), 4),
            "throughput_no_regression": gain >= REGRESSION_FLOOR,
            "ttft_no_regression": ttft_ratio <= TTFT_MAX_RATIO,
            "gain_at_least_1pct": gain >= GAIN_FLOOR,
        }

    gain = any(item["gain_at_least_1pct"] for item in comparisons.values())
    guards = all(
        item["throughput_no_regression"] and item["ttft_no_regression"]
        for item in comparisons.values()
    )
    accuracy_ok = accuracy >= ACCURACY_FLOOR
    passed = gain and guards and accuracy_ok
    target_met = (
        comparisons["4K"]["candidate_tokens_per_s"] >= 2700
        or comparisons["16K"]["candidate_tokens_per_s"] >= 1500
    )
    return {
        "passed": passed,
        "target_met": passed and target_met,
        "accuracy": accuracy,
        "accuracy_ok": accuracy_ok,
        "full_official_raw_runs_valid": True,
        "any_gain_at_least_1pct": gain,
        "other_workload_and_ttft_guards": guards,
        "comparison": comparisons,
        "source_paths": {
            "baseline_summary": str(args.baseline_summary),
            "baseline_raw": str(args.baseline_raw),
            "candidate_summary": str(args.candidate_summary),
            "candidate_raw": str(args.candidate_raw),
            "math500_log": str(args.math500_log),
        },
        "provenance_warning": (
            "Numeric validation only. Independently verify same-server A/B, "
            "model SHA, service config, logs, MATH-500 report and actual optimized kernel."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline-summary", "baseline-raw", "candidate-summary", "candidate-raw"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--math500-score", required=True, type=float)
    parser.add_argument("--math500-log", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    try:
        report = evaluate(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {"passed": False, "target_met": False, "error": str(exc)}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
