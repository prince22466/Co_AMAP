#!/usr/bin/env python3
"""Compare original 4K/16K half-run CSVs; numeric results are NOT a win gate.

Uses unmodified benchmark_throughput_serve_half.py (RUNS=2, SKIP_FIRST=0).
No MATH-500, repeatability or production-kernel verification is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CASES = {4096: (1024, 64, 256), 16384: (1024, 64, 128)}
TARGETS = {"4K": 2450.0, "16K": 1450.0}


def positive(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Invalid {label}: {value!r}")
    return result


def read_csv(path: Path, raw: bool) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {size: [] for size in CASES}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        for row in reader:
            size = int(row["Prefill"])
            if size not in CASES:
                raise ValueError(f"{path}: unexpected workload {size}")
            shape = tuple(int(row[col]) for col in ("Decode", "Conc", "Num Prompts"))
            if shape != CASES[size]:
                raise ValueError(f"{path}: {size} invalid workload shape {shape}")
            if raw:
                if row["Run Status"] != "SUCCESS":
                    raise ValueError(f"{path}: {size} failed run")
                if positive(row["Successful Requests"], "successful requests") != CASES[size][2]:
                    raise ValueError(f"{path}: {size} missing requests")
            grouped[size].append({
                "tok": positive(row["Total tok/s"], "Total tok/s"),
                "ttft": positive(row["Mean TTFT (ms)"], "Mean TTFT"),
            })
    required = 2 if raw else 1
    for size, rows in grouped.items():
        if len(rows) != required:
            raise ValueError(f"{path}: {size} needs exactly {required} {'raw runs' if raw else 'summary row'}")
    return grouped


def load_pair(summary_path: Path, raw_path: Path) -> dict[int, dict]:
    summary = read_csv(summary_path, raw=False)
    raw = read_csv(raw_path, raw=True)
    for size in CASES:
        for metric in ("tok", "ttft"):
            average = sum(row[metric] for row in raw[size]) / 2
            if abs(summary[size][0][metric] - average) > 0.05:
                raise ValueError(f"{size} {metric}: summary disagrees with raw half-runs")
    return {size: summary[size][0] for size in CASES}


def evaluate(args: argparse.Namespace) -> dict:
    baseline = load_pair(args.baseline_summary, args.baseline_raw)
    candidate = load_pair(args.candidate_summary, args.candidate_raw)
    comparison = {}
    for size in CASES:
        label = f"{size // 1024}K"
        a, b = baseline[size], candidate[size]
        comparison[label] = {
            "baseline_total_tok_s": a["tok"],
            "candidate_total_tok_s": b["tok"],
            "throughput_change_pct": round(100 * (b["tok"] / a["tok"] - 1), 4),
            "baseline_mean_ttft_ms": a["ttft"],
            "candidate_mean_ttft_ms": b["ttft"],
            "ttft_change_pct": round(100 * (b["ttft"] / a["ttft"] - 1), 4),
            "numeric_target_observed": b["tok"] >= TARGETS[label],
        }
    return {
        "valid_half_run_csvs": True,
        "numeric_gain_observed": any(
            item["candidate_total_tok_s"] > item["baseline_total_tok_s"]
            for item in comparison.values()
        ),
        "numeric_target_observed": any(item["numeric_target_observed"] for item in comparison.values()),
        "reproducibility_verified": False,
        "accuracy": "deferred_to_user",
        "comparison": comparison,
        "note": "CSV comparison only; repeat matched A/B and verify kernel tests/production trace before PR or stopping.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline-summary", "baseline-raw", "candidate-summary", "candidate-raw"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {"valid_half_run_csvs": False, "error": str(exc)}
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output)
    print(output, end="")
    return 0 if report["valid_half_run_csvs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
