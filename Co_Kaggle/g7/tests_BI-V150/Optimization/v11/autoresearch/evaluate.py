#!/usr/bin/env python3
"""Read-only CSV integrity and numeric comparison for v11 4K/16K half-runs.

The reported v10 reference is a FROZEN goal anchor, not matched machine data.
Require THREE pairs: the most recent matched accepted control, the preserved
v11 starting-code control (may be the same files for the initial candidate),
and the candidate. Does not run/modify any benchmark or establish repeatability.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CASES = {4096: (1024, 64, 256), 16384: (1024, 64, 128)}
V10_REPORTED = {"4K": 2372.38, "16K": 1872.70}
MINIMUM_GAIN = 0.05
STRETCH_GAIN = 0.10
PROGRESS_GAIN = 0.03
MINIMUM_TARGETS = {k: v * (1 + MINIMUM_GAIN) for k, v in V10_REPORTED.items()}
STRETCH_TARGETS = {k: v * (1 + STRETCH_GAIN) for k, v in V10_REPORTED.items()}


def positive(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Invalid {label}: {value!r}")
    return result


def read_csv(path: Path, *, raw: bool) -> dict[int, list[dict]]:
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
                raise ValueError(f"{path}: workload {size} invalid shape {shape}")
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
            raise ValueError(f"{path}: {size} needs exactly {required} "
                             f"{'raw runs' if raw else 'summary row'}")
    return grouped


def load_pair(summary_path: Path, raw_path: Path) -> dict[int, dict]:
    summary, raw = read_csv(summary_path, raw=False), read_csv(raw_path, raw=True)
    for size in CASES:
        for metric in ("tok", "ttft"):
            avg = sum(row[metric] for row in raw[size]) / 2
            if abs(summary[size][0][metric] - avg) > 0.05:
                raise ValueError(f"{size} {metric}: summary disagrees with raw half-runs")
    return {size: {
        **summary[size][0],
        "raw_total_tok_s": [row["tok"] for row in raw[size]],
    } for size in CASES}


def evaluate(args: argparse.Namespace) -> dict:
    baseline = load_pair(args.baseline_summary, args.baseline_raw)
    starting = load_pair(args.starting_summary, args.starting_raw)
    candidate = load_pair(args.candidate_summary, args.candidate_raw)
    comparisons = {}
    for size in CASES:
        name = f"{size // 1024}K"
        accepted, original, new = baseline[size], starting[size], candidate[size]
        tok = new["tok"]
        progress = tok + 1e-9 >= accepted["tok"] * (1 + PROGRESS_GAIN)
        fixed_min = tok + 1e-9 >= MINIMUM_TARGETS[name]
        fixed_stretch = tok + 1e-9 >= STRETCH_TARGETS[name]
        matched_min = tok + 1e-9 >= original["tok"] * (1 + MINIMUM_GAIN)
        matched_stretch = tok + 1e-9 >= original["tok"] * (1 + STRETCH_GAIN)
        comparisons[name] = {
            "accepted_matched_control_total_tok_s": accepted["tok"],
            "v11_starting_code_control_total_tok_s": original["tok"],
            "candidate_total_tok_s": tok,
            "candidate_raw_total_tok_s": new["raw_total_tok_s"],
            "reported_v10_anchor_total_tok_s": V10_REPORTED[name],
            "minimum_fixed_v10_target_total_tok_s": MINIMUM_TARGETS[name],
            "stretch_fixed_v10_target_total_tok_s": STRETCH_TARGETS[name],
            "change_pct_vs_accepted_matched_control": round(100 * (tok / accepted["tok"] - 1), 4),
            "change_pct_vs_v11_starting_code": round(100 * (tok / original["tok"] - 1), 4),
            "change_pct_vs_reported_v10": round(100 * (tok / V10_REPORTED[name] - 1), 4),
            "numeric_3pct_progress_vs_accepted": progress,
            "numeric_fixed_minimum_5pct_crossed": fixed_min,
            "numeric_fixed_stretch_10pct_crossed": fixed_stretch,
            "numeric_matched_minimum_5pct_crossed": matched_min,
            "numeric_matched_stretch_10pct_crossed": matched_stretch,
            "numeric_minimum_candidate": fixed_min and matched_min,
            "numeric_stretch_candidate": fixed_stretch and matched_stretch,
            "accepted_mean_ttft_ms": accepted["ttft"],
            "candidate_mean_ttft_ms": new["ttft"],
            "ttft_change_pct_vs_accepted": round(
                100 * (new["ttft"] / accepted["ttft"] - 1), 4
            ),
        }
    return {
        "valid_half_run_csvs": True,
        "numeric_progress_checkpoint_candidate": any(
            item["numeric_3pct_progress_vs_accepted"] for item in comparisons.values()
        ),
        "numeric_minimum_candidate": any(
            item["numeric_minimum_candidate"] for item in comparisons.values()
        ),
        "numeric_stretch_candidate": any(
            item["numeric_stretch_candidate"] for item in comparisons.values()
        ),
        "reproducibility_verified": False,
        "production_kernel_verified": False,
        "accuracy": "deferred_to_human_MATH500",
        "comparison": comparisons,
        "note": (
            "CSV integrity and numeric thresholds only: NO proof of repeatability, "
            "causal speedup, actual operator execution, accuracy or acceptable "
            "two-workload tradeoff. Check independently repeated A/B half-runs, "
            "source/benchmark provenance, GPU correctness and untouched benchmark "
            "before any progress or final PR. Exit code zero is NOT a performance win."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-summary", "baseline-raw",
                 "starting-summary", "starting-raw",
                 "candidate-summary", "candidate-raw"):
        parser.add_argument("--" + name, required=True, type=Path)
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
