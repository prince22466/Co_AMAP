#!/usr/bin/env python3
"""Validate unchanged two-run 4K/16K half-run CSVs and report numeric comparisons.

The FIXED v9 reference is reported in profiling/v9/summary.txt. The separate
baseline CLI files must be a matched last-accepted v10 control; never assume
the reported v9 summary is matched current-machine raw evidence.
The utility does not run/modify any benchmark or establish reproducibility.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CASES = {4096: (1024, 64, 256), 16384: (1024, 64, 128)}
V9_REPORTED = {"4K": 2256.31, "16K": 1518.06}
FINAL_GAIN = 0.06
TARGETS = {name: value * (1 + FINAL_GAIN) for name, value in V9_REPORTED.items()}
PROGRESS_GAIN = 0.05  # repeatable interim checkpoint vs last accepted matched v10 control


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
            raise ValueError(
                f"{path}: {size} needs exactly {required} "
                f"{'raw runs' if raw else 'summary row'}"
            )
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
        checkpoint = b["tok"] + 1e-9 >= a["tok"] * (1 + PROGRESS_GAIN)
        final_goal = b["tok"] + 1e-9 >= TARGETS[label]
        matched_uplift = b["tok"] > a["tok"] + 1e-9
        # The static v9 target may be crossed solely because of baseline drift.
        goal_with_uplift = final_goal and matched_uplift
        comparison[label] = {
            "matched_v10_control_total_tok_s": a["tok"],
            "candidate_total_tok_s": b["tok"],
            "throughput_change_pct_vs_matched_v10_control": round(
                100 * (b["tok"] / a["tok"] - 1), 4
            ),
            "v9_reported_total_tok_s": V9_REPORTED[label],
            "fixed_goal_total_tok_s": TARGETS[label],
            "throughput_change_pct_vs_reported_v9": round(
                100 * (b["tok"] / V9_REPORTED[label] - 1), 4
            ),
            "numeric_5pct_gain_observed": checkpoint,
            "numeric_6pct_vs_v9_goal_observed": final_goal,
            "numeric_matched_uplift_observed": matched_uplift,
            "numeric_goal_with_matched_uplift_candidate": goal_with_uplift,
            "baseline_mean_ttft_ms": a["ttft"],
            "candidate_mean_ttft_ms": b["ttft"],
            "ttft_change_pct": round(100 * (b["ttft"] / a["ttft"] - 1), 4),
        }

    five_pct = any(item["numeric_5pct_gain_observed"] for item in comparison.values())
    goal = any(item["numeric_6pct_vs_v9_goal_observed"] for item in comparison.values())
    goal_with_uplift = any(
        item["numeric_goal_with_matched_uplift_candidate"] for item in comparison.values()
    )
    return {
        "valid_half_run_csvs": True,
        "numeric_positive_change_observed": any(
            item["candidate_total_tok_s"] > item["matched_v10_control_total_tok_s"]
            for item in comparison.values()
        ),
        "numeric_5pct_gain_observed": five_pct,
        "numeric_6pct_vs_v9_goal_observed": goal,
        "numeric_goal_with_matched_uplift_candidate": goal_with_uplift,
        "numeric_checkpoint_candidate": five_pct or goal_with_uplift,
        "reproducibility_verified": False,
        "production_kernel_verified": False,
        "accuracy": "deferred_to_user",
        "comparison": comparison,
        "note": (
            "CSV validation/numeric observation ONLY. The v9 anchor is reported; "
            "prove actual gain with matched, repeated v10 control A/B, GPU correctness "
            "and production invocation. 5% checkpoint vs current v10 control; "
            "6% final threshold vs frozen reported v9; crossing it without "
            "positive matched-v10 uplift is NOT a checkpoint candidate. "
            "Numeric indicators do not establish causal gain, repeatability, "
            "model accuracy or acceptable combined competition scoring. "
            "Exit 0 is not success."
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
