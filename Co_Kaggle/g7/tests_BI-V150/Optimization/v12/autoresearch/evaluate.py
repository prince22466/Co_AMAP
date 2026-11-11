#!/usr/bin/env python3
"""Read-only v12 half-run CSV integrity and matched A/B numeric comparison.

This does not run benchmarks or certify repeatability, accuracy, or eligibility.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CASES = {4096: (1024, 64, 256), 16384: (1024, 64, 128)}
REPORTED_V10 = {4096: 2372.38, 16384: 1872.70}


def _positive(value: str, name: str) -> float:
    num = float(value)
    if not math.isfinite(num) or num <= 0:
        raise ValueError(f"Invalid {name}: {value!r}")
    return num


def read_rows(path: Path, *, raw: bool) -> dict[int, list[dict]]:
    results = {4096: [], 16384: []}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        for row in reader:
            size = int(row["Prefill"])
            if size not in CASES:
                raise ValueError(f"Unexpected workload {size} in {path}")
            shape = tuple(int(row[k]) for k in ("Decode", "Conc", "Num Prompts"))
            if shape != CASES[size]:
                raise ValueError(f"Wrong shape for {size}: {shape}")
            if raw:
                if row["Run Status"] != "SUCCESS":
                    raise ValueError(f"Failed run for {size}: {row['Run Status']}")
                if int(row["Successful Requests"]) != CASES[size][2]:
                    raise ValueError(f"Incomplete requests for {size}")
            results[size].append({
                "tok": _positive(row["Total tok/s"], "Total tok/s"),
                "ttft": _positive(row["Mean TTFT (ms)"], "Mean TTFT"),
            })
    expected_count = 2 if raw else 1
    for size in CASES:
        if len(results[size]) != expected_count:
            raise ValueError(
                f"Expected {expected_count} {'raw' if raw else 'summary'} "
                f"rows for {size}, found {len(results[size])}"
            )
    return results


def load_pair(summary_path: Path, raw_path: Path) -> dict[int, dict]:
    summary = read_rows(summary_path, raw=False)
    raw = read_rows(raw_path, raw=True)
    result = {}
    for size in CASES:
        for metric in ("tok", "ttft"):
            avg = sum(row[metric] for row in raw[size]) / 2
            if abs(summary[size][0][metric] - avg) > 0.05:
                raise ValueError(f"{size} {metric}: summary disagrees with raw runs")
        result[size] = {
            **summary[size][0],
            "raw_tok": [r["tok"] for r in raw[size]],
        }
    return result


def evaluate(args: argparse.Namespace) -> dict:
    control = load_pair(args.control_summary, args.control_raw)
    candidate = load_pair(args.candidate_summary, args.candidate_raw)
    rows = {}
    for size in CASES:
        base, trial = control[size], candidate[size]
        anchor = REPORTED_V10[size]
        gain = trial["tok"] / base["tok"] - 1
        rows[f"{size // 1024}K"] = {
            "control_tok_s": base["tok"],
            "candidate_tok_s": trial["tok"],
            "control_raw_tok_s": base["raw_tok"],
            "candidate_raw_tok_s": trial["raw_tok"],
            "reported_v10_anchor_tok_s": anchor,
            "matched_gain_pct": round(100 * gain, 4),
            "vs_v10_pct": round(100 * (trial["tok"] / anchor - 1), 4),
            "ttft_change_pct": round(
                100 * (trial["ttft"] / base["ttft"] - 1), 4
            ),
            "numeric_progress_3pct": gain + 1e-12 >= 0.03,
            "numeric_minimum_5pct": (
                gain + 1e-12 >= 0.05
                and trial["tok"] + 1e-9 >= anchor * 1.05
            ),
            "numeric_stretch_10pct": (
                gain + 1e-12 >= 0.10
                and trial["tok"] + 1e-9 >= anchor * 1.10
            ),
        }
    return {
        "valid_half_run_csvs": True,
        "numeric_progress_3pct": any(r["numeric_progress_3pct"] for r in rows.values()),
        "numeric_minimum_5pct": any(r["numeric_minimum_5pct"] for r in rows.values()),
        "numeric_stretch_10pct": any(r["numeric_stretch_10pct"] for r in rows.values()),
        "replication_verified": False,
        "model_quality_verified": False,
        "production_kernel_verified": False,
        "comparison": rows,
        "warning": "Numeric screen only; independent matched A/B, kernel provenance, "
                   "model ability and unchanged MATH-500 remain unverified.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("control-summary", "control-raw", "candidate-summary", "candidate-raw"):
        parser.add_argument("--" + opt, required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"valid_half_run_csvs": False, "error": str(exc)}
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output)
    print(output, end="")
    return 0 if report["valid_half_run_csvs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
