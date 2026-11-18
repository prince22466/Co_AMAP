#!/usr/bin/env python3
"""Read-only v16 half-run CSV integrity and matched A/B numeric comparison.

Uses the two reported raw repeats for numeric decisions, with summary consistency checks.
This does not run benchmarks or certify matched provenance, repeatability, accuracy, or eligibility.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CASES = {4096: (1024, 64, 256), 16384: (1024, 64, 128)}
MATCHED_4K_RATIO = 3011.12 / 2817.15
METRICS = {"tok": "Total tok/s", "ttft": "Mean TTFT (ms)", "tpot": "Mean TPOT (ms)"}


def _positive(value: str, name: str) -> float:
    num = float(value)
    if not math.isfinite(num) or num <= 0:
        raise ValueError(f"Invalid {name}: {value!r}")
    return num


def _integer(value: str, name: str) -> int:
    # Archived benchmark files serialize counts as e.g. "256.0".
    num = _positive(value, name)
    if not num.is_integer():
        raise ValueError(f"Non-integral {name}: {value!r}")
    return int(num)


def read_rows(path: Path, *, raw: bool) -> dict[int, list[dict]]:
    results = {4096: [], 16384: []}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        for row in reader:
            size = _integer(row["Prefill"], "Prefill")
            if size not in CASES:
                raise ValueError(f"Unexpected workload {size} in {path}")
            shape = tuple(_integer(row[k], k) for k in ("Decode", "Conc", "Num Prompts"))
            if shape != CASES[size]:
                raise ValueError(f"Wrong shape for {size}: {shape}")
            if raw:
                if row["Run Status"] != "SUCCESS":
                    raise ValueError(f"Failed run for {size}: {row['Run Status']}")
                if _integer(row["Successful Requests"], "Successful Requests") != CASES[size][2]:
                    raise ValueError(f"Incomplete requests for {size}")
            decode, _, prompts = CASES[size]
            for column, expected in (("Total Input Tokens", size * prompts),
                                     ("Total Output Tokens", decode * prompts)):
                if _integer(row[column], column) != expected:
                    raise ValueError(f"Wrong {column} for {size}: {row[column]}")
            results[size].append({
                key: _positive(row[column], column) for key, column in METRICS.items()
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
        means = {}
        for metric in METRICS:
            avg = math.fsum(row[metric] for row in raw[size]) / 2
            means[metric] = avg
            if abs(summary[size][0][metric] - avg) > 0.05:
                raise ValueError(f"{size} {metric}: summary disagrees with raw runs")
        result[size] = {
            **means,
            "raw": {metric: [r[metric] for r in raw[size]] for metric in METRICS},
        }
    return result


def evaluate(args: argparse.Namespace) -> dict:
    control = load_pair(args.control_summary, args.control_raw)
    candidate = load_pair(args.candidate_summary, args.candidate_raw)
    rows = {}
    for size in CASES:
        base, trial = control[size], candidate[size]
        gain = trial["tok"] / base["tok"] - 1
        rows[f"{size // 1024}K"] = {
            "control_tok_s": base["tok"],
            "candidate_tok_s": trial["tok"],
            "control_raw_tok_s": base["raw"]["tok"],
            "candidate_raw_tok_s": trial["raw"]["tok"],
            "control_mean_ttft_ms": base["ttft"],
            "candidate_mean_ttft_ms": trial["ttft"],
            "control_mean_tpot_ms": base["tpot"],
            "candidate_mean_tpot_ms": trial["tpot"],
            "matched_gain_pct": round(100 * gain, 4),
            "ttft_change_pct": round(
                100 * (trial["ttft"] / base["ttft"] - 1), 4
            ),
            "tpot_change_pct": round(100 * (trial["tpot"] / base["tpot"] - 1), 4),
        }
    goal = candidate[4096]["tok"] / control[4096]["tok"] + 1e-12 >= MATCHED_4K_RATIO
    throughput_ok = candidate[16384]["tok"] >= control[16384]["tok"] - 1e-9
    ttft_ok = all(candidate[s]["ttft"] <= control[s]["ttft"] * 1.05 + 1e-9 for s in CASES)
    tpot_ok = all(candidate[s]["tpot"] <= control[s]["tpot"] * 1.05 + 1e-9 for s in CASES)
    return {
        "valid_half_run_csvs": True,
        "numeric_goal_4k": goal,
        "matched_4k_target_gain_pct": 100 * (MATCHED_4K_RATIO - 1),
        "historical_full_run_4k_target_tok_s": 3011.12,
        "historical_full_run_16k_floor_tok_s": 2073.93,
        "historical_full_run_targets_verified": False,
        "throughput_regression_screen_pass": throughput_ok,  # 16K only
        "ttft_regression_screen_pass": ttft_ok,
        "tpot_regression_screen_pass": tpot_ok,
        "numeric_screen_pass": goal and throughput_ok and ttft_ok and tpot_ok,
        "replication_verified": False,
        "model_quality_verified": False,
        "production_kernel_verified": False,
        "comparison": rows,
        "warning": "Numeric target screen on means of reported raw repeats; zero exit only validates CSVs. "
                   "Matched conditions, independent replication, competition score/latency limits, kernel provenance, "
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
