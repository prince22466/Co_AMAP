"""Post-run S33 diagnostic triage (not an official grading or promotion test).

Checks fixed S1 task parity, retained wins, gross source safety and trace
red flags. Printed traces may omit tool arguments; review raw ATIF patches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from compare_submission24 import S1_IDS
from gate_submission29 import inspect_printout
from gate_submission32 import tool_telemetry


S29_WINS = frozenset(("rich_3905", "rich_3518", "rich_3454"))
S30_GAINS = frozenset(("rich_4077", "rich_3882"))
SUMMARY = re.compile(
    r"^\d+\t([^\t]+)\t(True|False)\t(\d+)\t([\d.]+)\t",
    re.MULTILINE,
)


def summary_rows(raw: str) -> dict[str, dict]:
    """Use final 15-row scored results, even when detailed events are elided."""
    rows: dict[str, dict] = {}
    for task, resolved, patch_chars, duration in SUMMARY.findall(raw):
        if task in rows:
            raise ValueError("Duplicate scored task row: " + task)
        rows[task] = {
            "id": task, "resolved": resolved == "True",
            "patch_chars": int(patch_chars),
            "duration_seconds": float(duration),
        }
    return rows


def assess(s29_raw: str, s30_raw: str, candidate_raw: str) -> dict:
    baseline29, baseline30, observed = [
        summary_rows(raw) for raw in (s29_raw, s30_raw, candidate_raw)
    ]
    detail = {
        row["id"]: row
        for row in inspect_printout(candidate_raw)["tasks"]
    }
    required = set(S1_IDS)
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    check(set(baseline29) == required and set(baseline30) == required,
          "baseline_task_set_mismatch")
    check(all(baseline29.get(i, {}).get("resolved") for i in S29_WINS),
          "baseline29_missing_expected_wins")
    check(all(baseline30.get(i, {}).get("resolved") for i in S30_GAINS),
          "baseline30_missing_expected_wins")
    check(set(observed) == required, "candidate_task_set_mismatch")
    check(len(observed) == len(S1_IDS),
          "duplicate_or_missing_task_rows")
    check(set(detail) == required, "incomplete_detailed_tool_events")

    wins = {
        name for name, row in observed.items() if row.get("resolved") is True
    }
    check(len(wins) >= 5, "below_five_resolved")
    check(S29_WINS <= wins, "previous_S29_wins_not_retained")
    check(bool(S30_GAINS & wins), "no_S30_new_win_retained")
    errors = 0
    unsafe = []
    too_slow = []
    for name, row in observed.items():
        recorded = detail.get(name, {})
        count = recorded.get("failed_edits") or 0
        errors += count
        if row.get("duration_seconds") is not None and row[
                "duration_seconds"] > 240:
            too_slow.append(name)
        check(count <= 8, "excessive_native_edit_errors:" + name)
        patch_size = row.get("patch_chars")
        check(patch_size is not None and 0 < patch_size <= 50000,
              "empty_or_oversize_patch:" + name)
        check((recorded.get("largest_diff_stat_lines") or 0) <= 1000,
              "large_diff_numstat:" + name)
        if recorded.get("unsafe_shell_commands"):
            unsafe.append(name)
    check(errors <= 45, "too_many_native_edit_errors")
    check(not unsafe, "shell_source_mutation_observed")

    trace = tool_telemetry(candidate_raw)
    check(trace["totals"]["graph_calls"] == 0, "unexpected_graph_tool")
    check(trace["totals"]["write_file_calls"] == 0, "unexpected_write_file")
    check(trace["totals"]["guard_markers"] == 0, "deprecated_shell_fallback")
    repeated_submits = [
        name for name, metrics in trace["by_task"].items()
        if metrics["submit_calls"] > 1
    ]
    check(not repeated_submits, "repeat_submit_patch_calls")

    return {
        "candidate": "submission_33",
        "evaluated_task_ids": sorted(observed),
        "resolved": len(wins),
        "resolved_ids": sorted(wins),
        "s29_retained": sorted(wins & S29_WINS),
        "s30_new_retained": sorted(wins & S30_GAINS),
        "empty_patches": sum((r.get("patch_chars") or 0) == 0
                             for r in observed.values()),
        "native_edit_failures_reported": errors,
        "unsafe_shell_tasks": sorted(unsafe),
        "repeated_submit_tasks": sorted(repeated_submits),
        "over_four_minute_tasks": sorted(too_slow),
        "printed_trace_details_present": sorted(trace["by_task"]),
        "scored_summary_rows": len(observed),
        "triage_passed": not failures,
        "failures": failures,
        "promotion_approved": False,
        "note": ("One printout is insufficient. Require two complete matched "
                 "real E4B runs, 4-minute comparison, exact patch and ATIF "
                 "manual review. No printed trace proves absence of unsafe "
                 "shell calls or correctness of behavior checks."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s29", type=Path)
    parser.add_argument("s30", type=Path)
    parser.add_argument("s33", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = assess(*(p.read_text(encoding="utf-8") for p in
                      (args.s29, args.s30, args.s33)))
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")
    if not report["triage_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
