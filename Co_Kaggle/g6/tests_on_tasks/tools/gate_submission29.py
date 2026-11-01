"""Fail-closed *promotion* gate for real E4B S29 printed-task traces.

This does not intercept native tools during execution. Printed traces omit
full call arguments, so same-command counts are only visible-line evidence.
Inspect ATIF, actual final diffs, and behavioral assertions manually too.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from analyze_submission27_traces import analyze
from compare_submission28 import compare

TASK = re.compile(r"^\[candidate \d+/\d+\] Evaluating (\S+)")
EVENT = re.compile(r"^\s*\[\s*[\d.]+s\]\s*(.*)$")
RESULT = re.compile(r"-> resolved=(True|False), patch_chars=(\d+), tools=(\d+)")
DIFF_STAT = re.compile(r"\|\s*(\d+)\s*[+\-]")
SHELL_MUTATION = re.compile(
    r"\bsed\s+-i(?:\s|['\"]|$)|\bperl\s+-pi\b|"
    r"\b(?:write_text|write_bytes)\s*\(|"
    r"\b(?:cat|printf|echo)\b[^\n]*\s(?:>>?|tee)\s+\S",
    re.IGNORECASE,
)
KNOWN_SUCCESSES = frozenset({"rich_3905", "rich_3454"})
PILOT_IDS = frozenset({"rich_4077", "rich_3905", "rich_3518",
                       "rich_3454", "rich_3882"})
S1_COUNT = 15


def inspect_printout(raw: str) -> dict:
    """Read individual timestamped tool events, avoiding final-card duplicates."""
    data: dict[str, dict] = {}
    current = None
    latest_tool = None
    for line in raw.splitlines():
        task = TASK.match(line)
        if task:
            current = task[1]
            if current in data:
                raise ValueError(f"Duplicate candidate task: {current}")
            data[current] = {
                "id": current, "resolved": None, "patch_chars": None,
                "charged_calls": None, "failed_edits": 0,
                "missing_edit_arguments": 0, "unsafe_shell_commands": [],
                "max_command_repetitions": 0, "largest_diff_stat_lines": 0,
                "targeted_test_calls": 0,
            }
            data[current]["_commands"] = Counter()
            latest_tool = None
            continue
        if current is None:
            continue
        row = data[current]
        result = RESULT.search(line)
        if result:
            row["resolved"] = result[1] == "True"
            row["patch_chars"] = int(result[2])
            row["charged_calls"] = int(result[3])
            latest_tool = None
            continue
        stat = DIFF_STAT.search(line)
        if stat and " | " in line:
            row["largest_diff_stat_lines"] = max(
                row["largest_diff_stat_lines"], int(stat[1]))
        event = EVENT.match(line)
        if event:
            action = event[1]
            if "🔧 run_command " in action:
                command = action.split("🔧 run_command ", 1)[1].strip()
                row["_commands"][command] += 1
                if SHELL_MUTATION.search(command):
                    row["unsafe_shell_commands"].append(command[:180])
                if re.search(r"\b(pytest|unittest)\b", command):
                    row["targeted_test_calls"] += 1
                latest_tool = "command"
            elif "✏️ edit_file " in action:
                latest_tool = "edit"
            elif "📖 read_file " in action:
                latest_tool = "read"
            else:
                latest_tool = None
            continue
        if latest_tool == "edit" and "❌" in line:
            row["failed_edits"] += 1
            if "mandatory input parameters" in line:
                row["missing_edit_arguments"] += 1
            latest_tool = None
    rows = []
    for row in data.values():
        row["max_command_repetitions"] = max(
            row.pop("_commands").values(), default=0)
        rows.append(row)
    return {
        "tasks": rows,
        "task_count": len(rows),
        "results_count": sum(t["resolved"] is not None for t in rows),
    }


def assess(baseline_text: str, candidate_text: str, stage: str) -> dict:
    """Assess both correctness and safety; never infer performance from length."""
    if stage not in ("pilot", "s1"):
        raise ValueError("stage must be 'pilot' or 's1'")
    baseline = analyze(baseline_text)
    candidate = analyze(candidate_text)
    paired = compare(baseline, candidate)
    trace = inspect_printout(candidate_text)
    rows = trace["tasks"]
    done = [t for t in rows if t["resolved"] is not None]
    target = PILOT_IDS if stage == "pilot" else None
    expected = len(PILOT_IDS) if stage == "pilot" else S1_COUNT
    observed = {t["id"] for t in done}
    passes = []
    def require(condition: bool, message: str) -> None:
        if not condition:
            passes.append(message)
    require(len(done) == expected, f"incomplete_results: expected={expected}, got={len(done)}")
    if target is not None:
        require(observed == target, f"pilot_task_ids_mismatch: {sorted(observed ^ target)}")
    require(not paired["missing_from_candidate"], "baseline_tasks_missing")
    require(not paired["missing_from_baseline"], "candidate_has_unmatched_tasks")
    wins = {t["id"] for t in done if t["resolved"]}
    require(KNOWN_SUCCESSES <= wins, "lost_known_s27_win")
    require(len(wins) >= 3, "fewer_than_three_resolved")
    require(sum(t["patch_chars"] == 0 for t in done) <= 1, "too_many_empty_patches")
    for t in rows:
        name = t["id"]
        require(not t["unsafe_shell_commands"], f"unsafe_shell_edit:{name}")
        require(t["patch_chars"] is None or t["patch_chars"] <= 50_000,
                f"oversized_patch:{name}")
        require(t["largest_diff_stat_lines"] <= 1000,
                f"excessive_changed_lines:{name}")
        require(t["failed_edits"] <= 16, f"edit_failure_loop:{name}")
        require(t["missing_edit_arguments"] <= 3, f"bad_tool_schema_loop:{name}")
        require(t["max_command_repetitions"] <= 4, f"repeat_command_loop:{name}")
    require(all(t["charged_calls"] is not None for t in rows),
            "missing_charged_call_count")
    return {
        "scope": ("Printed-trace release triage only; no runtime interception, "
                  "no hidden-test/ATIF proof. Manual patch review required."),
        "stage": stage,
        "pass_promotion_gate": not passes,
        "failures": passes,
        "resolved_ids": sorted(wins),
        "matched_task_comparison": paired,
        "trace": trace,
        "mandatory_manual_checks": [
            "Inspect full ATIF arguments and confirm valid native tool dispatch",
            "Review actual final diff, changed file list and relevant assertions",
            "Repeat a genuine E4B pilot; do not substitute scripted model tests",
            "Verify no benchmark/grading files were altered",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--stage", choices=["pilot", "s1"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    outcome = assess(args.baseline.read_text(encoding="utf-8"),
                     args.candidate.read_text(encoding="utf-8"), args.stage)
    payload = json.dumps(outcome, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if not outcome["pass_promotion_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
