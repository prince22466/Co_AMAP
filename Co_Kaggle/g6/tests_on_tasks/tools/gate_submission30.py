"""Post-run S30 promotion triage: >=5/15 genuine resolutions, retain S29 wins.

Printed logs omit full native tool arguments, actual diffs and precise test
assertions. This gate cannot enforce policy inside E4B or prove patch behavior;
manual ATIF/source and official result inspection remain mandatory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from compare_submission24 import S1_IDS
from gate_submission29 import inspect_printout, PILOT_IDS as OLD_PILOT

S29_WINS = frozenset({"rich_3905", "rich_3518", "rich_3454"})
PILOT_IDS = ("rich_4077", "rich_4076", "rich_3934", "rich_3905",
             "rich_3518", "rich_3454", "rich_3938")
GRAPH_TOOLS = ("search_similar_code", "get_code_neighbors", "get_code_subgraph")
EVENT = re.compile(r"^\s*\[\s*[\d.]+s\]\s*(.*)$")
TASK = re.compile(r"^\[candidate \d+/\d+\] Evaluating (\S+)")


def graph_usage(raw: str) -> dict:
    """Count timestamped calls only; ignore duplicate final detail cards."""
    calls = {name: 0 for name in GRAPH_TOOLS}
    by_task: dict[str, dict[str, int]] = {}
    current = None
    for line in raw.splitlines():
        m = TASK.match(line)
        if m:
            current = m[1]
            by_task[current] = {name: 0 for name in GRAPH_TOOLS}
            continue
        event = EVENT.match(line)
        if not event or current is None:
            continue
        action = event[1]
        for name in GRAPH_TOOLS:
            if re.search(r"\b" + name + r"\b", action) and not (
                "→" in action or "❌" in action or "✓" in action
            ):
                calls[name] += 1
                by_task[current][name] += 1
                break
    return {"tool_calls": calls, "calls_by_task": by_task,
            "tasks_using_graph": sum(any(v.values()) for v in by_task.values())}


def assess(baseline_text: str, candidate_text: str, stage: str) -> dict:
    if stage not in ("pilot", "s1"):
        raise ValueError("stage must be pilot or s1")
    baseline = inspect_printout(baseline_text)
    candidate = inspect_printout(candidate_text)
    old_rows = {r["id"]: r for r in baseline["tasks"]}
    rows = {r["id"]: r for r in candidate["tasks"]}
    expected = set(PILOT_IDS if stage == "pilot" else S1_IDS)
    failures = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    require(set(old_rows) == set(S1_IDS), "S29_baseline_not_complete_S1")
    require(set(rows) == expected, "candidate_task_ids_mismatch")
    require(all(r["resolved"] is not None and r["patch_chars"] is not None
                and r["charged_calls"] is not None for r in rows.values()),
            "incomplete_candidate_results")
    require(all(old_rows.get(n, {}).get("resolved") for n in S29_WINS),
            "S29_reference_wins_unverified")
    solved = {n for n, r in rows.items() if r["resolved"] is True}
    required = 5
    require(len(solved) >= required, f"resolved_below_{required}")
    require(S29_WINS <= solved, "lost_S29_confirmed_success")
    require(sum(r["patch_chars"] == 0 for r in rows.values()) <= 1,
            "too_many_empty_patches")
    for name, row in rows.items():
        require(row["patch_chars"] is None or row["patch_chars"] <= 50000,
                f"oversized_patch:{name}")
        require(row["largest_diff_stat_lines"] <= 1000,
                f"excessive_changed_lines:{name}")
        require(not row["unsafe_shell_commands"],
                f"unsafe_shell_mutation:{name}")
        require(row["failed_edits"] <= 8, f"edit_failure_loop:{name}")
        require(row["missing_edit_arguments"] <= 3,
                f"tool_schema_loop:{name}")
        require(row["max_command_repetitions"] <= 4,
                f"command_repetition:{name}")
    require(sum(r["failed_edits"] for r in rows.values()) <= 45,
            "aggregate_failed_edits_too_high")

    before = {n for n in expected if old_rows.get(n, {}).get("resolved")}
    usage = graph_usage(candidate_text)
    return {
        "stage": stage,
        "status": "PASS_PRINTED_TRACE_TRIAGE" if not failures else "REJECT",
        "promotion_approved": False,
        "pass_printed_trace_gate": not failures,
        "candidate_resolved": len(solved),
        "target": f">=5/{len(expected)} and retain S29's three wins in EACH matched repeat",
        "resolved_ids": sorted(solved),
        "gained_vs_s29": sorted(solved - before),
        "lost_vs_s29": sorted(before - solved),
        "empty_patches": sum(r["patch_chars"] == 0 for r in rows.values()),
        "failed_edits": sum(r["failed_edits"] for r in rows.values()),
        "failures": failures,
        "graph_usage": usage,
        "manual_checks": [
            "Inspect official resolved booleans and exact task IDs from every repeat",
            "Review ATIF full graph and edit arguments, actual final patch and diff",
            "Verify tests assert requested behavior, not merely pytest exit 0",
            "Inspect environment and package manifests; no grader changes",
            "Do not select best run: require 5/15 across BOTH S1 repeats",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="S29 15-task printed trace")
    parser.add_argument("candidate", type=Path, help="One matched S30 trace")
    parser.add_argument("--stage", choices=["pilot", "s1"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = assess(args.baseline.read_text(encoding="utf-8"),
                    args.candidate.read_text(encoding="utf-8"), args.stage)
    body = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")
    if not report["pass_printed_trace_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
