"""S32 printed-trace triage: compare S29, S30 and candidate on fixed IDs.

Logs are not enough to prove that a guarded shell edit was correct.
Treat the output as an investigation gate, never an auto-promote signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from compare_submission24 import S1_IDS
from gate_submission29 import inspect_printout
from gate_submission30 import graph_usage

S29_WINS = frozenset({"rich_3905", "rich_3518", "rich_3454"})
S30_GAIN_WINS = frozenset({"rich_4077", "rich_3882"})
PILOT_IDS = ("rich_4077", "rich_3882", "rich_3905", "rich_3518",
             "rich_3454", "rich_4076", "rich_3934")
TASK = re.compile(r"^\[candidate \d+/\d+\] Evaluating (\S+)")
EVENT = re.compile(r"^\s*\[\s*[\d.]+s\]\s*(.*)$")


def tool_telemetry(raw: str) -> dict:
    """Count visible native actions, failed edits and recovery markers."""
    by_task: dict[str, dict] = {}
    current = None
    for line in raw.splitlines():
        m = TASK.match(line)
        if m:
            current = m[1]
            by_task[current] = {
                "graph_calls": 0, "write_file_calls": 0,
                "edit_calls": 0, "native_edit_failures": 0,
                "guard_markers": 0, "submit_calls": 0,
            }
            continue
        e = EVENT.match(line)
        if not e or current is None:
            continue
        action = e[1]
        metrics = by_task[current]
        if re.search(r"\b(?:get_code_neighbors|search_similar_code|get_code_subgraph)\b",
                     action) and not any(x in action for x in ("→", "❌", "✓")):
            metrics["graph_calls"] += 1
        if "write_file" in action and not any(x in action for x in ("→", "❌", "✓")):
            metrics["write_file_calls"] += 1
        if "✏️ edit_file" in action:
            metrics["edit_calls"] += 1
        if "FileEditError" in action or (
                "❌" in action and "old_string not found" in action):
            metrics["native_edit_failures"] += 1
        if "S32_GUARDED_LINE_EDIT_OK" in action:
            metrics["guard_markers"] += 1
        if "🤖 submit_patch" in action:
            metrics["submit_calls"] += 1
    return {"by_task": by_task, "totals": {
        key: sum(x[key] for x in by_task.values())
        for key in ("graph_calls", "write_file_calls", "edit_calls",
                    "native_edit_failures", "guard_markers", "submit_calls")
    }}


def assess(s29_text: str, s30_text: str, candidate_text: str,
           stage: str, variant: str) -> dict:
    if stage not in ("pilot", "s1") or variant not in ("primary", "control"):
        raise ValueError("Invalid stage/variant")
    baseline = [inspect_printout(text) for text in (s29_text, s30_text)]
    candidate = inspect_printout(candidate_text)
    old = [{r["id"]: r for r in item["tasks"]} for item in baseline]
    rows = {r["id"]: r for r in candidate["tasks"]}
    expected = set(PILOT_IDS if stage == "pilot" else S1_IDS)
    failures: list[str] = []
    def require(ok: bool, code: str) -> None:
        if not ok:
            failures.append(code)

    require(all(set(d) == set(S1_IDS) for d in old),
            "baseline_task_set_mismatch")
    require(all(old[0].get(n, {}).get("resolved") for n in S29_WINS),
            "S29_successes_not_confirmed")
    require(all(old[1].get(n, {}).get("resolved") for n in S30_GAIN_WINS),
            "S30_new_successes_not_confirmed")
    require(set(rows) == expected, "candidate_task_set_mismatch")
    require(all(r["resolved"] is not None and r["patch_chars"] is not None
                and r["charged_calls"] is not None for r in rows.values()),
            "incomplete_task_outcomes")
    wins = {n for n, row in rows.items() if row["resolved"] is True}
    require(len(wins) >= (4 if stage == "pilot" else 5),
            "below_resolution_target")
    require(S29_WINS <= wins, "lost_S29_confirmed_win")
    require(bool(S30_GAIN_WINS & wins), "lost_both_S30_gains")
    require(sum(r["patch_chars"] == 0 for r in rows.values()) <= 1,
            "excessive_empty_patches")
    for name, row in rows.items():
        require(row["patch_chars"] is not None and row["patch_chars"] <= 50000,
                "oversized_patch:" + name)
        require(row["largest_diff_stat_lines"] <= 1000,
                "unbounded_lines:" + name)
        require(row["failed_edits"] <= 8, "edit_loop:" + name)
        require(row["missing_edit_arguments"] <= 3,
                "schema_loop:" + name)
        require(row["max_command_repetitions"] <= 4,
                "repeat_loop:" + name)
    require(sum(r["failed_edits"] for r in rows.values()) <= 45,
            "aggregate_edit_failures")
    telemetry = tool_telemetry(candidate_text)
    require(telemetry["totals"]["graph_calls"] == 0,
            "unexpected_graph_tools")
    require(telemetry["totals"]["write_file_calls"] == 0,
            "forbidden_write_file")
    shell_mutations = sorted(n for n, row in rows.items()
                             if row["unsafe_shell_commands"])
    # The guarded fallback is potentially treated as a shell mutation by
    # the older S29 parser, which cannot see complete commands. Allow a
    # preliminary result ONLY when a visible success marker exists; even
    # then mandatory ATIF review is needed before trusting the shell edit.
    guard_tasks = {name for name, metrics in telemetry["by_task"].items()
                   if metrics["guard_markers"]}
    if variant == "control":
        require(not shell_mutations, "control_used_shell_mutation")
        require(not guard_tasks, "control_used_guard")
    else:
        require(set(shell_mutations) <= guard_tasks,
                "shell_mutation_without_guard_evidence")
    return {
        "candidate": "submission_32",
        "variant": variant,
        "stage": stage,
        "preliminary_gate_pass": not failures,
        "promotion_approved": False,
        "resolved": len(wins),
        "resolved_ids": sorted(wins),
        "retained_S29": sorted(wins & S29_WINS),
        "retained_S30_new": sorted(wins & S30_GAIN_WINS),
        "new_vs_S30": sorted(wins - {n for n in expected
                                     if old[1].get(n, {}).get("resolved")}),
        "failures": failures,
        "empty_patches": sum(r["patch_chars"] == 0 for r in rows.values()),
        "failed_edits": sum(r["failed_edits"] for r in rows.values()),
        "shell_mutation_tasks": shell_mutations,
        "guard_marker_tasks": sorted(guard_tasks),
        "telemetry": telemetry,
        "manual_ATIF_review_required": bool(shell_mutations or guard_tasks),
        "manual_requirements": [
            "Check whole exact shell command, line matching and guard checks in ATIF",
            "Inspect post-edit source patch and reject unrelated file changes",
            "Verify issue-specific input/output and normal-case regression tests",
            "Require two matched real E4B repeats per variant and same time limit",
            "Do not compare to S29/S30 if time-limit protocols differ",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s29", type=Path)
    parser.add_argument("s30", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--stage", choices=["pilot", "s1"], required=True)
    parser.add_argument("--variant", choices=("primary", "control"),
                        required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = assess(args.s29.read_text(encoding="utf-8"),
                    args.s30.read_text(encoding="utf-8"),
                    args.candidate.read_text(encoding="utf-8"),
                    args.stage, args.variant)
    body = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")
    if not report["preliminary_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
