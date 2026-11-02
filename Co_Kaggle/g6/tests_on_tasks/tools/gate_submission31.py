"""Post-run printed-trace promotion triage for Submission 31.

This does NOT replace official resolved flags, ATIF tool argument inspection,
source diff/behavior review, or matched E4B repeats. Printed traces truncate
tool arguments and suppress some output; these are observable proxies only.
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
S30_NEW_WINS = frozenset({"rich_4077", "rich_3882"})
PILOT_IDS = ("rich_4077", "rich_3882", "rich_3518", "rich_3905",
             "rich_3454", "rich_4076", "rich_3934")
STAGE = re.compile(r"^\[candidate \d+/\d+\] Evaluating (\S+)")
CALL = re.compile(r"^\s*\[\s*[\d.]+s\]\s*🔧 run_command\s+(.+)$")


def inspect_verification(raw: str) -> dict:
    """Count test commands only AFTER a successful edit and BEFORE submit.

    Printed traces can establish timing and edit success, but cannot prove
    an assertion targets the issue or that a test actually passed. That
    still requires inspecting the full ATIF and requested behavior.
    """
    tasks: dict[str, list[str]] = {}
    current = None
    pending_edit = False
    edited = False
    submitted = False
    for line in raw.splitlines():
        match = STAGE.match(line)
        if match:
            current = match[1]
            tasks[current] = []
            pending_edit = edited = submitted = False
            continue
        if current is None:
            continue
        # Only timestamped lines are actual call/response events; card
        # summaries repeat tool names and must never count as new calls.
        timestamped = re.match(r"^\s*\[\s*[\d.]+s\]\s*(.*)$", line)
        if not timestamped:
            continue
        event = timestamped[1]
        if re.search(r"✏️ edit_file\b", event):
            pending_edit = True
            continue
        if pending_edit and re.search(r"✓\s*\{", event):
            edited = True
            pending_edit = False
            continue
        if pending_edit and "❌" in event:
            pending_edit = False
            continue
        if re.search(r"🤖 submit_patch\b", event):
            submitted = True
            continue
        if not edited or submitted:
            continue
        command = CALL.match(line)
        if command:
            cmd = command[1].strip()
            if (re.search(r"\bpytest\s+\S*(?:test_|tests/|::)", cmd)
                or re.search(r"\bpython(?:3)?\s+(?:-c\b|-\s*<<)", cmd)
                or re.search(r"\bpython(?:3)?\s+-m\s+unittest\s+\S+", cmd)):
                tasks[current].append(cmd[:200])
    return {
        "tasks_with_test_attempt": sum(bool(x) for x in tasks.values()),
        "task_commands": {name: commands for name, commands in tasks.items()
                          if commands},
        "qualification": (
            "Only post-successful-edit, pre-submit test attempts counted; "
            "their relevance and PASS/FAIL still require ATIF/manual review"
        ),
    }


def inspect_forbidden_tools(raw: str) -> dict:
    """Spot prohibited tool calls even if a broken package exposes them."""
    task = None
    writes: list[str] = []
    for line in raw.splitlines():
        found = STAGE.match(line)
        if found:
            task = found[1]
        if task and re.match(r"^\s*\[\s*[\d.]+s\]\s*📝 write_file\b", line):
            writes.append(task)
    return {"write_file_task_ids": sorted(set(writes))}


def assess(s29_text: str, s30_text: str, candidate_text: str,
           stage: str = "s1") -> dict:
    if stage not in ("pilot", "s1"):
        raise ValueError("stage must be 'pilot' or 's1'")
    previous = [inspect_printout(s) for s in (s29_text, s30_text)]
    candidate = inspect_printout(candidate_text)
    prior_rows = [{r["id"]: r for r in d["tasks"]} for d in previous]
    rows = {r["id"]: r for r in candidate["tasks"]}
    expected = set(PILOT_IDS if stage == "pilot" else S1_IDS)
    failures: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            failures.append(message)

    require(all(set(r) == set(S1_IDS) for r in prior_rows),
            "baseline_S1_mismatch")
    require(all(prior_rows[0].get(t, {}).get("resolved") for t in S29_WINS),
            "S29_reference_wins_mismatch")
    require(all(prior_rows[1].get(t, {}).get("resolved") for t in S30_NEW_WINS),
            "S30_reference_wins_mismatch")
    require(set(rows) == expected, "candidate_task_set_mismatch")
    require(all(r["resolved"] is not None and
                r["patch_chars"] is not None and
                r["charged_calls"] is not None for r in rows.values()),
            "candidate_incomplete_results")
    wins = {t for t, row in rows.items() if row["resolved"] is True}
    require(len(wins) >= 5, "below_5_resolved")
    require(S29_WINS <= wins, "lost_S29_win")
    require(bool(S30_NEW_WINS & wins), "lost_both_S30_gains")
    require(sum(row["patch_chars"] == 0 for row in rows.values()) <= 1,
            "too_many_empty_patches")
    for task, row in rows.items():
        require(not row["unsafe_shell_commands"], f"unsafe_shell:{task}")
        require(row["patch_chars"] is None or row["patch_chars"] <= 50_000,
                f"oversized_patch:{task}")
        require(row["largest_diff_stat_lines"] <= 1000,
                f"large_diff_stat:{task}")
        require(row["failed_edits"] <= 8, f"edit_failure_loop:{task}")
        require(row["missing_edit_arguments"] <= 3,
                f"edit_schema_loop:{task}")
        require(row["max_command_repetitions"] <= 4,
                f"repeated_command:{task}")
    require(sum(r["failed_edits"] for r in rows.values()) <= 45,
            "excessive_failed_edits")
    forbidden = inspect_forbidden_tools(candidate_text)
    require(not forbidden["write_file_task_ids"], "write_file_was_invoked")
    usage = graph_usage(candidate_text)
    # The former S30 architecture enabled these tools but called NONE;
    # at least 3 tasks exercising graph retrieval is a diagnostic target.
    require(usage["tasks_using_graph"] >= 3, "graph_tools_not_exercised")
    verification = inspect_verification(candidate_text)
    require(verification["tasks_with_test_attempt"] >= 4,
            "too_few_targeted_test_attempts")
    s29_wins = {t for t in expected if prior_rows[0].get(t, {}).get("resolved")}
    s30_wins = {t for t in expected if prior_rows[1].get(t, {}).get("resolved")}
    return {
        "stage": stage,
        "target": ">=5 resolved, all three S29 wins, >=1 S30-new win, 2 matched E4B repeats",
        "pass_printed_trace_triage": not failures,
        "promotion_approved": False,
        "failures": failures,
        "resolved": len(wins),
        "resolved_ids": sorted(wins),
        "retained_S29": sorted(wins & s29_wins),
        "retained_S30": sorted(wins & s30_wins),
        "new_vs_S30": sorted(wins - s30_wins),
        "lost_vs_S30": sorted(s30_wins - wins),
        "empty_patches": sum(r["patch_chars"] == 0 for r in rows.values()),
        "failed_edits": sum(r["failed_edits"] for r in rows.values()),
        "graph": usage,
        "verification_attempts": verification,
        "forbidden_tool_calls": forbidden,
        "manual_checks": [
            "Confirm actual model graph tool calls and semantic relevance in ATIF",
            "Review all changed sources and actual diff for destructive edits",
            "Review targeted test input/output and whether it passed",
            "Require 2 full E4B S1 repeats at the same timeout/seed protocol",
            "Do not approve from this printout gate alone",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s29", type=Path)
    parser.add_argument("s30", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--stage", choices=["pilot", "s1"], default="s1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(args.s29.read_text(encoding="utf-8"),
                    args.s30.read_text(encoding="utf-8"),
                    args.candidate.read_text(encoding="utf-8"), args.stage)
    body = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")
    if not result["pass_printed_trace_triage"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
