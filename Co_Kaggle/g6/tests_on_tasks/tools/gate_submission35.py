"""Fail-closed S35 post-run triage for official JSONL + optional printed traces.

This does not intercept agent tool calls, modify the competition evaluator,
prove behavior or approve promotion. Inspect raw ATIF arguments and diffs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from compare_submission24 import S1_IDS, load_rows, normalize


PILOT_IDS = ("rich_4077", "rich_3518", "rich_4076",
             "requests_7315", "rich_3905")
S29_WINS = frozenset(("rich_3905", "rich_3518", "rich_3454"))
DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
TASK = re.compile(r"^\[candidate \d+/\d+\] Evaluating (\S+)")
EVENT = re.compile(r"^\s*\[\s*[\d.]+s\]\s+(.*)$")
FORBIDDEN_PATCH_PATHS = (
    "pytest.ini", "conftest.py", "task_results.jsonl",
    ".github/", ".git/", "competition_material/", "tests_on_tasks/",
)


def patch_audit(patch: str) -> dict:
    """Conservative diff-only indicators. Does not validate task semantics."""
    if not patch.strip():
        return {"files": [], "changed_lines": 0, "flags": ["empty_patch"]}
    lines = patch.splitlines()
    paths, flags = [], []
    for line in lines:
        if line.startswith("diff --git "):
            m = DIFF_HEADER.match(line)
            if not m or m[1] != m[2]:
                flags.append("unparsable_or_rename_header")
            else:
                paths.append(m[1])
        if (line.startswith("deleted file mode") or
            line.startswith("GIT binary patch") or
            line.startswith("Binary files ")):
            flags.append("deletion_or_binary_patch")
    if not paths:
        flags.append("no_valid_diff_header")
    for path in paths:
        if (path.startswith("/")
            or ".." in Path(path).parts
            or any(path == bad or path.startswith(bad)
                   or path.endswith("/" + bad) for bad in FORBIDDEN_PATCH_PATHS)):
            flags.append("protected_path:" + path)
    changed = sum(
        line.startswith(("+", "-")) and
        not line.startswith(("+++", "---"))
        for line in lines
    )
    if len(patch) > 15000 or changed > 200 or len(paths) > 5:
        flags.append("oversized_patch")
    return {
        "files": sorted(set(paths)),
        "changed_lines": changed,
        "flags": sorted(set(flags)),
    }


def summarize(rows: list[dict], *, stage: str) -> dict:
    expected = PILOT_IDS if stage == "pilot" else S1_IDS
    checked = normalize(rows, expected=expected)
    wins, empty, flagged, errors = [], [], {}, []
    for task, row in checked.items():
        if row["resolved"]:
            wins.append(task)
        if not row["agent_patch"].strip():
            empty.append(task)
        audit = patch_audit(row["agent_patch"])
        if audit["flags"] and audit["flags"] != ["empty_patch"]:
            flagged[task] = audit
        if row.get("error") or row.get("error_message"):
            errors.append(task)
    return {
        "task_ids": sorted(checked),
        "resolved": len(wins),
        "solved_ids": sorted(wins),
        "empty_patch_ids": sorted(empty),
        "unsafe_or_unknown_patches": flagged,
        "agent_error_ids": sorted(errors),
        "retained_s29_wins": sorted(set(wins) & S29_WINS),
    }


def tool_events(text: str, *, stage: str) -> dict:
    """Printed trace proxy, not full ATIF arguments or repeat fingerprints."""
    expected = set(PILOT_IDS if stage == "pilot" else S1_IDS)
    by_task: dict[str, dict] = {}
    current: str | None = None
    pending: str | None = None
    failed_streak = 0
    streak_path: str | None = None
    for line in text.splitlines():
        m = TASK.match(line)
        if m:
            current = m[1]
            pending = None
            failed_streak = 0
            streak_path = None
            if current in by_task:
                raise ValueError("Duplicate detailed task: " + current)
            by_task[current] = dict(
                edits=0, edit_errors=0, missing_arguments=0, reads=0,
                commands=0, successful_edits=0, recovery_skill_calls=0,
                script_calls=0, max_failed_edit_streak=0,
                repeated_submit_calls=0, submits=0, diff_commands=0,
            )
            continue
        if current is None:
            continue
        match = EVENT.match(line)
        if not match:
            continue
        event = match[1]
        row = by_task[current]
        if "✏️ edit_file (" in event:
            pending = "edit"
            row["edits"] += 1
            path = event.split("✏️ edit_file (", 1)[1].split(")", 1)[0]
            if path != streak_path:
                failed_streak = 0
                streak_path = path
        elif "📖 read_file (" in event:
            row["reads"] += 1
            pending = "read"
            failed_streak = 0
        elif "🔧 run_command " in event:
            row["commands"] += 1
            pending = "command"
            failed_streak = 0
            if "git diff" in event:
                row["diff_commands"] += 1
        elif "submit_patch" in event and "🤖" in event:
            row["submits"] += 1
            pending = "submit"
        elif "load_skill" in event and "tool-recovery" in event:
            row["recovery_skill_calls"] += 1
            pending = "skill"
        elif "run_skill_script" in event:
            row["script_calls"] += 1
            pending = "script"

        failed = ("❌" in event and pending == "edit")
        if failed:
            row["edit_errors"] += 1
            row["missing_arguments"] += (
                "mandatory input parameters" in event
                or "missing required" in event.lower())
            failed_streak += 1
            row["max_failed_edit_streak"] = max(
                failed_streak, row["max_failed_edit_streak"])
            pending = None
        elif pending == "edit" and "✓ {'status': 'ok', 'filepath':" in event:
            row["successful_edits"] += 1
            failed_streak = 0
            pending = None

    for row in by_task.values():
        row["repeated_submit_calls"] = max(0, row["submits"] - 1)
    coverage = sorted(expected - set(by_task))
    extraneous = sorted(set(by_task) - expected)
    totals = {key: sum(row[key] for row in by_task.values())
              for key in ("edits", "edit_errors", "missing_arguments", "reads",
                          "commands", "successful_edits", "recovery_skill_calls",
                          "script_calls", "repeated_submit_calls", "diff_commands")}
    return {
        "coverage_missing": coverage,
        "extraneous": extraneous,
        "complete_detailed_events": not coverage and not extraneous,
        "totals": totals,
        "by_task": by_task,
        "caveat": "Truncated notebook printouts can omit detailed task blocks; same-path failed streak is a proxy.",
    }


def compare(primary: list[dict], control: list[dict], *, stage: str,
            primary_printout: str | None = None,
            control_printout: str | None = None) -> dict:
    a = summarize(primary, stage=stage)
    b = summarize(control, stage=stage)
    x, y = set(a["solved_ids"]), set(b["solved_ids"])
    min_resolved = 2 if stage == "pilot" else 5
    max_empty = 1 if stage == "pilot" else 2
    blockers = []
    if a["resolved"] < min_resolved:
        blockers.append("insufficient_resolved")
    if a["resolved"] <= b["resolved"]:
        blockers.append("no_gain_over_31b_control")
    if len(a["empty_patch_ids"]) > max_empty:
        blockers.append("too_many_empty_patches")
    if a["unsafe_or_unknown_patches"]:
        blockers.append("unsafe_or_unknown_patch")
    if a["agent_error_ids"]:
        blockers.append("agent_errors")
    if stage == "s1" and not S29_WINS <= x:
        blockers.append("lost_s29_wins")
    if primary_printout is not None:
        stats = tool_events(primary_printout, stage=stage)
        if not stats["complete_detailed_events"]:
            blockers.append("partial_tool_trace")
        if stats["totals"]["script_calls"]:
            blockers.append("unsupported_skill_script_call")
        if any(v["max_failed_edit_streak"] > 3
               for v in stats["by_task"].values()):
            blockers.append("repeated_edit_failure_loop")
        if stats["totals"]["missing_arguments"] > 4:
            blockers.append("tool_schema_failures")
    else:
        stats = None
        blockers.append("tool_traces_not_supplied")
    control_stats = (
        tool_events(control_printout, stage=stage)
        if control_printout is not None else None
    )
    return {
        "stage": stage,
        "primary": a, "no_skill_control": b,
        "gained": sorted(x - y), "lost": sorted(y - x),
        "delta_resolved": len(x) - len(y),
        "tool_telemetry": stats, "control_tool_telemetry": control_stats,
        "single_run_blockers": blockers,
        "single_run_screen_passed": not blockers,
        "promotion_approved": False,
        "limitations": [
            "Single run does not prove improvement; repeat paired real-model tasks.",
            "Printed telemetry cannot recover exact tool arguments, read ranges or patch semantics.",
            "31B model cannot be compared causally to historical E4B runs.",
            "Skill availability is not evidence the model loaded or followed it.",
            "Review full ATIF and actual final source diffs before promotion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--stage", choices=("pilot", "s1"), required=True)
    parser.add_argument("--primary-printout", type=Path)
    parser.add_argument("--control-printout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(load_rows(args.primary), load_rows(args.control),
                     stage=args.stage,
                     primary_printout=args.primary_printout.read_text()
                     if args.primary_printout else None,
                     control_printout=args.control_printout.read_text()
                     if args.control_printout else None)
    document = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")
    else:
        print(document, end="")


if __name__ == "__main__":
    main()
