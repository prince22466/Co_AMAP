"""Compare S29 and S34 real-model *printed diagnostics*, with limitations.

A printed trace omits full tool-call arguments, context history, and patches.
Do not confuse a shorter prompt with proven context-window management.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from compare_submission24 import S1_IDS


TASK = re.compile(r"^\[candidate \d+/\d+\] Evaluating (\S+)")
END = re.compile(r"^\s*-> resolved=(True|False), patch_chars=(\d+), "
                 r"tools=(\d+), duration=([\d.]+), input=(\d+), output=(\d+)")
EVENT = re.compile(r"^\s*\[\s*[\d.]+s\]\s*(.*)$")
SUMMARY = re.compile(r"^\d+\t([^\t]+)\t(True|False)\t(\d+)\t([\d.]+)\t",
                     re.MULTILINE)
S29_WINS = frozenset({"rich_3905", "rich_3518", "rich_3454"})


def parse(raw: str) -> dict:
    rows = {}
    for task, resolved, patch_chars, seconds in SUMMARY.findall(raw):
        if task in rows:
            raise ValueError("Duplicate summary row: " + task)
        rows[task] = {
            "resolved": resolved == "True", "patch_chars": int(patch_chars),
            "seconds": float(seconds),
        }
    if set(rows) != set(S1_IDS):
        raise ValueError("Incomplete or different task set: " +
                         str(sorted(set(rows) ^ set(S1_IDS))))

    metrics = {}
    current = None
    for line in raw.splitlines():
        task = TASK.match(line)
        if task:
            name = task[1]
            if name in metrics:
                raise ValueError("Duplicate task event block: " + name)
            metrics[name] = {
                "reads": 0, "commands": 0, "edits": 0, "successful_edits": 0,
                "failed_edits": 0, "missing_edit_arguments": 0,
                "writes": 0, "submits": 0, "bare_pytest": 0,
                "focused_pytest": 0, "diff_checks": 0, "token_limit_nudges": 0,
                "tools_charged": None, "input_tokens_reported": None,
                "output_tokens_reported": None, "first_edit_success": None,
            }
            current = name
            continue
        if current is None:
            continue
        m = END.match(line)
        if m:
            result = metrics[current]
            result["tools_charged"] = int(m[3])
            result["input_tokens_reported"] = int(m[5])
            result["output_tokens_reported"] = int(m[6])
            current = None
            continue
        event = EVENT.match(line)
        if not event:
            continue
        msg = event[1]
        t = metrics[current]
        if "📖 read_file " in msg:
            t["reads"] += 1
        elif "🔧 run_command " in msg:
            t["commands"] += 1
            command = msg.split("🔧 run_command ", 1)[1].strip()
            if re.fullmatch(r"(?:python -m )?pytest", command):
                t["bare_pytest"] += 1
            elif re.search(r"\bpytest\b", command):
                t["focused_pytest"] += 1
            if "git diff" in command:
                t["diff_checks"] += 1
        elif "✏️ edit_file " in msg:
            t["edits"] += 1
        elif "🤖 submit_patch" in msg:
            t["submits"] += 1
        elif "📝 write_file " in msg:
            t["writes"] += 1
        if ("FileEditError" in msg or "mandatory input parameters" in msg
                or "BudgetExceeded" in msg or "No changes to apply" in msg):
            t["failed_edits"] += 1
            if "mandatory input parameters" in msg:
                t["missing_edit_arguments"] += 1
            if t["first_edit_success"] is None and t["edits"] == 1:
                t["first_edit_success"] = False
        if "✓ {'status': 'ok', 'filepath':" in msg and t["edits"]:
            t["successful_edits"] += 1
            if t["first_edit_success"] is None and t["edits"] == 1:
                t["first_edit_success"] = True
        if "reached the token limit" in msg:
            t["token_limit_nudges"] += 1
    if set(metrics) != set(S1_IDS) or any(
            v["tools_charged"] is None for v in metrics.values()):
        raise ValueError("Incomplete scored detailed trace")
    return {"rows": rows, "traces": metrics}


def summarize(parsed: dict) -> dict:
    rows, traces = parsed["rows"], parsed["traces"]
    keys = ("reads", "commands", "edits", "successful_edits",
            "failed_edits", "missing_edit_arguments", "writes",
            "submits", "bare_pytest", "focused_pytest", "diff_checks",
            "token_limit_nudges", "tools_charged",
            "input_tokens_reported", "output_tokens_reported")
    return {
        "resolved": sum(r["resolved"] for r in rows.values()),
        "solved_ids": sorted(k for k, r in rows.items() if r["resolved"]),
        "empty_patches": sum(r["patch_chars"] == 0 for r in rows.values()),
        "largest_patch_chars": max(r["patch_chars"] for r in rows.values()),
        "total_duration_minutes": round(sum(
            r["seconds"] for r in rows.values()) / 60, 2),
        "over_four_minute_tasks": sum(
            r["seconds"] > 240 for r in rows.values()),
        "first_edit_successes": sum(
            t["first_edit_success"] is True for t in traces.values()),
        "first_edit_attempted_tasks": sum(
            t["first_edit_success"] is not None for t in traces.values()),
        "repeated_submit_tasks": sorted(
            k for k, t in traces.items() if t["submits"] > 1),
        "total": {key: sum(t[key] for t in traces.values()) for key in keys},
    }


def compare(before: str, after: str) -> dict:
    old, new = parse(before), parse(after)
    a, b = summarize(old), summarize(new)
    old_wins, new_wins = set(a["solved_ids"]), set(b["solved_ids"])
    # An explicit target for ONE matched run, not a promotion decision.
    provisional = (b["resolved"] >= 5 and S29_WINS <= new_wins
                   and b["largest_patch_chars"] <= 50000
                   and b["empty_patches"] <= 2)
    return {
        "baseline": a, "candidate": b,
        "gained": sorted(new_wins - old_wins),
        "lost": sorted(old_wins - new_wins),
        "delta_resolved": b["resolved"] - a["resolved"],
        "delta_reported_input_tokens": (
            b["total"]["input_tokens_reported"] -
            a["total"]["input_tokens_reported"]),
        "delta_reads": b["total"]["reads"] - a["total"]["reads"],
        "delta_failed_edits": (
            b["total"]["failed_edits"] - a["total"]["failed_edits"]),
        "single_run_target_met": provisional,
        "promotion_approved": False,
        "limitations": [
            "Printed events omit full tool arguments and final source diff.",
            "Input token totals reflect repeated context, not unique tokens.",
            "This does not measure the actual effective context window.",
            "Two real-model S1 repeats, diff inspection, and a timed run needed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline29", type=Path)
    parser.add_argument("candidate34", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.baseline29.read_text(encoding="utf-8"),
                     args.candidate34.read_text(encoding="utf-8"))
    data = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
