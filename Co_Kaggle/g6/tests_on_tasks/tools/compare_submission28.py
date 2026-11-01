"""Compare *printed-trace proxies* without claiming identical tool arguments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_submission27_traces import analyze


def compare(baseline: dict, candidate: dict) -> dict:
    """Report paired task-resolution changes plus limited action-count proxies."""
    before = {row["task_id"]: row for row in baseline["tasks"]
              if row["resolved"] is not None}
    after = {row["task_id"]: row for row in candidate["tasks"]
             if row["resolved"] is not None}
    common = sorted(before.keys() & after.keys())
    gained = [name for name in common
              if not before[name]["resolved"] and after[name]["resolved"]]
    lost = [name for name in common
            if before[name]["resolved"] and not after[name]["resolved"]]
    fields = ("reads", "edits", "failed_edits", "commands", "submits",
              "same_path_read_streaks", "same_path_failed_edit_streaks")
    totals = {
        field: {"baseline": sum(before[k][field] for k in common),
                "candidate": sum(after[k][field] for k in common)}
        for field in fields
    }
    return {
        "scope": "Paired printed-trace comparison; same-path proxy != identical arguments",
        "matched_task_count": len(common),
        "missing_from_candidate": sorted(before.keys() - after.keys()),
        "missing_from_baseline": sorted(after.keys() - before.keys()),
        "gained_task_ids": gained, "lost_task_ids": lost,
        "baseline_solved_on_matched": sum(before[k]["resolved"] for k in common),
        "candidate_solved_on_matched": sum(after[k]["resolved"] for k in common),
        "baseline_empty_on_matched": sum(before[k]["patch_chars"] == 0 for k in common),
        "candidate_empty_on_matched": sum(after[k]["patch_chars"] == 0 for k in common),
        "baseline_exhausted_on_matched": sum(before[k]["charged_calls"] == 40
                                            for k in common),
        "candidate_exhausted_on_matched": sum(after[k]["charged_calls"] == 40
                                             for k in common),
        "actions": totals,
    }


def objective_pilot_pass(run: dict) -> bool:
    """Require a genuine new success while retaining both demonstrated S27 wins.

    The legacy S25 checker requires rich_4077, which S27 did not solve.
    This gate therefore tests improvement over the *observed* S27 pilot.
    """
    solved = set(run.get("solved_ids", []))
    return (run.get("resolved", 0) >= 3
            and {"rich_3905", "rich_3454"} <= solved
            and run.get("empty_patches", 99) <= 1
            and not run.get("agent_error_ids"))


def main() -> None:
    """Read paired prints and emit a JSON report; do not alter evaluation files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(analyze(args.baseline.read_text(encoding="utf-8")),
                     analyze(args.candidate.read_text(encoding="utf-8")))
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
