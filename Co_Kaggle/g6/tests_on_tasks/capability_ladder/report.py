"""Compare paired assisted probes and unmodified S29 official autonomous results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


class IncompleteEvidence(ValueError):
    pass


def read_rows(path: Path) -> dict[str, dict]:
    if path.is_dir():
        candidates = list(path.rglob("notebook_task_results.jsonl"))
        if len(candidates) != 1:
            raise IncompleteEvidence("Choose one autonomous run directory")
        path = candidates[0]
    rows = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        key = item.get("instance_id")
        if not isinstance(key, str) or key in rows:
            raise IncompleteEvidence("Missing/duplicate task ID")
        rows[key] = item
    if not rows:
        raise IncompleteEvidence("Empty evidence file")
    return rows


def compare(a: dict[str, dict], b: dict[str, dict],
            autonomous: dict[str, dict] | None = None) -> dict:
    if set(a) != set(b) or (autonomous is not None and set(a) != set(autonomous)):
        raise IncompleteEvidence("Task IDs must match exactly across levels")
    rows = []
    for key in sorted(a):
        left, right = a[key], b[key]
        if left.get("level") != "reasoning" or right.get("level") != "tool":
            raise IncompleteEvidence("Wrong diagnostic level")
        valid = left.get("scorable") is True and right.get("scorable") is True
        if (left.get("resolved") not in (True, False, None)
                or right.get("resolved") not in (True, False, None)):
            raise IncompleteEvidence("Nonboolean verification verdict")
        if valid and (not isinstance(left.get("resolved"), bool)
                      or not isinstance(right.get("resolved"), bool)):
            raise IncompleteEvidence("Scorable row must have verdict")
        if autonomous is not None:
            ar = autonomous[key]
            if not isinstance(ar.get("resolved"), bool):
                raise IncompleteEvidence("Autonomous run missing boolean verdict")
            auto = ar["resolved"]
            if ar.get("error") or ar.get("error_message"):
                valid = False
        else:
            auto = None
        if not valid:
            category = "blocked_or_unmatched"
        elif left["resolved"] and right["resolved"] and auto is False:
            category = "autonomy_gap_candidate"
        elif left["resolved"] and not right["resolved"]:
            category = "tool_interface_gap_candidate"
        elif not left["resolved"] and right["resolved"]:
            category = "prompt_or_sampling_difference"
        elif not left["resolved"] and not right["resolved"]:
            category = "not_solved_under_assistance"
        elif left["resolved"] and right["resolved"] and auto is True:
            category = "all_levels_solved"
        else:
            category = "incomplete_autonomous_comparison"
        rows.append({
            "instance_id": key,
            "matched_scorable": valid,
            "reasoning_resolved": left.get("resolved"),
            "tool_resolved": right.get("resolved"),
            "autonomous_resolved": auto,
            "category": category,
        })
    comparable = [r for r in rows if r["matched_scorable"]]
    report = {
        "task_count": len(rows),
        "matched_scorable": len(comparable),
        "level1_resolved": sum(r["reasoning_resolved"] is True for r in comparable),
        "level2_resolved": sum(r["tool_resolved"] is True for r in comparable),
        "level3_resolved": (sum(r["autonomous_resolved"] is True for r in comparable)
                            if autonomous is not None else None),
        "categories": {
            key: sum(r["category"] == key for r in rows)
            for key in sorted({r["category"] for r in rows})
        },
        "results": rows,
        "caveats": [
            "L1 and L2 use oracle source-file locations; neither can be "
            "submitted as Kaggle agent performance.",
            "A one-turn, one-edit probe is not a model capability ceiling.",
            "Direct OpenAI API and ADK autonomy are different generation contexts.",
            "Scorable means empty patch demonstrably fails and reference passes "
            "in the same verifier environment; blocked tasks are not failures.",
            "One sample/run is not statistically significant. Use repeated seeds, "
            "holdout and inspect all diffs before drawing conclusions.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning", type=Path, required=True,
                        help="verified.jsonl from L1 verify.py")
    parser.add_argument("--tool", type=Path, required=True,
                        help="verified.jsonl from L2 verify.py")
    parser.add_argument("--autonomous", type=Path,
                        help="One S29 official candidate_1 JSONL or directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise IncompleteEvidence("Output exists; never overwrite experiment")
    result = compare(
        read_rows(args.reasoning), read_rows(args.tool),
        read_rows(args.autonomous) if args.autonomous else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in (
            "task_count", "matched_scorable", "level1_resolved",
            "level2_resolved", "level3_resolved", "categories")
    }, indent=2))


if __name__ == "__main__":
    main()
