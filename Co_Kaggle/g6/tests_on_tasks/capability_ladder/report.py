"""Compare paired assisted probes and unmodified S29 official autonomous results."""
from __future__ import annotations

import argparse
import json
import re
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


def require_matching_environment(reasoning: Path, tool: Path,
                                 autonomous: Path | None) -> dict:
    """Reject incomparable input/model/verifier builds before interpreting gaps."""
    left = reasoning.parent / "manifest.json"
    right = tool.parent / "manifest.json"
    if not left.is_file() or not right.is_file():
        raise IncompleteEvidence("Missing paired verifier manifests")
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (left, right))
    for field in ("tasks_sha256", "source_verifier_sha256",
                  "generator_seed", "served_model", "versions"):
        if a.get(field) is None or a.get(field) != b.get(field):
            raise IncompleteEvidence("Mismatch in paired verifier " + field)
    if (a.get("generator_level") != "reasoning"
            or b.get("generator_level") != "tool"):
        raise IncompleteEvidence("Wrong verifier generation levels")
    alignment = {"task_sha256": a["tasks_sha256"],
                 "served_model": a["served_model"],
                 "generator_seed": a["generator_seed"],
                 "verifier_sha256": a["source_verifier_sha256"]}
    if autonomous is not None:
        folder = autonomous if autonomous.is_dir() else autonomous.parent
        l3 = folder.parent / "run_manifest.json"
        if not l3.is_file():
            raise IncompleteEvidence("Missing official L3 runtime manifest")
        c = json.loads(l3.read_text(encoding="utf-8"))
        if (c.get("tasks_sha256") != a["tasks_sha256"]
                or c.get("served_model") != a["served_model"]):
            raise IncompleteEvidence("L3 model or tasks differ from assisted arms")
        if c.get("sandbox") != "docker":
            raise IncompleteEvidence("L3 must use Docker to match assisted verifier")
        runtime_versions = c.get("runtime_versions") or {}
        for package_name, version in a["versions"].items():
            if runtime_versions.get(package_name) != version:
                raise IncompleteEvidence(
                    "L3 verifier/runtime wheel mismatch: " + package_name)
        # Candidate runs are seeded 1, 2, ... by the unchanged S29 runner.
        # Reject a cross-repeat comparison even when task/model files match.
        match = re.fullmatch(r"candidate_(\d+)", folder.name)
        if match is None:
            raise IncompleteEvidence("L3 path must select a candidate_N run")
        if int(match.group(1)) != a["generator_seed"]:
            raise IncompleteEvidence("L3 candidate seed differs from assisted arms")
        l3_summary = folder.parent / "capability_l3_summary.json"
        if not l3_summary.is_file():
            raise IncompleteEvidence("Missing L3 verifier provenance; rerun run_autonomy.py")
        summary = json.loads(l3_summary.read_text(encoding="utf-8"))
        if (summary.get("submission") != "submission_29_unmodified"
                or summary.get("verifier_source_sha256")
                != a["source_verifier_sha256"]):
            raise IncompleteEvidence("L3 verifier source/S29 identity mismatch")
        alignment["autonomous_manifest"] = str(l3)
        alignment["autonomous_verifier_sha256"] = summary["verifier_source_sha256"]
    return alignment


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
    alignment = require_matching_environment(
        args.reasoning, args.tool, args.autonomous)
    result = compare(
        read_rows(args.reasoning), read_rows(args.tool),
        read_rows(args.autonomous) if args.autonomous else None)
    result["manifest_alignment"] = alignment
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
