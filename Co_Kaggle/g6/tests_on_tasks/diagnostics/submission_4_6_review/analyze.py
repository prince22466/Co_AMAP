"""Summarize the two supplied diagnostic archives without executing their code.

Run from any directory. Counts are observations, not causal failure labels.
Original archives and task outcomes are never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent


def load_run(version: int) -> tuple[dict, list[dict]]:
    path = HERE.parent / f"submission-{version}-zip_First30Tasks.zip"
    with ZipFile(path) as archive:
        root = next(n for n in archive.namelist() if n.endswith("/run_manifest.json")).rsplit("/", 1)[0]

        def read(name: str):
            return json.loads(archive.read(f"{root}/{name}"))

        manifest = read("run_manifest.json")
        selected = read("selected_tasks.json")
        tasks = read("report/task_summary.json")
        assert len({t["instance_id"] for t in tasks}) == len(tasks) == 30
        assert [t["instance_id"] for t in tasks] == manifest["task_ids"]
        rows, all_requests, helper_calls, errors = [], [], [], Counter()
        helper_names = set(manifest["submission"]["helper_names"])
        for task in tasks:
            tid = task["instance_id"]
            tools = read(f"report/{tid}_tools.json")
            requests = read(f"report/{tid}_requests.json")
            all_requests.extend(requests)
            tool_keys = [t.get("tool_call_id") for t in tools]
            assert all(tool_keys) and len(set(tool_keys)) == len(tool_keys), tid
            failures = [t for t in tools if t.get("status") == "error"]
            schema_errors = [t for t in failures if "mandatory input parameters" in json.dumps(t.get("result"))]
            traversal = [t for t in failures if t.get("tool_name") == "write_file"
                         and "escapes workspace root" in json.dumps(t.get("result"))]
            old_not_found = [t for t in failures if "old_string not found" in json.dumps(t.get("result"))]
            repeats = Counter((t.get("tool_name"), json.dumps(t.get("arguments"), sort_keys=True)) for t in failures)
            repeated_failures = [dict(tool=name, count=count, arguments=json.loads(args))
                                 for (name, args), count in repeats.items() if count > 1]
            for t in failures:
                result = t.get("result") or {}
                errors[(t.get("tool_name"), result.get("error_type", "argument_validation"))] += 1
            calls = [t for t in tools if t.get("tool_name") in helper_names]
            for t in calls:
                helper_calls.append({
                    "task": tid, "helper": t["tool_name"], "tool_wall_seconds": t.get("tool_wall_seconds"),
                    "nested_tool_calls": sum(x.get("agent") == t["tool_name"]
                                             and t["start_observer_seconds"] <= x["observer_elapsed_seconds"] <= t["end_observer_seconds"]
                                             for x in tools),
                    "nested_count_method": "matching helper author within observed delegation interval; calls are serial in these configurations",
                    "request": t.get("arguments"), "response": t.get("result"),
                })
            output = task.get("raw_result", {}).get("test_output", "") or ""
            patch_paths = re.findall(r"^diff --git a/(.*?) b/.*$", task.get("agent_patch", ""), re.M)
            row = {k: task.get(k) for k in (
                "instance_id", "resolved", "patch_chars", "tool_calls", "duration_seconds", "test_exit_code",
            )}
            root_runs = [r for r in read(f"report/{tid}_agent_runs.json")
                         if r.get("agent") == manifest["submission"]["root_agent"] and not r.get("parent_agent_run_id")]
            row.update(
                submission=version,
                context_error="ContextWindowExceeded" in (task.get("error") or ""),
                tool_error_responses=len(failures), schema_error_responses=len(schema_errors),
                write_outside_workspace_rejections=len(traversal), edit_match_failures=len(old_not_found),
                helper_invocations=len(calls), helper_nested_tools=sum(t.get("agent") in helper_names for t in tools),
                grader_missing_inline_snapshot="No module named 'inline_snapshot'" in output,
                grader_recursive_httpbin="recursive dependency involving fixture 'httpbin" in output,
                patch_paths=patch_paths,
                scratch_named_patch_paths=[p for p in patch_paths if re.search(r"(^|/)(repro|verify|check_|debug_)", p)],
                root_agent_observed_span_seconds=(max(r["end_observer_seconds"] for r in root_runs)
                                                 - min(r["start_observer_seconds"] for r in root_runs)) if root_runs else None,
                repeated_identical_error_calls=repeated_failures,
            )
            rows.append(row)

        def count(field):
            return sum(bool(r[field]) for r in rows)

        summary = {
            "archive": path.name, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "archive_root": root, "selected_tasks": len(tasks), "resolved": count("resolved"),
            "selected_task_identity_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
            "resolved_with_nonempty_patch": sum(r["resolved"] and r["patch_chars"] > 0 for r in rows),
            "resolved_task_ids": [r["instance_id"] for r in rows if r["resolved"]],
            "empty_patches": sum(r["patch_chars"] == 0 for r in rows),
            "tasks_at_40_calls": sum(r["tool_calls"] == 40 for r in rows),
            "context_errors": count("context_error"), "median_task_seconds": median(r["duration_seconds"] for r in rows),
            "helper_calls_by_name": dict(Counter(t["helper"] for t in helper_calls)),
            "tasks_with_helper_calls": count("helper_invocations"),
            "tool_error_responses": sum(r["tool_error_responses"] for r in rows),
            "schema_error_responses": sum(r["schema_error_responses"] for r in rows),
            "write_outside_workspace_rejections": sum(r["write_outside_workspace_rejections"] for r in rows),
            "edit_match_failures": sum(r["edit_match_failures"] for r in rows),
            "tasks_with_grader_missing_inline_snapshot": count("grader_missing_inline_snapshot"),
            "tasks_with_grader_recursive_httpbin": count("grader_recursive_httpbin"),
            "tasks_with_scratch_named_patch_paths": count("scratch_named_patch_paths"),
            "max_observed_response_output_tokens": max(r.get("output_tokens") or 0 for r in all_requests),
            "requests_without_output_usage": sum(r.get("output_tokens") is None for r in all_requests),
            "non_stop_finish_reasons": dict(Counter(r["finish_reason"] for r in all_requests if r.get("finish_reason") not in (None, "STOP"))),
            "effective_evaluation_limits": manifest["effective_evaluation_limits"],
            "declared_models": manifest["submission"]["declared_models"],
            "runtime_versions": manifest["runtime_versions"], "compaction": manifest["compaction"],
            "errors_by_tool_and_type": [{"tool": k[0], "type": k[1], "count": n} for k, n in errors.most_common()],
            "helper_call_evidence": helper_calls,
        }
        return summary, rows


def main():
    summaries, rows = {}, []
    for version in (4, 6):
        summary, version_rows = load_run(version)
        summaries[str(version)] = summary
        rows.extend(version_rows)
    ids4 = {r["instance_id"] for r in rows if r["submission"] == 4}
    ids6 = {r["instance_id"] for r in rows if r["submission"] == 6}
    assert ids4 == ids6
    assert summaries["4"]["effective_evaluation_limits"] == summaries["6"]["effective_evaluation_limits"]
    assert summaries["4"]["selected_task_identity_sha256"] == summaries["6"]["selected_task_identity_sha256"]
    task_sets_path = HERE.parents[1] / "task_inventory/test_sets.json"
    sets = json.loads(task_sets_path.read_text(encoding="utf-8"))["sets"]
    exposed_validation = sorted(ids4 & set(sets["validation"]))
    catalog = json.loads((task_sets_path.parent / "task_catalog.json").read_text(encoding="utf-8"))["tasks"]
    by_id = {t["instance_id"]: t for t in catalog}
    exposed_snapshots = {(by_id[tid]["repo"], by_id[tid]["base_commit"]) for tid in ids4}
    heldout = [tid for tid in sets["validation"]
               if (by_id[tid]["repo"], by_id[tid]["base_commit"]) not in exposed_snapshots]
    next_sets = {
        "note": "Original test_sets.json is unchanged. Previously viewed diagnostic tasks and their snapshot companions are excluded from this remaining holdout.",
        "feedback_30": [r["instance_id"] for r in rows if r["submission"] == 4],
        "smoke_4": sets["smoke_4"], "pilot_12": sets["pilot_12"], "extended_24": sets["extended_24"],
        "remaining_validation": heldout,
        "excluded_original_validation": [tid for tid in sets["validation"] if tid not in heldout],
    }
    (HERE / "next_evaluation_sets.json").write_text(json.dumps(next_sets, indent=2) + "\n", encoding="utf-8")
    summaries["comparison"] = {
        "same_task_ids": True, "same_effective_limits": True,
        "validation_ids_exposed_in_these_runs": exposed_validation,
        "remaining_validation_count_after_snapshot_exclusion": len(heldout),
        "note": "Model, prompts and agent structure differ; this is not an isolated model or architecture experiment."
    }
    (HERE / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (HERE / "task_evidence.json").write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (HERE / "task_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [k for k in rows[0] if k != "repeated_identical_error_calls"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    for version in ("4", "6"):
        print(json.dumps({k: v for k, v in summaries[version].items()
                          if k not in ("helper_call_evidence", "runtime_versions", "errors_by_tool_and_type", "effective_evaluation_limits", "compaction")}, indent=2))
    print(json.dumps(summaries["comparison"], indent=2))


if __name__ == "__main__":
    main()
