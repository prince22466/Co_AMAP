"""Compare supplied Submission 4 and 7 archives; never execute their contents."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "submission_4_6_review/analyze.py"
spec = importlib.util.spec_from_file_location("prior_review", PRIOR)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def additional_evidence(version: int, rows: list[dict]) -> dict:
    path = HERE.parent / f"submission-{version}-zip_First30Tasks.zip"
    tool_counts, schema_by_tool, finish_reasons = Counter(), Counter(), Counter()
    capped_requests, tool_failures = [], []
    calls_after_submit = []
    with ZipFile(path) as archive:
        root = next(n for n in archive.namelist() if n.endswith("/run_manifest.json")).rsplit("/", 1)[0]
        manifest = json.loads(archive.read(root + "/run_manifest.json"))
        for row in rows:
            tid = row["instance_id"]
            tools = json.loads(archive.read(f"{root}/report/{tid}_tools.json"))
            requests = json.loads(archive.read(f"{root}/report/{tid}_requests.json"))
            tool_counts.update(t["tool_name"] for t in tools)
            submitted = False
            for tool in tools:
                name = tool["tool_name"]
                result = tool.get("result") or {}
                if submitted:
                    calls_after_submit.append({"task": tid, "tool": name, "status": tool.get("status")})
                if name == "submit_patch" and result.get("status") == "ok":
                    submitted = True
                if tool.get("status") == "error":
                    message = result.get("error_message", result.get("error", ""))
                    if "mandatory input parameters" in message:
                        schema_by_tool[name] += 1
                    tool_failures.append({
                        "task": tid, "tool": name,
                        "type": result.get("error_type", "argument_validation"),
                        "message": message[:1200], "arguments": tool.get("arguments"),
                    })
            for request in requests:
                reason = request.get("finish_reason")
                if reason is not None:
                    finish_reasons[reason] += 1
                if reason not in (None, "STOP"):
                    capped_requests.append({"task": tid, **{k: request.get(k) for k in (
                        "request_id", "finish_reason", "output_tokens", "model_request_wall_seconds", "stage",
                    )}})
        source = HERE.parents[2] / "submission_nb" / f"submission_{version}"
        source_matches = {name: (source / name).exists() and hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
                          for name, digest in manifest["submission"]["file_sha256"].items()}
    return {
        "tool_calls_observed_by_name": dict(tool_counts),
        "argument_validation_errors_by_tool": dict(schema_by_tool),
        "finish_reasons": dict(finish_reasons), "non_stop_request_evidence": capped_requests,
        "tools_observed_after_successful_submit": calls_after_submit,
        "error_evidence": tool_failures,
        "local_source_matches_run_manifest": source_matches,
    }


def main():
    summaries, all_rows = {}, []
    for version in (4, 7):
        summary, rows = prior.load_run(version)
        summary["additional"] = additional_evidence(version, rows)
        summaries[str(version)] = summary
        all_rows.extend(rows)
    assert summaries["4"]["selected_task_identity_sha256"] == summaries["7"]["selected_task_identity_sha256"]
    assert summaries["4"]["effective_evaluation_limits"] == summaries["7"]["effective_evaluation_limits"]
    assert summaries["4"]["runtime_versions"] == summaries["7"]["runtime_versions"]
    assert all(summaries["7"]["additional"]["local_source_matches_run_manifest"].values())
    by_version = {v: {r["instance_id"]: r for r in all_rows if r["submission"] == v} for v in (4, 7)}
    paired = []
    for tid, old in by_version[4].items():
        new = by_version[7][tid]
        paired.append({
            "instance_id": tid, "s4_resolved": old["resolved"], "s7_resolved": new["resolved"],
            "s4_patch_chars": old["patch_chars"], "s7_patch_chars": new["patch_chars"],
            "s4_calls": old["tool_calls"], "s7_calls": new["tool_calls"],
            "s4_context_error": old["context_error"], "s7_context_error": new["context_error"],
            "s4_schema_errors": old["schema_error_responses"], "s7_schema_errors": new["schema_error_responses"],
            "s7_grader_missing_inline_snapshot": new["grader_missing_inline_snapshot"],
            "s7_grader_recursive_httpbin": new["grader_recursive_httpbin"],
        })
    summaries["comparison"] = {
        "same_tasks_commits_and_descriptions": True, "same_effective_limits_and_runtime_versions": True,
        "newly_resolved": [r["instance_id"] for r in paired if r["s7_resolved"] and not r["s4_resolved"]],
        "newly_unresolved": [r["instance_id"] for r in paired if r["s4_resolved"] and not r["s7_resolved"]],
        "note": "Both runs disable the task timer. Multiple config/prompt changes and one run per candidate prevent isolated causal attribution.",
    }
    (HERE / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (HERE / "task_evidence.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    with (HERE / "paired_tasks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    fields = ("resolved", "empty_patches", "context_errors", "schema_error_responses", "tasks_at_40_calls",
              "write_outside_workspace_rejections", "tasks_with_scratch_named_patch_paths", "helper_calls_by_name")
    print(json.dumps({v: {k: summaries[v][k] for k in fields} for v in ("4", "7")}, indent=2))
    print(json.dumps(summaries["comparison"], indent=2))


if __name__ == "__main__":
    main()
