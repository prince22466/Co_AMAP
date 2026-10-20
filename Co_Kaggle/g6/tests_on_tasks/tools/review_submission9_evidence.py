"""Recompute submission-9 design evidence from the committed diagnostic ZIPs.

This tool is development-only and is NEVER packaged in submission_9.zip.
It reads observations and task metadata, not reference fixes or test_patch.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def review(archive):
    with zipfile.ZipFile(archive) as z:
        def read(suffix):
            names = [n for n in z.namelist() if n.endswith("/" + suffix)]
            if len(names) != 1:
                raise ValueError(f"expected one {suffix}, found {len(names)}")
            return z.read(names[0]).decode("utf-8")

        tasks = [json.loads(line) for line in read("notebook_task_results.jsonl").splitlines()]
        manifest = json.loads(read("run_manifest.json"))
        tools, missing, helpers = Counter(), Counter(), Counter()
        errors = []
        protected_edits = []
        for name in sorted(z.namelist()):
            if "/report/" not in name or not name.endswith("_tools.json"):
                continue
            task = Path(name).name.removesuffix("_tools.json")
            for call in json.loads(z.read(name)):
                tool = call.get("tool_name") or call.get("function_name")
                tools[tool] += 1
                if tool.endswith("_agent"):
                    helpers[tool] += 1
                result = call.get("result") or {}
                text = json.dumps(result)
                if "mandatory input parameters are not present" in text:
                    missing[task] += 1
                    errors.append({"task": task, "tool": tool,
                                   "tool_call_id": call.get("tool_call_id")})
                path = (call.get("arguments") or {}).get("filepath", "")
                # Count confirmed mutations to existing test/config paths, not /tmp checks.
                if (tool == "edit_file" and isinstance(result, dict) and result.get("status") == "ok"
                        and (path.startswith(("tests/", "test/", "testing/")) or
                             Path(path).name in {"conftest.py", "pytest.ini", "pyproject.toml"})):
                    protected_edits.append({"task": task, "path": path})
        return {
            "archive": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "tasks": len(tasks), "resolved": sum(bool(t["resolved"]) for t in tasks),
            "resolved_ids": [t["instance_id"] for t in tasks if t["resolved"]],
            "empty_patch_ids": [t["instance_id"] for t in tasks if not t["agent_patch"]],
            "empty_patch_resolved_ids": [t["instance_id"] for t in tasks if t["resolved"] and not t["agent_patch"]],
            "counted_call_limit_reached": sum(t["tool_calls"] >= 40 for t in tasks),
            "missing_argument_errors": sum(missing.values()), "missing_arguments_by_task": dict(sorted(missing.items())),
            "missing_argument_events": errors, "observed_tool_invocations": dict(sorted(tools.items())),
            "helper_invocations": dict(helpers), "confirmed_protected_edit_events": protected_edits,
            "effective_settings": manifest["effective_settings_passed_to_evaluator"],
            "runtime_versions": manifest.get("runtime_versions"),
            "median_total_task_seconds": statistics.median(t["duration_seconds"] for t in tasks),
            "duration_note": "Total recorded task duration; not isolated inference time or a packaged-timer comparison.",
            "grading_environment_signals": {
                "missing_inline_snapshot": [t["instance_id"] for t in tasks if "No module named 'inline_snapshot'" in t["raw_result"].get("test_output", "")],
                "recursive_fixture": [t["instance_id"] for t in tasks if "recursive dependency involving fixture" in t["raw_result"].get("test_output", "")],
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    g6 = Path(__file__).resolve().parents[2]
    directory = g6 / "tests_on_tasks/diagnostics"
    results = {}
    for version in (4, 6, 7, 8):
        archives = list(directory.glob(f"submission-{version}-*.zip"))
        if len(archives) != 1:
            raise ValueError(f"expected one diagnostic archive for version {version}")
        results[f"submission_{version}"] = review(archives[0])
    output = {"source_commit": "cbe742951dcc089b56ba9331f64b3d64647ae595", "runs": results,
              "candidate_performance_measured": False,
              "warning": "Historical development diagnostics; not candidate results or independent held-out validation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {m: v[m] for m in ("tasks", "resolved", "missing_argument_errors", "counted_call_limit_reached")}
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
