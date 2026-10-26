"""Experimental focused public-test replay for Requests candidates.

Runs BASELINE and CANDIDATE on exact upstream commits with the published
test_patch in a separate CI environment. This is NOT the official Kaggle grader.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[2] / "competition_material/competition_data/tasks.jsonl"
# ROOT is .../g6/tests_on_tasks/diagnostics/gpt6_first15,
# ROOT.parents[2] = g6
IDS = ["requests_7505", "requests_7502", "requests_7433",
       "requests_7427", "requests_7315"]

def execute(command, cwd=None, env=None, timeout=120, check=False):
    p = subprocess.run(command, cwd=cwd, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"{command}: exit {p.returncode}: {p.stdout[-1500:]}")
    return p

def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    records = {}
    with DATA.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                obj = json.loads(line)
                if obj["instance_id"] in IDS:
                    records[obj["instance_id"]] = obj
    if set(records) != set(IDS):
        raise RuntimeError("Missing public tasks.jsonl records")

    results = {}
    for task_id in IDS:
        spec = manifest[task_id]
        record = records[task_id]
        result = {"task": task_id, "env": "GitHub Actions; NOT official SWE-gemma"}
        try:
            with tempfile.TemporaryDirectory(prefix="g6_behavior_") as temp:
                repo = Path(temp) / "checkout"
                repo.mkdir()
                execute(["git", "init", "-q"], cwd=repo, check=True)
                execute(["git", "remote", "add", "origin",
                         f"https://github.com/{spec['repo']}.git"], cwd=repo, check=True)
                execute(["git", "fetch", "-q", "--depth=1", "origin",
                         spec["base_commit"]], cwd=repo, timeout=300, check=True)
                execute(["git", "checkout", "-q", "--detach", "FETCH_HEAD"],
                        cwd=repo, check=True)
                tests = Path(temp) / "public_test.patch"
                tests.write_text(record["test_patch"].rstrip("\n") + "\n", encoding="utf-8")
                execute(["git", "apply", "--check", str(tests)], cwd=repo, check=True)
                execute(["git", "apply", str(tests)], cwd=repo, check=True)

                paths = re.findall(r"^\+\+\+ b/(.+)$", record["test_patch"], flags=re.M)
                names = re.findall(r"^\+\s*(?:async\s+)?def (test_\w+)", record["test_patch"], flags=re.M)
                if not paths or not names:
                    raise RuntimeError("No identifiable focused public test")
                target = paths[0]
                selector = " or ".join(names)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(repo / "src")
                env.pop("PYTHONWARNINGS", None)
                command = [sys.executable, "-m", "pytest", "-q", "-o", "addopts=",
                           target, "-k", selector, "--disable-warnings", "--maxfail=1"]
                baseline = execute(command, cwd=repo, env=env, timeout=45)
                result["baseline_exit"] = baseline.returncode
                result["baseline_tail"] = baseline.stdout[-2400:]
                patch = ROOT / spec["candidate_patch"]
                execute(["git", "apply", "--check", str(patch)], cwd=repo, check=True)
                execute(["git", "apply", str(patch)], cwd=repo, check=True)
                candidate = execute(command, cwd=repo, env=env, timeout=45)
                result["candidate_exit"] = candidate.returncode
                result["candidate_tail"] = candidate.stdout[-2400:]
                result["status"] = (
                    "focused_pass" if baseline.returncode != 0 and candidate.returncode == 0
                    else "focused_inconclusive_or_fail"
                )
        except Exception as err:
            result.update(status="environment_or_setup_blocked", error=str(err))
        results[task_id] = result
        print(f"{task_id}: {result['status']} "
              f"(baseline={result.get('baseline_exit')}, candidate={result.get('candidate_exit')})",
              flush=True)
        if result["status"] != "focused_pass":
            print(json.dumps(result, indent=2)[-3000:], flush=True)
    out = Path(os.environ.get("G6_RESULTS_DIR", str(ROOT)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "public_focus_results.json").write_text(json.dumps(results, indent=2)+"\n")
    success = sum(r["status"] == "focused_pass" for r in results.values())
    print(f"Focused public tests passing: {success}/{len(IDS)}; NOT official graded score.")
    return 0 if success == len(IDS) else 1

if __name__ == "__main__":
    sys.exit(main())
