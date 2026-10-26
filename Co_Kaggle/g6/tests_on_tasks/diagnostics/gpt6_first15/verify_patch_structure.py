"""Structural-only validation of independently drafted development patches.

Clones exact public base commits; checks git apply and Python syntax, not correctness.
Requires GitHub network in CI. Never reads the public gold reference patch.
"""
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def call(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)

def check(instance_id, spec):
    patch_file = ROOT / spec["candidate_patch"]
    changed = re.findall(r"^\+\+\+ b/(.+)$", patch_file.read_text(), flags=re.M)
    if not changed:
        raise RuntimeError("no changed files")
    with tempfile.TemporaryDirectory(prefix="g6_first15_") as td:
        workspace = Path(td) / "checkout"
        workspace.mkdir()
        call("git", "init", "-q", cwd=workspace)
        call("git", "remote", "add", "origin",
             f"https://github.com/{spec['repo']}.git", cwd=workspace)
        call("git", "fetch", "-q", "--depth=1", "origin",
             spec["base_commit"], cwd=workspace)
        call("git", "checkout", "-q", "--detach", "FETCH_HEAD", cwd=workspace)
        call("git", "apply", "--check", str(patch_file), cwd=workspace)
        call("git", "apply", str(patch_file), cwd=workspace)
        for path in changed:
            if path.endswith(".py"):
                code = (workspace / path).read_text(encoding="utf-8")
                ast.parse(code, filename=path)
        return {"status": "structural_pass", "paths": changed}

def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    results = {}
    for task_id, spec in manifest.items():
        if not spec["candidate_patch"]:
            results[task_id] = {"status": "not_attempted"}
            continue
        try:
            results[task_id] = check(task_id, spec)
        except Exception as exc:
            results[task_id] = {"status": "structural_fail", "error": str(exc)}
        print(task_id, results[task_id], flush=True)
    (ROOT / "structural_results.json").write_text(
        json.dumps(results, indent=2) + "\n")
    good = sum(x["status"]=="structural_pass" for x in results.values())
    attempted = sum(x["status"]!="not_attempted" for x in results.values())
    print(f"structural checks: {good}/{attempted}. This is NOT a Kaggle score.")
    return 0 if good == attempted else 1

if __name__ == "__main__":
    sys.exit(main())
