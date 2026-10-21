"""Budget-aware repair gate for a single coding task.

Executes the Submission 9 assertion checker and retains per-task evidence in
Git metadata, outside the captured implementation patch. Never edits code.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

_spec = importlib.util.spec_from_file_location(
    "submission10_check", Path(__file__).with_name("check.py"))
if _spec is None or _spec.loader is None:
    raise ImportError("missing sibling check.py")
_checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checker)
check = _checker.check

MAX_REPAIRS = 2
MIN_SECONDS_FOR_CHECK = 45
MIN_CALLS_FOR_CHECK = 6
MAX_REQUIREMENTS = 4


def identifiers(raw):
    names = [s.strip() for s in raw.split(",") if s.strip()]
    if len(names) > MAX_REQUIREMENTS or len(set(names)) != len(names):
        raise ValueError("use at most four distinct requirement IDs")
    if not all(re.fullmatch(r"[a-z][a-z0-9_]{0,39}", n) for n in names):
        raise ValueError("requirement IDs must be short lowercase identifiers")
    return names


def git(root, *args):
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, timeout=8, check=True)
    return done.stdout


def fingerprint(root):
    """Hash changed and untracked files; ignore the transient ADK wrapper."""
    changed = git(root, "status", "--porcelain=v1", "--untracked-files=all", "-z").split(b"\0")
    names = []
    for raw in changed:
        if not raw or len(raw) < 4:
            continue
        name = raw[3:].decode("utf-8", "surrogateescape")
        if re.fullmatch(r"\.adk_exec_[0-9a-f]{8}\.py", name):
            continue
        names.append(name)
    h = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        h.update(name.encode("utf-8", "surrogateescape") + b"\0")
        if path.is_symlink():
            h.update(b"SYMLINK")
        elif path.is_file():
            h.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            h.update(b"DELETED")
    return h.hexdigest(), len(names)


def state_file(root):
    directory = Path(git(root, "rev-parse", "--absolute-git-dir").decode().strip()).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("git metadata must be inside the sandbox repository")
    return directory / "submission10_repair_state.json"


def decide(workspace, requirements, covers, code, module=None, timeout=12,
           time_left=None, calls_left=None):
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir() or len(str(root)) > 1024:
        raise ValueError("invalid workspace")
    required, covered_by_check = identifiers(requirements), identifiers(covers)
    if not required or not set(covered_by_check).issubset(required):
        raise ValueError("define 1-4 requirements and test only named requirements")
    if ((time_left is not None and time_left < 0) or
            (calls_left is not None and calls_left < 0)):
        raise ValueError("remaining budgets must be nonnegative")
    if not 1 <= timeout <= 20:
        raise ValueError("timeout must be 1-20 seconds")
    patch_fingerprint, changed_count = fingerprint(root)
    location = state_file(root)
    state = json.loads(location.read_text()) if location.exists() else {
        "requirements": required, "repairs_requested": 0,
        "alternative_checks_requested": 0, "checked": {}, "last_failure": None}
    if state["requirements"] != required:
        raise ValueError("requirement IDs cannot change during the task")
    if state.get("patch_fingerprint") != patch_fingerprint:
        state["checked"] = {}  # An edit invalidates earlier successes.
        state["patch_fingerprint"] = patch_fingerprint
        state["last_failure"] = None
    enough_budget = (time_left is None or time_left >= MIN_SECONDS_FOR_CHECK) and (
        calls_left is None or calls_left >= MIN_CALLS_FOR_CHECK)
    result = {"next_action": None, "reason": "", "verification": None,
              "repairs_requested": state["repairs_requested"],
              "max_repairs": MAX_REPAIRS, "patch_changed": changed_count > 0,
              "budget_sufficient": enough_budget,
              "scope": "The coder supplies acceptance IDs; passing inline assertions do not prove hidden-test success."}
    if not enough_budget:
        result.update(next_action="audit_then_submit",
                      reason="reserve remaining time and calls for patch capture")
    else:
        result["verification"] = check(root, code, module, timeout)
        test = result["verification"]
        outcome = test["outcome"]
        if outcome == "passed" and test["assertions_passed"] > 0 and test["completed"]:
            for name in covered_by_check:
                state["checked"][name] = hashlib.sha256(code.encode()).hexdigest()[:12]
            missing = [r for r in required if r not in state["checked"]]
            result.update(
                next_action="verify_uncovered" if missing else "audit_then_submit",
                reason=("some acceptance cases remain unverified" if missing
                        else "all declared cases have passing executable assertions"))
        elif outcome == "failed":
            signature = hashlib.sha256(
                (patch_fingerprint + code + str(module)).encode()).hexdigest()
            if state["last_failure"] == signature:
                result.update(next_action="audit_then_submit",
                              reason="identical failed check on unchanged patch")
            elif state["repairs_requested"] >= MAX_REPAIRS:
                result.update(next_action="audit_then_submit",
                              reason="repair attempt limit reached")
            else:
                state["repairs_requested"] += 1
                result.update(next_action="repair_if_code_defect",
                              reason="check failed; diagnose before editing")
            state["last_failure"] = signature
        elif state["alternative_checks_requested"] == 0:
            state["alternative_checks_requested"] += 1
            result.update(next_action="alternative_check",
                          reason=f"{outcome} is inconclusive, not a pass")
        else:
            result.update(next_action="audit_then_submit",
                          reason="alternative-check allowance exhausted")
    result["repairs_requested"] = state["repairs_requested"]
    result["covered"] = sorted(state["checked"])
    result["uncovered"] = [r for r in required if r not in state["checked"]]
    location.write_text(json.dumps(state, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--covers", default="")
    parser.add_argument("--code", required=True)
    parser.add_argument("--module")
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--time-left", type=float)
    parser.add_argument("--calls-left", type=int)
    args = parser.parse_args()
    try:
        out = decide(args.workspace, args.requirements, args.covers, args.code,
                     args.module, args.timeout, args.time_left, args.calls_left)
    except (OSError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, json.JSONDecodeError, SyntaxError) as exc:
        out = {"next_action": "audit_then_submit", "error": str(exc)[:400],
               "reason": "decision gate unavailable; protect remaining budget"}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
