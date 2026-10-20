"""Read-only patch checks. Does not stage, remove, repair, or submit files."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess


PROTECTED = {
    "conftest.py", "pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini",
    "setup.cfg", "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py",
}


def protected(path):
    p = Path(path)
    return (p.name in PROTECTED or p.suffix == ".pth" or
            (p.suffix == ".py" and (p.name.startswith("test_") or
             p.name.endswith("_test.py") or
             any(part in {"test", "tests", "testing"} for part in p.parts[:-1]))))


def audit(workspace):
    root = Path(workspace).resolve(strict=True)

    def git(*args, allow_failure=False):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=10)
        if result.returncode and not allow_failure:
            raise ValueError(result.stderr.decode(errors="replace")[:400])
        return result

    if Path(git("rev-parse", "--show-toplevel").stdout.decode().strip()).resolve() != root:
        raise ValueError("workspace must be the repository root")

    def names(*args):
        return [p.decode("utf-8", errors="replace") for p in git(*args).stdout.split(b"\0") if p]

    tracked = names("diff", "HEAD", "--name-only", "-z")
    untracked = names("ls-files", "--others", "--exclude-standard", "-z")
    changed = sorted(set(tracked + untracked))
    syntax_errors, skipped, conflicts = [], [], []
    for name in changed[:80]:
        path = root / name
        if not path.exists():
            continue  # Deleted tracked file; the human/agent must inspect the diff.
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            skipped.append({"path": name, "reason": "symlink or outside workspace"})
            continue
        if not path.is_file() or path.stat().st_size > 1_000_000:
            skipped.append({"path": name, "reason": "not a file or exceeds 1 MB"})
            continue
        if path.suffix != ".py":
            continue
        try:
            raw = path.read_bytes()
            ast.parse(raw, filename=name)
            if re.search(rb"(?m)^(<<<<<<< |>>>>>>> )", raw):
                conflicts.append(name)
        except (SyntaxError, ValueError) as exc:
            syntax_errors.append({"path": name, "line": getattr(exc, "lineno", None),
                                  "error": str(exc)[:220]})
    whitespace = git("diff", "--check", "HEAD", allow_failure=True)
    protected_paths = [p for p in changed if protected(p)]
    scratch = [p for p in untracked if re.match(r"(?:repro|check_|tmp_|scratch|test_)", Path(p).name)]
    result = {
        "outcome": "structural_review_only", "empty_patch": not changed,
        "changed_count": len(changed), "changed_paths": changed[:16],
        "untracked_paths": untracked[:12], "protected_paths": protected_paths[:12],
        "possible_scratch_paths": scratch[:12], "syntax_errors": syntax_errors[:8],
        "skipped_paths": skipped[:8], "conflict_markers": conflicts[:8],
        "diff_check_returncode": whitespace.returncode,
        "diff_check_output": (whitespace.stdout + whitespace.stderr).decode(errors="replace")[:700],
        "truncated": (len(changed) > 16 or len(untracked) > 12 or len(protected_paths) > 12 or
                      len(scratch) > 12 or len(syntax_errors) > 8 or len(skipped) > 8),
        "scope": "No behavior checked. Inspect the actual diff; include intended new files at capture.",
    }
    # Keep a usable JSON object below the harness's 5000-character stdout limit.
    while len(json.dumps(result, ensure_ascii=False)) > 4400:
        key = max((k for k, v in result.items() if isinstance(v, list) and v),
                  key=lambda k: len(json.dumps(result[k])), default=None)
        if key is None:
            break
        result[key].pop()
        result["truncated"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    args = parser.parse_args()
    try:
        result = audit(args.workspace)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        result = {"outcome": "audit_error", "error": str(exc)[:500]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
