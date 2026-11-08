"""S38 sandbox-local repeat guard. Invoked by ADK run_skill_script only.

No host callbacks, grader access, external dependencies or network requests.
The ledger lives under the per-task repository .git directory, not in git diff.
Only calls routed through THIS script are guarded; native tools are untouched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile

MAX_FILE_BYTES = 1_000_000
MAX_EDIT_CHARS = 6000
MAX_EDIT_LINES = 80
MAX_READ_LINES = 100
MAX_RECORDS = 80
STATE_NAME = "s38_state_guard.json"


def emit(status, **fields):
    print(json.dumps({"status": status, **fields}, sort_keys=True, ensure_ascii=False))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def workspace():
    # ADK materializes Python skill files in a temporary directory and
    # changes cwd before runpy.run_path. NEVER use cwd as the repo root.
    # /workspace is Docker's root; subprocess mode exposes it in PYTHONPATH.
    candidates = [
        os.environ.get("S38_WORKSPACE"),
        # In subprocess mode /workspace may exist on the *host* but is not
        # necessarily the target repository. Prefer the sandbox's own paths.
        os.environ.get("PYTHONPATH", "").split(os.pathsep)[0],
        os.environ.get("PWD"),
        "/workspace",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root.is_dir() and not root.is_symlink()
                and (root / ".git").is_dir()
                and not (root / ".git").is_symlink()):
            return root.resolve()
    raise ValueError("workspace_unavailable: repository root not found")


def target_file(root, name):
    if not name or "\\" in name or "\x00" in name:
        raise ValueError("invalid_filepath")
    p = PurePosixPath(name)
    if (not p.parts or p.is_absolute()
            or any(x in ("..", ".", "") for x in p.parts)):
        raise ValueError("invalid_filepath")
    if ".git" in p.parts:
        # Nested repositories/submodules have their own internal metadata.
        raise ValueError("protected_git_internal")
    target = root.joinpath(*p.parts)
    if not target.is_relative_to(root):
        raise ValueError("path_outside_workspace")
    # Reject symlinks in the path, including symlinked parent folders.
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink_path_forbidden")
    if not target.is_file():
        raise ValueError("existing_regular_file_required")
    info = target.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
        raise ValueError("unsupported_or_oversized_file")
    if not target.resolve().is_relative_to(root):
        raise ValueError("path_outside_workspace")
    return target


@contextmanager
def ledger(root):
    git = root / ".git"
    lock_path = git / "s38_state_guard.lock"
    if lock_path.is_symlink():
        raise ValueError("unsafe_lock_path")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = git / STATE_NAME
        if path.is_symlink():
            raise ValueError("unsafe_state_path")
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("version") != 1:
                    raise ValueError("invalid_state_version")
                if not isinstance(data.get("attempts"), list):
                    raise ValueError("invalid_state_records")
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ValueError("corrupted_state_ledger") from exc
        else:
            data = {"version": 1, "attempts": [], "last": None}
        yield data
        data["attempts"] = data["attempts"][-MAX_RECORDS:]
        # Do not leave partial state after interruption or a broken encoder.
        fd, name = tempfile.mkstemp(prefix=".s38_state_", dir=git)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(data, output, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def record(data, action, file, fingerprint, reason):
    data["attempts"].append({
        "action": action, "file": file, "fingerprint": fingerprint,
        "reason": reason,
    })
    data["last"] = {"action": action, "file": file, "reason": reason}


def run(args):
    root = workspace()
    if args.action == "status":
        with ledger(root) as data:
            emit("ok", action="status", last=data["last"],
                 retained_attempts=len(data["attempts"]))
        return

    path = target_file(root, args.file)
    raw = path.read_bytes()
    digest = sha(raw)
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("non_utf8_source") from exc

    if args.action == "read":
        lines = content.splitlines()
        start = args.start or 1
        end = args.end or min(start + MAX_READ_LINES - 1, len(lines))
        if start <= 0 or end < start or end - start + 1 > MAX_READ_LINES:
            raise ValueError("invalid_read_range")
        key = sha(json.dumps(["read", args.file, digest, start, end]).encode())
        with ledger(root) as data:
            if any(x.get("fingerprint") == key for x in data["attempts"]):
                emit("blocked_duplicate", action="read", file=args.file,
                     reason="Unchanged source and identical line range already read; change line range or search target.")
                return
            record(data, "read", args.file, key, "read_delivered")
            excerpt = "\n".join(
                f"{i}: {lines[i-1]}" for i in range(start, min(end, len(lines)) + 1)
            )
            emit("ok", action="read", file=args.file,
                 start=start, end=end, source_sha256=digest[:16],
                 content=excerpt[:8000])
        return

    if args.action != "edit":
        raise ValueError("invalid_action")
    old, new = args.old, args.new
    if (not old or old == new or
            max(len(old), len(new)) > MAX_EDIT_CHARS or
            max(len(old.splitlines()), len(new.splitlines())) > MAX_EDIT_LINES):
        raise ValueError("empty_noop_or_oversized_edit")
    key = sha(json.dumps(
        ["edit", args.file, digest, old, new], ensure_ascii=False).encode("utf-8"))
    with ledger(root) as data:
        if any(x.get("fingerprint") == key for x in data["attempts"]):
            emit("blocked_duplicate", action="edit", file=args.file,
                 reason="This exact edit failed against the same file revision. Read current lines and choose a NEW anchor.")
            return
        occurrences = content.count(old)
        if occurrences != 1:
            reason = "old_string_not_found" if occurrences == 0 else "ambiguous_anchor"
            record(data, "edit", args.file, key, reason)
            emit("rejected", action="edit", file=args.file, reason=reason,
                 occurrences=occurrences,
                 next_action="Read current source lines; supply a unique exact substring.")
            return
        modified = content.replace(old, new, 1).encode("utf-8")
        if len(modified) > MAX_FILE_BYTES:
            raise ValueError("oversized_result")
        # Recheck before committing. Another tool/process could have changed
        # the target between our read and the write. Never clobber it.
        if sha(path.read_bytes()) != digest:
            emit("rejected", action="edit", file=args.file,
                 reason="source_changed_since_read")
            return
        # Atomic replacement leaves tracked work as a normal git diff.
        info = path.stat()
        fd, temporary = tempfile.mkstemp(prefix=".s38_edit_", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(modified)
                out.flush()
                os.fsync(out.fileno())
            os.chmod(temporary, stat.S_IMODE(info.st_mode))
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        # Success changes the source revision and thus naturally invalidates
        # earlier failed fingerprints for this file.
        data["last"] = {"action": "edit", "file": args.file,
                        "reason": "applied"}
        emit("ok", action="edit", file=args.file,
             occurrences=1, source_sha256=sha(modified)[:16],
             next_action="Verify requested behavior and inspect git diff.")


def skill_argv(values):
    """ADK supplies --key, value pairs even when source starts with '-'.

    argparse misinterprets an old/new string beginning with '-' as an
    option. Join only the three free-form field pairs using --key=value,
    preserving the exact bytes of the original argument strings.
    """
    normalized = []
    i = 0
    while i < len(values):
        token = values[i]
        if token in ("--file", "--old", "--new") and i + 1 < len(values):
            normalized.append(token + "=" + values[i + 1])
            i += 2
        else:
            normalized.append(token)
            i += 1
    return normalized


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # A malformed model-generated tool call must produce structured
        # feedback rather than a traceback/usage block on stderr.
        raise ValueError("invalid_arguments: " + message)


def main():
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("status", "read", "edit"),
                        required=True)
    parser.add_argument("--file", default="")
    parser.add_argument("--old", default="")
    parser.add_argument("--new", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    try:
        run(parser.parse_args(skill_argv(sys.argv[1:])))
    except (OSError, ValueError) as exc:
        emit("error", reason=str(exc)[:200])


if __name__ == "__main__":
    main()
