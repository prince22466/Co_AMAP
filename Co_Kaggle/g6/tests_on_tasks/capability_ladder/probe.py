"""L1 (no tools) and L2 (one parsed edit_file tool call) E4B probes.

Research only. Reference patches choose *locations*, never proposed edits.
Only pristine snapshot bytes, issue and hints are sent to the model.
L3 autonomous is the unchanged S29 official-runner experiment.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time
from urllib.request import Request, urlopen

MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
S1 = (
    "fastapi_14258", "requests_7315", "requests_7205", "rich_4077",
    "rich_4076", "rich_4075", "rich_3718", "rich_3934", "rich_3882",
    "rich_3905", "rich_3518", "rich_3454", "rich_3938", "rich_3676",
    "rich_3052",
)
PILOT = ("rich_3905", "rich_3454", "rich_3518", "rich_4077", "requests_7315")
FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)\s*$")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
MAX_INPUT_CHARS = 18000
MAX_REPLACEMENT_CHARS = 10000
MAX_PATCH_CHARS = 20000

EDIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Replace a unique exact string in an existing source file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "allow_multiple": {"type": "boolean", "enum": [False]},
            },
            "required": ["filepath", "old_string", "new_string"],
            "additionalProperties": False,
        },
    },
}

SYSTEM = (
    "Fix the described repository issue using the supplied ORIGINAL code. "
    "The file/line locations are an assisted diagnostic hint, not a proposed "
    "solution. Produce one minimal exact replacement in one of those files. "
    "Preserve behavior outside the issue. Do not alter tests, grading or config."
)


class ProbeError(ValueError):
    """Invalid diagnostic input or unsafe edit; not evidence of model failure."""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_path(name: str) -> str:
    value = PurePosixPath(name)
    if (not name or name.startswith("/") or name.startswith("\\")
            or "\\" in name or value.as_posix() != name
            or any(part in ("..", ".") for part in name.split("/"))):
        raise ProbeError("Unsafe or noncanonical path")
    return name


def read_tasks(path: Path) -> dict[str, dict]:
    tasks = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("instance_id")
        if not isinstance(key, str) or key in tasks:
            raise ProbeError("Missing/duplicate task ID")
        tasks[key] = row
    return tasks


def oracle_locations(reference_patch: str) -> list[tuple[str, int]]:
    """Read ONLY diff headers and hunk old-line numbers, never diff content."""
    if not isinstance(reference_patch, str):
        raise ProbeError("Reference patch unavailable")
    current = None
    result = []
    for line in reference_patch.splitlines():
        match = FILE.match(line)
        if match:
            current = safe_path(match[1]) if match[1] == match[2] else None
            continue
        hunk = HUNK.match(line)
        if hunk and current:
            if current.endswith((".py", ".pyi")) and (
                    "/test" not in "/" + current and
                    not PurePosixPath(current).name.startswith("test_")):
                item = (current, int(hunk[1]))
                if item not in result:
                    result.append(item)
    return result


def snapshot_path(snapshots: Path, task_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
        raise ProbeError("Unsafe task ID")
    matches = [snapshots / (task_id + ext)
               for ext in (".tgz", ".tar.gz", ".tar")
               if (snapshots / (task_id + ext)).is_file()]
    if len(matches) != 1:
        raise ProbeError("Missing or ambiguous task snapshot: " + task_id)
    return matches[0]


def pristine_source(archive: Path, filepath: str) -> str:
    """Read one uniquely matching ordinary file, without extracting archives."""
    safe_path(filepath)
    with tarfile.open(archive, mode="r:*") as tar:
        members = [
            m for m in tar.getmembers()
            if m.isfile() and m.size <= 3_000_000
            and (m.name.lstrip("./") == filepath
                 or m.name.endswith("/" + filepath))
        ]
        if len(members) != 1:
            raise ProbeError("Missing/ambiguous original source: " + filepath)
        stream = tar.extractfile(members[0])
        if stream is None:
            raise ProbeError("Could not read source")
        data = stream.read(3_000_001)
        if len(data) > 3_000_000:
            raise ProbeError("Source exceeds limit")
        return data.decode("utf-8")


def make_case(task: dict, snapshots: Path) -> dict:
    """Return model-visible, non-reference source only (no patch/test_patch)."""
    locations = oracle_locations(task.get("patch", ""))
    if not locations:
        raise ProbeError("No source location found in reference headers")
    archive = snapshot_path(snapshots, task["instance_id"])
    sections = []
    sources = {}
    for file, line in locations:
        if file in sources:
            continue
        source = pristine_source(archive, file)
        lines = source.splitlines(keepends=True)
        begin, end = max(0, line - 31), min(len(lines), line + 70)
        window = "".join(lines[begin:end])[:6500]
        sections.append({
            "filepath": file, "line_from": begin + 1, "line_to": end,
            "excerpt": window,
        })
        sources[file] = source
        if len(sections) >= 2:
            break
    issue = str(task.get("problem_statement") or "")
    hints = str(task.get("hints_text") or "")
    if not issue.strip():
        raise ProbeError("No issue description")
    # Bound the model-visible budget; never truncate source/replacements
    # after the model has used them. If the issue exceeds budget, block.
    user = (
        f"Repository: {task['repo']}\nIssue:\n{issue}\n"
        f"Hints:\n{hints}\n\n"
        "Pristine source excerpt(s), with original line ranges:\n"
        + "\n\n".join(
            f"FILE {s['filepath']} L{s['line_from']}-{s['line_to']}\n"
            + s["excerpt"] for s in sections
        )
    )
    if len(user) > MAX_INPUT_CHARS:
        raise ProbeError("Input too long; do not silently truncate task")
    return {
        "task_id": task["instance_id"], "repo": task["repo"],
        "base_commit": task.get("base_commit"),
        "message": user, "sources": sources,
        "oracle_files": list(sources),
    }


def validate_edit(raw: object, sources: dict[str, str]) -> tuple[str, str]:
    if not isinstance(raw, dict) or not {"filepath", "old_string", "new_string"} <= raw.keys():
        raise ProbeError("Missing edit_file parameters")
    if set(raw) - {"filepath", "old_string", "new_string", "allow_multiple"}:
        raise ProbeError("Unknown edit_file parameters")
    file, old, new = (raw["filepath"], raw["old_string"], raw["new_string"])
    if not isinstance(file, str) or safe_path(file) not in sources:
        raise ProbeError("File not in oracle-provided source locations")
    if raw.get("allow_multiple", False) is not False:
        raise ProbeError("allow_multiple must be false")
    if not isinstance(old, str) or not isinstance(new, str):
        raise ProbeError("old/new must be strings")
    if not old or old == new:
        raise ProbeError("Empty/no-op edit")
    if len(old) > MAX_REPLACEMENT_CHARS or len(new) > MAX_REPLACEMENT_CHARS:
        raise ProbeError("Oversized replacement")
    before = sources[file]
    if before.count(old) != 1:
        raise ProbeError("Exact old_string not uniquely found")
    after = before.replace(old, new, 1)
    if not before.endswith("\n") or not after.endswith("\n"):
        raise ProbeError("Newline-less files unsupported in this probe")
    diff = "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/" + file, tofile="b/" + file, n=3,
    ))
    if not diff or len(diff) > MAX_PATCH_CHARS:
        raise ProbeError("Empty/oversized unified diff")
    return file, diff


def parse_json_edit(message: str) -> object:
    # Do not silently repair JSON; malformed responses are the measured result.
    if not isinstance(message, str):
        raise ProbeError("No final assistant message")
    return json.loads(message.strip())


def parse_tool_edit(choice: dict) -> object:
    msg = choice.get("message") or {}
    calls = msg.get("tool_calls") or []
    if len(calls) != 1:
        raise ProbeError("Expected exactly one parsed tool call")
    f = calls[0].get("function") or {}
    if f.get("name") != "edit_file":
        raise ProbeError("Wrong tool function")
    data = f.get("arguments")
    return json.loads(data) if isinstance(data, str) else data


def post_json(endpoint: str, data: dict, token: str, timeout: int = 240) -> dict:
    req = Request(
        endpoint,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def model_probe(case: dict, level: str, api: str, served: str, token: str,
                seed: int) -> dict:
    if level not in ("reasoning", "tool"):
        raise ProbeError("Unknown level")
    instruction = SYSTEM + (
        " Return ONLY a JSON object with filepath, old_string and new_string "
        "using ordinary JSON escaping. No Markdown or commentary."
        if level == "reasoning" else
        " Call edit_file ONCE with filepath, old_string and new_string. "
        "Do not output a patch in prose."
    )
    body = {
        "model": served, "temperature": 0.0, "top_p": 0.95,
        "max_tokens": 4096, "seed": seed,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": case["message"]},
        ],
    }
    if level == "tool":
        body["tools"] = [EDIT_SCHEMA]
        body["tool_choice"] = "auto"
    start = time.monotonic()
    response = post_json(api.rstrip("/") + "/chat/completions", body, token)
    elapsed = time.monotonic() - start
    choices = response.get("choices") or []
    if not choices:
        raise ProbeError("No model choices")
    choice = choices[0]
    answer = choice.get("message") or {}
    # Return the raw response for audit, but never copy a reference solution.
    return {
        "raw": response, "duration_seconds": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "choice": choice,
    }


def verify_server(api: str, served: str, token: str) -> None:
    req = Request(
        api.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + token},
    )
    with urlopen(req, timeout=20) as response:
        models = json.load(response)
    ids = [m.get("id") for m in models.get("data", [])]
    if served not in ids or MODEL not in served:
        raise ProbeError("Server does not expose exact E4B QAT model")


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ProbeError("Output path already exists; never overwrite evidence")
    tasks = read_tasks(args.tasks)
    ids = tuple(args.task_ids or (PILOT if args.stage == "pilot" else S1))
    if len(ids) != len(set(ids)) or any(i not in tasks for i in ids):
        raise ProbeError("Duplicate or missing requested tasks")
    if args.seed < 0:
        raise ProbeError("Seed must be >=0")
    verify_server(args.model_url, args.served_model, args.api_key)
    cases = []
    for task_id in ids:
        try:
            cases.append((task_id, make_case(tasks[task_id], args.snapshots), None))
        except (ProbeError, UnicodeError, tarfile.TarError) as exc:
            cases.append((task_id, None, str(exc)))
    args.output.mkdir(parents=True)
    raw_dir = args.output / "raw_model"; raw_dir.mkdir()
    manifest = {
        "level": args.level, "task_ids": list(ids), "seed": args.seed,
        "model": MODEL, "served_model": args.served_model,
        "model_url": args.model_url, "temperature": 0.0,
        "top_p": 0.95, "max_tokens": 4096,
        "tasks_sha256": digest(args.tasks.read_bytes()),
        "oracle_location": "reference diff paths/hunk numbers only",
        "reference_patch_sent_to_model": False,
        "test_patch_sent_to_model": False,
        "max_tools": 0 if args.level == "reasoning" else 1,
        "behavior_verified": False,
        "status": "research data, not official competition submission",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (args.output / "probes.jsonl").open("x", encoding="utf-8") as out:
        for task_id, case, blocked in cases:
            row = {
                "instance_id": task_id, "level": args.level,
                "location_available": case is not None,
                "oracle_files": [] if case is None else case["oracle_files"],
                "status": "blocked_input" if blocked else "not_started",
                "error": blocked, "patch": "", "edit_valid": False,
                "resolved": None,  # NEVER infer from successful edit alone.
            }
            if case is not None:
                try:
                    resp = model_probe(case, args.level, args.model_url,
                                       args.served_model, args.api_key, args.seed)
                    (raw_dir / (task_id + ".json")).write_text(
                        json.dumps(resp["raw"], indent=2) + "\n", encoding="utf-8")
                    row.update({
                        "duration_seconds": resp["duration_seconds"],
                        "finish_reason": resp["finish_reason"],
                        "usage": resp["usage"],
                        "status": "invalid_edit",
                    })
                    parsed = (parse_json_edit(
                        (resp["choice"].get("message") or {}).get("content"))
                        if args.level == "reasoning"
                        else parse_tool_edit(resp["choice"]))
                    path, patch = validate_edit(parsed, case["sources"])
                    row.update({
                        "edit_valid": True, "status": "edit_applied_to_pristine",
                        "edited_file": path, "patch": patch,
                        "patch_sha256": digest(patch.encode()),
                    })
                except (ProbeError, ValueError, TypeError, KeyError) as exc:
                    row["status"] = "invalid_response_or_edit"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    row["status"] = "model_or_transport_error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(task_id, row["status"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("reasoning", "tool"), required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--served-model", default=MODEL)
    parser.add_argument("--api-key", default=os.environ.get("LOCAL_API_KEY", "EMPTY"))
    parser.add_argument("--stage", choices=("pilot", "s1"), default="pilot")
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
