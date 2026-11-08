"""Build a Kaggle-valid S37 ZIP: S36 primary plus one prompt-only addendum.

Unlike the separate research-only S37 host callback, this ZIP is actually
loadable by the competition YAML compiler. It does not enforce retries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from build_submission36 import (
    ARTIFACT_DIR, G6, MODEL, ROOT, payload as s36_payload,
)

SOURCE = G6 / "submission_nb" / "submission_37"
ADDENDUM = SOURCE / "failure_memory_addendum.md"
ARCHIVE = ARTIFACT_DIR / "submission_37.zip"


def payload() -> dict[str, bytes]:
    """Keep S36's model, tools, generation budget and prompt byte-identical."""
    base = s36_payload("primary")
    note = ADDENDUM.read_bytes()
    if not note.startswith(b"\nS37 FAILURE MEMORY"):
        raise ValueError("S37 addendum header/paragraph separation missing")
    if len(note) > 2200:
        raise ValueError("S37 memory guidance too long for E4B experiment")
    for phrase in (
        b"most recent tool RESULT", b"DO NOT repeat", b"edit_file",
        b"Missing arguments", b"two failed edits", b"submit",
    ):
        if phrase not in note:
            raise ValueError("Incomplete S37 failure recovery instructions")
    # No task-specific solutions, reference patches, new tool names or
    # undocumented tool parameters are allowed in this addendum.
    for forbidden in (
        b"rich_3905", b"rich_3518", b"rich_3454",
        b"fastapi_14258", b"requests_7315",
        b"before_tool_callback", b"after_tool_callback",
        b"plugin_manager", b"run_skill_script",
    ):
        if forbidden in note:
            raise ValueError("Disallowed benchmark or unsupported hook reference")
    return {
        **base,
        "instruction.md": base["instruction.md"] + note,
    }


def build(output: Path = ARCHIVE) -> dict:
    members = payload()
    if tuple(sorted(members)) != tuple(sorted(ROOT)):
        raise ValueError("Unexpected Kaggle archive members")
    # Byte-for-byte control on the competition-facing model and limits.
    base = s36_payload("primary")
    for name in ("agent.yaml", "eval_config.yaml"):
        if members[name] != base[name]:
            raise ValueError("S36 configuration drift: " + name)
    if members["instruction.md"] != (
        base["instruction.md"] + ADDENDUM.read_bytes()
    ):
        raise ValueError("S37 prompt differs beyond the intended addendum")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as target:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            target.writestr(info, members[name])
    # Check the archive we will actually hand to the evaluator, not just
    # the in-memory member dict.
    with zipfile.ZipFile(output, "r") as target:
        if target.testzip() is not None:
            raise ValueError("Corrupted ZIP member")
        if target.namelist() != sorted(ROOT):
            raise ValueError("Unexpected ZIP root layout")
        for name, expected in members.items():
            if target.read(name) != expected:
                raise ValueError("ZIP payload mismatch: " + name)
    manifest = {
        "submission": 37,
        "status": "Kaggle-loadable S1 candidate, not yet evaluated by E4B",
        "model": MODEL,
        "source_control": "S36 primary",
        "only_model_facing_change": "failure_memory_addendum.md appended",
        "deterministic_repetition_prevention_in_zip": False,
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "members_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(members.items())
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
