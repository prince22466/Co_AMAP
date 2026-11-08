"""Build S39: S36 + reconciled S37 failure memory + exact S38 skill.

S39 changes only the instruction string; agent.yaml, eval_config.yaml, E4B
model, tools, and packaged skill bytes are identical to S38. To remove an
inherited S36 contradiction, allowlist ONLY the registered state-guard skill.
Build an additional skill-enabled S38 control so S37's contribution remains
independently testable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from build_submission36 import ARTIFACT_DIR, G6, payload as s36_payload
import build_submission38 as s38


SOURCE = G6 / "submission_nb/submission_39"
RECONCILED_S37 = SOURCE / "s37_failure_memory_reconciled.md"
ARCHIVE = ARTIFACT_DIR / "submission_39.zip"
CONTROL = ARTIFACT_DIR / "submission_39_control_s38.zip"
SKILL_CONTROL = ARTIFACT_DIR / "submission_39_control_skill_enabled.zip"

# These are the reviewed original Git blob hashes, NOT task-specific data.
S37_ORIGINAL_BLOB = "19f91644808bee28454131853f32ad74d377b60d"
S38_SKILL_BLOB = "4d500774dea94dccf2b58e1b3ae190fbf8bb5cb9"
S38_SCRIPT_BLOB = "ee70f7eb4ffefe929e88908c603cbc524fb7fe67"
# Pin the S38 activation paragraph too, so neither arm silently changes.
S38_ACTIVATION_BLOB = "e27899240a1c358baf11fd7019a519b25c1b4aa3"

HEADER_S37_BEFORE = "S37 FAILURE MEMORY (model-guided; no new tools):"
HEADER_S37_AFTER = (
    "S37 FAILURE MEMORY (model-guided native fast path; "
    "S38 skill recovery available):"
)
LEDGER_BEFORE = "Do not spend tool calls\nwriting a ledger or narrating a plan."
LEDGER_AFTER = (
    "Do not spend native-tool calls\n"
    "manually writing a ledger or narrating a plan. Loading the skill is one\n"
    "setup call; the two-call recovery limit covers only run_skill_script actions\n"
    "(read current source, then one guarded edit), not load_skill itself."
)

# S36's otherwise-retained text forbids ALL skills and scripted edits, which
# contradicts S38's explicitly registered skill. Fix these exact two spans
# only inside the S39 payload, never in the S36/S38 baseline or repository.
S36_SKILL_BAN = (
    b"Never request a skill,\n"
    b"wrapper, invented tool or another agent."
)
S39_SKILL_ALLOW = (
    b"Do not request unregistered skills,\n"
    b"wrappers, invented tools or another agent. The registered state-guard skill\n"
    b"is allowed only for recovery after a failed native edit or read loop."
)
S36_SCRIPT_BAN = (
    b"Python Path.write_text or scripted insertion. These uncontrolled fallbacks"
)
S39_SCRIPT_ALLOW = (
    b"Python Path.write_text or unguarded scripted insertion. Only the registered\n"
    b"  state-guard skill's bounded exact edit is permitted. Other fallbacks"
)


def git_blob_sha(data: bytes) -> str:
    """Compute Git's source blob hash to prove copy/parity without git history."""
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 (Git blob)


def allow_registered_skill(s36_instruction: bytes) -> bytes:
    """Preserve every S36 instruction except its two incompatible bans."""
    updated = s36_instruction
    for forbidden, exception in (
        (S36_SKILL_BAN, S39_SKILL_ALLOW),
        (S36_SCRIPT_BAN, S39_SCRIPT_ALLOW),
    ):
        if updated.count(forbidden) != 1:
            raise ValueError("S36 skill exclusion changed; review the base prompt")
        updated = updated.replace(forbidden, exception, 1)
    if S36_SKILL_BAN in updated or S36_SCRIPT_BAN in updated:
        raise ValueError("S36 skill exclusion remains after reconciliation")
    return updated


def reconciled_memory() -> bytes:
    if not RECONCILED_S37.is_file() or RECONCILED_S37.is_symlink():
        raise ValueError("Missing or unsafe S39 S37 memory file")
    text = RECONCILED_S37.read_text(encoding="utf-8")
    if not text.startswith("\n" + HEADER_S37_AFTER + "\n"):
        raise ValueError("Unexpected S37 memory header")
    if text.count(HEADER_S37_AFTER) != 1 or text.count(LEDGER_AFTER) != 1:
        raise ValueError("S37/S38 reconciliation markers missing or repeated")
    # Reverse precisely the two documented compatibility edits; the complete
    # remaining S37 text must match the original source byte-for-byte.
    original = text.replace(HEADER_S37_AFTER, HEADER_S37_BEFORE)
    original = original.replace(LEDGER_AFTER, LEDGER_BEFORE)
    if git_blob_sha(original.encode("utf-8")) != S37_ORIGINAL_BLOB:
        raise ValueError("Unintended change to the original S37 recovery rules")
    return text.encode("utf-8")


def payload(variant: str = "primary") -> dict[str, bytes]:
    if variant not in ("primary", "control", "skill_enabled_control"):
        raise ValueError("Unknown S39 variant")
    s38_base = s38.payload("primary")
    note38 = s38.read_source(s38.NOTE)
    if git_blob_sha(note38) != S38_ACTIVATION_BLOB:
        raise ValueError("S38 reviewed skill-activation instruction drifted")
    expected_s38 = {
        "skills/state-guard/SKILL.md": S38_SKILL_BLOB,
        "skills/state-guard/scripts/state_guard.py": S38_SCRIPT_BLOB,
    }
    for filename, expected in expected_s38.items():
        if git_blob_sha(s38_base[filename]) != expected:
            raise ValueError("S38 reviewed skill implementation drifted: " + filename)
    if variant == "control":
        return s38_base

    original = s36_payload("primary")
    allowed_base = allow_registered_skill(original["instruction.md"])
    if s38_base["instruction.md"] != original["instruction.md"] + note38:
        raise ValueError("S38 original instruction is not S36 + S38 addendum")
    if variant == "skill_enabled_control":
        # Exactly S38 with the inherited skill ban repaired, no S37 rules.
        return {**s38_base, "instruction.md": allowed_base + note38}

    note37 = reconciled_memory()
    if len(note37) > 1800 or len(note38) > 1200:
        raise ValueError("Excessive agent instruction overhead")
    combined = allowed_base + note37 + note38

    result = {**s38_base, "instruction.md": combined}
    if result["agent.yaml"] != s38_base["agent.yaml"]:
        raise ValueError("Agent configuration drift")
    if result["eval_config.yaml"] != s38_base["eval_config.yaml"]:
        raise ValueError("Evaluation budget drift")
    if set(result) != set(s38.FILES):
        raise ValueError("Wrong S39 archive file list")

    # Guard against accidental reintroduction of benchmark-specific fixes,
    # unapproved model callbacks or a model/budget change.
    for forbidden in (
        b"rich_3518", b"rich_3938", b"rich_4077", b"requests_7315",
        b"before_tool_callback", b"after_tool_callback",
    ):
        if forbidden in note37 + note38:
            raise ValueError("Task-specific content or unsupported callback")
    return result


def path_for(variant: str) -> Path:
    if variant == "primary":
        return ARCHIVE
    if variant == "control":
        return CONTROL
    if variant == "skill_enabled_control":
        return SKILL_CONTROL
    raise ValueError("Unknown S39 variant")


def build(variant: str = "primary", output: Path | None = None) -> dict:
    members = payload(variant)
    output = output or path_for(variant)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in sorted(members.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)

    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt S39 archive")
        if archive.namelist() != sorted(s38.FILES):
            raise ValueError("Unexpected ZIP members or ordering")
        for name, expected in members.items():
            if archive.read(name) != expected:
                raise ValueError("S39 ZIP differs from its verified payload: " + name)

    manifest = {
        "submission": 39,
        "variant": variant,
        "model": s38.MODEL,
        "comparison": "S39 stack vs S38 original and S38 with skill permitted",
        "s36_primary_settings_unchanged": True,
        "s38_skill_bytes_unchanged": True,
        "s37_compatibility_edits": 2 if variant == "primary" else 0,
        "s36_skill_exception_spans": 0 if variant == "control" else 2,
        "native_tool_interception": False,
        "enforcement_scope": "skill-mediated operations only",
        "score": "unscored; E4B S1 evaluation required",
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "member_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(members.items())
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("primary", "control", "skill_enabled_control", "all"),
        default="all"
    )
    args = parser.parse_args()
    variants = (
        ("primary", "control", "skill_enabled_control")
        if args.variant == "all" else (args.variant,)
    )
    print(json.dumps({v: build(v) for v in variants}, indent=2))


if __name__ == "__main__":
    main()
