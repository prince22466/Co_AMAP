"""Build competition-valid S38 E4B stateful skill ZIP and S36 control.

Only S38 registers an allowed ADK skill (sandbox-persistent script). The
original S36 native tools/model/budgets/instructions are otherwise preserved.
No arbitrary Python callbacks, organizer harness changes or benchmark fixes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import yaml

from build_submission36 import (
    G6, ARTIFACT_DIR, ROOT, MODEL, IncludeLoader,
    payload as s36_payload,
)

SOURCE = G6 / "submission_nb/submission_38"
FILES = (
    "agent.yaml", "eval_config.yaml", "instruction.md",
    "skills/state-guard/SKILL.md",
    "skills/state-guard/scripts/state_guard.py",
)
SKILL_PATH = "skills/state-guard"
NOTE = "stateful_recovery_addendum.md"
ARCHIVE = ARTIFACT_DIR / "submission_38.zip"
CONTROL = ARTIFACT_DIR / "submission_38_control_s36.zip"


def read_source(name: str) -> bytes:
    path = SOURCE / name
    if (not path.is_file() or path.is_symlink()
            or not path.resolve().is_relative_to(SOURCE.resolve())):
        raise ValueError("Missing or unsafe S38 source: " + name)
    return path.read_bytes()


def payload(variant="primary") -> dict[str, bytes]:
    if variant not in ("primary", "control"):
        raise ValueError("Unknown variant")
    s36 = s36_payload("primary")
    if variant == "control":
        return s36
    result = {
        "agent.yaml": read_source("agent.yaml"),
        "eval_config.yaml": read_source("eval_config.yaml"),
        "instruction.md": s36["instruction.md"] + read_source(NOTE),
        "skills/state-guard/SKILL.md": read_source(
            "skills/state-guard/SKILL.md"),
        "skills/state-guard/scripts/state_guard.py": read_source(
            "skills/state-guard/scripts/state_guard.py"),
    }
    validate(result, s36)
    return result


def validate(result: dict[str, bytes], base: dict[str, bytes]) -> None:
    if set(result) != set(FILES):
        raise ValueError("Wrong S38 archive member list")
    for name, data in result.items():
        p = PurePosixPath(name)
        if (p.is_absolute() or ".." in p.parts or not data
                or len(data) > 50_000):
            raise ValueError("Invalid member: " + name)
    if result["eval_config.yaml"] != base["eval_config.yaml"]:
        raise ValueError("S36 time/tool/turn budgets changed")
    reference = yaml.load(base["agent.yaml"], Loader=IncludeLoader)
    candidate = yaml.load(result["agent.yaml"], Loader=IncludeLoader)
    expected = {**reference, "skills": [SKILL_PATH]}
    if candidate != expected:
        raise ValueError("Unexpected model, sampling, native tools or skill config")
    if candidate["model"] != MODEL:
        raise ValueError("E4B model changed")
    note = read_source(NOTE)
    if not note.startswith(b"\nS38 COMPETITION-COMPATIBLE STATEFUL RECOVERY"):
        raise ValueError("S38 prompt must be one appended paragraph")
    if len(note) > 1200:
        raise ValueError("Prompt too large for the targeted ablation")
    if result["instruction.md"] != base["instruction.md"] + note:
        raise ValueError("S36 baseline instructions changed")
    skill = result["skills/state-guard/SKILL.md"].decode("utf-8")
    if not skill.startswith("---\n"):
        raise ValueError("Missing skill frontmatter")
    frontmatter = yaml.safe_load(skill.split("---", 2)[1])
    if (frontmatter.get("name") != "state-guard"
            or not frontmatter.get("description")):
        raise ValueError("Invalid skill frontmatter")
    script = result["skills/state-guard/scripts/state_guard.py"].decode()
    for required in (
        "def workspace(", "def target_file(", "def ledger(",
        "blocked_duplicate", "source_changed_since_read",
        'STATE_NAME = "s38_state_guard.json"',
    ):
        if required not in script:
            raise ValueError("Incomplete persistent state script: " + required)
    for forbidden in (
        b"rich_3518", b"rich_3905", b"rich_4077",
        b"fastapi_14258", b"requests_7315",
        b"before_tool_callback", b"after_tool_callback",
    ):
        if any(forbidden in result[x] for x in result):
            raise ValueError("Benchmark leak or unsupported callback")


def archive_path(variant: str) -> Path:
    return ARCHIVE if variant == "primary" else CONTROL


def build(variant="primary", output: Path | None = None) -> dict:
    parts = payload(variant)
    output = output or archive_path(variant)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as z:
        for name in sorted(parts):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            z.writestr(info, parts[name])
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None or set(z.namelist()) != set(parts):
            raise ValueError("Bad ZIP members")
        for name, content in parts.items():
            if z.read(name) != content:
                raise ValueError("Bad ZIP payload")
    manifest = {
        "submission": 38, "variant": variant,
        "model": MODEL, "control": "S36 primary",
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "member_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(parts.items())
        },
        "native_tool_interception": False,
        "enforced_state_scope": (
            "run_skill_script operations ONLY" if variant == "primary"
            else "none"
        ),
        "score": "unscored; requires E4B S1 and held-out verification",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps({kind: build(kind)
                      for kind in ("control", "primary")}, indent=2))
