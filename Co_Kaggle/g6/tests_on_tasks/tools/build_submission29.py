"""Build a deterministic, self-contained native-tool Submission 29 archive.

This validates package mechanics, never task-solving effectiveness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / "submission_nb/submission_29"
ARTIFACT = G6 / "submission_nb/artifacts/submission_29.zip"
BASELINE = G6 / "submission_nb/submission_27"
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
FILES = ("agent.yaml", "eval_config.yaml", "instruction.md")
NATIVE_TOOLS = ("run_command", "read_file", "edit_file", "write_file",
                "get_status", "submit_patch")


class IncludeLoader(yaml.SafeLoader):
    """Allow filename-only !include for declarative ADK YAML parsing."""


IncludeLoader.add_constructor(
    "!include", lambda loader, node: loader.construct_scalar(node))


def validate(payload: dict[str, bytes]) -> None:
    """Reject unexpected schema/model/limits and harmful tool substitutions."""
    for filename in ("agent.yaml", "eval_config.yaml"):
        assert payload[filename] == (BASELINE / filename).read_bytes(), (
            f"S29 must preserve the known S27 {filename}")
    agent = yaml.load(payload["agent.yaml"], Loader=IncludeLoader)
    cfg = yaml.safe_load(payload["eval_config.yaml"])
    assert agent["model"] == MODEL and "adapter" not in agent
    assert agent["name"] == "repair_agent"
    assert agent["instruction"] == "instruction.md"
    assert agent["include_contents"] == "default"
    assert tuple(agent["tools"]) == NATIVE_TOOLS
    assert "skills" not in agent and "sub_agents" not in agent
    assert cfg["evaluation"] == {
        "timeout_seconds": 30, "max_tool_calls": 40,
        "max_time_minutes": 4, "max_turns": 48,
    }
    instruction = payload["instruction.md"].decode("utf-8")
    assert 1000 < len(instruction) < 4800
    required = ("{problem_description}", "{hints?}", "read_file",
                "start_line", "end_line", "edit_file", "old_string",
                "new_string", "submit_patch", "git diff --numstat",
                "git diff --check", "behavior", "sed -i")
    for phrase in required:
        assert phrase in instruction, f"Missing instruction contract: {phrase}"
    assert "run_skill_script" not in instruction
    assert "do not patch grader" in instruction.lower()


def build() -> dict[str, object]:
    """Produce byte-stable ZIP_STORED archive and separate local manifest."""
    payload: dict[str, bytes] = {}
    for filename in FILES:
        path = SOURCE / filename
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError(f"Unsafe submission path: {filename}")
        payload[filename] = path.read_bytes()
    validate(payload)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARTIFACT, "w") as archive:
        for filename, data in sorted(payload.items()):
            info = zipfile.ZipInfo(filename, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    manifest = {
        "model": MODEL,
        "archive_sha256": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(payload.items())},
        "evidence": "S29 unscored; packaging tests do not demonstrate E4B wins",
    }
    ARTIFACT.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
