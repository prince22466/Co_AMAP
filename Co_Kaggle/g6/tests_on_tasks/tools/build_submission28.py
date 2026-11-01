"""Build a deterministic self-contained S28 archive; no model inference involved."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / "submission_nb/submission_28"
ARCHIVE = G6 / "submission_nb/artifacts/submission_28.zip"
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
FILES = ("agent.yaml", "eval_config.yaml", "instruction.md")
TOOLS = ("run_command", "read_file", "edit_file", "write_file",
         "get_status", "submit_patch")


class IncludeLoader(yaml.SafeLoader):
    """Parse ADK's !include filename without executing external YAML constructors."""


IncludeLoader.add_constructor("!include",
                              lambda loader, node: loader.construct_scalar(node))


def validate(payload: dict[str, bytes]) -> None:
    """Reject wrong models, unsupported agents, changed limits and tool removal."""
    agent = yaml.load(payload["agent.yaml"], Loader=IncludeLoader)
    limits = yaml.safe_load(payload["eval_config.yaml"])
    assert agent.get("agent_class", "LlmAgent") == "LlmAgent"
    assert agent["name"] == "repair_agent"
    assert agent["model"] == MODEL and "adapter" not in agent
    assert agent["include_contents"] == "default"
    assert agent["instruction"] == "instruction.md"
    assert tuple(agent["tools"]) == TOOLS
    assert not agent.get("skills") and not agent.get("sub_agents")
    assert agent["generate_content_config"] == {
        "temperature": 0.0, "top_p": 0.95, "max_output_tokens": 4096,
        "thinking_config": {"thinking_budget": 512, "include_thoughts": False},
    }
    assert limits["evaluation"] == {
        "timeout_seconds": 30, "max_tool_calls": 40,
        "max_time_minutes": 4, "max_turns": 48,
    }
    instruction = payload["instruction.md"].decode("utf-8")
    assert 0 < len(instruction) < 3700
    for phrase in ("{problem_description}", "{hints?}", "run_command",
                   "read_file", "sed -n", "edit_file", "git diff",
                   "submit_patch", "BLOCKED"):
        assert phrase in instruction


def build() -> dict[str, object]:
    """Validate source files, write a stable ZIP_STORED package and manifest."""
    payload: dict[str, bytes] = {}
    for name in FILES:
        path = SOURCE / name
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError(f"Unsafe package path: {name}")
        payload[name] = path.read_bytes()
    validate(payload)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w") as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    manifest = {
        "model": MODEL,
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "files": {p: hashlib.sha256(data).hexdigest()
                  for p, data in sorted(payload.items())},
        "evidence": "Unscored S28 prompt experiment; no E4B performance measured",
    }
    ARCHIVE.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
