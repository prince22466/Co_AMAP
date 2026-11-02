"""Build the reproducible E4B S30 native-code-graph submission archive.

Packaging tests establish format and configuration parity, NOT task accuracy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / "submission_nb/submission_30"
BASELINE = G6 / "submission_nb/submission_29"
ARTIFACT = G6 / "submission_nb/artifacts/submission_30.zip"
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
FILES = ("agent.yaml", "eval_config.yaml", "instruction.md")
NATIVE = ("run_command", "read_file", "edit_file", "write_file",
          "get_status", "submit_patch")
GRAPH = ("get_code_neighbors", "search_similar_code", "get_code_subgraph")


class IncludeLoader(yaml.SafeLoader):
    """Read ADK !include paths without executing imports or templates."""


IncludeLoader.add_constructor(
    "!include", lambda loader, node: loader.construct_scalar(node))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(payload: dict[str, bytes]) -> None:
    require(set(payload) == set(FILES), "Archive file set differs from allowlist")
    agent = yaml.load(payload["agent.yaml"], Loader=IncludeLoader)
    baseline = yaml.load((BASELINE / "agent.yaml").read_bytes(),
                         Loader=IncludeLoader)
    require(isinstance(agent, dict), "agent.yaml must contain a mapping")
    require(payload["eval_config.yaml"] ==
            (BASELINE / "eval_config.yaml").read_bytes(),
            "Evaluation limits must remain identical to S29")
    for key in ("name", "model", "include_contents", "instruction",
                "generate_content_config"):
        require(agent.get(key) == baseline.get(key),
                f"S30 unexpectedly changed S29 {key}")
    require(agent["model"] == MODEL, "Wrong competition model")
    require(tuple(baseline["tools"]) == NATIVE, "Baseline changed unexpectedly")
    require(tuple(agent.get("tools", [])) == NATIVE + GRAPH,
            "Only the three native graph tools may be added")
    require(not any(x in agent for x in ("adapter", "skills", "sub_agents")),
            "No adapters, skills or subagents in this experiment")
    require(yaml.safe_load(payload["eval_config.yaml"])["evaluation"] == {
        "timeout_seconds": 30, "max_tool_calls": 40,
        "max_time_minutes": 4, "max_turns": 48,
    }, "Unexpected evaluation limits")
    prompt = payload["instruction.md"].decode("utf-8")
    require(900 < len(prompt) < 6500, "Prompt outside bounded size")
    for phrase in ("{problem_description}", "{hints?}", "search_similar_code",
                   "get_code_neighbors", "get_code_subgraph", "k=3",
                   "max_neighbors=8", "old_string", "new_string",
                   "allow_multiple=false", "start_line", "end_line",
                   "git diff --check", "submit_patch", "sed -i"):
        require(phrase in prompt, "Missing S30 instruction contract: " + phrase)


def build() -> dict:
    payload = {}
    for name in FILES:
        path = SOURCE / name
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(SOURCE.resolve()),
                f"Invalid source path: {path}")
        payload[name] = path.read_bytes()
    validate(payload)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARTIFACT, "w") as z:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    manifest = {
        "model": MODEL,
        "archive_sha256": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(payload.items())},
        "evidence": "S30: real Gemma E4B task evaluation NOT RUN",
    }
    ARTIFACT.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
