"""Deterministic Submission 33 package: one E4B native repair agent.

The package validator checks configuration, not correctness on issues.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import yaml


G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / "submission_nb/submission_33"
BASELINE = G6 / "submission_nb/submission_32"
ARTIFACT = G6 / "submission_nb/artifacts/submission_33.zip"
FILES = ("agent.yaml", "eval_config.yaml", "instruction.md")
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
TOOLS = ("run_command", "read_file", "edit_file", "get_status", "submit_patch")


class IncludeLoader(yaml.SafeLoader):
    """Parse ADK include references as paths without loading other files."""


IncludeLoader.add_constructor(
    "!include", lambda loader, node: loader.construct_scalar(node)
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def payload() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for name in FILES:
        path = SOURCE / name
        require(
            path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(SOURCE.resolve()),
            "Unsafe or missing package file: " + name,
        )
        files[name] = path.read_bytes()
    validate(files)
    return files


def validate(files: dict[str, bytes]) -> None:
    require(set(files) == set(FILES), "Unexpected root archive members")
    config = yaml.load(files["agent.yaml"], Loader=IncludeLoader)
    baseline = yaml.load((BASELINE / "agent.yaml").read_bytes(),
                         Loader=IncludeLoader)
    require(isinstance(config, dict), "agent.yaml must be a mapping")
    require(tuple(config.get("tools") or ()) == TOOLS,
            "Unexpected tools (no graph/write_file/wrapper allowed)")
    for name in ("name", "model", "include_contents", "instruction",
                 "generate_content_config"):
        require(config.get(name) == baseline.get(name),
                "Unexpected divergence from S32: " + name)
    require(config["model"] == MODEL, "Wrong E4B model")
    require(not any(k in config for k in (
        "sub_agents", "skills", "adapter", "agent_class")),
        "Unexpected multi-agent or adapter configuration")
    require(files["eval_config.yaml"] ==
            (BASELINE / "eval_config.yaml").read_bytes(),
            "Evaluation budget drifted from S32")
    require(yaml.safe_load(files["eval_config.yaml"])["evaluation"] == {
        "timeout_seconds": 30, "max_tool_calls": 40,
        "max_time_minutes": 4, "max_turns": 48,
    }, "Unsupported evaluation limits")

    instruction = files["instruction.md"].decode("utf-8")
    require(2000 < len(instruction) < 6000, "Unbounded prompt size")
    for required in ("{problem_description}", "{hints?}", "old_string",
                     "new_string", "allow_multiple=false", "start_line",
                     "end_line", "git diff --check", "submit_patch",
                     "normal/control", "behavior"):
        require(required in instruction, "Missing instruction: " + required)
    for forbidden in ("rich_3518", "rich_4077", "fastapi_14258",
                      "S32_GUARDED_LINE_EDIT_OK"):
        require(forbidden not in instruction,
                "Benchmark leak or deprecated recovery: " + forbidden)


def build(output: Path = ARTIFACT) -> dict:
    files = payload()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in sorted(files.items()):
            member = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "submission": 33,
        "model": MODEL,
        "sha256": digest,
        "files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(files.items())
        },
        "status": "Packaging only; official compilation and real E4B NOT RUN",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
