"""Build S36 parser-aware and exact S29 control ZIPs, deterministically.

This changes model-facing string serialization guidance only. Packaging tests
cannot establish real E4B tool-call validity, correctness or Kaggle score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import yaml


G6 = Path(__file__).resolve().parents[2]
BASE = G6 / "submission_nb/submission_29"
SOURCE = G6 / "submission_nb/submission_36"
ARTIFACT_DIR = G6 / "submission_nb/artifacts"
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
TOOLS = ("run_command", "read_file", "edit_file", "write_file",
         "get_status", "submit_patch")
ROOT = ("agent.yaml", "eval_config.yaml", "instruction.md")
ARCHIVES = {
    "control": "submission_36_control.zip",
    "primary": "submission_36.zip",
}
BUDGET = {"timeout_seconds": 30, "max_tool_calls": 40,
          "max_time_minutes": 4, "max_turns": 48}


class IncludeLoader(yaml.SafeLoader):
    """Treat !include as a filename, without opening or evaluating it."""


IncludeLoader.add_constructor(
    "!include", lambda loader, node: loader.construct_scalar(node)
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def safe_read(folder: Path, filename: str) -> bytes:
    path = folder / filename
    require(path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(folder.resolve()),
            "Missing or unsafe input: " + str(path))
    return path.read_bytes()


def payload(variant: str) -> dict[str, bytes]:
    require(variant in ARCHIVES, "Unknown S36 variant")
    data = {filename: safe_read(BASE, filename) for filename in ROOT}
    if variant == "primary":
        data["instruction.md"] += safe_read(SOURCE, "parser_aware_addendum.md")
    validate(variant, data)
    return data


def validate(variant: str, data: dict[str, bytes]) -> None:
    require(variant in ARCHIVES, "Unknown S36 variant")
    require(set(data) == set(ROOT), "Unexpected archive members")
    for name, content in data.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts
                and len(content) <= 100_000, "Unsafe member: " + name)

    for name in ("agent.yaml", "eval_config.yaml"):
        require(data[name] == safe_read(BASE, name),
                "S29 baseline drift in " + name)

    control_prompt = safe_read(BASE, "instruction.md")
    expected = control_prompt
    if variant == "primary":
        note = safe_read(SOURCE, "parser_aware_addendum.md")
        require(note.startswith(b"\nGEMMA 4 TOOL-ARGUMENT STRING FORMAT"),
                "Incorrect addendum")
        for part in (b'<|"|>', b"filepath:", b"old_string:",
                     b"new_string:", b"allow_multiple:false",
                     b"NOT a shell command", b"actual native tool interface"):
            require(part in note, "Incomplete parser-aware guidance: " + str(part))
        for forbidden in (b"rich_3518", b"rich_3454", b"rich_4077",
                          b"requests_7315", b"fastapi_14258",
                          b"<|tool_call>"):
            require(forbidden not in note, "Benchmark leak / call-template risk")
        expected += note
    require(data["instruction.md"] == expected,
            "S36 must differ from S29 only by the exact addendum")

    agent = yaml.load(data["agent.yaml"], Loader=IncludeLoader)
    require(isinstance(agent, dict) and set(agent) == {
        "name", "model", "include_contents", "instruction",
        "tools", "generate_content_config"}, "Unexpected agent schema")
    require(agent["name"] == "repair_agent"
            and agent["model"] == MODEL
            and agent["include_contents"] == "default"
            and agent["instruction"] == "instruction.md",
            "Unexpected model or agent identity")
    require(tuple(agent["tools"]) == TOOLS, "Native tool set differs from S29")
    require(agent["generate_content_config"] == {
        "temperature": 0.0, "top_p": 0.95, "max_output_tokens": 4096,
        "thinking_config": {"thinking_budget": 512, "include_thoughts": False},
    }, "Sampling or thinking differs from S29")
    require(yaml.safe_load(data["eval_config.yaml"]) == {
        "evaluation": BUDGET}, "Evaluation budget differs from S29")


def archive_path(variant: str) -> Path:
    require(variant in ARCHIVES, "Unknown S36 variant")
    return ARTIFACT_DIR / ARCHIVES[variant]


def build(variant: str, output: Path | None = None) -> dict:
    data = payload(variant)
    output = archive_path(variant) if output is None else output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in sorted(data.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    manifest = {
        "submission": 36, "variant": variant, "model": MODEL,
        "control": "submission_29",
        "only_model_facing_change": ("none" if variant == "control"
                                     else "parser_aware_addendum.md"),
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "members_sha256": {name: hashlib.sha256(content).hexdigest()
                           for name, content in sorted(data.items())},
        "status": "unscored; CPU packaging checks are not E4B task evidence",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps({variant: build(variant) for variant in ARCHIVES}, indent=2))


if __name__ == "__main__":
    main()
