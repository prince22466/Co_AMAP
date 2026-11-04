"""Build deterministic S34 ZIP. Packaging correctness is NOT E4B performance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import yaml


G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / "submission_nb/submission_34"
BASELINE = G6 / "submission_nb/submission_29"
ARTIFACT = G6 / "submission_nb/artifacts/submission_34.zip"
FILES = ("agent.yaml", "eval_config.yaml", "instruction.md")
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
NATIVE_TOOLS = ("run_command", "read_file", "edit_file", "write_file",
                "get_status", "submit_patch")


class IncludeLoader(yaml.SafeLoader):
    """Treat !include only as an inert relative path in schema validation."""


IncludeLoader.add_constructor(
    "!include", lambda loader, node: loader.construct_scalar(node))


def require(condition: bool, why: str) -> None:
    if not condition:
        raise ValueError(why)


def validate(files: dict[str, bytes]) -> None:
    require(set(files) == set(FILES), "Unexpected/missing archive member")
    # Keep S29 as a true ablation control: only instruction.md may change.
    for name in ("agent.yaml", "eval_config.yaml"):
        require(files[name] == (BASELINE / name).read_bytes(),
                "S34 must match S29 byte for byte: " + name)
    cfg = yaml.load(files["agent.yaml"], Loader=IncludeLoader)
    require(isinstance(cfg, dict), "Invalid agent.yaml")
    require(cfg.get("name") == "repair_agent"
            and cfg.get("model") == MODEL, "Wrong single E4B agent")
    require(cfg.get("instruction") == "instruction.md"
            and cfg.get("include_contents") == "default", "Config mismatch")
    require(tuple(cfg.get("tools") or ()) == NATIVE_TOOLS,
            "Native tool set changed")
    require("skills" not in cfg and "sub_agents" not in cfg
            and "adapter" not in cfg and "agent_class" not in cfg,
            "Additional agents, skills or adapters are not part of S34")
    budgets = yaml.safe_load(files["eval_config.yaml"])
    require(budgets == {"evaluation": {
        "timeout_seconds": 30, "max_tool_calls": 40,
        "max_time_minutes": 4, "max_turns": 48,
    }}, "Evaluation budgets changed")

    prompt = files["instruction.md"].decode("utf-8")
    old = (BASELINE / "instruction.md").read_text(encoding="utf-8")
    require(700 <= len(prompt) <= int(0.65 * len(old)),
            "S34 prompt is not compact enough for this ablation")
    for phrase in ("{problem_description}", "{hints?}",
                   "start_line", "end_line", "old_string", "new_string",
                   "allow_multiple=false", "write_file",
                   "git diff --check", "git diff --numstat",
                   "submit_patch", "requested", "pytest"):
        require(phrase in prompt, "Missing compact contract: " + phrase)
    for forbidden in ("run_skill_script", "get_code_neighbors",
                      "search_similar_code", "get_code_subgraph",
                      "rich_3518", "rich_3454", "rich_4077"):
        require(forbidden not in prompt, "Extra tool or benchmark leak")


def payload() -> dict[str, bytes]:
    result = {}
    for name in FILES:
        path = SOURCE / name
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(SOURCE.resolve()),
                "Missing or unsafe file: " + name)
        result[name] = path.read_bytes()
    validate(result)
    return result


def build(output: Path = ARTIFACT) -> dict[str, object]:
    files = payload()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    manifest: dict[str, object] = {
        "submission": 34,
        "model": MODEL,
        "control": "submission_29",
        "only_intended_change": "instruction.md",
        "prompt_chars": len(files["instruction.md"].decode("utf-8")),
        "baseline_prompt_chars": len(
            (BASELINE / "instruction.md").read_text(encoding="utf-8")),
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(files.items())},
        "status": "CPU packaging only; E4B issue-resolution NOT TESTED",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
