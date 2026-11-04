"""Build deterministic S35 E4B archives (skill-enabled and no-skill control).

CPU/package validation does not establish model tool-use or task resolution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import yaml


G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / "submission_nb/submission_35"
ARTIFACT_DIR = G6 / "submission_nb/artifacts"
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
TOOLS = ("run_command", "read_file", "edit_file", "get_status", "submit_patch")
EVAL_BUDGET = {
    "timeout_seconds": 30, "max_tool_calls": 40,
    "max_time_minutes": 4, "max_turns": 48,
}
VARIANTS = {
    "primary": ("agent.yaml", "submission_35.zip", True),
    "control": ("agent_control.yaml", "submission_35_control.zip", False),
}
ROOT_FILES = ("agent.yaml", "eval_config.yaml", "instruction.md")
SKILL_FILE = "skills/tool-recovery/SKILL.md"


class IncludeLoader(yaml.SafeLoader):
    """Read include directives as inert filenames, never resolve arbitrary files."""


IncludeLoader.add_constructor("!include", lambda loader, node:
                              loader.construct_scalar(node))


def require(ok: bool, explanation: str) -> None:
    if not ok:
        raise ValueError(explanation)


def source_bytes(filename: str) -> bytes:
    path = SOURCE / filename
    require(path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(SOURCE.resolve()),
            "Missing or unsafe source: " + filename)
    return path.read_bytes()


def build_payload(variant: str) -> dict[str, bytes]:
    require(variant in VARIANTS, "Unknown S35 variant")
    agent_source, _, with_skill = VARIANTS[variant]
    payload = {
        "agent.yaml": source_bytes(agent_source),
        "eval_config.yaml": source_bytes("eval_config.yaml"),
        "instruction.md": source_bytes("instruction.md"),
    }
    if with_skill:
        payload[SKILL_FILE] = source_bytes(SKILL_FILE)
    validate(variant, payload)
    return payload


def validate(variant: str, payload: dict[str, bytes]) -> None:
    require(variant in VARIANTS, "Unknown S35 variant")
    expected = set(ROOT_FILES)
    if VARIANTS[variant][2]:
        expected.add(SKILL_FILE)
    require(set(payload) == expected, "Wrong archive file set")
    for name in payload:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts,
                "Unsafe archive member")
        require(len(payload[name]) < 100_000, "Excessive packaged file size")

    agent = yaml.load(payload["agent.yaml"], Loader=IncludeLoader)
    require(isinstance(agent, dict), "Invalid agent.yaml")
    require(set(agent) == {"name", "model", "include_contents", "instruction",
                            "skills", "tools", "generate_content_config"}
            if VARIANTS[variant][2]
            else set(agent) == {"name", "model", "include_contents", "instruction",
                                "tools", "generate_content_config"},
            "Unexpected or missing agent config keys")
    require(agent["name"] == "repair_agent"
            and agent["model"] == MODEL
            and agent["include_contents"] == "default"
            and agent["instruction"] == "instruction.md",
            "Unexpected agent settings")
    require(tuple(agent["tools"]) == TOOLS,
            "Tool interface changed or dangerous tool reintroduced")
    generation = agent["generate_content_config"]
    require(generation == {
        "temperature": 0.0, "top_p": 0.95, "max_output_tokens": 4096,
        "thinking_config": {"thinking_budget": 512, "include_thoughts": False},
    }, "Unexpected sampling config")
    require(yaml.safe_load(payload["eval_config.yaml"]) == {
        "evaluation": EVAL_BUDGET}, "Evaluation limits changed")

    with_skill = VARIANTS[variant][2]
    if with_skill:
        require(agent.get("skills") == ["skills/tool-recovery"],
                "Wrong optional skill registration")
        skill = payload[SKILL_FILE].decode("utf-8")
        require(skill.startswith("---\n"), "Skill frontmatter missing")
        parts = skill.split("---", 2)
        require(len(parts) == 3, "Malformed skill frontmatter")
        metadata = yaml.safe_load(parts[1])
        require(metadata.get("name") == "tool-recovery"
                and "description" in metadata and metadata["description"],
                "Skill metadata invalid")
        require("run_skill_script" in skill and "git apply --check" in skill
                and "FileEditError" in skill, "Recovery skill incomplete")
    else:
        require("skills" not in agent, "Control must not register a skill")

    prompt = payload["instruction.md"].decode("utf-8")
    require(1600 <= len(prompt) <= 4200, "Root prompt length unexpected")
    for token in ("{problem_description}", "{hints?}",
                  "old_string", "new_string", "allow_multiple=false",
                  "Missing", "FileEditError", "git diff --check",
                  "submit_patch", "tool-recovery"):
        require(token in prompt, "Missing execution contract: " + token)
    for forbidden in ("run_skill_script(", "write_file(", "rich_3518",
                      "rich_4076", "requests_7315"):
        require(forbidden not in prompt, "Unsupported wrapper or task leak")


def archive_path(variant: str) -> Path:
    require(variant in VARIANTS, "Unknown S35 variant")
    return ARTIFACT_DIR / VARIANTS[variant][1]


def build_variant(variant: str, *, output: Path | None = None) -> dict:
    payload = build_payload(variant)
    output = output or archive_path(variant)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as z:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    manifest = {
        "submission": 35,
        "variant": variant,
        "model": MODEL,
        "tool_count": len(TOOLS),
        "optional_skill": VARIANTS[variant][2],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sources": {name: hashlib.sha256(content).hexdigest()
                    for name, content in sorted(payload.items())},
        "status": "unscored; CPU validation is not real-model evidence",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_all() -> dict[str, dict]:
    return {variant: build_variant(variant) for variant in VARIANTS}


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
