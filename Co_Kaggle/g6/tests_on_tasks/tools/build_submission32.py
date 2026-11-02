"""Build deterministic S32 guarded-edit and native-only control packages.

No official compiler invocation or real Gemma E4B scoring is implied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / "submission_nb/submission_32"
ARTIFACT_DIR = G6 / "submission_nb/artifacts"
BASELINE = G6 / "submission_nb/submission_30"
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
FILES = ("agent.yaml", "eval_config.yaml", "instruction.md")
TOOLS = ("run_command", "read_file", "edit_file", "get_status", "submit_patch")
VARIANTS = {
    "primary": ("instruction.md", "submission_32.zip"),
    "control": ("instruction_control.md", "submission_32_control.zip"),
}


class IncludeLoader(yaml.SafeLoader):
    """Allow !include path to remain literal without loading arbitrary files."""


IncludeLoader.add_constructor(
    "!include", lambda loader, node: loader.construct_scalar(node))


def check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def payload(variant: str) -> dict[str, bytes]:
    check(variant in VARIANTS, "Unknown S32 variant")
    instruction, _ = VARIANTS[variant]
    names = {
        "agent.yaml": "agent.yaml",
        "eval_config.yaml": "eval_config.yaml",
        "instruction.md": instruction,
    }
    result = {}
    for destination, source_name in names.items():
        source = SOURCE / source_name
        check(source.is_file() and not source.is_symlink()
              and source.resolve().is_relative_to(SOURCE.resolve()),
              "Missing, symlinked or unsafe source: " + source_name)
        result[destination] = source.read_bytes()
    validate(result, variant)
    return result


def validate(files: dict[str, bytes], variant: str) -> None:
    check(set(files) == set(FILES), "Invalid root archive files")
    config = yaml.load(files["agent.yaml"], Loader=IncludeLoader)
    base = yaml.load((BASELINE / "agent.yaml").read_bytes(),
                     Loader=IncludeLoader)
    check(isinstance(config, dict), "Agent config must be mapping")
    check(config.get("model") == MODEL, "Wrong E4B model")
    check(tuple(config.get("tools") or ()) == TOOLS, "Native tool set changed")
    check(config.get("instruction") == "instruction.md", "Wrong include path")
    check(config.get("name") == base.get("name"), "Unexpected ADK name")
    check(config.get("include_contents") == "default", "Context mode changed")
    check(config.get("generate_content_config") ==
          base.get("generate_content_config"), "Uncontrolled sampling change")
    check(not any(k in config for k in
                  ("sub_agents", "skills", "adapter", "agent_class")),
          "Must be exactly one native agent")
    check(files["eval_config.yaml"] ==
          (BASELINE / "eval_config.yaml").read_bytes(),
          "S32 budget drift versus S30")
    check(yaml.safe_load(files["eval_config.yaml"])["evaluation"] == {
        "timeout_seconds": 30, "max_tool_calls": 40,
        "max_time_minutes": 4, "max_turns": 48,
    }, "Unexpected task evaluation limits")
    prompt = files["instruction.md"].decode("utf-8")
    check(1100 < len(prompt) < 7500, "Prompt outside bounded length")
    for phrase in ("{problem_description}", "{hints?}", "old_string",
                   "new_string", "read_file", "edit_file", "submit_patch",
                   "git diff --check", "run_command"):
        check(phrase in prompt, "Missing prompt contract: " + phrase)
    for token in ("rich_3518", "rich_4077", "fastapi_14258",
                  "get_code_neighbors(", "search_similar_code("):
        check(token not in prompt, "Task-specific or graph instruction leaked")
    if variant == "primary":
        for token in ("S32_GUARDED_LINE_EDIT_OK", "ast.parse",
                      "tempfile.NamedTemporaryFile", "os.replace",
                      'p.stat().st_size < 200000'):
            check(token in prompt, "Missing bounded recovery guard: " + token)
    else:
        check("S32_GUARDED_LINE_EDIT_OK" not in prompt,
              "Control must not include fallback")
        check("NO shell fallback" in prompt,
              "Control must explicitly exclude the treatment")


def build_variant(variant: str) -> dict:
    content = payload(variant)
    _, filename = VARIANTS[variant]
    archive_path = ARTIFACT_DIR / filename
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as z:
        for name, data in sorted(content.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    manifest = {
        "variant": variant,
        "model": MODEL,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(content.items())},
        "confidence": "Packaging only. Real E4B inference NOT RUN.",
    }
    archive_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build() -> dict:
    return {name: build_variant(name) for name in VARIANTS}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
