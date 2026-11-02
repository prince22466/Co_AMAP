"""Deterministic, schema-checked Submission 31 packaging.

Local configuration tests are NOT evidence of Gemma E4B task correctness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / "submission_nb/submission_31"
ARCHIVE = G6 / "submission_nb/artifacts/submission_31.zip"
BASELINE = G6 / "submission_nb/submission_30"
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
FILES = (
    "agent.yaml", "eval_config.yaml",
    "agents/navigator.yaml", "agents/navigator.md",
    "agents/implementer.yaml", "agents/implementer.md",
    "agents/verifier.yaml", "agents/verifier.md",
)
GRAPH_TOOLS = ("get_code_neighbors", "search_similar_code", "get_code_subgraph")
IMPLEMENTER_TOOLS = ("read_file", "edit_file", "run_command", "get_status")
VERIFIER_TOOLS = ("run_command", "read_file", "get_status", "submit_patch")


class IncludeLoader(yaml.SafeLoader):
    """Treat !include values as paths, never execute YAML content as Python."""


IncludeLoader.add_constructor(
    "!include", lambda loader, node: loader.construct_scalar(node))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(payload: dict[str, bytes]) -> None:
    require(set(payload) == set(FILES), "Unexpected submission files")
    configs = {
        name: yaml.load(content, Loader=IncludeLoader)
        for name, content in payload.items() if name.endswith(".yaml")
    }
    root = configs["agent.yaml"]
    require(root.get("agent_class") == "SequentialAgent",
            "S31 must have an ADK SequentialAgent root")
    names = ("navigator", "implementer", "verifier")
    require([x.get("config_path") for x in root.get("sub_agents", [])] ==
            [f"agents/{name}.yaml" for name in names],
            "Stage order/paths changed")
    stages = [configs[f"agents/{name}.yaml"] for name in names]
    for name, stage in zip(names, stages):
        require(stage.get("model") == MODEL, "Non-E4B model in " + name)
        require(stage.get("include_contents") == "none",
                "Stage must use isolated context: " + name)
        require(stage.get("disallow_transfer_to_parent") is True
                and stage.get("disallow_transfer_to_peers") is True,
                "Unexpected agent transfer permissions in " + name)
        require(not any(k in stage for k in
                        ("adapter", "skills", "sub_agents", "agent_class")),
                "No adapters, skills or nested agents in " + name)
        require(stage.get("instruction") == f"{name}.md",
                "Stage prompt include must be relative to agents/: " + name)
        require((SOURCE / "agents" / stage["instruction"]).is_file(),
                "Missing stage prompt: " + name)
        require("write_file" not in stage.get("tools", []),
                "Unsafe overwrite tool exposed in " + name)
        require(stage.get("generate_content_config", {}).get("temperature") == 0.0,
                "Sampling parity drift: " + name)
    require(tuple(stages[0]["tools"]) == GRAPH_TOOLS,
            "Navigation may use only native code graph tools")
    require(tuple(stages[1]["tools"]) == IMPLEMENTER_TOOLS,
            "Implementer must use bounded native editing, no submission")
    require(tuple(stages[2]["tools"]) == VERIFIER_TOOLS,
            "Verifier must independently inspect and exclusively submit")
    require(stages[0]["output_key"] == "navigation_evidence" and
            stages[1]["output_key"] == "implementation_evidence" and
            "output_key" not in stages[2], "Stage handoff keys changed")
    require(sum("submit_patch" in stage["tools"] for stage in stages) == 1,
            "Exactly one stage may submit")
    require(payload["eval_config.yaml"] ==
            (BASELINE / "eval_config.yaml").read_bytes(),
            "Budget differs from S30")
    require(configs["eval_config.yaml"]["evaluation"] == {
        "timeout_seconds": 30, "max_tool_calls": 40,
        "max_time_minutes": 4, "max_turns": 48,
    }, "Unexpected evaluation limits")
    for name, config in configs.items():
        parent = Path(name).parent
        for child in config.get("sub_agents", []):
            require((parent / child["config_path"]).as_posix() in payload,
                    f"Missing referenced child from {name}")
        if isinstance(config.get("instruction"), str):
            require((parent / config["instruction"]).as_posix() in payload,
                    f"Missing instruction include from {name}")
    prompt = "\n".join(payload[f"agents/{name}.md"].decode()
                       for name in names)
    for phrase in ("{problem_description}", "{hints?}",
                   "{navigation_evidence?}", "{implementation_evidence?}",
                   "get_code_neighbors", "search_similar_code",
                   "old_string", "new_string", "git diff --numstat",
                   "submit_patch"):
        require(phrase in prompt, "Missing critical prompt contract: " + phrase)
    # Prevent public development reference patches or task-specific rules
    # from becoming hard-coded into the generated agent.
    for phrase in ("rich_3518", "rich_4077", "fastapi_14258"):
        require(phrase not in prompt, "Task-specific prompt leak: " + phrase)


def build() -> dict:
    payload: dict[str, bytes] = {}
    for name in FILES:
        path = SOURCE / name
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(SOURCE.resolve()),
                f"Unsafe or missing file: {name}")
        payload[name] = path.read_bytes()
    validate(payload)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w") as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    result = {
        "model": MODEL,
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(payload.items())},
        "evidence": "S31 E4B score: NOT TESTED. Do not promote.",
    }
    ARCHIVE.with_suffix(".manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
