"""Compile S31 with official runtime and inspect real ADK tool binding.

Requires the Kaggle competition wheels; no E4B inference or model download.
Exit nonzero if the native graph tools or stage permissions fail to compile.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

from build_submission31 import ARCHIVE, MODEL, build


def check_compiler() -> dict:
    from adk_submission import ModelRegistry, ToolRegistry, compile_submission
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import build_submission_limits
    from swegemma.context import SwegemmaContext
    from swegemma.tools import create_tools

    build()
    registry = ToolRegistry()
    available = create_tools(SwegemmaContext())
    for name, fn in available.items():
        registry.register(name, fn)
    required = {"get_code_neighbors", "search_similar_code",
                "get_code_subgraph", "edit_file", "run_command",
                "read_file", "get_status", "submit_patch"}
    missing = required - set(available)
    if missing:
        raise RuntimeError(f"Native tool missing in evaluator: {sorted(missing)}")
    models = ModelRegistry()
    models.register(MODEL, LiteLlm(
        model="openai/" + MODEL,
        api_base="http://localhost:8000/v1", api_key="EMPTY"))
    limits, constraints = build_submission_limits()

    with tempfile.TemporaryDirectory() as folder:
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(folder)
        agent = compile_submission(
            Path(folder), registry, models,
            limits=limits, generation_constraints=constraints)
        assert type(agent).__name__ == "SequentialAgent", type(agent).__name__
        children = list(agent.sub_agents)
        assert len(children) == 3
        assert [x.name for x in children] == [
            "graph_navigator", "bounded_implementer", "independent_verifier"]
        tools = [{getattr(t, "__name__", type(t).__name__) for t in x.tools}
                 for x in children]
        assert {"get_code_neighbors", "search_similar_code",
                "get_code_subgraph"} <= tools[0], tools[0]
        assert {"read_file", "edit_file", "run_command"} <= tools[1], tools[1]
        assert "submit_patch" not in tools[1]
        assert "submit_patch" in tools[2]
        assert all("write_file" not in ts for ts in tools)
        return {
            "compiler": "PASS",
            "root": type(agent).__name__,
            "agents": [x.name for x in children],
            "tools": [sorted(x) for x in tools],
            "warning": "Compiler check does NOT run E4B or prove 5/15",
        }


if __name__ == "__main__":
    print(json.dumps(check_compiler(), indent=2))
