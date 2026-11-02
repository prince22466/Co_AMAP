"""Official ADK compiler smoke test for both S32 variants (no model inference).

Requires competition adk-submission, swegemma and google-adk wheels.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

from build_submission32 import ARTIFACT_DIR, MODEL, VARIANTS, build


def smoke() -> dict:
    from adk_submission import ModelRegistry, ToolRegistry, compile_submission
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import build_submission_limits
    from swegemma.context import SwegemmaContext
    from swegemma.tools import create_tools

    build()
    tools = create_tools(SwegemmaContext())
    registry = ToolRegistry()
    for name, fn in tools.items():
        registry.register(name, fn)
    models = ModelRegistry()
    models.register(MODEL, LiteLlm(
        model="openai/local", api_base="http://localhost:8000/v1",
        api_key="EMPTY"))
    limits, constraints = build_submission_limits()
    records = {}
    expected = {"run_command", "read_file", "edit_file",
                "get_status", "submit_patch"}
    for variant, (_, filename) in VARIANTS.items():
        with tempfile.TemporaryDirectory() as folder:
            with zipfile.ZipFile(ARTIFACT_DIR / filename) as z:
                z.extractall(folder)
            agent = compile_submission(
                Path(folder), registry, models,
                limits=limits, generation_constraints=constraints)
            names = {getattr(t, "__name__", type(t).__name__)
                     for t in agent.tools}
            assert names == expected, (variant, names)
            assert agent.name == "repair_agent"
            assert not agent.sub_agents
            assert agent.include_contents == "default"
            assert agent.instruction
            assert "get_code_neighbors" not in names
            assert "write_file" not in names
            records[variant] = sorted(names)
    return {"status": "COMPILER_ONLY_PASS", "tools": records,
            "warning": "No Gemma E4B task was executed"}


if __name__ == "__main__":
    print(json.dumps(smoke(), indent=2))
