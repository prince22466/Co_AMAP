"""S39 executable ZIP smoke test with pinned official restricted compiler.

The test requires the upstream adk_submission package in PYTHONPATH.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
import zipfile

from adk_submission.compiler import compile_submission
from adk_submission.registry import ModelRegistry
from google.adk.tools.skill_toolset import SkillToolset

import build_submission39 as s39
import build_submission38 as s38


NATIVE_TOOLS = (
    "run_command", "read_file", "edit_file", "write_file",
    "get_status", "submit_patch",
)


def fake_native(name):
    def function(command: str = "") -> str:
        return command
    function.__name__ = name
    function.__qualname__ = name
    return function


class RestrictedCompilerSmokeTest(unittest.TestCase):
    def test_competition_restricted_compiler_loads_s39_and_s38_control(self):
        models = ModelRegistry()
        models.register(s38.MODEL, s38.MODEL)
        registry = {name: fake_native(name) for name in NATIVE_TOOLS}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for variant in ("primary", "control"):
                with self.subTest(variant=variant):
                    target = root / variant
                    target.mkdir()
                    output = root / f"{variant}.zip"
                    s39.build(variant, output)
                    with zipfile.ZipFile(output) as archive:
                        archive.extractall(target)
                    agent = compile_submission(
                        submission_dir=target,
                        tool_registry=registry,
                        model_registry=models,
                    )
                    self.assertEqual(agent.name, "repair_agent")
                    self.assertEqual(agent.model, s38.MODEL)
                    self.assertEqual(len(agent.tools), 7)
                    skills = [
                        t for t in agent.tools if isinstance(t, SkillToolset)
                    ]
                    self.assertEqual(len(skills), 1)
                    tools = asyncio.run(skills[0].get_tools())
                    tool_names = {t.name for t in tools}
                    self.assertIn("load_skill", tool_names)
                    self.assertIn("run_skill_script", tool_names)
                    skill = skills[0]._get_skill("state-guard")
                    self.assertIsNotNone(skill)
                    self.assertIn("state_guard.py", skill.resources.list_scripts())

                    if variant == "primary":
                        self.assertIn(
                            "S37 FAILURE MEMORY", agent.instruction
                        )
                    else:
                        self.assertNotIn(
                            "S37 FAILURE MEMORY", agent.instruction
                        )


if __name__ == "__main__":
    unittest.main()
