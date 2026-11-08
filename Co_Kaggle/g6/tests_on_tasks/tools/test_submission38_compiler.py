"""Compile S38 through the pinned organizer restricted adk-submission compiler.

CI checks out the organizer's adk_submission source at an exact Git SHA.
Unlike independent YAML validation, this catches registry/schema/skill errors
that would prevent an extracted competition ZIP from loading.
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

import build_submission38 as s38


NATIVE_TOOLS = (
    "run_command", "read_file", "edit_file", "write_file",
    "get_status", "submit_patch",
)


def make_callable(name):
    def native_tool(command: str = "") -> str:
        return command
    native_tool.__name__ = name
    native_tool.__qualname__ = name
    return native_tool


class OfficialCompilerSmokeTest(unittest.TestCase):
    def test_restricted_compiler_loads_both_zip_variants(self):
        models = ModelRegistry()
        models.register(s38.MODEL, s38.MODEL)
        registry = {name: make_callable(name) for name in NATIVE_TOOLS}

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for variant in ("control", "primary"):
                with self.subTest(variant=variant):
                    archive = base / f"{variant}.zip"
                    s38.build(variant=variant, output=archive)
                    submission = base / variant
                    submission.mkdir()
                    with zipfile.ZipFile(archive) as z:
                        z.extractall(submission)
                    agent = compile_submission(
                        submission_dir=submission,
                        tool_registry=registry,
                        model_registry=models,
                    )
                    self.assertEqual(agent.name, "repair_agent")
                    self.assertEqual(len(agent.tools),
                                     7 if variant == "primary" else 6)
                    skills = [
                        tool for tool in agent.tools
                        if isinstance(tool, SkillToolset)
                    ]
                    self.assertEqual(len(skills),
                                     1 if variant == "primary" else 0)
                    if skills:
                        tools = asyncio.run(skills[0].get_tools())
                        names = {tool.name for tool in tools}
                        self.assertIn("run_skill_script", names)
                        self.assertIn("load_skill", names)
                        skill = skills[0]._get_skill("state-guard")
                        self.assertIsNotNone(skill)
                        self.assertIn("state_guard.py",
                                      skill.resources.list_scripts())


if __name__ == "__main__":
    unittest.main()
