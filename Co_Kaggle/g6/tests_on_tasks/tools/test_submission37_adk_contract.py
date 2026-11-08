"""Pinned google-adk==1.36.1 callback integration without model/GPU.

This is intentionally a separate test from fake-ADK tests: it checks the
real PluginManager callback dispatch and Pydantic App validation contract.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.plugin_manager import PluginManager

from submission37_repetition_guard import (
    new_adk_plugin, patch_app_initializer,
)

ARGS = {"filepath": "rich/panel.py", "old_string": "x = 1",
        "new_string": "x = 2", "allow_multiple": False}
FAILED = {"status": "error", "error_type": "FileEditError",
          "error_message": "old_string not found"}


class PinnedADKIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_real_app_plugin_manager_and_tool_callback_contract(self):
        plugins = []
        original_init = App.__init__
        patched_init = patch_app_initializer(
            original_init, new_adk_plugin, plugins)
        with patch.object(App, "__init__", patched_init):
            app = App(
                name="swegemma_eval",
                root_agent=LlmAgent(name="repair_agent", model="gemma-4-e4b-it-qat-w4a16-ct"),
                plugins=[],
            )
        self.assertEqual(len(app.plugins), 1)
        self.assertIs(app.plugins[0], plugins[0])
        self.assertEqual(app.plugins[0].name, "s37_duplicate_failed_edit")

        manager = PluginManager(plugins=app.plugins)
        tool = SimpleNamespace(name="edit_file")
        context = SimpleNamespace(_invocation_context=SimpleNamespace(
            session=SimpleNamespace(id="session-1")))

        self.assertIsNone(await manager.run_before_tool_callback(
            tool=tool, tool_args=ARGS, tool_context=context))
        self.assertIsNone(await manager.run_after_tool_callback(
            tool=tool, tool_args=ARGS, tool_context=context, result=FAILED))
        rejected = await manager.run_before_tool_callback(
            tool=tool, tool_args=ARGS, tool_context=context)
        self.assertEqual(rejected["error_type"], "RepeatedFailedEdit")
        self.assertEqual(plugins[0].guard.stats["duplicate_failures_blocked"], 1)

        # A shell invocation must invalidate before it could mutate.
        command = SimpleNamespace(name="run_command")
        await manager.run_before_tool_callback(
            tool=command, tool_args={"command": "touch x"}, tool_context=context)
        self.assertIsNone(await manager.run_before_tool_callback(
            tool=tool, tool_args=ARGS, tool_context=context))


if __name__ == "__main__":
    unittest.main()
