"""S37 CPU tests: deterministic memory, tool callback semantics and parity."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from submission37_repetition_guard import (
    RepetitionGuard, arguments_digest, error_kind, patch_app_initializer,
    session_id, was_synthetically_blocked,
)


EDIT = {"filepath": "rich/panel.py", "old_string": "x = 1",
        "new_string": "x = 2", "allow_multiple": False}
BAD = {"status": "error", "error_type": "FileEditError",
       "error_message": "Failed to replace: old_string not found"}
OK = {"status": "ok", "filepath": "rich/panel.py"}
BLOCK = {"status": "error", "error_type": "RepeatedFailedEdit"}


class CoreTests(unittest.TestCase):
    def test_blocks_only_exact_failed_edits_and_preserves_ledger(self):
        g = RepetitionGuard()
        self.assertIsNone(g.before("task-1", "edit_file", EDIT))
        g.after("task-1", "edit_file", EDIT, BAD)
        denied = g.before("task-1", "edit_file", dict(reversed(list(EDIT.items()))))
        self.assertEqual(denied["error_type"], "RepeatedFailedEdit")
        self.assertIsNone(g.before("task-1", "edit_file",
                                   {**EDIT, "new_string": "x = 3"}))
        # Some ADK versions invoke after_tool_callback on intercepted results:
        # a synthetic denial MUST NOT erase the original failed anchor.
        g.after("task-1", "edit_file", EDIT, denied)
        self.assertIsNotNone(g.before("task-1", "edit_file", EDIT))
        self.assertEqual(g.stats["duplicate_failures_blocked"], 2)
        self.assertEqual(g.stats["deterministic_edit_failures"], 1)
        self.assertNotIn("x = 1", json.dumps(g.report()))

    def test_block_cap_prevents_endless_uncharged_retry_loop(self):
        g = RepetitionGuard()
        g.after("task", "edit_file", EDIT, BAD)
        self.assertIsNotNone(g.before("task", "edit_file", EDIT))
        self.assertIsNotNone(g.before("task", "edit_file", EDIT))
        # After two synthetic denials, return control to native edit_file
        # so its usual charged call budget can stop further looping.
        self.assertIsNone(g.before("task", "edit_file", EDIT))
        self.assertEqual(g.stats["block_cap_reached"], 1)
        g.after("task", "run_command", {"command": "git diff"}, {"status": "ok"})
        g.after("task", "edit_file", EDIT, BAD)
        self.assertIsNotNone(g.before("task", "edit_file", EDIT))

    def test_read_does_not_reset_but_any_shell_can_mutate(self):
        g = RepetitionGuard()
        g.after("s", "edit_file", EDIT, {"result": json.dumps(BAD)})
        g.after("s", "read_file", {"filepath": EDIT["filepath"]}, OK)
        self.assertIsNotNone(g.before("s", "edit_file", EDIT))
        g.after("s", "run_command", {"command": "git status"}, {"status": "ok"})
        self.assertIsNone(g.before("s", "edit_file", EDIT))

    def test_success_and_write_invalidate_failed_anchors(self):
        g = RepetitionGuard()
        g.after("s", "edit_file", EDIT, BAD)
        g.after("s", "edit_file", {**EDIT, "new_string": "x = 4"}, OK)
        self.assertIsNone(g.before("s", "edit_file", EDIT))
        g.after("s", "edit_file", EDIT, BAD)
        g.after("s", "write_file", {"filepath": "a.py"}, {"status": "error"})
        self.assertIsNone(g.before("s", "edit_file", EDIT))

    def test_session_isolation(self):
        g = RepetitionGuard()
        g.after("one", "edit_file", EDIT, BAD)
        self.assertIsNone(g.before("two", "edit_file", EDIT))
        self.assertIsNotNone(g.before("one", "edit_file", EDIT))
        self.assertIsNone(g.before(None, "edit_file", EDIT))

    def test_fail_open_on_unknown_errors_and_uncanonicalizable_arguments(self):
        g = RepetitionGuard()
        g.after("s", "edit_file", EDIT, {
            "status": "error", "error_type": "BudgetExceeded"})
        self.assertIsNone(g.before("s", "edit_file", EDIT))
        weird = {**EDIT, "old_string": object()}
        self.assertIsNone(arguments_digest("edit_file", weird))
        g.after("s", "edit_file", weird, BAD)
        self.assertIsNone(g.before("s", "edit_file", weird))

    def test_only_specific_error_taxonomy_is_cached(self):
        self.assertEqual(error_kind(BAD), "FileEditError")
        self.assertEqual(error_kind({"result": json.dumps(BAD)}), "FileEditError")
        self.assertEqual(error_kind({
            "error_message": "mandatory input parameters are not provided"
        }), "MissingArguments")
        self.assertIsNone(error_kind({
            "status": "error", "error_type": "BudgetExceeded"}))
        self.assertTrue(was_synthetically_blocked({"result": json.dumps(BLOCK)}))
        self.assertFalse(was_synthetically_blocked(BAD))

    def test_exception_caching_only_for_file_edit_error(self):
        g = RepetitionGuard()
        FakeEditError = type("FileEditError", (Exception,), {})
        g.on_error("s", "edit_file", EDIT, FakeEditError("not found"))
        self.assertIsNotNone(g.before("s", "edit_file", EDIT))
        g.on_error("s", "run_command", {"command": "echo test"},
                   RuntimeError("fail"))
        self.assertIsNone(g.before("s", "edit_file", EDIT))

    def test_context_id_must_exist(self):
        ok = SimpleNamespace(_invocation_context=SimpleNamespace(
            session=SimpleNamespace(id="task-id")))
        self.assertEqual(session_id(ok), "task-id")
        self.assertIsNone(session_id(object()))

    def test_app_injection_keeps_original_plugins(self):
        class FakeApp:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        injected = []
        factory = lambda: object()
        patched = patch_app_initializer(FakeApp.__init__, factory, injected)
        obj = object.__new__(FakeApp)
        current = object()
        patched(obj, plugins=[current], name="test", root_agent="agent")
        self.assertEqual(obj.kwargs["plugins"], [current, injected[0]])
        self.assertEqual(obj.kwargs["name"], "test")
        with self.assertRaises(RuntimeError):
            patched(obj, "positional")
        with self.assertRaises(RuntimeError):
            patched(obj, plugins="invalid")
        self.assertEqual(len(injected), 1)


if __name__ == "__main__":
    unittest.main()
