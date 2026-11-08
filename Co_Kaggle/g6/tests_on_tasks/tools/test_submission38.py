"""S38 CPU contract + real sandbox-script behavior tests.

These do NOT prove E4B invokes run_skill_script; inspect live S1 traces.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import build_submission38 as s38
from build_submission36 import payload as s36_payload


class PackagingTests(unittest.TestCase):
    def test_s36_parity_except_one_skill_registration_and_prompt_addendum(self):
        base = s36_payload("primary")
        candidate = s38.payload()
        self.assertEqual(candidate["eval_config.yaml"], base["eval_config.yaml"])
        self.assertEqual(
            candidate["instruction.md"],
            base["instruction.md"] + s38.read_source(s38.NOTE))
        self.assertIn(b"skills/state-guard", candidate["agent.yaml"])
        self.assertEqual(s38.payload("control"), base)

    def test_archives_are_deterministic_and_contain_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind in ("primary", "control"):
                a = root / (kind + "a.zip")
                b = root / (kind + "b.zip")
                manifest = s38.build(kind, a)
                s38.build(kind, b)
                self.assertEqual(a.read_bytes(), b.read_bytes())
                self.assertEqual(
                    manifest["archive_sha256"],
                    hashlib.sha256(a.read_bytes()).hexdigest())
                with zipfile.ZipFile(a) as z:
                    self.assertEqual(set(z.namelist()), set(s38.payload(kind)))
                    for name, data in s38.payload(kind).items():
                        self.assertEqual(z.read(name), data)


SCRIPT = s38.SOURCE / "skills/state-guard/scripts/state_guard.py"


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "workspace"
        self.root.mkdir()
        (self.root / ".git").mkdir()
        (self.root / "pkg").mkdir()
        self.source = self.root / "pkg" / "module.py"
        self.source.write_text("first = 1\nsecond = 2\n", encoding="utf-8")
        self.runner_cwd = Path(self.tmp.name) / "script_temp"
        self.runner_cwd.mkdir()

    def call(self, action, **kwargs):
        cmd = [sys.executable, str(SCRIPT), "--action", action]
        for k, v in kwargs.items():
            cmd += ["--" + k, str(v)]
        env = os.environ.copy()
        env["S38_WORKSPACE"] = str(self.root)
        done = subprocess.run(
            cmd, cwd=self.runner_cwd, env=env, capture_output=True, text=True,
            check=True, timeout=10)
        self.assertEqual(done.stderr, "")
        return json.loads(done.stdout.strip())

    def test_failed_edit_is_durably_blocked_on_same_file_revision(self):
        kwargs = {"file": "pkg/module.py", "old": "missing = 1",
                  "new": "missing = 2"}
        one = self.call("edit", **kwargs)
        self.assertEqual(one["status"], "rejected")
        self.assertEqual(one["reason"], "old_string_not_found")
        two = self.call("edit", **kwargs)
        self.assertEqual(two["status"], "blocked_duplicate")
        self.assertEqual(self.source.read_text(), "first = 1\nsecond = 2\n")
        state_file = self.root / ".git/s38_state_guard.json"
        self.assertTrue(state_file.exists())
        self.assertNotIn("missing = 1", state_file.read_text())

        # Native tools may alter source. Revised hash permits retry.
        self.source.write_text("missing = 1\n", encoding="utf-8")
        three = self.call("edit", **kwargs)
        self.assertEqual(three["status"], "ok")
        self.assertEqual(self.source.read_text(), "missing = 2\n")

    def test_edit_applies_once_and_preserves_other_source(self):
        reply = self.call("edit", file="pkg/module.py", old="second = 2",
                          new="second = 3")
        self.assertEqual(reply["status"], "ok")
        self.assertEqual(self.source.read_text(),
                         "first = 1\nsecond = 3\n")
        self.assertEqual(self.call("edit", file="pkg/module.py",
                                   old="second = 2", new="second = 3")["reason"],
                         "old_string_not_found")
        self.assertEqual(self.call("edit", file="pkg/module.py",
                                   old="second = 2", new="second = 3")["status"],
                         "blocked_duplicate")

    def test_duplicate_reads_use_current_source_digest_and_line_range(self):
        args = {"file": "pkg/module.py", "start": 1, "end": 2}
        result = self.call("read", **args)
        self.assertEqual(result["status"], "ok")
        self.assertIn("2: second = 2", result["content"])
        self.assertEqual(self.call("read", **args)["status"], "blocked_duplicate")
        self.assertEqual(
            self.call("read", file="pkg/module.py", start=2, end=2)["status"],
            "ok")
        self.source.write_text("first = 99\nsecond = 2\n")
        self.assertEqual(self.call("read", **args)["status"], "ok")

    def test_ambiguous_anchor_and_no_mutation(self):
        self.source.write_text("x = 1\nx = 1\n")
        r = self.call("edit", file="pkg/module.py", old="x = 1", new="x = 2")
        self.assertEqual(r["reason"], "ambiguous_anchor")
        self.assertEqual(self.source.read_text(), "x = 1\nx = 1\n")

    def test_path_traversal_git_internals_and_symlinks_are_refused(self):
        other = Path(self.tmp.name) / "elsewhere.py"
        other.write_text("old\n")
        (self.root / "pkg" / "link.py").symlink_to(other)
        for file in (
            "../elsewhere.py", ".git/s38_state_guard.json", "pkg/link.py",
            ".", "",
        ):
            with self.subTest(file=file):
                r = self.call("edit", file=file, old="old", new="new")
                self.assertEqual(r["status"], "error")
        self.assertEqual(other.read_text(), "old\n")

    def test_bounded_edits_and_reads(self):
        r = self.call("edit", file="pkg/module.py", old="first = 1",
                      new="new\n" * 100)
        self.assertEqual(r["status"], "error")
        self.assertEqual(
            self.call("read", file="pkg/module.py",
                      start=1, end=150)["status"], "error")
        self.assertEqual(self.source.read_text(), "first = 1\nsecond = 2\n")

    def test_no_workspace_detected_when_script_runs_in_temp_dir(self):
        # ADK runs scripts from a newly materialized temporary directory;
        # PWD is NOT the correct repository root.
        env = os.environ.copy()
        env.pop("S38_WORKSPACE", None)
        env["PWD"] = str(self.runner_cwd)
        env["PYTHONPATH"] = str(self.runner_cwd)
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "--action", "status"],
            cwd=self.runner_cwd, env=env, capture_output=True, text=True,
            check=True)
        self.assertEqual(json.loads(done.stdout)["status"], "error")

    def test_state_resets_across_task_sandboxes(self):
        self.call("edit", file="pkg/module.py", old="missing",
                  new="other")
        other_root = Path(self.tmp.name) / "new_task"
        other_root.mkdir()
        (other_root / ".git").mkdir()
        (other_root / "pkg").mkdir()
        (other_root / "pkg/module.py").write_text("missing\n")
        env = os.environ.copy()
        env["S38_WORKSPACE"] = str(other_root)
        done = subprocess.run([
            sys.executable, str(SCRIPT), "--action", "edit",
            "--file", "pkg/module.py", "--old", "missing", "--new", "other",
        ], cwd=self.runner_cwd, env=env, capture_output=True, text=True,
                            check=True)
        self.assertEqual(json.loads(done.stdout)["status"], "ok")
        self.assertEqual(
            (other_root / "pkg/module.py").read_text(), "other\n")


class RealADKSkillLoaderTests(unittest.TestCase):
    def test_skill_files_are_understood_by_pinned_google_adk(self):
        from google.adk.skills import load_skill_from_dir
        skill = load_skill_from_dir(s38.SOURCE / "skills/state-guard")
        self.assertEqual(skill.name, "state-guard")
        self.assertIn("state_guard.py", skill.resources.list_scripts())
        self.assertIsNotNone(skill.resources.get_script("state_guard.py"))


class RealADKRunSkillScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_adk_script_tool_passes_args_and_persists_state(self):
        from google.adk.code_executors.code_execution_utils import CodeExecutionResult
        from google.adk.skills import load_skill_from_dir
        from google.adk.tools.skill_toolset import SkillToolset
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            (root / ".git").mkdir(parents=True)
            (root / "pkg").mkdir()
            (root / "pkg/module.py").write_text("alpha = 1\n", encoding="utf-8")

            class LocalSandboxExecutor:
                def execute_code(self, invocation_context, code_execution_input):
                    env = os.environ.copy()
                    env["S38_WORKSPACE"] = str(root)
                    proc = subprocess.run(
                        [sys.executable, "-c", code_execution_input.code],
                        cwd=root, env=env, text=True, capture_output=True,
                        timeout=10)
                    return CodeExecutionResult(
                        stdout=proc.stdout, stderr=proc.stderr)

            skill = load_skill_from_dir(s38.SOURCE / "skills/state-guard")
            toolset = SkillToolset(
                skills=[skill], code_executor=LocalSandboxExecutor())
            available = {tool.name: tool
                         for tool in await toolset.get_tools()}
            self.assertIn("run_skill_script", available)
            tool = available["run_skill_script"]
            context = SimpleNamespace(
                invocation_id="test-invocation",
                _invocation_context=object(),
            )

            async def invoke(action, **kwargs):
                response = await tool.run_async(
                    args={
                        "skill_name": "state-guard",
                        "file_path": "scripts/state_guard.py",
                        "args": {"action": action, **kwargs},
                    },
                    tool_context=context)
                self.assertEqual(response["status"], "success")
                self.assertFalse(response["stderr"])
                return json.loads(response["stdout"].strip())

            # This is exactly the tool signature the small model sees.
            args = {"file": "pkg/module.py", "old": "not_present",
                    "new": "new_value"}
            first = await invoke("edit", **args)
            second = await invoke("edit", **args)
            self.assertEqual(first["status"], "rejected")
            self.assertEqual(second["status"], "blocked_duplicate")
            self.assertEqual(
                (root / "pkg/module.py").read_text(), "alpha = 1\n")

            read_args = {"file": "pkg/module.py", "start": 1, "end": 1}
            self.assertEqual((await invoke("read", **read_args))["status"],
                             "ok")
            self.assertEqual((await invoke("read", **read_args))["status"],
                             "blocked_duplicate")

            edited = await invoke(
                "edit", file="pkg/module.py",
                old="alpha = 1", new="alpha = 2")
            self.assertEqual(edited["status"], "ok")
            self.assertEqual(
                (root / "pkg/module.py").read_text(), "alpha = 2\n")


if __name__ == "__main__":
    unittest.main()
