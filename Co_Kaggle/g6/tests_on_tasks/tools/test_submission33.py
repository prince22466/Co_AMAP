"""CPU tests for S33 package reproducibility and safety/config contracts.

No test here can establish real E4B issue-resolution performance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from build_submission33 import FILES, SOURCE, build, payload, validate


class Submission33Packaging(unittest.TestCase):
    def test_deterministic_zip_and_exact_root_file_set(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "candidate1.zip"
            second = Path(directory) / "candidate2.zip"
            manifest1 = build(first)
            manifest2 = build(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(manifest1["sha256"], manifest2["sha256"])
            self.assertEqual(manifest1["sha256"],
                             hashlib.sha256(first.read_bytes()).hexdigest())
            with zipfile.ZipFile(first) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), set(FILES))
                for name, data in payload().items():
                    self.assertEqual(archive.read(name), data)

    def test_no_new_agents_tools_adapters_or_budget_changes(self):
        files = payload()
        agent = files["agent.yaml"].decode()
        for excluded in ("  - write_file\n", "get_code_neighbors",
                         "get_code_subgraph", "search_similar_code",
                         "sub_agents:", "adapter:", "skills:"):
            self.assertNotIn(excluded, agent)
        for token in ("run_command", "read_file", "edit_file",
                      "get_status", "submit_patch"):
            self.assertIn("  - " + token + "\n", agent)
        instruction = files["instruction.md"].decode()
        self.assertNotIn("S32_GUARDED_LINE_EDIT_OK", instruction)
        self.assertIn("behavior", instruction)
        self.assertEqual((SOURCE / "instruction.md").read_bytes(),
                         files["instruction.md"])

    def test_reject_modified_tool_set_or_config(self):
        files = payload()
        cases = {
            "write_file": {"agent.yaml": files["agent.yaml"].replace(
                b"  - edit_file\n", b"  - edit_file\n  - write_file\n")},
            "graph": {"agent.yaml": files["agent.yaml"].replace(
                b"  - edit_file\n", b"  - edit_file\n  - get_code_neighbors\n")},
            "adapter": {"agent.yaml": files["agent.yaml"] + b"adapter: x\n"},
            "budget": {"eval_config.yaml": files["eval_config.yaml"].replace(
                b"max_tool_calls: 40", b"max_tool_calls: 80")},
            "leak": {"instruction.md": files["instruction.md"] +
                     b"\nrich_3518\n"},
            "missing_prompt": {"instruction.md": b"Hello"},
        }
        for name, replacement in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate({**files, **replacement})

    def test_reject_missing_or_extra_zip_members(self):
        files = payload()
        for change in ({k: v for k, v in files.items()
                        if k != "eval_config.yaml"},
                       {**files, "unwanted.py": b"x"}):
            with self.assertRaises(ValueError):
                validate(change)


if __name__ == "__main__":
    unittest.main()
