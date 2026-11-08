"""S39 stack regression tests: S37 mental memory + exact S38 ADK skill.

This is CPU/package verification, not proof that Gemma E4B uses the skill.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import build_submission38 as s38
import build_submission39 as s39
from build_submission36 import payload as s36_payload


class Submission39StackTests(unittest.TestCase):
    def test_source_s37_is_retained_except_two_documented_reconciliations(self):
        memo = s39.reconciled_memory().decode("utf-8")
        self.assertEqual(memo.count(s39.HEADER_S37_AFTER), 1)
        self.assertEqual(memo.count(s39.LEDGER_AFTER), 1)
        original = memo.replace(
            s39.HEADER_S37_AFTER, s39.HEADER_S37_BEFORE
        ).replace(
            s39.LEDGER_AFTER, s39.LEDGER_BEFORE
        )
        self.assertEqual(
            s39.git_blob_sha(original.encode("utf-8")),
            s39.S37_ORIGINAL_BLOB
        )
        self.assertIn("old_string not found", memo)
        self.assertIn("After two failed edits", memo)

    def test_s39_exactly_stacks_s37_in_front_of_s38_recovery(self):
        base36 = s36_payload("primary")
        base38 = s38.payload("primary")
        primary = s39.payload("primary")
        control = s39.payload("control")

        self.assertEqual(set(primary), set(base38))
        self.assertEqual(control, base38)
        for name in base38:
            if name != "instruction.md":
                self.assertEqual(primary[name], base38[name])
        expected_instruction = (
            base36["instruction.md"]
            + s39.reconciled_memory()
            + s38.read_source(s38.NOTE)
        )
        self.assertEqual(primary["instruction.md"], expected_instruction)
        self.assertEqual(primary["agent.yaml"], base38["agent.yaml"])
        self.assertEqual(
            primary["eval_config.yaml"], base36["eval_config.yaml"]
        )
        self.assertIn(b"load_skill(skill_name=", primary["instruction.md"])
        self.assertIn(b"state-guard", primary["instruction.md"])

    def test_s38_skill_source_remains_pinned(self):
        parts = s39.payload()
        self.assertEqual(s39.git_blob_sha(
            parts["skills/state-guard/SKILL.md"]), s39.S38_SKILL_BLOB)
        self.assertEqual(s39.git_blob_sha(
            parts["skills/state-guard/scripts/state_guard.py"]),
            s39.S38_SCRIPT_BLOB)

    def test_reject_undocumented_mutation_of_s37_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "modified.md"
            original = s39.reconciled_memory()
            path.write_bytes(original.replace(
                b"Use the most recent tool RESULT",
                b"Ignore previous tool results",
            ))
            with patch.object(s39, "RECONCILED_S37", path):
                with self.assertRaisesRegex(
                    ValueError, "Unintended change"
                ):
                    s39.payload("primary")

    def test_build_reproducible_zips_and_manifests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for variant in ("primary", "control"):
                with self.subTest(variant=variant):
                    output = root / (variant + ".zip")
                    other = root / (variant + "_repeat.zip")
                    manifest = s39.build(variant, output)
                    s39.build(variant, other)
                    self.assertEqual(output.read_bytes(), other.read_bytes())
                    self.assertEqual(
                        manifest["archive_sha256"],
                        hashlib.sha256(output.read_bytes()).hexdigest()
                    )
                    self.assertFalse(manifest["native_tool_interception"])
                    self.assertEqual(
                        manifest["s37_compatibility_edits"],
                        2 if variant == "primary" else 0,
                    )
                    with zipfile.ZipFile(output) as archive:
                        self.assertEqual(
                            archive.namelist(), sorted(s38.FILES)
                        )
                        for name, expected in s39.payload(variant).items():
                            self.assertEqual(archive.read(name), expected)
                    self.assertEqual(
                        json.loads(output.with_suffix(".manifest.json").read_text()),
                        manifest,
                    )

    def test_bad_variant_fails_closed(self):
        with self.assertRaises(ValueError):
            s39.payload("unknown")
        with self.assertRaises(ValueError):
            s39.path_for("unknown")
        with self.assertRaises(ValueError):
            s39.build("unknown")


if __name__ == "__main__":
    unittest.main()
