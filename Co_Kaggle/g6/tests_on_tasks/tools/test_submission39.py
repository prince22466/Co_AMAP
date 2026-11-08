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
        self.assertIn("the two-call recovery limit covers only run_skill_script", memo)
        self.assertIn("not load_skill itself", memo)

    def test_s39_exactly_stacks_s37_in_front_of_s38_recovery(self):
        base36 = s36_payload("primary")
        base38 = s38.payload("primary")
        primary = s39.payload("primary")
        control = s39.payload("control")
        skill_control = s39.payload("skill_enabled_control")

        self.assertEqual(set(primary), set(base38))
        self.assertEqual(control, base38)
        self.assertEqual(
            s39.git_blob_sha(s38.read_source(s38.NOTE)),
            s39.S38_ACTIVATION_BLOB,
        )
        for name in base38:
            if name != "instruction.md":
                self.assertEqual(primary[name], base38[name])
        allowed_s36 = s39.allow_registered_skill(base36["instruction.md"])
        expected_instruction = (
            allowed_s36
            + s39.reconciled_memory()
            + s38.read_source(s38.NOTE)
        )
        self.assertEqual(primary["instruction.md"], expected_instruction)
        self.assertEqual(
            skill_control["instruction.md"],
            allowed_s36 + s38.read_source(s38.NOTE)
        )
        for name in base38:
            if name != "instruction.md":
                self.assertEqual(skill_control[name], base38[name])
        # The primary-vs-skill-enabled-control difference is JUST S37.
        self.assertEqual(
            primary["instruction.md"],
            skill_control["instruction.md"].replace(
                s38.read_source(s38.NOTE),
                s39.reconciled_memory() + s38.read_source(s38.NOTE),
                1,
            )
        )
        self.assertIn(s39.S36_SKILL_BAN, control["instruction.md"])
        self.assertNotIn(s39.S36_SKILL_BAN, primary["instruction.md"])
        self.assertIn(b"state-guard skill", primary["instruction.md"])
        self.assertNotIn(b"Never request a skill", primary["instruction.md"])
        self.assertNotIn(s39.S36_SCRIPT_BAN, primary["instruction.md"])
        self.assertEqual(primary["agent.yaml"], base38["agent.yaml"])
        self.assertEqual(
            primary["eval_config.yaml"], base36["eval_config.yaml"]
        )
        self.assertIn(b"load_skill(skill_name=", primary["instruction.md"])
        self.assertIn(b"state-guard", primary["instruction.md"])

    def test_only_registered_skill_is_allowlisted(self):
        original = s36_payload("primary")["instruction.md"]
        allowed = s39.allow_registered_skill(original)
        self.assertEqual(original.count(s39.S36_SKILL_BAN), 1)
        self.assertEqual(original.count(s39.S36_SCRIPT_BAN), 1)
        restored = allowed.replace(s39.S39_SKILL_ALLOW, s39.S36_SKILL_BAN)
        restored = restored.replace(s39.S39_SCRIPT_ALLOW, s39.S36_SCRIPT_BAN)
        self.assertEqual(restored, original)
        self.assertIn(b"Do not request unregistered skills", allowed)
        self.assertIn(b"bounded exact edit", allowed)
        with self.assertRaisesRegex(ValueError, "skill exclusion changed"):
            s39.allow_registered_skill(
                original.replace(s39.S36_SKILL_BAN, b"other instruction")
            )

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

    def test_changed_s38_activation_instructions_are_rejected(self):
        # S38's own validator checks structure and consistency but doesn't
        # pin the exact activation text. S39's A/B comparison must pin it.
        read_original = s38.read_source

        def mutated_read(name):
            data = read_original(name)
            return data + b" " if name == s38.NOTE else data

        with patch.object(s38, "read_source", side_effect=mutated_read):
            for variant in ("primary", "control", "skill_enabled_control"):
                with self.subTest(variant=variant):
                    with self.assertRaisesRegex(
                        ValueError, "activation instruction drifted"
                    ):
                        s39.payload(variant)

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
                    self.assertEqual(
                        manifest["s36_skill_exception_spans"],
                        0 if variant == "control" else 2,
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
