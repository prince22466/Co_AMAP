"""CPU checks for the Kaggle-loadable S37 S1 ZIP (no model inference)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from build_submission36 import payload as s36_payload, ROOT
import build_submission37 as s37


class Submission37PackageTests(unittest.TestCase):
    def test_exact_s36_config_and_single_additive_prompt(self):
        reference = s36_payload("primary")
        new = s37.payload()
        self.assertEqual(set(new), set(ROOT))
        for name in ("agent.yaml", "eval_config.yaml"):
            self.assertEqual(reference[name], new[name])
        note = s37.ADDENDUM.read_bytes()
        self.assertEqual(new["instruction.md"], reference["instruction.md"] + note)
        self.assertNotEqual(new["instruction.md"], reference["instruction.md"])

    def test_deterministic_zip_and_integrity_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1, p2 = Path(tmp) / "a.zip", Path(tmp) / "b.zip"
            m1, m2 = s37.build(p1), s37.build(p2)
            self.assertEqual(p1.read_bytes(), p2.read_bytes())
            self.assertEqual(m1["archive_sha256"], m2["archive_sha256"])
            self.assertEqual(m1["archive_sha256"], hashlib.sha256(p1.read_bytes()).hexdigest())
            self.assertFalse(m1["deterministic_repetition_prevention_in_zip"])
            self.assertEqual(m1["model"], "gemma-4-e4b-it-qat-w4a16-ct")
            with zipfile.ZipFile(p1) as z:
                self.assertEqual(z.namelist(), ["agent.yaml", "eval_config.yaml", "instruction.md"])
                for name, expected in s37.payload().items():
                    self.assertEqual(z.read(name), expected)
            self.assertEqual(
                json.loads(p1.with_suffix(".manifest.json").read_text()),
                m1,
            )

    def test_reject_leaked_benchmark_hints_or_unsupported_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / "note.md"
            base = s37.ADDENDUM.read_bytes()
            for injection in (
                b"rich_3518", b"fastapi_14258", b"before_tool_callback",
                b"run_skill_script",
            ):
                with self.subTest(injection=injection):
                    note.write_bytes(base + injection)
                    with patch.object(s37, "ADDENDUM", note):
                        with self.assertRaises(ValueError):
                            s37.payload()

    def test_reject_wrong_addendum_header_or_missing_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / "note.md"
            for candidate in (
                b"No change at all",
                b"\nS37 FAILURE MEMORY\nOnly the title",
                b"\nS37 FAILURE MEMORY\n" + b"Z" * 2400,
            ):
                with self.subTest(size=len(candidate)):
                    note.write_bytes(candidate)
                    with patch.object(s37, "ADDENDUM", note):
                        with self.assertRaises(ValueError):
                            s37.payload()


if __name__ == "__main__":
    unittest.main()
