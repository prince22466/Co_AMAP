"""CPU-level S34 controls; these DO NOT establish Gemma effectiveness."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from build_submission34 import (ARTIFACT, BASELINE, FILES, SOURCE,
                                build, payload, validate)


class Submission34Contract(unittest.TestCase):
    def test_exact_config_parity_and_prompt_only_ablation(self):
        for name in ("agent.yaml", "eval_config.yaml"):
            self.assertEqual((SOURCE / name).read_bytes(),
                             (BASELINE / name).read_bytes())
        self.assertNotEqual((SOURCE / "instruction.md").read_bytes(),
                            (BASELINE / "instruction.md").read_bytes())
        self.assertLessEqual(
            len((SOURCE / "instruction.md").read_text()),
            0.65 * len((BASELINE / "instruction.md").read_text()))
        self.assertFalse((SOURCE / "skills").exists())
        self.assertFalse((SOURCE / "sub_agents").exists())
        self.assertFalse((SOURCE / "adapters").exists())

    def test_reproducible_archive_matches_committed_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = (Path(tmp) / filename
                        for filename in ("one.zip", "two.zip"))
            first, second = build(one), build(two)
            self.assertEqual(first, second)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            self.assertTrue(ARTIFACT.is_file(), "Commit the reviewable ZIP")
            self.assertEqual(one.read_bytes(), ARTIFACT.read_bytes())
            self.assertEqual(first["archive_sha256"],
                             hashlib.sha256(one.read_bytes()).hexdigest())
            with zipfile.ZipFile(one) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), set(FILES))
                for name, source in payload().items():
                    self.assertEqual(archive.read(name), source)

    def test_rejects_tool_or_budget_changes(self):
        originals = payload()
        changes = (
            {"agent.yaml": originals["agent.yaml"].replace(
                b"  - edit_file\n", b"  - get_code_neighbors\n")},
            {"agent.yaml": originals["agent.yaml"].replace(
                b"  - write_file\n", b"")},
            {"eval_config.yaml": originals["eval_config.yaml"].replace(
                b"max_tool_calls: 40", b"max_tool_calls: 80")},
            {"instruction.md": originals["instruction.md"] + b"rich_3518"},
            {"instruction.md": b"small but missing contract"},
        )
        for update in changes:
            with self.subTest(update=tuple(update)), self.assertRaises(
                    ValueError):
                validate({**originals, **update})

    def test_rejects_extra_or_missing_archive_content(self):
        files = payload()
        for variant in ({k: v for k, v in files.items()
                         if k != "instruction.md"},
                        {**files, "unsafe.py": b"print(1)"}):
            with self.assertRaises(ValueError):
                validate(variant)


if __name__ == "__main__":
    unittest.main()
