"""S36 CPU-only contracts; these do NOT score real Gemma 4 performance."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

from build_submission36 import (
    ARCHIVES, BASE, MODEL, ROOT, SOURCE, TOOLS, archive_path,
    build, payload, validate,
)


class Submission36Tests(unittest.TestCase):
    def test_true_single_variable_ablation(self) -> None:
        control = payload("control")
        primary = payload("primary")
        self.assertEqual(set(control), set(ROOT))
        self.assertEqual(set(primary), set(ROOT))
        for file in ("agent.yaml", "eval_config.yaml"):
            self.assertEqual(control[file], primary[file])
            self.assertEqual(control[file], (BASE / file).read_bytes())
        baseline = (BASE / "instruction.md").read_bytes()
        note = (SOURCE / "parser_aware_addendum.md").read_bytes()
        self.assertEqual(control["instruction.md"], baseline)
        self.assertEqual(primary["instruction.md"], baseline + note)
        self.assertEqual(MODEL, "gemma-4-e4b-it-qat-w4a16-ct")
        self.assertEqual(len(TOOLS), 6)

    def test_example_uses_delimiters_for_all_strings(self) -> None:
        note = (SOURCE / "parser_aware_addendum.md").read_text()
        # Validate the *authored example*, not Gemma generations or the live
        # vLLM parser. A real server/tool trace is necessary for that.
        expected = (
            'filepath:<|"|>pkg/module.py<|"|>,'
            'old_string:<|"|>x = values[0]<|"|>,'
            'new_string:<|"|>x = values[1]<|"|>,allow_multiple:false'
        )
        self.assertIn(expected, note)
        self.assertEqual(note.count("filepath:<|"), 1)
        self.assertEqual(note.count("old_string:<|"), 1)
        self.assertEqual(note.count("new_string:<|"), 1)
        self.assertNotIn("load_skill(", note)
        self.assertNotIn("run_skill_script(", note)
        self.assertNotIn("<|tool_call>", note)

    def test_reject_modified_model_tools_budget_and_prompt(self) -> None:
        pristine = payload("primary")
        bad = (
            {"agent.yaml": pristine["agent.yaml"].replace(
                b"gemma-4-e4b-it-qat-w4a16-ct",
                b"gemma-4-31b-it-qat-w4a16-ct")},
            {"agent.yaml": pristine["agent.yaml"].replace(
                b"  - edit_file\n", b"  - get_code_subgraph\n")},
            {"eval_config.yaml": pristine["eval_config.yaml"].replace(
                b"max_time_minutes: 4", b"max_time_minutes: 40")},
            {"instruction.md": pristine["instruction.md"] + b"extra instruction"},
            {"instruction.md": (BASE / "instruction.md").read_bytes()},
            {"unknown.py": b"print('extra')"},
        )
        for mutation in bad:
            with self.subTest(fields=tuple(mutation)), self.assertRaises(ValueError):
                validate("primary", {**pristine, **mutation})
        with self.assertRaises(ValueError):
            payload("typo")

    def test_build_deterministically_and_validate_zip_contents(self) -> None:
        for variant in ARCHIVES:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                a, b = Path(tmp) / "a.zip", Path(tmp) / "b.zip"
                first = build(variant, a)
                second = build(variant, b)
                self.assertEqual(first, second)
                self.assertEqual(a.read_bytes(), b.read_bytes())
                self.assertEqual(first["archive_sha256"],
                                 hashlib.sha256(a.read_bytes()).hexdigest())
                self.assertEqual(first["variant"], variant)
                with zipfile.ZipFile(a) as z:
                    self.assertIsNone(z.testzip())
                    self.assertEqual(set(z.namelist()), set(ROOT))
                    self.assertEqual({name: z.read(name) for name in z.namelist()},
                                     payload(variant))
                    self.assertTrue(all(x.compress_type == zipfile.ZIP_STORED
                                        for x in z.infolist()))
                self.assertEqual(archive_path(variant).name, ARCHIVES[variant])


if __name__ == "__main__":
    unittest.main()
