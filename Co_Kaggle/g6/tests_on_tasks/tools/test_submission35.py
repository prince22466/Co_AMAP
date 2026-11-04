"""S35 CPU regression tests: packaging/constraints, not Gemma resolution."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

import yaml

from build_submission35 import (
    ARTIFACT_DIR, MODEL, ROOT_FILES, SKILL_FILE, TOOLS,
    VARIANTS, archive_path, build_payload, build_variant,
    validate,
)


class Submission35PackageTests(unittest.TestCase):
    def test_primary_and_control_only_differ_in_skill_surface(self) -> None:
        primary = build_payload("primary")
        control = build_payload("control")
        self.assertEqual(set(primary) - set(control), {SKILL_FILE})
        for file in ("instruction.md", "eval_config.yaml"):
            self.assertEqual(primary[file], control[file])
        a, b = yaml.safe_load(primary["agent.yaml"].replace(
            b"!include", b"")), yaml.safe_load(control["agent.yaml"].replace(
            b"!include", b""))
        self.assertEqual(a.pop("skills"), ["skills/tool-recovery"])
        self.assertEqual(a, b)
        self.assertEqual(a["model"], MODEL)
        self.assertEqual(tuple(a["tools"]), TOOLS)
        self.assertNotIn("write_file", a["tools"])
        self.assertNotIn("agent_tool", str(a))
        self.assertNotIn("sub_agents", a)

    def test_all_archives_are_deterministic_and_committed(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                a, b = Path(temp) / "a.zip", Path(temp) / "b.zip"
                manifest = build_variant(variant, output=a)
                replay = build_variant(variant, output=b)
                self.assertEqual(manifest, replay)
                self.assertEqual(a.read_bytes(), b.read_bytes())
                self.assertEqual(manifest["sha256"],
                                 hashlib.sha256(a.read_bytes()).hexdigest())
                expected = archive_path(variant)
                self.assertTrue(expected.is_file(), "Commit S35 ZIP: " + str(expected))
                self.assertEqual(a.read_bytes(), expected.read_bytes())
                with zipfile.ZipFile(a) as z:
                    self.assertIsNone(z.testzip())
                    self.assertEqual(set(z.namelist()), set(build_payload(variant)))
                    for name, content in build_payload(variant).items():
                        self.assertEqual(z.read(name), content)
                    self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED
                                        for i in z.infolist()))

    def test_rejects_unsafe_or_unexpected_agent_config(self) -> None:
        originals = build_payload("primary")
        edits = (
            {"agent.yaml": originals["agent.yaml"].replace(
                b"  - edit_file\n", b"  - write_file\n")},
            {"agent.yaml": originals["agent.yaml"].replace(
                b"gemma-4-31b-it-qat-w4a16-ct",
                b"gemma-4-e4b-it-qat-w4a16-ct")},
            {"agent.yaml": originals["agent.yaml"].replace(
                b"skills/tool-recovery", b"skills/random-skill")},
            {"eval_config.yaml": originals["eval_config.yaml"].replace(
                b"max_time_minutes: 4", b"max_time_minutes: 40")},
            {"instruction.md": originals["instruction.md"] + b"rich_3518"},
            {SKILL_FILE: originals[SKILL_FILE].replace(
                b"name: tool-recovery", b"name: other")},
        )
        for mutated in edits:
            with self.subTest(names=tuple(mutated)), self.assertRaises(ValueError):
                validate("primary", {**originals, **mutated})

    def test_rejects_extra_files_or_skill_in_control(self) -> None:
        primary = build_payload("primary")
        control = build_payload("control")
        with self.assertRaises(ValueError):
            validate("primary", {**primary, "agent.py": b"print(1)"})
        with self.assertRaises(ValueError):
            validate("control", {**control, SKILL_FILE: primary[SKILL_FILE]})
        with self.assertRaises(ValueError):
            validate("primary", {k: v for k, v in primary.items()
                                  if k != SKILL_FILE})
        with self.assertRaises(ValueError):
            validate("unknown", control)


if __name__ == "__main__":
    unittest.main()
