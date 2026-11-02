"""CPU unit tests for S32 packaging and guarded shell template.

These tests do not claim Gemma E4B follows the prompt or fixes SWE tasks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile

from build_submission32 import (ARTIFACT_DIR, FILES, SOURCE, VARIANTS,
                                build, build_variant, payload, validate)


def guarded_command(path: Path, line: int, old: str, new: str) -> str:
    prompt = (SOURCE / "instruction.md").read_text(encoding="utf-8")
    match = re.search(r"```bash\n(.*?)\n```", prompt, re.DOTALL)
    if not match:
        raise AssertionError("Guarded inline template missing from prompt")
    code = match.group(1)
    for target, replacement in (
        ('p = Path("PATH_TO_EXISTING_PY_FILE")',
         f'p = Path({path.name!r})'),
        ("n = 123", f"n = {line}"),
        ('old = "EXACT_EXISTING_LINE_WITHOUT_NEWLINE"',
         f"old = {old!r}"),
        ('new = "REPLACEMENT_LINE_WITHOUT_NEWLINE"',
         f"new = {new!r}"),
    ):
        if target not in code:
            raise AssertionError("Missing guard placeholder: " + target)
        code = code.replace(target, replacement, 1)
    return code


def run_guard(path: Path, number: int, old: str, new: str):
    command = guarded_command(path, number, old, new)
    assert command.startswith("python - <<'PY'\n")
    assert command.endswith("\nPY")
    return subprocess.run(command, shell=True, cwd=path.parent,
                          text=True, capture_output=True,
                          executable="/bin/bash", timeout=8)


class Packaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals = {
            variant: (ARTIFACT_DIR / filename).read_bytes()
            for variant, (_, filename) in VARIANTS.items()
        }
        cls.manifests = build()

    def test_both_committed_zips_equal_deterministic_builder(self):
        for variant, (_, filename) in VARIANTS.items():
            with self.subTest(variant=variant):
                raw = (ARTIFACT_DIR / filename).read_bytes()
                self.assertEqual(raw, self.originals[variant])
                self.assertEqual(hashlib.sha256(raw).hexdigest(),
                                 self.manifests[variant]["archive_sha256"])
                self.assertEqual(build_variant(variant)["archive_sha256"],
                                 self.manifests[variant]["archive_sha256"])
                with zipfile.ZipFile(ARTIFACT_DIR / filename) as z:
                    self.assertEqual(set(z.namelist()), set(FILES))
                    self.assertIsNone(z.testzip())
                    for name, data in payload(variant).items():
                        self.assertEqual(z.read(name), data)

    def test_control_is_strictly_instruction_only_ablation(self):
        primary = payload("primary")
        control = payload("control")
        self.assertEqual(primary["agent.yaml"], control["agent.yaml"])
        self.assertEqual(primary["eval_config.yaml"], control["eval_config.yaml"])
        self.assertNotEqual(primary["instruction.md"], control["instruction.md"])
        treatment = primary["instruction.md"].decode()
        comparator = control["instruction.md"].decode()
        self.assertEqual(treatment.split(
            "BOUNDED RECOVERY FOR ONE-LINE PYTHON REPLACEMENTS ONLY:")[0],
            comparator.split(
                "NATIVE-ONLY RECOVERY (this control has NO shell fallback):")[0])
        self.assertEqual(treatment.split(
            "PROVE / SUBMIT (reserve 6+ calls):")[1],
            comparator.split("PROVE / SUBMIT (reserve 6+ calls):")[1])
        config = primary["agent.yaml"].decode()
        self.assertNotIn("write_file", config)
        self.assertNotIn("search_similar_code", config)
        self.assertNotIn("run_skill_script", config)
        self.assertIn("submit_patch", config)

    def test_reject_unexpected_native_tool_or_lora(self):
        files = payload("primary")
        invalid = dict(files)
        invalid["agent.yaml"] = files["agent.yaml"].replace(
            b"  - edit_file\n", b"  - edit_file\n  - write_file\n")
        with self.assertRaises(ValueError):
            validate(invalid, "primary")
        invalid["agent.yaml"] = files["agent.yaml"] + b"adapter: bad\n"
        with self.assertRaises(ValueError):
            validate(invalid, "primary")

    def test_guard_success_single_line_and_no_other_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "example.py"
            p.write_text("def answer():\n    return 1\n\nVALUE = 10\n")
            result = run_guard(p, 2, "    return 1", "    return 2")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("S32_GUARDED_LINE_EDIT_OK", result.stdout)
            self.assertEqual(p.read_text(),
                             "def answer():\n    return 2\n\nVALUE = 10\n")

    def test_guard_rejects_wrong_anchor_line_and_invalid_python(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "example.py"
            original = "def answer():\n    return 1\n"
            for number, old, new in (
                (2, "    return 999", "    return 2"),
                (4, "    return 1", "    return 2"),
                (2, "    return 1", "    return ("),
                (2, "    return 1", "    return 1\nX = 2"),
                (2, "    return 1", "    return 2" * 500),
            ):
                with self.subTest(number=number, new=new[:30]):
                    p.write_text(original)
                    before = p.read_bytes()
                    result = run_guard(p, number, old, new)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(p.read_bytes(), before)

    def test_guard_rejects_new_files_non_py_and_large_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "example.py"
            self.assertNotEqual(run_guard(p, 1, "a = 1", "a = 2").returncode, 0)
            p = Path(folder) / "example.txt"
            p.write_text("a = 1\n")
            self.assertNotEqual(run_guard(p, 1, "a = 1", "a = 2").returncode, 0)
            p = Path(folder) / "example.py"
            p.write_text("VALUE = 1\n" * 30000)
            before = p.read_bytes()
            self.assertNotEqual(run_guard(p, 1, "VALUE = 1", "VALUE = 2").returncode, 0)
            self.assertEqual(p.read_bytes(), before)

    def test_guard_rejects_symlink_and_parent_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.py"
            source.write_text("VALUE = 1\n")
            link = root / "alias.py"
            link.symlink_to(source)
            result = run_guard(link, 1, "VALUE = 1", "VALUE = 2")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_text(), "VALUE = 1\n")
            # This is a pure workspace-relative command: absolute paths and
            # ../ escapes fail before any source file can be written.
            command = guarded_command(source, 1, "VALUE = 1", "VALUE = 2")
            self.assertIn("p = Path('source.py')", command)
            outside = command.replace("p = Path('source.py')",
                                      "p = Path('../outside.py')", 1)
            self.assertNotEqual(outside, command,
                                "Traversal test must actually change the path")
            result = subprocess.run(outside, shell=True, cwd=root,
                                    text=True, capture_output=True,
                                    executable="/bin/bash", timeout=8)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(source.read_text(), "VALUE = 1\\n")

    def test_guard_preserves_crlf_and_missing_final_newline(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "example.py"
            p.write_bytes(b"def answer():\r\n    return 1\r\n")
            result = run_guard(p, 2, "    return 1", "    return 2")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(p.read_bytes(), b"def answer():\r\n    return 2\r\n")
            p.write_bytes(b"VALUE = 1")
            result = run_guard(p, 1, "VALUE = 1", "VALUE = 2")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(p.read_bytes(), b"VALUE = 2")


if __name__ == "__main__":
    unittest.main()
