"""CPU-only behavior tests for the packaged skill, using synthetic repositories."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from build_submission9 import CANDIDATE, build, validate


SCRIPTS = CANDIDATE / "skills/repo_navigation/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location("skill_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inspect = load("inspect_source").inspect
check = load("check").check
audit = load("audit_patch").audit


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], stderr=subprocess.STDOUT)

    def init_git(self):
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Skill test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "synthetic baseline")


class InspectionTests(Fixture):
    def test_qualified_symbol_and_no_source_execution(self):
        self.write("module.py", "raise RuntimeError('must not execute')\nclass API:\n    def run(self):\n        return 1\n")
        result = inspect(self.root, "module.py", "API.run")
        self.assertEqual(result["start_line"], 3)
        self.assertIn("return 1", result["source"])

    def test_ambiguous_short_symbol(self):
        self.write("module.py", "class A:\n def run(self): pass\nclass B:\n def run(self): pass\n")
        self.assertEqual(len(inspect(self.root, "module.py", "run")["matches"]), 2)

    def test_bounded_pagination(self):
        self.write("module.py", "def f():\n" + "    x = '" + "x" * 300 + "'\n" + "    pass\n" * 200)
        first = inspect(self.root, "module.py", "f", lines=30)
        second = inspect(self.root, "module.py", "f", start=first["next_line"], lines=30)
        self.assertEqual(second["start_line"], first["end_line"] + 1)
        self.assertLess(len(json.dumps(first)), 4800)

    def test_path_escape_and_symlink(self):
        self.write("module.py", "x = 1\n")
        with self.assertRaises(ValueError):
            inspect(self.root, "../secret.py")
        outside = Path(self.temp.name) / "outside.py"
        outside.write_text("x = 2\n")
        (self.root / "link.py").symlink_to(outside)
        with self.assertRaises(ValueError):
            inspect(self.root, "link.py")

    def test_explicit_workspace_with_changed_cwd(self):
        self.write("module.py", "answer = 42\n")
        output = subprocess.check_output([sys.executable, "-B", str(SCRIPTS / "inspect_source.py"),
                                          "--workspace", str(self.root), "--path", "module.py"],
                                         cwd=self.temp.name)
        self.assertIn("42", json.loads(output)["source"])


class CheckTests(Fixture):
    def setUp(self):
        super().setUp()
        self.write("src/example_pkg/__init__.py", "value = 42\n")

    def test_src_import_and_no_workspace_artifacts(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        result = check(self.root, "import example_pkg\nassert example_pkg.value == 42", "example_pkg")
        self.assertEqual(result["outcome"], "passed")
        self.assertIn("src/example_pkg", result["module_origin"])
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))

    def test_rejects_external_import_origin(self):
        self.assertEqual(check(self.root, "assert True", "json")["outcome"], "import_origin_error")

    def test_assertion_failure_is_not_outer_tool_success(self):
        self.assertEqual(check(self.root, "assert False", "example_pkg")["outcome"], "failed")

    def test_print_only_is_not_behavioral_pass(self):
        self.assertEqual(check(self.root, "print('looks fine')", "example_pkg")["outcome"], "exit_zero_without_assertions")

    def test_early_exit_is_incomplete(self):
        self.assertEqual(check(self.root, "import sys\nsys.exit(0)\nassert False", "example_pkg")["outcome"], "incomplete")

    def test_runtime_missing_import_is_detected(self):
        self.write("src/example_pkg/__init__.py", "def run():\n return MovedSymbol()\n")
        result = check(self.root, "import example_pkg\nassert example_pkg.run()", "example_pkg")
        self.assertEqual(result["outcome"], "failed")
        self.assertIn("NameError", result["stderr_tail"])

    def test_timeout(self):
        result = check(self.root, "while True: pass", "example_pkg", timeout=1)
        self.assertEqual(result["outcome"], "timeout")

    def test_output_stays_valid_and_bounded(self):
        result = check(self.root, "print('\\\\\\n' * 10000)\nassert True", "example_pkg")
        self.assertLess(len(json.dumps(result, ensure_ascii=False)), 4500)
        self.assertTrue(result["output_truncated"])


class AuditTests(Fixture):
    def setUp(self):
        super().setUp()
        self.write("module.py", "value = 1\n")
        self.write("tests/test_module.py", "assert True\n")
        self.write("pyproject.toml", "[project]\nname = 'fixture'\n")
        self.init_git()

    def test_detects_protected_changes_and_scratch(self):
        self.write("tests/test_module.py", "assert 1\n")
        self.write("pyproject.toml", "[project]\nname = 'changed'\n")
        self.write("repro.py", "print('scratch')\n")
        result = audit(self.root)
        self.assertEqual(result["protected_paths"], ["pyproject.toml", "tests/test_module.py"])
        self.assertEqual(result["possible_scratch_paths"], ["repro.py"])

    def test_detects_syntax_and_whitespace_errors(self):
        self.write("module.py", "def broken(:   \n")
        result = audit(self.root)
        self.assertTrue(result["syntax_errors"])
        self.assertNotEqual(result["diff_check_returncode"], 0)

    def test_audit_does_not_stage_or_modify(self):
        self.write("new_module.py", "value = 2\n")
        before = self.git("status", "--porcelain=v1")
        result = audit(self.root)
        self.assertFalse(result["empty_patch"])
        self.assertEqual(before, self.git("status", "--porcelain=v1"))
        self.assertEqual(self.git("diff", "--cached"), b"")

    def test_empty_patch_is_reported(self):
        self.assertTrue(audit(self.root)["empty_patch"])

    def test_large_change_list_is_bounded(self):
        for i in range(120):
            self.write(f"new_{i}.py", "pass\n")
        result = audit(self.root)
        self.assertTrue(result["truncated"])
        self.assertLess(len(json.dumps(result)), 4500)


class PackageTests(unittest.TestCase):
    def test_configuration_and_baseline_preservation(self):
        _, preserved = validate()
        self.assertEqual(len(preserved), 8)

    def test_repeatable_zip_and_only_active_files(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.zip", Path(directory) / "b.zip"
            build(a)
            build(b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with zipfile.ZipFile(a) as z:
                self.assertIn("agent.yaml", z.namelist())
                self.assertIn("skills/repo_navigation/SKILL.md", z.namelist())
                self.assertFalse(any("diagnostics" in p or p.endswith("README.md") for p in z.namelist()))


if __name__ == "__main__":
    unittest.main()
