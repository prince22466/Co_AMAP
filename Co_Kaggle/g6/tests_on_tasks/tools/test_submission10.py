"""Offline regression tests for Submission 10. No Gemma inference or Kaggle grading."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from build_submission10 import BASE, CANDIDATE, build, validate

SCRIPT = CANDIDATE / "skills/repo_navigation/scripts/repair_gate.py"
spec = importlib.util.spec_from_file_location("submission10_repair_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class RepoFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.git("init", "-q")
        self.write("src/example_pkg/__init__.py", "def answer():\n    return 42\n")
        self.git("add", ".")
        self.git("-c", "user.name=Gate Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "baseline")
        self.write("src/example_pkg/__init__.py", "def answer():\n    return 43\n")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args],
                                       stderr=subprocess.STDOUT)

    def write(self, path, content):
        dest = self.root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        return dest

    def run_gate(self, code="from example_pkg import answer\nassert isinstance(answer(), int)", requirements="behavior", covers="behavior", **kw):
        return gate.decide(self.root, requirements, covers, code, **kw)


class GateTests(RepoFixture):
    def test_pass_and_actual_checker(self):
        out = self.run_gate(code="from example_pkg import answer\nassert answer() == 43",
                            module="example_pkg")
        self.assertEqual(out["verification"]["outcome"], "passed")
        self.assertEqual(out["next_action"], "audit_then_submit")
        self.assertEqual(out["covered"], ["behavior"])
        self.assertEqual(out["repairs_requested"], 0)

    def test_partial_coverage_and_edit_invalidation(self):
        first = self.run_gate(requirements="behavior,default", covers="behavior")
        self.assertEqual(first["next_action"], "verify_uncovered")
        self.assertEqual(first["uncovered"], ["default"])
        self.write("src/example_pkg/__init__.py", "def answer():\n    return 44\n")
        after = self.run_gate(requirements="behavior,default", covers="default")
        self.assertEqual(after["uncovered"], ["behavior"])

    def test_failed_diagnostic_is_not_a_repair(self):
        first = self.run_gate(code="assert False")
        self.assertEqual(first["next_action"], "repair_if_code_defect")
        self.assertEqual(first["repairs_requested"], 0)
        second = self.run_gate(code="assert 2 == 3")
        self.assertEqual(second["next_action"], "repair_if_code_defect")
        self.assertEqual(second["repairs_requested"], 0)
        self.write("src/example_pkg/__init__.py", "def answer():\n    return 45\n")
        third = self.run_gate(code="assert False")
        self.assertEqual(third["repairs_requested"], 1)

    def test_identical_failed_check_is_not_repeated(self):
        self.run_gate(code="assert False")
        out = self.run_gate(code="assert False")
        self.assertEqual(out["next_action"], "audit_then_submit")

    def test_max_two_actual_repairs(self):
        for value in (44, 45):
            self.run_gate(code="assert False")
            self.write("src/example_pkg/__init__.py",
                       "def answer():\n    return %d\n" % value)
        out = self.run_gate(code="assert False")
        self.assertEqual(out["repairs_requested"], 2)
        self.assertEqual(out["next_action"], "audit_then_submit")

    def test_no_edit_after_failure_then_pass_clears_pending(self):
        self.run_gate(code="assert False")
        out = self.run_gate()
        self.assertEqual(out["repairs_requested"], 0)
        self.assertFalse(out["pending_repair"])

    def test_budget_reservation(self):
        out = self.run_gate(time_left=44, calls_left=40)
        self.assertEqual(out["next_action"], "audit_then_submit")
        self.assertIsNone(out["verification"])
        out = self.run_gate(time_left=120, calls_left=5)
        self.assertEqual(out["next_action"], "audit_then_submit")

    def test_vacuous_assertion_does_not_claim_acceptance(self):
        result = self.run_gate(code="assert True")
        self.assertEqual(result["next_action"], "alternative_check")
        self.assertEqual(result["covered"], [])
        self.assertEqual(result["uncovered"], ["behavior"])

    def test_inconclusive_allows_one_alternative(self):
        first = self.run_gate(code="pass")
        self.assertEqual(first["next_action"], "alternative_check")
        second = self.run_gate(code="pass")
        self.assertEqual(second["next_action"], "audit_then_submit")

    def test_immutable_and_validated_requirements(self):
        self.run_gate(requirements="behavior,default", covers="behavior")
        with self.assertRaises(ValueError):
            self.run_gate(requirements="different", covers="different")
        with self.assertRaises(ValueError):
            gate.identifiers("UPPERCASE")
        with self.assertRaises(ValueError):
            gate.identifiers("a,a")

    def test_state_write_is_atomic_and_outside_patch(self):
        self.run_gate()
        self.assertTrue(gate.state_file(self.root).exists())
        self.assertFalse(gate.state_file(self.root).with_name(
            "submission10_repair_state.json.tmp").exists())
        self.assertNotIn("submission10_repair_state.json",
                         self.git("status", "--porcelain").decode())

    def test_fingerprint_size_is_bounded(self):
        with patch.object(gate, "MAX_HASH_BYTES", 1):
            with self.assertRaises(ValueError):
                gate.fingerprint(self.root)

    def test_staged_only_change_changes_fingerprint(self):
        before = gate.fingerprint(self.root)[0]
        self.git("add", "src/example_pkg/__init__.py")
        self.write("src/example_pkg/__init__.py", "def answer():\n    return 42\n")
        after = gate.fingerprint(self.root)[0]
        self.assertNotEqual(before, after)

    def test_git_rename_changes_fingerprint(self):
        before = gate.fingerprint(self.root)[0]
        self.git("mv", "src/example_pkg/__init__.py", "src/example_pkg/renamed.py")
        after = gate.fingerprint(self.root)[0]
        self.assertNotEqual(before, after)

    def test_cli_accepts_adk_underscore_budget_fields(self):
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "--workspace", str(self.root),
             "--requirements", "behavior", "--covers", "behavior",
             "--code", "from example_pkg import answer\nassert answer() == 43", "--time_left", "180", "--calls_left", "30"],
            capture_output=True, text=True, check=True, timeout=20)
        result = json.loads(done.stdout)
        self.assertEqual(result["next_action"], "audit_then_submit")
        self.assertTrue(result["budget_sufficient"])

    def test_untracked_change_invalidates_previous_checks(self):
        self.run_gate(requirements="behavior,default", covers="behavior")
        self.write("new_implementation.py", "x = 1\n")
        out = self.run_gate(requirements="behavior,default", covers="default")
        self.assertEqual(out["uncovered"], ["behavior"])

    def test_worktree_git_metadata_can_be_outside_workspace(self):
        self.git("add", ".")
        self.git("-c", "user.name=Gate Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "second")
        other = Path(self.temp.name) / "other"
        self.git("worktree", "add", "-q", "--detach", str(other))
        self.assertFalse(gate.state_file(other).is_relative_to(other))
        out = gate.decide(other, "behavior", "behavior", "from example_pkg import answer\nassert answer() == 43")
        self.assertEqual(out["next_action"], "audit_then_submit")


class PackagingTests(unittest.TestCase):
    def test_static_structure_and_baseline(self):
        files, preserved = validate()
        self.assertGreater(len(files), 10)
        self.assertGreater(len(preserved), 8)

    def test_repeatable_archive_and_skill_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.zip", Path(directory) / "b.zip"
            build(a)
            build(b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with zipfile.ZipFile(a) as z:
                names = set(z.namelist())
                self.assertIn("agent.yaml", names)
                self.assertIn("skills/repo-navigation/scripts/repair_gate.py", names)
                self.assertIn("skills/repo-navigation/references/contracts.md", names)
                self.assertNotIn("README.md", names)
                self.assertNotIn("skills/repo_navigation", z.read("agent.yaml").decode())
                self.assertFalse(any("__pycache__" in name for name in names))
                self.assertLess(sum(x.file_size for x in z.infolist()), 3 * 1024**3)

    def test_builder_rejects_extra_active_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "submission_10"
            shutil.copytree(CANDIDATE, root)
            extra = root / "skills/repo_navigation/scripts/extra.py"
            extra.write_text("pass\n")
            with self.assertRaises(ValueError):
                validate(root=root, baseline=BASE)

    def test_builder_rejects_modified_baseline_checker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "submission_10"
            shutil.copytree(CANDIDATE, root)
            checker = root / "skills/repo_navigation/scripts/check.py"
            checker.write_text(checker.read_text() + "\n# unexpected edit\n")
            with self.assertRaises(ValueError):
                validate(root=root, baseline=BASE)

    def test_self_contained_gate_for_runpy_executor(self):
        text = SCRIPT.read_text()
        self.assertNotIn("importlib.util", text)
        self.assertIn("def check(", text)
        self.assertIn("def decide(", text)


if __name__ == "__main__":
    unittest.main()
