"""S28 packaging and comparison regressions; not real model benchmarks."""
import hashlib
import unittest
import zipfile

import yaml
from build_submission28 import ARCHIVE, FILES, IncludeLoader, SOURCE, TOOLS, build
from compare_submission28 import compare, objective_pilot_pass
from analyze_submission27_traces import analyze


class Submission28Tests(unittest.TestCase):
    """Keep the small E4B package reproducible and comparison claims scoped."""

    @classmethod
    def setUpClass(cls):
        """Assemble the package under test once."""
        cls.manifest = build()

    def test_package_deterministic_and_complete(self):
        """Rebuild produces byte-identical ZIP with only three approved files."""
        initial = ARCHIVE.read_bytes()
        self.assertEqual(build()["archive_sha256"],
                         hashlib.sha256(initial).hexdigest())
        self.assertEqual(ARCHIVE.read_bytes(), initial)
        with zipfile.ZipFile(ARCHIVE) as archive:
            self.assertEqual(sorted(archive.namelist()), sorted(FILES))
            self.assertIsNone(archive.testzip())
            for name in FILES:
                self.assertEqual(archive.read(name), (SOURCE / name).read_bytes())

    def test_tool_schema_and_model_unchanged(self):
        """S28 uses the proven native tool interface and same model settings."""
        with zipfile.ZipFile(ARCHIVE) as archive:
            cfg = yaml.load(archive.read("agent.yaml"), Loader=IncludeLoader)
        self.assertEqual(tuple(cfg["tools"]), TOOLS)
        self.assertEqual(cfg["model"], "gemma-4-e4b-it-qat-w4a16-ct")
        self.assertNotIn("skills", cfg)
        self.assertNotIn("sub_agents", cfg)
        baseline = SOURCE.parent / "submission_27"
        for file in ("agent.yaml", "eval_config.yaml"):
            self.assertEqual((SOURCE / file).read_bytes(),
                             (baseline / file).read_bytes())

    def test_concrete_source_navigation_and_release(self):
        """Check instruction emphasizes bounded source access and real edits."""
        text = (SOURCE / "instruction.md").read_text()
        for required in ("rg -n", "sed -n", "FIRST 150 lines",
                         "edit_file", "old_string", "git diff --check",
                         "submit_patch", "BLOCKED", "benchmark/grading"):
            with self.subTest(required=required):
                self.assertIn(required, text)
        self.assertLess(len(text), 3700)

    def test_task_comparison_counts_gains_and_regressions(self):
        """Never hide lost tasks behind a total solved count."""
        baseline = analyze(
            "[candidate 1/2] Evaluating a (repo)\n"
            "  [ 1.0s] 📖 read_file (a.py)\n"
            "  -> resolved=True, patch_chars=100, tools=10\n"
            "[candidate 2/2] Evaluating b (repo)\n"
            "  -> resolved=False, patch_chars=0, tools=40\n")
        candidate = analyze(
            "[candidate 1/2] Evaluating a (repo)\n"
            "  -> resolved=False, patch_chars=0, tools=40\n"
            "[candidate 2/2] Evaluating b (repo)\n"
            "  [ 1.0s] ✏️ edit_file (b.py)\n"
            "  [ 1.1s]   ✓ ok\n"
            "  -> resolved=True, patch_chars=75, tools=3\n")
        result = compare(baseline, candidate)
        self.assertEqual(result["matched_task_count"], 2)
        self.assertEqual(result["gained_task_ids"], ["b"])
        self.assertEqual(result["lost_task_ids"], ["a"])
        self.assertEqual(result["candidate_empty_on_matched"], 1)
        self.assertEqual(result["baseline_exhausted_on_matched"], 1)
        self.assertEqual(result["actions"]["reads"]["baseline"], 1)

    def test_pilot_gate_requires_both_retained_wins_plus_one_new(self):
        """Do not accidentally inherit S25's different regression gate."""
        good = {"resolved": 3,
                "solved_ids": ["rich_3905", "rich_3454", "rich_3882"],
                "empty_patches": 1, "agent_error_ids": []}
        self.assertTrue(objective_pilot_pass(good))
        self.assertFalse(objective_pilot_pass(
            {**good, "solved_ids": ["rich_3905", "rich_3882", "rich_4077"]}))
        self.assertFalse(objective_pilot_pass({**good, "resolved": 2}))
        self.assertFalse(objective_pilot_pass({**good, "empty_patches": 2}))
        self.assertFalse(objective_pilot_pass(
            {**good, "agent_error_ids": ["rich_3518"]}))

    def test_unmatched_tasks_are_reported(self):
        """A truncated candidate trace must not appear to be a full run."""
        before = analyze("[candidate 1/1] Evaluating a (repo)\n"
                         "  -> resolved=True, patch_chars=50, tools=3\n")
        after = analyze("[candidate 1/1] Evaluating b (repo)\n"
                        "  -> resolved=True, patch_chars=50, tools=3\n")
        report = compare(before, after)
        self.assertEqual(report["matched_task_count"], 0)
        self.assertEqual(report["missing_from_candidate"], ["a"])
        self.assertEqual(report["missing_from_baseline"], ["b"])


if __name__ == "__main__":
    unittest.main()
