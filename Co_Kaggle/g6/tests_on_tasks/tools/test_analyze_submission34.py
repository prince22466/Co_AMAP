"""Synthetic tests for S34 scored-log comparison (no model claims)."""
from __future__ import annotations

import unittest

from analyze_submission34 import compare, parse
from compare_submission24 import S1_IDS


def fake(wins: set[str], *, invalid_first: bool = False,
         truncated: bool = False, duplicate: bool = False) -> str:
    rows = []
    for index, name in enumerate(S1_IDS, 1):
        if truncated and index == len(S1_IDS):
            break
        rows += [
            f"[candidate {index}/15] Evaluating {name} (repo)",
            "  [ 1.0s] 📖 read_file (code.py)",
            "  [ 2.0s] ✏️ edit_file (code.py)",
            ("  [ 2.1s]   ❌ {'error': 'mandatory input parameters'}"
             if invalid_first and index == 1 else
             "  [ 2.1s]   ✓ {'status': 'ok', 'filepath': 'code.py'}"),
            "  [ 2.3s] 🔧 run_command pytest -q tests/test_code.py",
            "  [ 2.6s] 🔧 run_command git diff --check",
            "  [ 3.0s] 🤖 submit_patch",
            (f"  -> resolved={name in wins}, patch_chars=500, tools=5, "
             "duration=10.0, input=100, output=20, stop=not recorded"),
        ]
    rows.append("idx\tinstance_id\tresolved\tpatch_chars\tseconds\tinput")
    for index, name in enumerate(S1_IDS):
        if truncated and index == len(S1_IDS) - 1:
            break
        rows.append(f"{index}\t{name}\t{name in wins}\t500\t10.0\t100")
    if duplicate:
        rows.append(rows[-1])
    return "\n".join(rows) + "\n"


class Submission34Telemetry(unittest.TestCase):
    def test_baseline_candidate_metrics_and_no_false_promotion(self):
        old = fake({"rich_3905", "rich_3518", "rich_3454"})
        new = fake({"rich_3905", "rich_3518", "rich_3454",
                    "rich_4077", "rich_3882"})
        result = compare(old, new)
        self.assertEqual(result["delta_resolved"], 2)
        self.assertEqual(result["candidate"]["first_edit_successes"], 15)
        self.assertTrue(result["single_run_target_met"])
        self.assertFalse(result["promotion_approved"])
        self.assertEqual(result["candidate"]["total"]["diff_checks"], 15)

    def test_failure_metric_and_retention_gate(self):
        old = fake({"rich_3905", "rich_3518", "rich_3454"})
        candidate = fake({"rich_3905", "rich_3454",
                          "rich_4077", "rich_3882", "rich_3934"},
                         invalid_first=True)
        result = compare(old, candidate)
        self.assertEqual(result["candidate"]["total"][
            "missing_edit_arguments"], 1)
        self.assertEqual(result["candidate"]["first_edit_successes"], 14)
        self.assertIn("rich_3518", result["lost"])
        self.assertFalse(result["single_run_target_met"])

    def test_later_file_reads_are_not_counted_as_edits(self):
        raw = fake({"rich_3905"})
        raw = raw.replace(
            "  [ 2.3s] 🔧 run_command pytest -q tests/test_code.py",
            "  [ 2.2s] 📖 read_file (code.py)\n"
            "  [ 2.21s]   ✓ {'status': 'ok', 'filepath': 'code.py'}\n"
            "  [ 2.3s] 🔧 run_command pytest -q tests/test_code.py",
            1)
        result = parse(raw)
        self.assertEqual(result["traces"][S1_IDS[0]]["successful_edits"], 1)
        self.assertTrue(result["traces"][S1_IDS[0]]["first_edit_success"])

    def test_reject_incomplete_and_duplicate_logs(self):
        good = fake({"rich_3905"})
        for bad in (fake(set(), truncated=True),
                    fake(set(), duplicate=True)):
            with self.assertRaises(ValueError):
                compare(good, bad)
        self.assertEqual(len(parse(good)["rows"]), 15)


if __name__ == "__main__":
    unittest.main()
