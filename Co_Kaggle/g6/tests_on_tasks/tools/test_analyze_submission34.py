"""Synthetic tests for S34 scored-log comparison (no model claims)."""
from __future__ import annotations

import unittest
from pathlib import Path

from analyze_submission34 import compare, parse, compare_jsonl, summarize_jsonl
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


def evaluation_rows(wins: set[str]) -> list[dict]:
    """Representative official evaluator JSONL rows."""
    return [
        {
            "instance_id": task,
            "resolved": task in wins,
            "agent_patch": "diff --git a/example.py b/example.py\n+fix\n",
            "duration_seconds": 12.0,
            "error": None,
        }
        for task in S1_IDS
    ]



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

    def test_real_historical_printouts_parse(self):
        diagnostics = Path(__file__).resolve().parents[1] / "diagnostics"
        base = diagnostics / "submission-29-zip_s1TasksPrintouts.txt"
        candidate = diagnostics / "submission-33-zip_s1TasksPrintouts.txt"
        if not base.exists() or not candidate.exists():
            self.skipTest("Historical printed diagnostics not mounted")
        report = compare(base.read_text(encoding="utf-8"),
                         candidate.read_text(encoding="utf-8"))
        self.assertEqual(report["baseline"]["resolved"], 3)
        self.assertEqual(report["candidate"]["resolved"], 2)
        self.assertFalse(report["promotion_approved"])

    def test_official_jsonl_results_can_be_compared(self):
        old = evaluation_rows({"rich_3905", "rich_3518", "rich_3454"})
        new = evaluation_rows({
            "rich_3905", "rich_3518", "rich_3454",
            "rich_4077", "rich_3882",
        })
        report = compare_jsonl(old, new)
        self.assertEqual(report["input_format"], "official_runner_jsonl")
        self.assertEqual(report["delta_resolved"], 2)
        self.assertTrue(report["single_run_target_met"])
        self.assertFalse(report["promotion_approved"])

    def test_jsonl_incomplete_duplicate_or_invalid_duration_rejected(self):
        baseline = evaluation_rows({"rich_3905"})
        for candidate in (
            baseline[:-1],
            baseline + [dict(baseline[0])],
        ):
            with self.subTest(rows=len(candidate)), self.assertRaises(ValueError):
                compare_jsonl(baseline, candidate)
        invalid_duration = evaluation_rows(set())
        invalid_duration[0]["duration_seconds"] = None
        with self.assertRaisesRegex(ValueError, "task durations"):
            summarize_jsonl(invalid_duration)

    def test_jsonl_oversized_patch_never_passes_provisional_gate(self):
        baseline = evaluation_rows({"rich_3905", "rich_3518", "rich_3454"})
        candidate = evaluation_rows({
            "rich_3905", "rich_3518", "rich_3454", "rich_3882", "rich_4077",
        })
        candidate[0]["agent_patch"] = "a" * 50001
        self.assertFalse(compare_jsonl(baseline, candidate)["single_run_target_met"])

    def test_reject_incomplete_and_duplicate_logs(self):
        good = fake({"rich_3905"})
        for bad in (fake(set(), truncated=True),
                    fake(set(), duplicate=True)):
            with self.assertRaises(ValueError):
                compare(good, bad)
        self.assertEqual(len(parse(good)["rows"]), 15)


if __name__ == "__main__":
    unittest.main()
