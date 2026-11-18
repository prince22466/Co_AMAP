"""Offline tests for v16 read-only matched A/B CSV validator."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from autoresearch.evaluate import evaluate


class CsvEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, tok4, tok16, *, raw, fail=False, incomplete=False, tpot=100, ttft=10000):
        path = self.root / name
        rows = []
        for size, prompts, tok in ((4096, 256, tok4), (16384, 128, tok16)):
            for rep in range(2 if raw else 1):
                row = {
                    "Prefill": size, "Decode": 1024, "Conc": 64,
                    "Num Prompts": prompts, "Total tok/s": tok,
                    "Mean TTFT (ms)": ttft,
                    "Mean TPOT (ms)": tpot,
                    "Total Input Tokens": float(size * prompts),
                    "Total Output Tokens": float(1024 * prompts),
                }
                if raw:
                    bad = size == 16384 and rep == 1
                    row["Run Status"] = "FAILED" if fail and bad else "SUCCESS"
                    row["Successful Requests"] = (
                        float(prompts - 1 if incomplete and bad else prompts)
                    )
                rows.append(row)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def args(self, *, base4=2817.15, base16=2073.93,
             cand4=2817.15, cand16=2073.93, fail=False, incomplete=False, tpot=100, ttft=10000):
        return SimpleNamespace(
            control_summary=self.write("a_summary.csv", base4, base16, raw=False),
            control_raw=self.write("a_raw.csv", base4, base16, raw=True),
            candidate_summary=self.write("b_summary.csv", cand4, cand16, raw=False, tpot=tpot, ttft=ttft),
            candidate_raw=self.write(
                "b_raw.csv", cand4, cand16, raw=True,
                fail=fail, incomplete=incomplete, tpot=tpot, ttft=ttft,
            ),
        )

    def test_4k_goal_and_strict_16k_floor(self):
        report = evaluate(self.args(cand4=3011.12))
        self.assertTrue(report["numeric_goal_4k"])
        self.assertTrue(report["numeric_screen_pass"])
        self.assertFalse(report["historical_full_run_targets_verified"])
        self.assertFalse(report["replication_verified"])
        report = evaluate(self.args(cand16=2073.93 * 1.20))
        self.assertFalse(report["numeric_goal_4k"])
        self.assertFalse(report["numeric_screen_pass"])
        report = evaluate(self.args(cand4=3011.12, cand16=2073.92))
        self.assertTrue(report["numeric_goal_4k"])
        self.assertFalse(report["throughput_regression_screen_pass"])
        self.assertFalse(report["numeric_screen_pass"])

    def test_below_4k_goal(self):
        self.assertFalse(evaluate(self.args(cand4=3011.11))["numeric_goal_4k"])

    def test_unchanged_control_is_not_a_gain(self):
        report = evaluate(self.args(
            base4=3000, cand4=3000, base16=2200, cand16=2200
        ))
        self.assertFalse(report["numeric_goal_4k"])

    def test_historical_reference_alone_does_not_prove_gain(self):
        report = evaluate(self.args(base4=3000, cand4=2959))
        self.assertFalse(report["numeric_goal_4k"])

    def test_matched_gain_can_be_below_historical_official_reference(self):
        report = evaluate(self.args(base4=2000, cand4=2300))
        self.assertTrue(report["numeric_goal_4k"])

    def test_discloses_other_case_regression(self):
        report = evaluate(self.args(cand4=3100, cand16=1500))
        self.assertLess(report["comparison"]["16K"]["matched_gain_pct"], 0)
        self.assertFalse(report["throughput_regression_screen_pass"])
        self.assertFalse(report["numeric_screen_pass"])

    def test_incomplete_or_failed_raw_rejected(self):
        with self.assertRaisesRegex(ValueError, "Failed run"):
            evaluate(self.args(fail=True))
        with self.assertRaisesRegex(ValueError, "Incomplete requests"):
            evaluate(self.args(incomplete=True))

    def change_cell(self, path, column, value):
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0][column] = value
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_real_float_request_counts_and_token_totals(self):
        self.assertTrue(evaluate(self.args())["valid_half_run_csvs"])
        for column, value in (("Successful Requests", "256.5"),
                              ("Total Output Tokens", "262143.0"),
                              ("Total Input Tokens", "1048575.0")):
            with self.subTest(column=column):
                a = self.args()
                self.change_cell(a.candidate_raw, column, value)
                with self.assertRaises(ValueError):
                    evaluate(a)

    def test_tpot_and_ttft_regressions_block_numeric_screen(self):
        for kwargs, key in (({"tpot": 106}, "tpot_regression_screen_pass"),
                            ({"ttft": 10600}, "ttft_regression_screen_pass")):
            with self.subTest(key=key):
                report = evaluate(self.args(cand4=3100, **kwargs))
                self.assertTrue(report["numeric_goal_4k"])
                self.assertFalse(report[key])
                self.assertFalse(report["numeric_screen_pass"])

    def test_rounded_summary_cannot_hide_16k_loss(self):
        a = self.args(cand4=3100, cand16=2073.92)
        with a.candidate_summary.open(newline="") as f:
            rows = list(csv.DictReader(f))
        rows[1]["Total tok/s"] = 2073.93  # within summary tolerance
        with a.candidate_summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = evaluate(a)
        self.assertEqual(report["comparison"]["16K"]["candidate_tok_s"], 2073.92)
        self.assertFalse(report["throughput_regression_screen_pass"])
        self.assertFalse(report["numeric_screen_pass"])

    def test_rounded_summary_cannot_create_4k_goal(self):
        a = self.args(cand4=3011.11)
        self.change_cell(a.candidate_summary, "Total tok/s", 3011.12)
        self.assertFalse(evaluate(a)["numeric_goal_4k"])

    def test_raw_repeat_mean_is_used(self):
        a = self.args(cand4=3100)
        with a.candidate_raw.open(newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["Total tok/s"] = 3000
        rows[1]["Total tok/s"] = 3200
        with a.candidate_raw.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = evaluate(a)
        self.assertEqual(report["comparison"]["4K"]["candidate_tok_s"], 3100)
        self.assertEqual(report["comparison"]["4K"]["candidate_raw_tok_s"], [3000, 3200])

    def test_tpot_summary_mismatch(self):
        a = self.args()
        self.change_cell(a.candidate_summary, "Mean TPOT (ms)", 120)
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            evaluate(a)

    def test_raw_summary_mismatch_rejected(self):
        a = self.args()
        with a.candidate_summary.open(newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["Total tok/s"] = 9999
        with a.candidate_summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            evaluate(a)


if __name__ == "__main__":
    unittest.main()
