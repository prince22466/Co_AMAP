"""Synthetic tests for the v10 post-hoc half-run CSV evaluator; no GPU needed."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from autoresearch.evaluate import TARGETS, evaluate, load_pair


class HalfRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def csv(self, name, rows):
        path = self.root / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def args(self, *, cand4=2256.31, cand16=1518.06,
             base4=2256.31, base16=1518.06,
             failed=False, ttft_scale=1.0):
        def lines(four, sixteen, ttft, raw=False):
            result = []
            for size, prompts, tok in ((4096, 256, four), (16384, 128, sixteen)):
                for index in range(2 if raw else 1):
                    row = {"Prefill": size, "Decode": 1024, "Conc": 64,
                           "Num Prompts": prompts, "Total tok/s": tok,
                           "Mean TTFT (ms)": ttft}
                    if raw:
                        error = failed and size == 16384 and index == 1
                        row["Run Status"] = "FAILED" if error else "SUCCESS"
                        row["Successful Requests"] = prompts - 1 if error else prompts
                    result.append(row)
            return result
        return SimpleNamespace(
            baseline_summary=self.csv("base_s.csv", lines(base4, base16, 10000)),
            baseline_raw=self.csv("base_r.csv", lines(base4, base16, 10000, True)),
            candidate_summary=self.csv("cand_s.csv", lines(cand4, cand16, 10000 * ttft_scale)),
            candidate_raw=self.csv("cand_r.csv", lines(cand4, cand16, 10000 * ttft_scale, True)),
        )

    def test_fixed_thresholds_exact_not_prematurely_rounded(self):
        self.assertAlmostEqual(TARGETS["4K"], 2391.6886)
        self.assertAlmostEqual(TARGETS["16K"], 1609.1436)

    def test_exact_4k_goal_numeric_only(self):
        report = evaluate(self.args(cand4=TARGETS["4K"]))
        self.assertTrue(report["numeric_6pct_vs_v9_goal_observed"])
        self.assertTrue(report["comparison"]["4K"]["numeric_6pct_vs_v9_goal_observed"])
        self.assertFalse(report["reproducibility_verified"])
        self.assertFalse(report["production_kernel_verified"])

    def test_exact_16k_goal(self):
        report = evaluate(self.args(cand16=TARGETS["16K"]))
        self.assertTrue(report["comparison"]["16K"]["numeric_6pct_vs_v9_goal_observed"])
        self.assertTrue(report["numeric_checkpoint_candidate"])

    def test_two_decimal_below_goal_rejected(self):
        report = evaluate(self.args(cand4=2391.68, cand16=1609.14))
        self.assertFalse(report["numeric_6pct_vs_v9_goal_observed"])

    def test_5pct_interim_checkpoint_without_goal(self):
        report = evaluate(self.args(cand4=2256.31 * 1.05))
        self.assertTrue(report["numeric_5pct_gain_observed"])
        self.assertFalse(report["numeric_6pct_vs_v9_goal_observed"])
        self.assertTrue(report["numeric_checkpoint_candidate"])

    def test_below_5pct_not_checkpoint(self):
        report = evaluate(self.args(cand4=2256.31 * 1.049))
        self.assertFalse(report["numeric_5pct_gain_observed"])
        self.assertFalse(report["numeric_checkpoint_candidate"])

    def test_final_goal_exception_below_5pct_from_current_control(self):
        report = evaluate(self.args(base4=2390, cand4=2391.69))
        self.assertTrue(report["numeric_6pct_vs_v9_goal_observed"])
        self.assertFalse(report["numeric_5pct_gain_observed"])
        self.assertTrue(report["numeric_checkpoint_candidate"])

    def test_fixed_goal_not_moving_with_v10_baseline(self):
        report = evaluate(self.args(base4=2300, cand4=2391.69))
        self.assertTrue(report["numeric_6pct_vs_v9_goal_observed"])
        self.assertFalse(report["comparison"]["4K"]["numeric_5pct_gain_observed"])

    def test_no_gain_valid_but_not_win(self):
        report = evaluate(self.args())
        self.assertTrue(report["valid_half_run_csvs"])
        self.assertFalse(report["numeric_positive_change_observed"])
        self.assertFalse(report["numeric_checkpoint_candidate"])

    def test_regression_and_ttft_disclosed(self):
        report = evaluate(self.args(cand4=2400, cand16=1400, ttft_scale=1.2))
        self.assertLess(report["comparison"]["16K"]["throughput_change_pct_vs_matched_v10_control"], 0)
        self.assertGreater(report["comparison"]["4K"]["ttft_change_pct"], 0)

    def test_failed_requests_rejected(self):
        with self.assertRaisesRegex(ValueError, "failed run"):
            evaluate(self.args(failed=True))

    def test_missing_case_rejected(self):
        args = self.args()
        with args.candidate_summary.open() as stream:
            rows = list(csv.DictReader(stream))
        self.csv("cand_s.csv", rows[:1])
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            evaluate(args)

    def test_missing_raw_repeat_rejected(self):
        args = self.args()
        with args.candidate_raw.open() as stream:
            rows = list(csv.DictReader(stream))
        self.csv("cand_r.csv", rows[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            evaluate(args)

    def test_summary_mismatch_rejected(self):
        args = self.args()
        with args.candidate_summary.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["Total tok/s"] = "9999"
        self.csv("cand_s.csv", rows)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_pair(args.candidate_summary, args.candidate_raw)

    def test_nonfinite_rejected(self):
        args = self.args()
        with args.candidate_raw.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["Total tok/s"] = "NaN"
        self.csv("cand_r.csv", rows)
        with self.assertRaisesRegex(ValueError, "Invalid Total"):
            evaluate(args)


if __name__ == "__main__":
    unittest.main()
