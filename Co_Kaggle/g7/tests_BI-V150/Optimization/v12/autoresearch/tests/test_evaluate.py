"""Synthetic tests of v12 read-only post-hoc half-run CSV checker; no GPU."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from autoresearch.evaluate import (
    MINIMUM_TARGETS, STRETCH_TARGETS, evaluate, load_pair,
)


class HalfRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, rows):
        p = self.root / name
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return p

    def args(self, *, base4=2372.38, base16=1872.7,
             start4=2372.38, start16=1872.7,
             cand4=2372.38, cand16=1872.7,
             failed=False, bad_requests=False, ttft_scale=1.0):
        def rows(four, sixteen, ttft, raw=False):
            items = []
            for size, prompts, tok in ((4096, 256, four), (16384, 128, sixteen)):
                for rep in range(2 if raw else 1):
                    row = {
                        "Prefill": size, "Decode": 1024, "Conc": 64,
                        "Num Prompts": prompts, "Total tok/s": tok,
                        "Mean TTFT (ms)": ttft,
                    }
                    if raw:
                        error = size == 16384 and rep == 1
                        row["Run Status"] = "FAILED" if failed and error else "SUCCESS"
                        row["Successful Requests"] = (
                            prompts - 1 if bad_requests and error else prompts
                        )
                    items.append(row)
            return items

        return SimpleNamespace(
            baseline_summary=self.write("a_s.csv", rows(base4, base16, 10000)),
            baseline_raw=self.write("a_r.csv", rows(base4, base16, 10000, True)),
            starting_summary=self.write("start_s.csv", rows(start4, start16, 10000)),
            starting_raw=self.write("start_r.csv", rows(start4, start16, 10000, True)),
            candidate_summary=self.write("b_s.csv", rows(cand4, cand16, 10000 * ttft_scale)),
            candidate_raw=self.write(
                "b_r.csv", rows(cand4, cand16, 10000 * ttft_scale, True)
            ),
        )

    def test_exact_frozen_thresholds_not_prematurely_rounded(self):
        self.assertAlmostEqual(MINIMUM_TARGETS["4K"], 2490.999)
        self.assertAlmostEqual(MINIMUM_TARGETS["16K"], 1966.335)
        self.assertAlmostEqual(STRETCH_TARGETS["4K"], 2609.618)
        self.assertAlmostEqual(STRETCH_TARGETS["16K"], 2059.97)

    def test_exact_4k_minimum_numeric_only(self):
        report = evaluate(self.args(cand4=MINIMUM_TARGETS["4K"]))
        self.assertTrue(report["comparison"]["4K"]["numeric_minimum_candidate"])
        self.assertTrue(report["numeric_minimum_candidate"])
        self.assertFalse(report["reproducibility_verified"])
        self.assertFalse(report["production_kernel_verified"])

    def test_exact_16k_minimum_numeric_only(self):
        report = evaluate(self.args(cand16=MINIMUM_TARGETS["16K"]))
        self.assertTrue(report["numeric_minimum_candidate"])

    def test_exact_4k_stretch(self):
        report = evaluate(self.args(cand4=STRETCH_TARGETS["4K"]))
        self.assertTrue(report["numeric_stretch_candidate"])

    def test_exact_16k_stretch(self):
        report = evaluate(self.args(cand16=STRETCH_TARGETS["16K"]))
        self.assertTrue(report["numeric_stretch_candidate"])

    def test_below_fixed_goal_not_rounded_up(self):
        r = evaluate(self.args(cand4=2490.99, cand16=1966.33))
        self.assertFalse(r["numeric_minimum_candidate"])

    def test_progress_at_3pct_without_minimum(self):
        r = evaluate(self.args(cand4=2372.38 * 1.03))
        self.assertTrue(r["numeric_progress_checkpoint_candidate"])
        self.assertFalse(r["numeric_minimum_candidate"])

    def test_sub3pct_not_progress(self):
        r = evaluate(self.args(cand4=2372.38 * 1.0299))
        self.assertFalse(r["numeric_progress_checkpoint_candidate"])

    def test_fixed_threshold_alone_not_enough_without_matched_gain(self):
        r = evaluate(self.args(start4=2500, base4=2500, cand4=2491.0))
        self.assertTrue(r["comparison"]["4K"]["numeric_fixed_minimum_5pct_crossed"])
        self.assertFalse(r["numeric_minimum_candidate"])

    def test_matched_gain_alone_not_enough_without_fixed_threshold(self):
        r = evaluate(self.args(start4=2000, base4=2000, cand4=2400))
        self.assertTrue(r["comparison"]["4K"]["numeric_matched_minimum_5pct_crossed"])
        self.assertFalse(r["numeric_minimum_candidate"])

    def test_different_starting_and_accepted_baselines(self):
        r = evaluate(self.args(base4=2500, start4=2372.38, cand4=2575))
        self.assertTrue(r["comparison"]["4K"]["numeric_3pct_progress_vs_accepted"])
        self.assertTrue(r["numeric_minimum_candidate"])
        self.assertFalse(r["numeric_stretch_candidate"])

    def test_unchanged_baseline_above_target_is_not_success(self):
        r = evaluate(self.args(base4=2600, start4=2600, cand4=2600))
        self.assertFalse(r["numeric_progress_checkpoint_candidate"])
        self.assertFalse(r["numeric_minimum_candidate"])

    def test_other_workload_regression_disclosed(self):
        r = evaluate(self.args(cand4=2600, cand16=1500, ttft_scale=1.2))
        self.assertLess(r["comparison"]["16K"]["change_pct_vs_accepted_matched_control"], 0)
        self.assertGreater(r["comparison"]["4K"]["ttft_change_pct_vs_accepted"], 0)

    def test_no_gain_is_valid_but_no_win(self):
        r = evaluate(self.args())
        self.assertTrue(r["valid_half_run_csvs"])
        self.assertFalse(r["numeric_progress_checkpoint_candidate"])
        self.assertFalse(r["numeric_minimum_candidate"])

    def test_failed_status_rejected(self):
        with self.assertRaisesRegex(ValueError, "failed run"):
            evaluate(self.args(failed=True))

    def test_incomplete_request_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing requests"):
            evaluate(self.args(bad_requests=True))

    def test_missing_case_rejected(self):
        args = self.args()
        with args.candidate_summary.open() as f:
            rows = list(csv.DictReader(f))
        self.write("b_s.csv", rows[:1])
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            evaluate(args)

    def test_missing_raw_repetition_rejected(self):
        args = self.args()
        with args.candidate_raw.open() as f:
            rows = list(csv.DictReader(f))
        self.write("b_r.csv", rows[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            evaluate(args)

    def test_summary_disagrees_with_raw_rejected(self):
        args = self.args()
        with args.candidate_summary.open() as f:
            rows = list(csv.DictReader(f))
        rows[0]["Total tok/s"] = "9999"
        self.write("b_s.csv", rows)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_pair(args.candidate_summary, args.candidate_raw)

    def test_nonfinite_rejected(self):
        args = self.args()
        with args.candidate_raw.open() as f:
            rows = list(csv.DictReader(f))
        rows[0]["Total tok/s"] = "NaN"
        self.write("b_r.csv", rows)
        with self.assertRaisesRegex(ValueError, "Invalid Total"):
            evaluate(args)


if __name__ == "__main__":
    unittest.main()
