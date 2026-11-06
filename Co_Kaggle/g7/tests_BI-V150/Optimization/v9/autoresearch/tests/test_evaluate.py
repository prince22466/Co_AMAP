"""Synthetic half-run CSV tests; no GPU/benchmark/MATH-500 needed."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from autoresearch.evaluate import evaluate, load_pair


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

    def args(self, *, cand4=2450, cand16=1300, base4=2350, base16=1300,
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

    def test_4k_threshold_is_numeric_only(self):
        report = evaluate(self.args())
        self.assertTrue(report["numeric_target_observed"])
        self.assertTrue(report["numeric_gain_observed"])
        self.assertFalse(report["reproducibility_verified"])
        self.assertNotIn("passed", report)

    def test_16k_threshold(self):
        self.assertTrue(evaluate(self.args(cand4=2350, cand16=1450))["comparison"]["16K"]["numeric_target_observed"])

    def test_gain_below_threshold(self):
        report = evaluate(self.args(cand4=2380))
        self.assertTrue(report["numeric_gain_observed"])
        self.assertFalse(report["numeric_target_observed"])

    def test_no_gain_csv_is_still_valid(self):
        report = evaluate(self.args(cand4=2350))
        self.assertTrue(report["valid_half_run_csvs"])
        self.assertFalse(report["numeric_gain_observed"])

    def test_regression_and_ttft_disclosed(self):
        report = evaluate(self.args(cand16=1200, ttft_scale=1.2))
        self.assertLess(report["comparison"]["16K"]["throughput_change_pct"], 0)
        self.assertGreater(report["comparison"]["4K"]["ttft_change_pct"], 0)

    def test_failed_request_rejected(self):
        with self.assertRaisesRegex(ValueError, "failed run"):
            evaluate(self.args(failed=True))

    def test_missing_case_rejected(self):
        args = self.args()
        with args.candidate_summary.open() as stream:
            rows = list(csv.DictReader(stream))
        self.csv("cand_s.csv", rows[:1])
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            evaluate(args)

    def test_missing_raw_run_rejected(self):
        args = self.args()
        with args.candidate_raw.open() as stream:
            rows = list(csv.DictReader(stream))
        self.csv("cand_r.csv", rows[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            evaluate(args)

    def test_summary_raw_mismatch_rejected(self):
        args = self.args()
        with args.candidate_summary.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["Total tok/s"] = "9999"
        self.csv("cand_s.csv", rows)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_pair(args.candidate_summary, args.candidate_raw)

    def test_nonfinite_raw_rejected(self):
        args = self.args()
        with args.candidate_raw.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["Total tok/s"] = "NaN"
        self.csv("cand_r.csv", rows)
        with self.assertRaisesRegex(ValueError, "Invalid Total"):
            evaluate(args)


if __name__ == "__main__":
    unittest.main()
