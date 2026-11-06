"""Offline synthetic fixture checks for the numeric benchmark acceptance gate."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from autoresearch.evaluate import evaluate, summary, validate_raw


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.math = self.root / "math500.log"
        self.math.write_text("Test fixture only; not a real accuracy result\n")

    def csvfile(self, name, rows):
        path = self.root / name
        with path.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def make_args(self, base4=2331.29, base16=1251.22,
                  cand4=2701.0, cand16=1251.22, cand_ttft_scale=1.0,
                  accuracy=0.96, failure=False):
        def summary_rows(a, b, ttft_scale):
            return [
                {"Prefill": 4096, "Decode": 1024, "Conc": 64,
                 "Num Prompts": 256, "Total tok/s": a,
                 "Mean TTFT (ms)": 8738.76 * ttft_scale},
                {"Prefill": 16384, "Decode": 1024, "Conc": 64,
                 "Num Prompts": 128, "Total tok/s": b,
                 "Mean TTFT (ms)": 428970.85 * ttft_scale},
            ]

        def raw_rows():
            result = []
            for size, prompts in ((4096, 256), (16384, 128)):
                for index in range(4):
                    failed = failure and size == 16384 and index == 2
                    result.append({
                        "Prefill": size, "Decode": 1024, "Conc": 64,
                        "Num Prompts": prompts,
                        "Run Status": "FAILED" if failed else "SUCCESS",
                        "Successful Requests": prompts - 1 if failed else prompts,
                    })
            return result

        return SimpleNamespace(
            baseline_summary=self.csvfile("base_summary.csv", summary_rows(base4, base16, 1)),
            candidate_summary=self.csvfile("cand_summary.csv", summary_rows(cand4, cand16, cand_ttft_scale)),
            baseline_raw=self.csvfile("base_raw.csv", raw_rows() if not failure else [
                {**r, "Run Status": "SUCCESS", "Successful Requests": r["Num Prompts"]}
                for r in raw_rows()
            ]),
            candidate_raw=self.csvfile("cand_raw.csv", raw_rows()),
            math500_log=self.math,
            math500_score=accuracy,
        )

    def test_goal_passes_with_both_guards(self):
        result = evaluate(self.make_args())
        self.assertTrue(result["passed"])
        self.assertTrue(result["target_met"])

    def test_improvement_below_goal_is_accepted(self):
        result = evaluate(self.make_args(cand4=2400))
        self.assertTrue(result["passed"])
        self.assertFalse(result["target_met"])

    def test_accuracy_floor_rejects(self):
        result = evaluate(self.make_args(accuracy=0.949))
        self.assertFalse(result["passed"])

    def test_other_case_regression_rejects(self):
        result = evaluate(self.make_args(cand16=1200))
        self.assertFalse(result["passed"])

    def test_ttft_regression_rejects(self):
        result = evaluate(self.make_args(cand_ttft_scale=1.02))
        self.assertFalse(result["passed"])

    def test_post_warmup_failure_rejects(self):
        args = self.make_args(failure=True)
        with self.assertRaisesRegex(ValueError, "failed post-warmup"):
            evaluate(args)

    def test_missing_official_shape_rejects(self):
        args = self.make_args()
        self.csvfile("cand_summary.csv", [
            {"Prefill":4096, "Decode":1024, "Conc":64,
             "Num Prompts":256,"Total tok/s":2701,
             "Mean TTFT (ms)":8738.76},
        ])
        with self.assertRaisesRegex(ValueError, "BOTH"):
            summary(args.candidate_summary)

    def test_bad_raw_run_count_rejects(self):
        args = self.make_args()
        with args.candidate_raw.open() as f:
            data = list(csv.DictReader(f))
        self.csvfile("cand_raw.csv", data[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            validate_raw(args.candidate_raw)


if __name__ == "__main__":
    unittest.main()
