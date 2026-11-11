"""Offline tests for v12 read-only matched A/B CSV validator."""
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

    def write(self, name, tok4, tok16, *, raw, fail=False, incomplete=False):
        path = self.root / name
        rows = []
        for size, prompts, tok in ((4096, 256, tok4), (16384, 128, tok16)):
            for rep in range(2 if raw else 1):
                row = {
                    "Prefill": size, "Decode": 1024, "Conc": 64,
                    "Num Prompts": prompts, "Total tok/s": tok,
                    "Mean TTFT (ms)": 10000,
                }
                if raw:
                    bad = size == 16384 and rep == 1
                    row["Run Status"] = "FAILED" if fail and bad else "SUCCESS"
                    row["Successful Requests"] = (
                        prompts - 1 if incomplete and bad else prompts
                    )
                rows.append(row)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def args(self, *, base4=2372.38, base16=1872.70,
             cand4=2372.38, cand16=1872.70, fail=False, incomplete=False):
        return SimpleNamespace(
            control_summary=self.write("a_summary.csv", base4, base16, raw=False),
            control_raw=self.write("a_raw.csv", base4, base16, raw=True),
            candidate_summary=self.write("b_summary.csv", cand4, cand16, raw=False),
            candidate_raw=self.write(
                "b_raw.csv", cand4, cand16, raw=True,
                fail=fail, incomplete=incomplete,
            ),
        )

    def test_5pct_goal_either_workload(self):
        for cand4, cand16 in (
            (2372.38 * 1.05, 1872.70),
            (2372.38, 1872.70 * 1.05),
        ):
            with self.subTest(cand4=cand4, cand16=cand16):
                report = evaluate(self.args(cand4=cand4, cand16=cand16))
                self.assertTrue(report["numeric_goal_5pct"])
                self.assertFalse(report["replication_verified"])

    def test_less_than_5pct_is_not_goal(self):
        report = evaluate(self.args(cand4=2372.38 * 1.0499))
        self.assertFalse(report["numeric_goal_5pct"])

    def test_control_above_anchor_is_not_a_gain(self):
        report = evaluate(self.args(
            base4=2600, cand4=2600, base16=2100, cand16=2100
        ))
        self.assertFalse(report["numeric_goal_5pct"])

    def test_fixed_anchor_alone_does_not_prove_gain(self):
        report = evaluate(self.args(base4=2500, cand4=2491))
        self.assertFalse(report["numeric_goal_5pct"])

    def test_matched_gain_alone_does_not_cross_fixed_anchor(self):
        report = evaluate(self.args(base4=2000, cand4=2300))
        self.assertFalse(report["numeric_goal_5pct"])

    def test_discloses_other_case_regression(self):
        report = evaluate(self.args(cand4=2600, cand16=1500))
        self.assertLess(report["comparison"]["16K"]["matched_gain_pct"], 0)

    def test_incomplete_or_failed_raw_rejected(self):
        with self.assertRaisesRegex(ValueError, "Failed run"):
            evaluate(self.args(fail=True))
        with self.assertRaisesRegex(ValueError, "Incomplete requests"):
            evaluate(self.args(incomplete=True))

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
