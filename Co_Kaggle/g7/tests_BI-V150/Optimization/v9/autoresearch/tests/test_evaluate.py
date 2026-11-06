"""Synthetic half-run CSV acceptance tests. No GPU, benchmark or MATH-500 required."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from autoresearch.evaluate import evaluate, summary, validate_raw


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def csvfile(self, name, entries):
        path = self.root / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
        return path

    def args(self, base4=2350, base16=1300, cand4=2451, cand16=1300,
             ttft_scale=1.0, fail=False):
        def s(a, b, factor):
            return [
                {"Prefill": k, "Decode": 1024, "Conc": 64, "Num Prompts": n,
                 "Total tok/s": val, "Mean TTFT (ms)": 10000 * factor}
                for k, n, val in ((4096, 256, a), (16384, 128, b))
            ]

        def r(failure):
            return [
                {"Prefill": k, "Decode": 1024, "Conc": 64, "Num Prompts": n,
                 "Run Status": "FAILED" if failure and k == 16384 and i == 1 else "SUCCESS",
                 "Successful Requests": n - 1 if failure and k == 16384 and i == 1 else n,
                 "Total tok/s": 1200}
                for k, n in ((4096, 256), (16384, 128)) for i in range(2)
            ]

        return SimpleNamespace(
            baseline_summary=self.csvfile("b_s.csv", s(base4, base16, 1.0)),
            candidate_summary=self.csvfile("c_s.csv", s(cand4, cand16, ttft_scale)),
            baseline_raw=self.csvfile("b_r.csv", r(False)),
            candidate_raw=self.csvfile("c_r.csv", r(fail)),
        )

    def test_4k_goal(self):
        gate = evaluate(self.args())
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["target_met"])
        self.assertEqual(gate["runs_per_case"], 2)

    def test_16k_goal_only(self):
        gate = evaluate(self.args(cand4=2350, cand16=1450))
        self.assertTrue(gate["target_met"])
        self.assertTrue(gate["passed"])

    def test_gain_below_goal(self):
        gate = evaluate(self.args(cand4=2380))
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["target_met"])

    def test_no_gain(self):
        gate = evaluate(self.args(cand4=2350))
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["target_met"])

    def test_other_workload_and_ttft_are_disclosed_not_extra_gates(self):
        gate = evaluate(self.args(cand16=1200, ttft_scale=1.2))
        self.assertTrue(gate["target_met"])
        self.assertLess(gate["comparison"]["16K"]["throughput_change_pct"], 0)
        self.assertGreater(gate["comparison"]["4K"]["ttft_change_pct"], 0)

    def test_failed_request_rejects(self):
        with self.assertRaisesRegex(ValueError, "failed run"):
            evaluate(self.args(fail=True))

    def test_missing_shape_rejects(self):
        a = self.args()
        self.csvfile("c_s.csv", [
            {"Prefill": 4096, "Decode": 1024, "Conc": 64, "Num Prompts": 256,
             "Total tok/s": 2451, "Mean TTFT (ms)": 10000}
        ])
        with self.assertRaisesRegex(ValueError, "BOTH"):
            summary(a.candidate_summary)

    def test_only_one_raw_run_rejects(self):
        a = self.args()
        with a.candidate_raw.open() as f:
            entries = list(csv.DictReader(f))
        self.csvfile("c_r.csv", entries[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            validate_raw(a.candidate_raw)


if __name__ == "__main__":
    unittest.main()
