"""Offline safety tests for the v11 GPU session guard and checkpoint writer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from autoresearch import session_budget as budget
from autoresearch import checkpoint as checkpoints


class SessionBudgetTests(unittest.TestCase):
    def test_330_minute_session_admits_two_estimated_runs(self):
        state = budget.start_session(330, None, now=1000)
        report = budget.decision(state, runs=2, now=1000)
        self.assertEqual(report["required_minutes"], 275)
        self.assertTrue(report["admitted"])

    def test_90_and_105_minute_floors_are_rejected(self):
        state = budget.start_session(330, None, now=1000)
        for remaining in (90, 105, 125, 149.99):
            now = state["deadline_epoch"] - 60 * remaining
            self.assertFalse(budget.decision(state, now=now)["admitted"])
        self.assertTrue(budget.decision(
            state, now=state["deadline_epoch"] - 60 * 150
        )["admitted"])

    def test_long_observed_run_raises_admission_requirement(self):
        state = budget.start_session(330, None, now=1000)
        state["observed_runs_minutes"].extend([105, 132])
        report = budget.decision(state, runs=2, now=1000)
        self.assertEqual(report["required_minutes"], 329)
        self.assertTrue(report["admitted"])
        report = budget.decision(state, runs=2, now=1120)
        self.assertFalse(report["admitted"])

    def test_expired_session_never_admitted(self):
        state = budget.start_session(330, None, now=1000)
        result = budget.decision(state, now=state["deadline_epoch"] + 1)
        self.assertEqual(result["remaining_minutes"], 0)
        self.assertFalse(result["admitted"])

    def test_missing_and_bad_deadline_fail_closed(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Unknown GPU deadline"):
                budget.read_state(Path(folder) / "missing.json")
        with self.assertRaises(ValueError):
            budget.start_session(None, None, now=100)
        with self.assertRaises(ValueError):
            budget.start_session(-1, None, now=100)
        with self.assertRaises(ValueError):
            budget.start_session(None, "2026-11-09T17:00:00", now=100)

    def test_deadline_utc_including_offset(self):
        state = budget.start_session(None, "2026-11-10T17:00:00+01:00", now=100)
        self.assertEqual(
            state["deadline_epoch"],
            budget.datetime(2026, 11, 10, 16, 0, tzinfo=budget.timezone.utc).timestamp(),
        )

    def test_state_roundtrip(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "budget.json"
            state = budget.start_session(330, None, now=1000)
            budget.save(path, state)
            self.assertEqual(budget.read_state(path), state)

    def test_invalid_one_run_admission_does_not_start_benchmark(self):
        state = budget.start_session(1, None)
        with TemporaryDirectory() as folder, patch.object(
            budget, "BENCHMARK", Path(folder) / "does_not_exist"
        ):
            rc = budget.execute_half(state, Path(folder) / "state.json")
            self.assertEqual(rc, 2)


class CheckpointTests(unittest.TestCase):
    def test_handoff_and_append_only_ledger(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            notes = root / "autoresearch"
            notes.mkdir()
            ledger = notes / "EXPERIMENTS.md"
            ledger.write_text("# Prior evidence\n")
            checkpoints.checkpoint(
                root, "Kernel 16K improved provisionally", "log/runA.csv",
                "Run matched control", "provisional", 40,
            )
            checkpoints.checkpoint(
                root, "Test was rejected", "log/correctness.txt",
                "Restore baseline", "rejected", 30,
            )
            text = ledger.read_text()
            self.assertIn("# Prior evidence", text)
            self.assertIn("Kernel 16K improved", text)
            self.assertIn("Test was rejected", text)
            handoff = (notes / "HANDOFF.md").read_text()
            self.assertIn("Restore baseline", handoff)
            self.assertIn("not verify accuracy", handoff)

    def test_small_raw_csv_is_preserved_with_hash(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            raw_folder = root / ".autoresearch-work" / "bench_runs" / "guard_A" / "benchmark_results"
            raw_folder.mkdir(parents=True)
            raw = raw_folder / "raw_runs_001.csv"
            raw.write_text(
                "Prefill,Decode,Conc,Num Prompts,Total tok/s,Mean TTFT (ms),"
                "Run Status,Successful Requests\n"
                "4096,1024,64,256,2000,1000,SUCCESS,256\n"
            )
            checkpoints.checkpoint(
                root, "Run completed but not replicated", str(raw), "Check results",
                csv_paths=[raw],
            )
            evidence = list((root / "autoresearch" / "evidence").glob("*.csv"))
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].read_bytes(), raw.read_bytes())
            self.assertIn("sha256", (root / "autoresearch" / "EXPERIMENTS.md").read_text())

    def test_csv_outside_worktree_or_invalid_schema_refused(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            external = root / "unrelated" / "raw_runs.csv"
            external.parent.mkdir()
            external.write_text("a,b\n1,2\n")
            with self.assertRaisesRegex(ValueError, "outside ignored work tree"):
                checkpoints.checkpoint(root, "a", "b", "c", csv_paths=[external])
            internal = root / ".autoresearch-work" / "raw_runs.csv"
            internal.parent.mkdir()
            internal.write_text("a,b\n1,2\n")
            with self.assertRaisesRegex(ValueError, "lacks expected benchmark columns"):
                checkpoints.checkpoint(root, "a", "b", "c", csv_paths=[internal])

    def test_missing_evidence_is_rejected(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                checkpoints.checkpoint(Path(folder), "a", "", "next")

    def test_push_never_uses_main_and_only_stages_evidence(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []
            def fake_git(_root, *args):
                calls.append(args)
                if args[:2] == ("rev-parse", "HEAD"):
                    return "123"
                if args[:2] == ("branch", "--show-current"):
                    return "codex/v11-terra61-autoresearch-setup"
                return ""
            with patch.object(checkpoints, "git", side_effect=fake_git):
                checkpoints.checkpoint(root, "gain", "raw.csv", "repeat", push=True)
            self.assertTrue(any(args[:2] == ("commit", "--only") for args in calls))
            self.assertTrue(any(args[:2] == ("push", "origin") for args in calls))
            self.assertFalse(any("kernel.py" in str(args) for args in calls))

        with TemporaryDirectory() as folder:
            with patch.object(checkpoints, "git", side_effect=["123", "main"]):
                with self.assertRaises(RuntimeError):
                    checkpoints.checkpoint(
                        Path(folder), "gain", "raw.csv", "repeat", push=True,
                    )


if __name__ == "__main__":
    unittest.main()
