"""Printed-trace triage regressions, including the real S30 destruction case."""
from pathlib import Path
import unittest

from compare_submission24 import S1_IDS
from gate_submission32 import (PILOT_IDS, S29_WINS, S30_GAIN_WINS,
                               assess, tool_telemetry)

DIAG = Path(__file__).resolve().parents[1] / "diagnostics"


def trace(ids, solved=(), *, empty=(), unsafe=False, guard=False, graph=False):
    result = []
    for i, name in enumerate(ids, 1):
        result.append(f"[candidate {i}/{len(ids)}] Evaluating {name} (repo)")
        result.append("  [  0.1s] 📖 read_file (source.py)")
        if graph and i == 1:
            result.append('  [  0.2s] 🤖 get_code_neighbors {"node":"F"}')
        if unsafe and i == 1:
            result.append('  [  0.3s] 🔧 run_command python -c "Path(\"x\").write_text(\"BAD\")"')
        if name not in empty:
            result.append("  [  0.4s] ✏️ edit_file (source.py)")
            result.append("  [  0.5s]   ✓ {'status':'ok'}")
        if guard and i == 1:
            result.append('  [  0.7s]   → S32_GUARDED_LINE_EDIT_OK source.py 2')
        result.append("  [  1.0s] 🤖 submit_patch")
        result.append(f"  -> resolved={name in solved}, "
                      f"patch_chars={0 if name in empty else 512}, tools=4, "
                      "duration=9, input=400, output=80")
    return "\n".join(result) + "\n"


class GateTests(unittest.TestCase):
    def setUp(self):
        self.s29 = trace(S1_IDS, S29_WINS)
        self.s30 = trace(S1_IDS, set(S30_GAIN_WINS) |
                         {"rich_3905", "rich_3454"})
        self.wins = set(S29_WINS) | set(S30_GAIN_WINS)

    def test_five_wins_with_retention_is_only_preliminary_pass(self):
        candidate = trace(S1_IDS, self.wins)
        result = assess(self.s29, self.s30, candidate, "s1", "control")
        self.assertEqual(result["resolved"], 5)
        self.assertTrue(result["preliminary_gate_pass"], result["failures"])
        self.assertFalse(result["promotion_approved"])
        self.assertEqual(result["telemetry"]["totals"]["graph_calls"], 0)

    def test_reject_regression_empty_and_overactive_graph(self):
        for reason, result in (
            ("below_resolution_target", trace(S1_IDS, set(S29_WINS))),
            ("lost_S29_confirmed_win", trace(
                S1_IDS, (self.wins - {"rich_3518"}) | {"rich_4076"})),
            ("lost_both_S30_gains", trace(
                S1_IDS, (self.wins - set(S30_GAIN_WINS)) |
                {"rich_4076", "rich_3934"})),
            ("excessive_empty_patches", trace(
                S1_IDS, self.wins, empty={"rich_4075", "rich_4076"})),
            ("unexpected_graph_tools", trace(S1_IDS, self.wins, graph=True)),
        ):
            with self.subTest(reason=reason):
                self.assertIn(reason,
                    assess(self.s29, self.s30, result, "s1", "control")["failures"])

    def test_unsafe_control_and_primary_unverified_shell_are_rejected(self):
        raw = trace(S1_IDS, self.wins, unsafe=True)
        a = assess(self.s29, self.s30, raw, "s1", "control")
        self.assertIn("control_used_shell_mutation", a["failures"])
        b = assess(self.s29, self.s30, raw, "s1", "primary")
        self.assertIn("shell_mutation_without_guard_evidence", b["failures"])
        self.assertFalse(b["promotion_approved"])

    def test_recovery_marker_demands_manual_review_even_if_prelim_passes(self):
        # Marker + unsafe shell is only preliminary, NOT certified safe.
        raw = trace(S1_IDS, self.wins, unsafe=True, guard=True)
        result = assess(self.s29, self.s30, raw, "s1", "primary")
        self.assertTrue(result["preliminary_gate_pass"], result["failures"])
        self.assertTrue(result["manual_ATIF_review_required"])
        self.assertFalse(result["promotion_approved"])

    def test_fixed_pilot_and_real_s30_trace_fail(self):
        self.assertEqual(len(PILOT_IDS), 7)
        candidate = trace(PILOT_IDS, self.wins)
        self.assertTrue(assess(self.s29, self.s30, candidate, "pilot",
                               "control")["preliminary_gate_pass"])
        real_s29 = (DIAG / "submission-29-zip_s1TasksPrintouts.txt").read_text()
        real_s30 = (DIAG / "submission-30-zip_s1TasksPrintouts.txt").read_text()
        result = assess(real_s29, real_s30, real_s30, "s1", "control")
        self.assertIn("below_resolution_target", result["failures"])
        self.assertIn("lost_S29_confirmed_win", result["failures"])
        self.assertIn("oversized_patch:fastapi_14258", result["failures"])
        self.assertFalse(result["preliminary_gate_pass"])

    def test_missing_tasks_are_not_accepted(self):
        candidate = trace(S1_IDS[:-1], self.wins)
        result = assess(self.s29, self.s30, candidate, "s1", "control")
        self.assertIn("candidate_task_set_mismatch", result["failures"])


if __name__ == "__main__":
    unittest.main()
