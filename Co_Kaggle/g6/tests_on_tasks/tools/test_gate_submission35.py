"""Synthetic S35 gate tests; no simulated result is evidence of Gemma skill use."""
from __future__ import annotations

import unittest

from compare_submission24 import S1_IDS
from gate_submission35 import (
    PILOT_IDS, compare, patch_audit, summarize, tool_events,
)


PATCH = ("diff --git a/pkg/module.py b/pkg/module.py\n"
         "--- a/pkg/module.py\n"
         "+++ b/pkg/module.py\n"
         "@@ -2 +2 @@\n"
         "-return False\n"
         "+return True\n")


def results(ids: tuple[str, ...], wins: set[str], patch: str = PATCH) -> list[dict]:
    return [
        {"instance_id": task, "resolved": task in wins,
         "agent_patch": patch, "duration_seconds": 40.0, "error": None}
        for task in ids
    ]


def printout(ids: tuple[str, ...], *, failures: int = 0,
             skip_last: bool = False, use_skill: bool = False) -> str:
    lines = []
    for i, task in enumerate(ids, 1):
        if skip_last and i == len(ids):
            continue
        lines.extend([
            f"[candidate {i}/{len(ids)}] Evaluating {task} (repo)",
            "  [ 1.0s] 📖 read_file (pkg/module.py)",
            "  [ 2.0s] ✏️ edit_file (pkg/module.py)",
        ])
        if failures and i == 1:
            for retry in range(failures):
                lines.extend([
                    f"  [ {2.1+retry}s]   ❌ {{'status': 'error', 'error_type': 'FileEditError'}}",
                    f"  [ {2.2+retry}s] ✏️ edit_file (pkg/module.py)",
                ])
        lines.append(
            "  [ 4.0s]   ✓ {'status': 'ok', 'filepath': 'pkg/module.py'}")
        if use_skill and i == 1:
            lines.append("  [ 5.0s] 📚 load_skill (tool-recovery)")
        lines.extend([
            "  [ 6.0s] 🔧 run_command git diff --check",
            "  [ 7.0s] 🤖 submit_patch",
            f"  -> resolved={task in {'rich_4077', 'rich_3905'}}, "
            "patch_chars=100, tools=4, duration=10.0, input=100, output=10",
        ])
    return "\n".join(lines) + "\n"


class S35GateTests(unittest.TestCase):
    def test_patch_audit_rejects_destructive_or_unparsable_diffs(self):
        self.assertEqual(patch_audit(PATCH)["flags"], [])
        self.assertIn("empty_patch", patch_audit("")["flags"])
        for bad in (
            PATCH.replace("pkg/module.py", "pytest.ini"),
            PATCH + "deleted file mode 100644\n",
            PATCH + ("+a\n" * 220),
            PATCH.replace("diff --git a/pkg/module.py b/pkg/module.py",
                          "diff --git not-a-diff"),
        ):
            with self.subTest(patch=bad[:50]):
                self.assertTrue(patch_audit(bad)["flags"])

    def test_complete_pilot_scores_and_tool_telemetry(self):
        primary = results(PILOT_IDS, {"rich_4077", "rich_3905", "rich_3518"})
        control = results(PILOT_IDS, {"rich_4077", "rich_3905"})
        ptext = printout(PILOT_IDS, use_skill=True)
        ctext = printout(PILOT_IDS)
        out = compare(primary, control, stage="pilot",
                      primary_printout=ptext, control_printout=ctext)
        self.assertEqual(out["delta_resolved"], 1)
        self.assertEqual(out["gained"], ["rich_3518"])
        self.assertEqual(out["primary"]["resolved"], 3)
        self.assertTrue(out["single_run_screen_passed"])
        self.assertFalse(out["promotion_approved"])
        self.assertEqual(out["tool_telemetry"]["totals"]["recovery_skill_calls"], 1)

    def test_repeated_edits_and_missing_tool_trace_block_screen(self):
        primary = results(PILOT_IDS, {"rich_4077", "rich_3905", "rich_3518"})
        control = results(PILOT_IDS, {"rich_4077", "rich_3905"})
        out = compare(primary, control, stage="pilot",
                      primary_printout=printout(PILOT_IDS, failures=4,
                                              skip_last=True))
        self.assertIn("partial_tool_trace", out["single_run_blockers"])
        self.assertIn("repeated_edit_failure_loop", out["single_run_blockers"])

    def test_rereading_same_file_does_not_clear_edit_failure_streak(self):
        txt = printout(PILOT_IDS, failures=4)
        txt = txt.replace(
            "  [ 2.2s] ✏️ edit_file (pkg/module.py)",
            "  [ 2.15s] 📖 read_file (pkg/module.py)\n"
            "  [ 2.2s] ✏️ edit_file (pkg/module.py)",
            1,
        )
        telemetry = tool_events(txt, stage="pilot")
        self.assertGreaterEqual(
            telemetry["by_task"][PILOT_IDS[0]]["max_failed_edit_streak"], 4
        )

    def test_missing_arguments_detected_without_claiming_tool_enforcement(self):
        txt = printout(PILOT_IDS)
        txt = txt.replace(
            "  [ 4.0s]   ✓ {'status': 'ok', 'filepath': 'pkg/module.py'}",
            "  [ 4.0s] ❌ {'error': 'mandatory input parameters are not present'}",
            1)
        tel = tool_events(txt, stage="pilot")
        self.assertEqual(tel["totals"]["missing_arguments"], 1)
        self.assertEqual(tel["totals"]["edit_errors"], 1)

    def test_s1_task_parity_and_no_false_promotion(self):
        base = results(S1_IDS, {"rich_4077"})
        primary = results(S1_IDS, {"rich_4077", "rich_3905", "rich_3518",
                                   "rich_3454", "rich_3882"})
        out = compare(primary, base, stage="s1",
                      primary_printout=printout(S1_IDS))
        self.assertTrue(out["single_run_screen_passed"], out["single_run_blockers"])
        self.assertFalse(out["promotion_approved"])
        broken = primary[:-1]
        with self.assertRaisesRegex(ValueError, "Task set mismatch"):
            summarize(broken, stage="s1")
        duplicate = primary + [dict(primary[0])]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(duplicate, stage="s1")

    def test_missing_tool_transcript_blocks_screen_but_keeps_scores(self):
        primary = results(PILOT_IDS, {"rich_4077", "rich_3518", "rich_3905"})
        control = results(PILOT_IDS, {"rich_4077", "rich_3905"})
        report = compare(primary, control, stage="pilot")
        self.assertEqual(report["delta_resolved"], 1)
        self.assertFalse(report["single_run_screen_passed"])
        self.assertIn("tool_traces_not_supplied", report["single_run_blockers"])
        self.assertFalse(report["promotion_approved"])

    def test_empty_patch_or_agent_error_blocks(self):
        primary = results(PILOT_IDS, {"rich_4077", "rich_3905", "rich_3518"})
        control = results(PILOT_IDS, {"rich_4077"})
        primary[2]["agent_patch"] = ""
        primary[3]["agent_patch"] = ""
        primary[4]["error"] = "InvalidArguments"
        out = compare(primary, control, stage="pilot",
                      primary_printout=printout(PILOT_IDS))
        self.assertIn("too_many_empty_patches", out["single_run_blockers"])
        self.assertIn("agent_errors", out["single_run_blockers"])


if __name__ == "__main__":
    unittest.main()
