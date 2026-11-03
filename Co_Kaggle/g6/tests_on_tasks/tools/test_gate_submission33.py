"""Synthetic diagnostic-gate regression tests; NOT Gemma task evaluations."""
from __future__ import annotations

import unittest

from compare_submission24 import S1_IDS
from gate_submission33 import S29_WINS, S30_GAINS, assess, summary_rows


def diagnostic(wins: set[str] | frozenset[str], *, empty: int = 0,
               oversized: str | None = None,
               include_detail: bool = True) -> str:
    """Minimal complete scored run; tool activity may be shown or missing."""
    if empty and not set(S1_IDS[:empty]).isdisjoint(wins):
        raise ValueError("Fixture cannot have a resolved task with empty patch")
    lines = []
    if include_detail:
        for index, task in enumerate(S1_IDS, 1):
            patch = (0 if index <= empty else
                     100001 if task == oversized else 100)
            lines.extend((
                f"[candidate {index}/15] Evaluating {task} (test)",
                "  [   0.1s] 🤖 submit_patch",
                (f"  -> resolved={task in wins}, patch_chars={patch}, "
                 "tools=1, duration=10.0, input=0, output=0"),
            ))
    lines.append(
        "instance_id\tresolved\tpatch_chars\tduration_seconds\ttotal_tokens")
    for index, task in enumerate(S1_IDS):
        patch = (0 if index < empty else
                 100001 if task == oversized else 100)
        lines.append(f"{index}\t{task}\t{task in wins}\t{patch}\t10.0\t0")
    return "\n".join(lines) + "\n"


class Submission33GateTests(unittest.TestCase):
    def setUp(self):
        self.baseline29 = diagnostic(S29_WINS, include_detail=False)
        self.baseline30 = diagnostic(
            set(S30_GAINS) | {"rich_3905", "rich_3454"},
            include_detail=False)
        self.five_wins = set(S29_WINS | S30_GAINS)

    def check_candidate(self, **options):
        return assess(self.baseline29, self.baseline30,
                      diagnostic(self.five_wins, **options))

    def test_scores_from_complete_summary_with_one_empty_patch(self):
        result = self.check_candidate(empty=1)
        self.assertTrue(result["triage_passed"], result["failures"])
        self.assertEqual(result["resolved"], 5)
        self.assertEqual(result["empty_patches"], 1)
        self.assertFalse(result["promotion_approved"])

    def test_two_empty_patches_fail(self):
        result = self.check_candidate(empty=2)
        self.assertFalse(result["triage_passed"])
        self.assertIn("excessive_empty_patches", result["failures"])

    def test_oversize_patch_fails(self):
        result = self.check_candidate(oversized="rich_3882")
        self.assertFalse(result["triage_passed"])
        self.assertIn("missing_or_oversize_patch:rich_3882", result["failures"])

    def test_missing_detail_fails_closed_without_atif(self):
        result = self.check_candidate(include_detail=False)
        self.assertFalse(result["triage_passed"])
        self.assertIn("incomplete_detailed_tool_events", result["failures"])

    def test_prior_win_retention_is_not_only_count(self):
        updated_wins = self.five_wins - {"rich_3518"} | {"rich_3934"}
        result = assess(self.baseline29, self.baseline30,
                        diagnostic(updated_wins))
        self.assertEqual(result["resolved"], 5)
        self.assertIn("previous_S29_wins_not_retained", result["failures"])

    def test_duplicate_summary_row_is_rejected(self):
        data = diagnostic(self.five_wins)
        duplicate = data + data.splitlines()[-1] + "\n"
        with self.assertRaisesRegex(ValueError, "Duplicate scored task"):
            summary_rows(duplicate)


if __name__ == "__main__":
    unittest.main()
