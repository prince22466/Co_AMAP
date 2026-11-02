"""S29 local unit/regression tests; not evidence of E4B task-solving."""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest
import zipfile

from build_submission29 import ARTIFACT, BASELINE, FILES, SOURCE, build
from gate_submission29 import assess, inspect_printout, PILOT_IDS

DIAGNOSTICS = Path(__file__).resolve().parents[1] / "diagnostics"


def synthetic(ids: list[str], wins: set[str], patch: int = 500) -> str:
    """Produce a minimal matched printed evaluation fixture."""
    lines = []
    for ix, name in enumerate(ids, 1):
        lines.extend((f"[candidate {ix}/{len(ids)}] Evaluating {name} (repo)",
                      f"  [ 1.0s] 📖 read_file (code.py)",
                      f"  [ 2.0s] ✏️ edit_file (code.py)",
                      "  [ 2.1s]   ✓ {'status': 'ok'}",
                      "  [ 2.2s] 🤖 submit_patch",
                      f"  -> resolved={name in wins}, patch_chars={patch}, "
                      "tools=3, duration=5, input=100, output=100"))
    return "\n".join(lines) + "\n"


class Submission29Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build()

    def test_archive_is_deterministic_and_reproduces_sources(self):
        first = ARTIFACT.read_bytes()
        self.assertEqual(self.manifest["archive_sha256"],
                         hashlib.sha256(first).hexdigest())
        self.assertEqual(build()["archive_sha256"], self.manifest["archive_sha256"])
        self.assertEqual(first, ARTIFACT.read_bytes())
        with zipfile.ZipFile(ARTIFACT) as z:
            self.assertEqual(sorted(z.namelist()), sorted(FILES))
            self.assertIsNone(z.testzip())
            for name in FILES:
                self.assertEqual(z.read(name), (SOURCE / name).read_bytes())

    def test_native_runtime_parity(self):
        for name in ("agent.yaml", "eval_config.yaml"):
            with self.subTest(name=name):
                self.assertEqual((SOURCE / name).read_bytes(),
                                 (BASELINE / name).read_bytes())
        self.assertFalse((SOURCE / "skills").exists())
        self.assertFalse((SOURCE / "sub_agents").exists())

    def test_contract_addresses_s28_failure_modes(self):
        text = (SOURCE / "instruction.md").read_text()
        for phrase in ("old_string", "new_string", "allow_multiple",
                       "start_line", "end_line", "sed -i",
                       "git diff --numstat", "git diff --check",
                       "requested behavior", "submit_patch"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_s28_real_printout_is_rejected(self):
        baseline = DIAGNOSTICS / "submission-27-zip_s1TasksPrintouts.txt.txt"
        candidate = DIAGNOSTICS / "submission-28-zip_s1TasksPrintouts.txt.txt"
        if not baseline.exists() or not candidate.exists():
            self.skipTest("Real S27/S28 printouts not mounted")
        outcome = assess(baseline.read_text(encoding="utf-8"),
                         candidate.read_text(encoding="utf-8"), "s1")
        self.assertFalse(outcome["pass_promotion_gate"])
        self.assertIn("lost_known_s27_win", outcome["failures"])
        self.assertIn("oversized_patch:fastapi_14258", outcome["failures"])
        self.assertIn("unsafe_shell_edit:fastapi_14258", outcome["failures"])
        self.assertIn("edit_failure_loop:rich_3718", outcome["failures"])
        self.assertIn("bad_tool_schema_loop:rich_3454", outcome["failures"])
        self.assertIn("repeat_command_loop:rich_3676", outcome["failures"])

    def test_pilot_accepts_only_matched_verified_synthetic_outcome(self):
        ids = sorted(PILOT_IDS)
        base = synthetic(ids, {"rich_3905", "rich_3454"})
        candidate = synthetic(ids, {"rich_3905", "rich_3454", "rich_3882"})
        result = assess(base, candidate, "pilot")
        self.assertTrue(result["pass_promotion_gate"], result["failures"])
        lost = synthetic(ids, {"rich_3905", "rich_3882", "rich_4077"})
        self.assertIn("lost_known_s27_win",
                      assess(base, lost, "pilot")["failures"])
        self.assertFalse(assess(base, candidate.replace(
            "Evaluating rich_4077", "Evaluating unexpected"), "pilot")[
            "pass_promotion_gate"])

    def test_safety_flags_and_no_false_positive_for_readonly_sed(self):
        ids = sorted(PILOT_IDS)
        base = synthetic(ids, {"rich_3905", "rich_3454"})
        candidate = synthetic(ids, {"rich_3905", "rich_3454", "rich_3882"})
        inserted = (
            "  [ 1.3s] 🔧 run_command sed -n '1,40p' rich/panel.py\n"
            "  [ 1.4s] 🔧 run_command sed -i '1iBAD' rich/panel.py\n"
            "  [ 1.5s] ✏️ edit_file (rich/panel.py)\n"
            "  [ 1.6s]   ❌ {'error': 'mandatory input parameters missing'}\n"
        )
        modified = candidate.replace(
            "  [ 2.0s] ✏️ edit_file (code.py)", inserted +
            "  [ 2.0s] ✏️ edit_file (code.py)", 1)
        outcome = assess(base, modified, "pilot")
        self.assertIn("unsafe_shell_edit:rich_3454", outcome["failures"])
        row = inspect_printout(modified)["tasks"][0]
        self.assertEqual(row["missing_edit_arguments"], 1)
        self.assertEqual(row["unsafe_shell_commands"],
                         ["sed -i '1iBAD' rich/panel.py"])

    def test_incomplete_result_cannot_pass(self):
        ids = sorted(PILOT_IDS)
        base = synthetic(ids, {"rich_3905", "rich_3454"})
        truncated = synthetic(ids[:4], {"rich_3905", "rich_3454", "rich_3882"})
        result = assess(base, truncated, "pilot")
        self.assertFalse(result["pass_promotion_gate"])
        self.assertTrue(any(x.startswith("incomplete_results") for x in result["failures"]))


if __name__ == "__main__":
    unittest.main()
