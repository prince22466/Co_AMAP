"""S31 deterministic package and trace-gate tests, not E4B scoring."""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest
import zipfile

from build_submission31 import (ARCHIVE, FILES, SOURCE, GRAPH_TOOLS,
                                IMPLEMENTER_TOOLS, VERIFIER_TOOLS,
                                IncludeLoader, build, validate)
from compare_submission24 import S1_IDS
from gate_submission31 import (PILOT_IDS, S29_WINS, S30_NEW_WINS,
                               assess, inspect_verification, inspect_forbidden_tools)
import yaml

DIAG = Path(__file__).resolve().parents[1] / "diagnostics"


def synthetic(ids, solved, *, graph_count=0, checks=0, empty=(),
              extra_events=None):
    """Generate an explicit scored printed-trace fixture."""
    out = []
    for i, name in enumerate(ids, 1):
        out.append(f"[candidate {i}/{len(ids)}] Evaluating {name} (repo)")
        out.append("  [   0.1s] 📖 read_file (source.py)")
        if i <= graph_count:
            out.append("  [   0.3s] 🔗 get_code_neighbors (package.Class)")
        if name not in empty:
            out.append("  [   1.0s] ✏️ edit_file (source.py)")
            out.append("  [   1.1s]   ✓ {'status':'ok'}")
        if i <= checks:
            out.append("  [   1.3s] 🔧 run_command python -c 'assert 2 == 2'")
        if extra_events and i == 1:
            out.extend(extra_events)
        out.append("  [   1.8s] 🤖 submit_patch")
        out.append(f"  -> resolved={name in solved}, "
                   f"patch_chars={0 if name in empty else 600}, tools=5, "
                   "duration=9, input=400, output=80")
    return "\n".join(out) + "\n"


class Submission31Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checked_in_archive = ARCHIVE.read_bytes()
        cls.manifest = build()

    def test_archived_bytes_match_builder_and_are_deterministic(self):
        data = ARCHIVE.read_bytes()
        self.assertEqual(self.checked_in_archive, data,
                         "Committed S31 ZIP differs from built source")
        self.assertEqual(self.manifest["archive_sha256"],
                         hashlib.sha256(data).hexdigest())
        self.assertEqual(self.manifest["archive_sha256"],
                         build()["archive_sha256"])
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(set(z.namelist()), set(FILES))
            for name in FILES:
                self.assertEqual(z.read(name), (SOURCE / name).read_bytes())

    def test_three_stage_tool_permissions_and_no_write_file(self):
        root = yaml.load((SOURCE / "agent.yaml").read_bytes(), Loader=IncludeLoader)
        agents = [yaml.load((SOURCE / "agents" / f"{name}.yaml").read_bytes(),
                            Loader=IncludeLoader)
                  for name in ("navigator", "implementer", "verifier")]
        self.assertEqual(root["agent_class"], "SequentialAgent")
        self.assertEqual(tuple(agents[0]["tools"]), GRAPH_TOOLS)
        self.assertEqual(tuple(agents[1]["tools"]), IMPLEMENTER_TOOLS)
        self.assertEqual(tuple(agents[2]["tools"]), VERIFIER_TOOLS)
        self.assertTrue(all("write_file" not in x["tools"] for x in agents))
        self.assertTrue(all("submit_patch" not in x["tools"] for x in agents[:2]))
        self.assertEqual(sum("submit_patch" in x["tools"] for x in agents), 1)
        self.assertEqual(agents[0]["output_key"], "navigation_evidence")
        self.assertEqual(agents[1]["output_key"], "implementation_evidence")

    def test_validate_rejects_write_and_misconfigured_order(self):
        original = {name: (SOURCE / name).read_bytes() for name in FILES}
        validate(original)
        unauthorized = dict(original)
        unauthorized["agents/implementer.yaml"] = unauthorized[
            "agents/implementer.yaml"].replace(
            b"  - edit_file\n", b"  - edit_file\n  - write_file\n")
        with self.assertRaises(ValueError):
            validate(unauthorized)
        ordered = dict(original)
        ordered["agent.yaml"] = ordered["agent.yaml"].replace(
            b"config_path: agents/navigator.yaml",
            b"config_path: agents/verifier.yaml", 1)
        with self.assertRaises(ValueError):
            validate(ordered)
        authorized = dict(original)
        authorized["agents/verifier.yaml"] = authorized[
            "agents/verifier.yaml"].replace(
            b"  - submit_patch\n", b"  - edit_file\n  - submit_patch\n")
        with self.assertRaises(ValueError):
            validate(authorized)

    def test_real_s30_failures_cannot_pass_as_s31(self):
        s29 = (DIAG / "submission-29-zip_s1TasksPrintouts.txt")
        s30 = (DIAG / "submission-30-zip_s1TasksPrintouts.txt")
        self.assertTrue(s29.is_file(), "S29 diagnostics must be available")
        self.assertTrue(s30.is_file(), "S30 diagnostics must be available")
        result = assess(s29.read_text(), s30.read_text(),
                        s30.read_text(), "s1")
        self.assertFalse(result["pass_printed_trace_triage"])
        self.assertIn("below_5_resolved", result["failures"])
        self.assertIn("lost_S29_win", result["failures"])
        self.assertIn("graph_tools_not_exercised", result["failures"])
        self.assertIn("oversized_patch:fastapi_14258", result["failures"])
        self.assertIn("write_file_was_invoked", result["failures"])

    def test_complete_five_wins_with_graph_and_checks_passes_triage(self):
        s29 = synthetic(S1_IDS, S29_WINS)
        s30 = synthetic(S1_IDS, {"rich_3905", "rich_3454"} | S30_NEW_WINS)
        wins = set(S29_WINS) | set(S30_NEW_WINS)
        cand = synthetic(S1_IDS, wins, graph_count=4, checks=5)
        result = assess(s29, s30, cand, "s1")
        self.assertTrue(result["pass_printed_trace_triage"], result["failures"])
        self.assertEqual(result["resolved"], 5)
        self.assertFalse(result["promotion_approved"])
        self.assertGreaterEqual(result["graph"]["tasks_using_graph"], 3)
        self.assertEqual(result["verification_attempts"]["tasks_with_test_attempt"], 5)

    def test_reject_lost_wins_missing_graphs_checks_and_bad_counts(self):
        s29 = synthetic(S1_IDS, S29_WINS)
        s30 = synthetic(S1_IDS, {"rich_3905", "rich_3454"} | S30_NEW_WINS)
        wins = set(S29_WINS) | set(S30_NEW_WINS)
        candidate = synthetic(S1_IDS, wins, graph_count=4, checks=5)
        cases = [
            ("lost_S29_win", synthetic(S1_IDS, (wins - {"rich_3518"}) | {"rich_4076"},
                                        graph_count=4, checks=5)),
            ("lost_both_S30_gains", synthetic(S1_IDS, (wins - S30_NEW_WINS) |
                                            {"rich_4076", "rich_3934"},
                                            graph_count=4, checks=5)),
            ("graph_tools_not_exercised", synthetic(S1_IDS, wins, checks=5)),
            ("too_few_targeted_test_attempts", synthetic(S1_IDS, wins,
                                                         graph_count=4)),
            ("candidate_task_set_mismatch", synthetic(S1_IDS[:-1], wins,
                                                       graph_count=4, checks=5)),
            ("write_file_was_invoked", synthetic(S1_IDS, wins, graph_count=4,
                                                 checks=5, extra_events=[
                "  [   1.2s] 📝 write_file (fastapi/routing.py, 2 lines)"
            ])),
            ("unsafe_shell:fastapi_14258", synthetic(S1_IDS, wins,
                  graph_count=4, checks=5, extra_events=[
                "  [   0.8s] 🔧 run_command sed -i '1iBad' code.py"
            ])),
            ("edit_failure_loop:fastapi_14258", synthetic(S1_IDS, wins,
                  graph_count=4, checks=5, extra_events=[
                line for _ in range(9) for line in
                ("  [   0.9s] ✏️ edit_file (source.py)",
                 "  [   0.9s]   ❌ {'error': 'old_string not found'}")
            ])),
        ]
        for failure, raw in cases:
            with self.subTest(failure=failure):
                self.assertIn(failure,
                    assess(s29, s30, raw, "s1")["failures"])
        self.assertTrue(assess(s29, s30, candidate)["pass_printed_trace_triage"])

    def test_pilot_requires_five_of_seven_and_retention(self):
        s29 = synthetic(S1_IDS, S29_WINS)
        s30 = synthetic(S1_IDS, {"rich_3905", "rich_3454"} | S30_NEW_WINS)
        self.assertEqual(len(PILOT_IDS), 7)
        candidate = synthetic(PILOT_IDS, S29_WINS | S30_NEW_WINS,
                              graph_count=3, checks=4)
        self.assertTrue(assess(s29, s30, candidate, "pilot")[
            "pass_printed_trace_triage"])
        fewer = synthetic(PILOT_IDS, S29_WINS | {"rich_4077"},
                          graph_count=3, checks=4)
        self.assertIn("below_5_resolved",
                      assess(s29, s30, fewer, "pilot")["failures"])

    def test_verification_must_follow_a_successful_edit_and_precede_submission(self):
        s29 = synthetic(S1_IDS, S29_WINS)
        s30 = synthetic(S1_IDS, {"rich_3905", "rich_3454"} | S30_NEW_WINS)
        wins = set(S29_WINS) | set(S30_NEW_WINS)
        valid = synthetic(S1_IDS, wins, graph_count=4, checks=5)
        self.assertTrue(assess(s29, s30, valid)["pass_printed_trace_triage"])

        before_edit = valid.replace(
            "  [   1.0s] ✏️ edit_file (source.py)\\n"
            "  [   1.1s]   ✓ {'status':'ok'}\\n"
            "  [   1.3s] 🔧 run_command python -c 'assert 2 == 2'",
            "  [   0.5s] 🔧 run_command python -c 'assert 2 == 2'\\n"
            "  [   1.0s] ✏️ edit_file (source.py)\\n"
            "  [   1.1s]   ✓ {'status':'ok'}",
        )
        self.assertIn("too_few_targeted_test_attempts",
                      assess(s29, s30, before_edit)["failures"])
        after_submit = valid.replace(
            "  [   1.3s] 🔧 run_command python -c 'assert 2 == 2'\\n"
            "  [   1.8s] 🤖 submit_patch",
            "  [   1.8s] 🤖 submit_patch\\n"
            "  [   2.1s] 🔧 run_command python -c 'assert 2 == 2'",
        )
        self.assertIn("too_few_targeted_test_attempts",
                      assess(s29, s30, after_submit)["failures"])

        failed_edit = valid.replace(
            "  [   1.1s]   ✓ {'status':'ok'}",
            "  [   1.1s]   ❌ {'status':'error'}",
        )
        self.assertIn("too_few_targeted_test_attempts",
                      assess(s29, s30, failed_edit)["failures"])

    def test_qualification_is_not_behavioral_proof(self):
        raw = synthetic(S1_IDS, set(), graph_count=5, checks=6)
        self.assertEqual(inspect_verification(raw)[
            "tasks_with_test_attempt"], 6)
        self.assertEqual(inspect_forbidden_tools(raw)[
            "write_file_task_ids"], [])


if __name__ == "__main__":
    unittest.main()
