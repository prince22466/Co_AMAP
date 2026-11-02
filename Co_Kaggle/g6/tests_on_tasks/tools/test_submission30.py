"""S30 packaging and release-gate tests. They are NOT model evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest
import zipfile

from build_submission30 import (ARTIFACT, BASELINE, FILES, GRAPH, NATIVE,
                                SOURCE, build, validate, IncludeLoader)
from gate_submission30 import (PILOT_IDS, S29_WINS, assess, graph_usage)
from compare_submission24 import S1_IDS
import yaml

DIAGNOSTICS = Path(__file__).resolve().parents[1] / "diagnostics"


def synthetic(ids: tuple[str, ...], winners: set[str], *,
              graph: bool = False, empty: set[str] | None = None) -> str:
    lines = []
    for i, name in enumerate(ids, 1):
        lines.append(f"[candidate {i}/{len(ids)}] Evaluating {name} (repo)")
        lines.append("  [  1.0s] 📖 read_file (implementation.py)")
        if graph:
            lines.append("  [  1.5s] 🧩 search_similar_code (Symbol)")
            lines.append("  [  1.8s] 🧩 get_code_neighbors (package.Symbol)")
        if name not in (empty or set()):
            lines.append("  [  2.0s] ✏️ edit_file (implementation.py)")
            lines.append("  [  2.1s]   ✓ {'status': 'ok'}")
        lines.append("  [  3.0s] 🤖 submit_patch")
        patch = 0 if name in (empty or set()) else 400
        lines.append(f"  -> resolved={name in winners}, patch_chars={patch}, "
                     "tools=4, duration=8, input=100, output=40")
    return "\n".join(lines) + "\n"


class Submission30Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build()

    def test_zip_deterministic_complete_and_unchanged(self):
        data = ARTIFACT.read_bytes()
        self.assertEqual(self.manifest["archive_sha256"],
                         hashlib.sha256(data).hexdigest())
        self.assertEqual(self.manifest["archive_sha256"],
                         build()["archive_sha256"])
        self.assertEqual(data, ARTIFACT.read_bytes())
        with zipfile.ZipFile(ARTIFACT) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(sorted(archive.namelist()), sorted(FILES))
            for name in FILES:
                self.assertEqual(archive.read(name), (SOURCE / name).read_bytes())

    def test_only_graph_tools_and_instruction_changed(self):
        base = yaml.load((BASELINE / "agent.yaml").read_bytes(),
                         Loader=IncludeLoader)
        cur = yaml.load((SOURCE / "agent.yaml").read_bytes(),
                        Loader=IncludeLoader)
        self.assertEqual(tuple(base["tools"]), NATIVE)
        self.assertEqual(tuple(cur["tools"]), NATIVE + GRAPH)
        for name in ("name", "model", "include_contents", "instruction",
                     "generate_content_config"):
            self.assertEqual(cur[name], base[name])
        self.assertEqual((SOURCE / "eval_config.yaml").read_bytes(),
                         (BASELINE / "eval_config.yaml").read_bytes())
        self.assertFalse((SOURCE / "skills").exists())
        self.assertFalse((SOURCE / "sub_agents").exists())

    def test_validator_rejects_extra_tool_or_changed_budget(self):
        originals = {name: (SOURCE / name).read_bytes() for name in FILES}
        validate(originals)
        altered = dict(originals)
        altered["agent.yaml"] = altered["agent.yaml"].replace(
            b"  - get_code_subgraph\n",
            b"  - get_code_subgraph\n  - run_skill_script\n")
        with self.assertRaises(ValueError):
            validate(altered)
        altered = dict(originals)
        altered["eval_config.yaml"] = altered["eval_config.yaml"].replace(
            b"max_tool_calls: 40", b"max_tool_calls: 100")
        with self.assertRaises(ValueError):
            validate(altered)

    def test_real_s29_diagnostics_regression_gate_fails(self):
        path = DIAGNOSTICS / "submission-29-zip_s1TasksPrintouts.txt"
        self.assertTrue(path.is_file(), "S29 real trace MUST be present")
        raw = path.read_text(encoding="utf-8")
        report = assess(raw, raw, "s1")
        self.assertFalse(report["pass_printed_trace_gate"])
        self.assertIn("resolved_below_5", report["failures"])
        self.assertIn("too_many_empty_patches", report["failures"])

    def test_minimum_five_of_fifteen_retaining_three(self):
        base = synthetic(S1_IDS, set(S29_WINS))
        wins = set(S29_WINS) | {"rich_4077", "rich_4076"}
        candidate = synthetic(S1_IDS, wins, graph=True)
        result = assess(base, candidate, "s1")
        self.assertTrue(result["pass_printed_trace_gate"], result["failures"])
        self.assertEqual(result["candidate_resolved"], 5)
        self.assertEqual(result["gained_vs_s29"], ["rich_4076", "rich_4077"])
        self.assertFalse(result["promotion_approved"])
        self.assertEqual(result["graph_usage"]["tasks_using_graph"], 15)
        four = synthetic(S1_IDS, wins - {"rich_4076"})
        self.assertIn("resolved_below_5", assess(base, four, "s1")["failures"])
        lost = synthetic(S1_IDS, (wins - {"rich_3518"}) | {"rich_3718"})
        self.assertIn("lost_S29_confirmed_success",
                      assess(base, lost, "s1")["failures"])

    def test_pilot_predeclared_and_five_of_seven(self):
        self.assertEqual(len(PILOT_IDS), 7)
        base = synthetic(S1_IDS, set(S29_WINS))
        wins = set(S29_WINS) | {"rich_4077", "rich_4076"}
        candidate = synthetic(PILOT_IDS, wins)
        self.assertTrue(assess(base, candidate, "pilot")[
            "pass_printed_trace_gate"])
        incomplete = synthetic(PILOT_IDS[:-1], wins)
        self.assertIn("candidate_task_ids_mismatch",
                      assess(base, incomplete, "pilot")["failures"])

    def test_unsafe_shell_oversized_and_failed_edits_rejected(self):
        base = synthetic(S1_IDS, set(S29_WINS))
        wins = set(S29_WINS) | {"rich_4077", "rich_4076"}
        candidate = synthetic(S1_IDS, wins, graph=True)
        shell = candidate.replace(
            "  [  1.0s] 📖 read_file (implementation.py)",
            "  [  1.0s] 🔧 run_command sed -i '1iBAD' rich/console.py\n"
            "  [  1.1s] 📖 read_file (implementation.py)", 1)
        self.assertIn("unsafe_shell_mutation:fastapi_14258",
                      assess(base, shell, "s1")["failures"])
        edits = candidate.replace(
            "  [  1.0s] 📖 read_file (implementation.py)",
            "  [  1.0s] 📖 read_file (implementation.py)\n" +
            ("  [  1.5s] ✏️ edit_file (implementation.py)\n"
             "  [  1.6s]   ❌ {'status':'error'}\n") * 9, 1)
        self.assertIn("edit_failure_loop:fastapi_14258",
                      assess(base, edits, "s1")["failures"])

    def test_empty_patches_and_mismatched_tasks_rejected(self):
        base = synthetic(S1_IDS, set(S29_WINS))
        wins = set(S29_WINS) | {"rich_4077", "rich_4076"}
        empty = synthetic(S1_IDS, wins,
                          empty={"fastapi_14258", "rich_4075"})
        self.assertIn("too_many_empty_patches",
                      assess(base, empty, "s1")["failures"])
        mismatch = synthetic(S1_IDS[1:] + ("unknown_task",), wins)
        self.assertIn("candidate_task_ids_mismatch",
                      assess(base, mismatch, "s1")["failures"])

    def test_graph_call_count_not_equated_to_success(self):
        base = synthetic(S1_IDS, set(S29_WINS))
        no_wins = synthetic(S1_IDS, set(), graph=True)
        self.assertEqual(graph_usage(no_wins)["tool_calls"][
            "search_similar_code"], 15)
        self.assertIn("resolved_below_5",
                      assess(base, no_wins, "s1")["failures"])


if __name__ == "__main__":
    unittest.main()
