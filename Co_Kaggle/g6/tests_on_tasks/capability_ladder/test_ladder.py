"""CPU tests: no Gemma inference, official runtime, or GPU needed."""
from __future__ import annotations

import hashlib
from argparse import Namespace
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from probe import (EDIT_SCHEMA, ProbeError, make_case, model_probe, oracle_locations,
                   parse_json_edit, parse_tool_edit, pristine_source, read_tasks,
                   safe_path, validate_edit, run as run_probe)
from report import IncompleteEvidence, compare, read_rows, require_matching_environment
from verify import VerificationBlocked, load_probes, preflight_result, reference_control

REF = (
    "diff --git a/pkg/mod.py b/pkg/mod.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -2,1 +2,1 @@\n"
    "-x = 1\n"
    "+SOLUTION_CANARY_MUST_NOT_LEAK = 2\n"
)
SRC = "def value():\n    x = 1\n    return x\n"
TASK = {
    "instance_id": "rich_123", "repo": "Textualize/rich",
    "base_commit": "a" * 40, "problem_statement": "Fix value.",
    "hints_text": "In production code", "patch": REF,
    "test_patch": "TEST_PATCH_CANARY_MUST_NOT_LEAK",
}


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        archive = self.root / "rich_123.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            data = SRC.encode()
            info = tarfile.TarInfo("repo/pkg/mod.py")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        self.tasks = self.root / "tasks.jsonl"
        self.tasks.write_text(json.dumps(TASK) + "\n")

    def test_source_excerpt_never_exposes_reference_fix_or_tests(self):
        case = make_case(TASK, self.root)
        self.assertEqual(case["oracle_files"], ["pkg/mod.py"])
        self.assertIn("x = 1", case["message"])
        self.assertNotIn("SOLUTION_CANARY", case["message"])
        self.assertNotIn("TEST_PATCH_CANARY", case["message"])
        self.assertEqual(case["sources"]["pkg/mod.py"], SRC)
        self.assertEqual(oracle_locations(REF), [("pkg/mod.py", 2)])
        self.assertEqual(pristine_source(self.root / "rich_123.tgz", "pkg/mod.py"), SRC)

    def test_unsafe_sources_are_blocked(self):
        for value in ("../secret.py", "/etc/passwd", "pkg//mod.py", "pkg\\mod.py"):
            with self.subTest(value=value), self.assertRaises(ProbeError):
                safe_path(value)
        with self.assertRaises(ProbeError):
            pristine_source(self.root / "rich_123.tgz", "pkg/absent.py")
        with self.assertRaises(ProbeError):
            make_case({**TASK, "patch": "unstructured oracle info"}, self.root)
        with self.assertRaises(ProbeError):
            make_case({**TASK, "problem_statement": ""}, self.root)

    def test_exact_unique_edit_and_bounded_diff(self):
        src = {"pkg/mod.py": SRC}
        valid = {"filepath": "pkg/mod.py", "old_string": "x = 1",
                 "new_string": "x = 2"}
        path, diff = validate_edit(valid, src)
        self.assertEqual(path, "pkg/mod.py")
        self.assertIn("-    x = 1\n", diff)
        self.assertIn("+    x = 2\n", diff)
        self.assertIn("+++ b/pkg/mod.py", diff)
        self.assertEqual(SRC, src[path])
        bad = [
            {**valid, "filepath": "../pkg/mod.py"},
            {**valid, "filepath": "tests/test_mod.py"},
            {**valid, "old_string": "not found"},
            {**valid, "old_string": ""},
            {**valid, "new_string": "x = 1"},
            {**valid, "allow_multiple": True},
            {**valid, "new_string": "z" * 10001},
            {**valid, "extra": "unrecognized"},
            {**valid, "old_string": 0},
        ]
        for row in bad:
            with self.subTest(field=str(row)[:60]), self.assertRaises(ProbeError):
                validate_edit(row, src)
        with self.assertRaises(ProbeError):
            validate_edit({**valid, "old_string": "x"}, {"pkg/mod.py": "xx\n"})

    def test_model_response_parsing_is_strict(self):
        raw = '{"filepath":"pkg/mod.py","old_string":"x = 1","new_string":"x = 2"}'
        self.assertEqual(parse_json_edit(raw)["filepath"], "pkg/mod.py")
        with self.assertRaises((ValueError, ProbeError)):
            parse_json_edit(chr(96) * 3 + "json\n" + raw)
        call = {"function": {"name": "edit_file", "arguments": raw}}
        choice = {"message": {"tool_calls": [call]}}
        self.assertEqual(parse_tool_edit(choice)["new_string"], "x = 2")
        for item in (
            {"message": {"tool_calls": []}},
            {"message": {"tool_calls": [call, call]}},
            {"message": {"tool_calls": [{"function": {
                "name": "wrong", "arguments": raw}}]}},
        ):
            with self.subTest(item=item), self.assertRaises(ProbeError):
                parse_tool_edit(item)
        self.assertEqual(EDIT_SCHEMA["function"]["parameters"]["required"],
                         ["filepath", "old_string", "new_string"])

    def test_model_request_source_parity(self):
        case = make_case(TASK, self.root)
        answer = {"choices": [{"message": {"content": "{}"},
                               "finish_reason": "stop"}], "usage": {}}
        for level in ("reasoning", "tool"):
            with self.subTest(level=level), patch(
                "probe.post_json", return_value=answer) as mocked:
                result = model_probe(case, level, "http://localhost:8000/v1",
                                     "gemma-4-e4b-it-qat-w4a16-ct", "EMPTY", 2)
                self.assertEqual(result["raw"], answer)
                params = mocked.call_args.args[1]
                self.assertIn("x = 1", params["messages"][1]["content"])
                self.assertNotIn("SOLUTION_CANARY", json.dumps(params))
                self.assertEqual(bool(params.get("tools")), level == "tool")

    def test_end_to_end_generation_writes_unscored_evidence(self):
        output = self.root / "probe_out"
        response = {"choices": [{"message": {"content": json.dumps({
            "filepath": "pkg/mod.py", "old_string": "x = 1",
            "new_string": "x = 2"})},
            "finish_reason": "stop"}], "usage": {}}
        args = Namespace(
            level="reasoning", tasks=self.tasks, snapshots=self.root,
            model_url="http://localhost:8000/v1",
            served_model="gemma-4-e4b-it-qat-w4a16-ct", api_key="EMPTY",
            stage="pilot", task_ids=["rich_123"], seed=1, output=output)
        with patch("probe.verify_server"), patch("probe.post_json", return_value=response):
            run_probe(args)
        rows = [json.loads(x) for x in (output / "probes.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["edit_valid"])
        self.assertEqual(rows[0]["status"], "edit_applied_to_pristine")
        self.assertIsNone(rows[0]["resolved"])
        self.assertNotIn("SOLUTION_CANARY", rows[0]["patch"])
        self.assertTrue((output / "raw_model" / "rich_123.json").is_file())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertFalse(manifest["reference_patch_sent_to_model"])
        self.assertEqual(manifest["task_ids"], ["rich_123"])
        with self.assertRaises(ProbeError):
            run_probe(args)

    def test_out_of_range_reference_hunk_cannot_produce_empty_or_wrong_window(self):
        far = REF.replace("@@ -2,1 +2,1 @@", "@@ -9999,1 +9999,1 @@")
        with self.assertRaisesRegex(ProbeError, "outside original source"):
            make_case({**TASK, "patch": far}, self.root)

    def test_task_loader_rejects_duplicates(self):
        self.assertEqual(read_tasks(self.tasks)["rich_123"]["repo"], "Textualize/rich")
        self.tasks.write_text(json.dumps(TASK) + "\n" + json.dumps(TASK) + "\n")
        with self.assertRaises(ProbeError):
            read_tasks(self.tasks)


class VerificationTests(unittest.TestCase):
    def test_controls_require_real_negative_and_positive_verification(self):
        bad = SimpleNamespace(resolved=False, test_exit_code=1,
                              test_output="FAILED tests/test_mod.py", error=None)
        good = SimpleNamespace(resolved=True, test_exit_code=0,
                               test_output="1 passed", error=None)
        self.assertTrue(preflight_result(bad))
        self.assertTrue(reference_control(good))
        self.assertFalse(preflight_result(SimpleNamespace(
            resolved=False, test_exit_code=1,
            test_output="ModuleNotFoundError missing library", error=None)))
        self.assertFalse(preflight_result(SimpleNamespace(
            resolved=False, test_exit_code=1, test_output="", error=None)))
        self.assertFalse(reference_control(SimpleNamespace(
            resolved=True, test_exit_code=0, error="environment broken")))

    def test_edit_integrity_and_no_unverified_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probes.jsonl"
            diff = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-a\n+b\n"
            row = {
                "instance_id": "rich_123", "level": "reasoning",
                "edit_valid": True, "status": "edit_applied_to_pristine",
                "patch": diff, "resolved": None,
                "patch_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            }
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(load_probes(path, {"rich_123": {}})), 1)
            for mutation in (
                {"resolved": True}, {"patch_sha256": "wrong"},
                {"status": "failed"}, {"edit_valid": False},
            ):
                with self.subTest(mutation=mutation):
                    path.write_text(json.dumps({**row, **mutation}) + "\n")
                    with self.assertRaises(VerificationBlocked):
                        load_probes(path, {"rich_123": {}})


class ReportTests(unittest.TestCase):
    def test_matched_report_excludes_blocked(self):
        a = {
            "rich_1": {"instance_id": "rich_1", "level": "reasoning",
                       "scorable": True, "resolved": True},
            "rich_2": {"instance_id": "rich_2", "level": "reasoning",
                       "scorable": False, "resolved": None},
        }
        b = {
            "rich_1": {"instance_id": "rich_1", "level": "tool",
                       "scorable": True, "resolved": False},
            "rich_2": {"instance_id": "rich_2", "level": "tool",
                       "scorable": False, "resolved": None},
        }
        auto = {"rich_1": {"resolved": False}, "rich_2": {"resolved": False}}
        report = compare(a, b, auto)
        self.assertEqual(report["matched_scorable"], 1)
        self.assertEqual(report["level1_resolved"], 1)
        self.assertEqual(report["level2_resolved"], 0)
        self.assertEqual(report["categories"]["tool_interface_gap_candidate"], 1)
        with self.assertRaises(IncompleteEvidence):
            compare(a, {"rich_1": b["rich_1"]}, auto)
        with self.assertRaises(IncompleteEvidence):
            compare(a, {**b, "rich_1": {**b["rich_1"], "level": "reasoning"}}, auto)

    def test_manifest_pairing_rejects_seed_or_dataset_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            left, right = base / "l1", base / "l2"
            left.mkdir(); right.mkdir()
            a = {
                "tasks_sha256": "identical", "source_verifier_sha256": "verifier",
                "generator_seed": 1, "served_model": "gemma-4-e4b-it-qat-w4a16-ct",
                "versions": {"swegemma": "0.2.11"},
                "generator_level": "reasoning",
            }
            b = {**a, "generator_level": "tool"}
            (left / "manifest.json").write_text(json.dumps(a))
            (right / "manifest.json").write_text(json.dumps(b))
            report = require_matching_environment(
                left / "verified.jsonl", right / "verified.jsonl", None)
            self.assertEqual(report["generator_seed"], 1)
            (right / "manifest.json").write_text(
                json.dumps({**b, "generator_seed": 2}))
            with self.assertRaises(IncompleteEvidence):
                require_matching_environment(
                    left / "verified.jsonl", right / "verified.jsonl", None)

    def test_autonomous_seed_manifest_matches_paired_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            l1, l2, autonomous = base / "l1", base / "l2", base / "candidate_2"
            for folder in (l1, l2, autonomous):
                folder.mkdir()
            attrs = {
                "tasks_sha256": "same", "source_verifier_sha256": "code",
                "generator_seed": 1, "served_model": "gemma-4-e4b-it-qat-w4a16-ct",
                "versions": {"swegemma": "0.2.11"},
            }
            (l1 / "manifest.json").write_text(
                json.dumps({**attrs, "generator_level": "reasoning"}))
            (l2 / "manifest.json").write_text(
                json.dumps({**attrs, "generator_level": "tool"}))
            (base / "run_manifest.json").write_text(json.dumps({
                "tasks_sha256": "same", "served_model": attrs["served_model"],
                "sandbox": "docker",
                "runtime_versions": attrs["versions"],
            }))
            with self.assertRaisesRegex(IncompleteEvidence, "seed differs"):
                require_matching_environment(
                    l1 / "verified.jsonl", l2 / "verified.jsonl", autonomous)
            candidate1 = base / "candidate_1"
            candidate1.mkdir()
            matched = require_matching_environment(
                l1 / "verified.jsonl", l2 / "verified.jsonl", candidate1)
            self.assertEqual(matched["generator_seed"], 1)
            (base / "run_manifest.json").write_text(json.dumps({
                "tasks_sha256": "same", "served_model": attrs["served_model"],
                "sandbox": "docker",
                "runtime_versions": {"swegemma": "wrong"},
            }))
            with self.assertRaisesRegex(IncompleteEvidence, "wheel mismatch"):
                require_matching_environment(
                    l1 / "verified.jsonl", l2 / "verified.jsonl", candidate1)
            (base / "run_manifest.json").write_text(json.dumps({
                "tasks_sha256": "same", "served_model": attrs["served_model"],
                "sandbox": "subprocess",
            }))
            with self.assertRaisesRegex(IncompleteEvidence, "Docker"):
                require_matching_environment(
                    l1 / "verified.jsonl", l2 / "verified.jsonl", candidate1)

    def test_duplicate_autonomous_results_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            row = '{"instance_id":"rich_1","resolved":false}\n'
            path.write_text(row + row)
            with self.assertRaises(IncompleteEvidence):
                read_rows(path)


if __name__ == "__main__":
    unittest.main()
