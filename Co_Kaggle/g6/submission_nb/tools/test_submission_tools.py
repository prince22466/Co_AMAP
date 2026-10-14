"""Meaningful local tests; no Gemma weights, runtime package, or network required."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from submission_tools import (
    ALLOWED_MODELS, CORE_TOOLS, MODEL, ValidationError, build_submission, main, make_report,
    trace_metrics, validate_submission,
)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.root = self.workspace / "agent"
        self.root.mkdir()
        (self.root / "prompts").mkdir()
        (self.root / "prompts" / "system.md").write_text("Inspect, edit, verify, submit.\n", encoding="utf-8")
        self.base = {"name": "coder", "model": MODEL, "instruction": "!include prompts/system.md", "tools": sorted(CORE_TOOLS)}
        self.write_agent()

    def write_agent(self, **overrides):
        config = {**self.base, **overrides}
        text = yaml.safe_dump(config).replace("'!include prompts/system.md'", "!include prompts/system.md")
        (self.root / "agent.yaml").write_text(text, encoding="utf-8")

    def test_build_dependency_closure_and_root_archive_layout(self):
        (self.root / "sub_agents").mkdir()
        (self.root / "prompts" / "analyzer.md").write_text("Locate root cause.\n", encoding="utf-8")
        (self.root / "sub_agents" / "analyzer.yaml").write_text(
            f"name: code_analyzer_agent\nmodel: {MODEL}\ninstruction: !include ../prompts/analyzer.md\ntools: [read_file]\n", encoding="utf-8")
        skill = self.root / "skills" / "testing"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: testing\n---\nUse focused checks.\n", encoding="utf-8")
        (skill / "check.py").write_text("print('ok')\n", encoding="utf-8")
        (skill / "reference.txt").write_text("Resource.\n", encoding="utf-8")
        (self.root / "README.md").write_text("Local development documentation.\n", encoding="utf-8")
        (self.root / "eval_config.yaml").write_text("evaluation:\n  max_time_minutes: 4\n  max_tool_calls: 40\n", encoding="utf-8")
        self.write_agent(skills=["skills/testing"], tools=[*sorted(CORE_TOOLS), {"agent_tool": {"config_path": "sub_agents/analyzer.yaml", "skip_summarization": True}}])
        output = self.workspace / "submission.zip"
        result = build_submission(self.root, output)
        with zipfile.ZipFile(output) as archive:
            names = set(archive.namelist())
            self.assertIn("agent.yaml", names)
            self.assertIn("eval_config.yaml", names)
            self.assertIn("prompts/analyzer.md", names)
            self.assertIn("skills/testing/reference.txt", names)
            self.assertIn("skills/testing/check.py", names)
            self.assertNotIn("README.md", names)
            self.assertTrue(all(not name.startswith("agent/") for name in names))
        self.assertEqual(result["adapters"], [])
        self.assertEqual(result["agent_tool_names"], ["code_analyzer_agent"])
        self.assertEqual(result["omitted_files"], ["README.md"])

    def test_rejects_escaping_include_and_windows_paths(self):
        (self.workspace / "outside.md").write_text("outside", encoding="utf-8")
        for ref in ("../outside.md", "..\\outside.md", "C:\\outside.md", "/outside.md"):
            with self.subTest(ref=ref):
                (self.root / "agent.yaml").write_text(f"name: coder\nmodel: {MODEL}\ninstruction: !include '{ref}'\n", encoding="utf-8")
                with self.assertRaises(ValidationError):
                    validate_submission(self.root)

    def test_rejects_include_cycle_and_agent_cycle(self):
        (self.root / "agent.yaml").write_text("!include agent.yaml\n", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "cycle"):
            validate_submission(self.root)
        self.write_agent(tools=[{"agent_tool": {"config_path": "agent.yaml"}}])
        with self.assertRaisesRegex(ValidationError, "cycle"):
            validate_submission(self.root)

    def test_rejects_unsupported_tools_model_generation_and_binary(self):
        scenarios = [
            {"tools": ["internet_tool"]}, {"model": "other-model"},
            {"generate_content_config": {"http_options": {}}},
            {"generate_content_config": {"max_output_tokens": 32769}},
            {"generate_content_config": {"thinking_config": {"thinking_budget": -1}}},
            {"python_entrypoint": "agent.py"},
        ]
        for overrides in scenarios:
            with self.subTest(overrides=overrides):
                self.write_agent(**overrides)
                with self.assertRaises(ValidationError):
                    validate_submission(self.root)
        self.write_agent()
        (self.root / "model.bin").write_bytes(b"not valid weights")
        with self.assertRaisesRegex(ValidationError, "extension"):
            validate_submission(self.root)

    def test_stray_standalone_other_model_is_rejected_before_runtime(self):
        (self.root / "unused_agent.yaml").write_text("name: unused\nmodel: other\n", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "Standalone agent"):
            validate_submission(self.root)

    def test_all_allowed_variants_can_be_shared_by_coder_and_helper(self):
        (self.root / "sub_agents").mkdir()
        for model in sorted(ALLOWED_MODELS):
            with self.subTest(model=model):
                (self.root / "sub_agents" / "helper.yaml").write_text(
                    f"name: helper\nmodel: {model}\ninstruction: Inspect code.\ntools: [read_file]\n", encoding="utf-8")
                self.write_agent(model=model, tools=["read_file", {"agent_tool": {"config_path": "sub_agents/helper.yaml"}}])
                manifest = validate_submission(self.root)
                self.assertEqual(manifest["model"], model)
                self.assertEqual(manifest["agents"], 2)

    def test_mixing_allowed_variants_is_rejected_even_for_unreferenced_agents(self):
        smallest = "gemma-4-e2b-it-qat-w4a16-ct"
        (self.root / "sub_agents").mkdir()
        (self.root / "sub_agents" / "helper.yaml").write_text(
            f"name: helper\nmodel: {smallest}\ninstruction: Inspect code.\ntools: [read_file]\n", encoding="utf-8")
        for tools in [["read_file"], ["read_file", {"agent_tool": {"config_path": "sub_agents/helper.yaml"}}]]:
            with self.subTest(tools=tools):
                self.write_agent(tools=tools)
                with self.assertRaisesRegex(ValidationError, "same model variant"):
                    validate_submission(self.root)

    def test_inline_helper_cannot_mix_an_allowed_variant(self):
        self.write_agent(tools=[{
            "name": "inline_helper", "model": "gemma-4-e2b-it-qat-w4a16-ct",
            "instruction": "Inspect code.", "tools": ["read_file"],
        }])
        with self.assertRaisesRegex(ValidationError, "same model variant"):
            validate_submission(self.root)

    def test_stale_adapter_assets_rejected_and_referenced_adapter_included(self):
        adapter = self.root / "adapters" / "trained"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"fixture-format-not-validated")
        with self.assertRaisesRegex(ValidationError, "Unreferenced adapter"):
            validate_submission(self.root)
        self.write_agent(adapter="trained")
        manifest = validate_submission(self.root)
        self.assertEqual(manifest["adapters"], ["trained"])
        self.assertIn("adapters/trained/adapter_model.safetensors", manifest["package_files"])
        # The local check deliberately does not claim PEFT/tensor compatibility.

    def test_symlinks_rejected_when_platform_permits_creation(self):
        link = self.root / "linked.md"
        try:
            link.symlink_to(self.root / "prompts" / "system.md")
        except OSError as exc:
            self.skipTest(f"Platform cannot create symlinks: {exc}")
        with self.assertRaisesRegex(ValidationError, "Symlink"):
            validate_submission(self.root)

    def test_build_output_inside_source_rejected(self):
        with self.assertRaisesRegex(ValidationError, "outside"):
            build_submission(self.root, self.root / "submission.zip")

    def test_cli_validate_and_build(self):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            status = main(["validate", "--submission-dir", str(self.root)])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(captured.getvalue())["valid"])
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["build", "--submission-dir", str(self.root), "--output", str(self.workspace / "submission.zip")])
        self.assertEqual(status, 0)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "results"
        self.root.mkdir()
        for name in ("patches", "logs", "test_outputs", "traces"):
            (self.root / name).mkdir()
        self.output = Path(self.temp.name) / "report"

    def write_results(self, rows, *, notebook=False):
        filename = "notebook_task_results.jsonl" if notebook else "task_results.jsonl"
        (self.root / filename).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def write_patch(self, identifier, content="diff --git a/code.py b/code.py\n"):
        (self.root / "patches" / f"{identifier}.patch").write_text(content, encoding="utf-8")

    def test_evidence_categories_recovered_patch_and_missing_metadata(self):
        rows = [
            {"instance_id": "solved", "resolved": True, "test_exit_code": 0},
            {"task_id": "timeout_patch", "resolved": False, "test_exit_code": 1, "duration_seconds": 300},
            {"id": "empty", "resolved": False, "test_exit_code": -1, "tool_calls": 4, "duration_seconds": 76.1},
            {"instance_id": "tests", "resolved": False, "test_exit_code": 1},
            {"instance_id": "unknown", "resolved": False},
            {"instance_id": "zero_exit", "resolved": False, "test_exit_code": 0},
        ]
        self.write_results(rows)
        for identifier in ("solved", "timeout_patch", "tests", "zero_exit"):
            self.write_patch(identifier)
        self.write_patch("empty", "")
        (self.root / "logs" / "observer_timeout_patch.log").write_text("ERROR:swegemma.agent_runner: Time budget exhausted; automatically extracting patch", encoding="utf-8")
        report = make_report(self.root, self.output)
        by_id = {task["instance_id"]: task for task in report["tasks"]}
        self.assertEqual(by_id["solved"]["failure_category"], "solved")
        self.assertEqual(by_id["timeout_patch"]["failure_category"], "timeout_with_patch")
        self.assertTrue(by_id["timeout_patch"]["patch_present"])
        self.assertEqual(by_id["timeout_patch"]["verification_status"], "failed")
        self.assertEqual(by_id["empty"]["failure_category"], "empty_patch")
        self.assertEqual(by_id["empty"]["stop_cause"], "unknown")
        self.assertIsNone(by_id["empty"]["agent_elapsed_seconds"])
        self.assertIsNone(by_id["unknown"]["patch_present"])
        self.assertIsNone(by_id["unknown"]["tool_calls"])
        self.assertEqual(by_id["tests"]["failure_category"], "verification_failure")
        self.assertEqual(by_id["zero_exit"]["failure_category"], "verification_failure")
        self.assertAlmostEqual(report["summary"]["resolution_rate"], 1 / 6)
        for filename in ("diagnostics.json", "diagnostics.csv", "diagnostics.md"):
            self.assertTrue((self.output / filename).is_file())
        self.assertNotIn("diff --git", (self.output / "diagnostics.json").read_text(encoding="utf-8"))

    def test_per_request_tokens_are_distinct_from_cumulative_and_free_calls(self):
        self.write_results([{"instance_id": "trace", "resolved": True}])
        self.write_patch("trace")
        trace = {
            "final_metrics": {"total_prompt_tokens": 9000, "total_completion_tokens": 1500},
            "steps": [
                {"source": "agent", "elapsed_seconds": 1, "metrics": {"prompt_tokens": 1000, "completion_tokens": 100},
                 "tool_calls": [{"tool_call_id": "s", "function_name": "get_status"}, {"tool_call_id": "a", "function_name": "code_analyzer_agent"}]},
                {"source": "agent", "elapsed_seconds": 3, "metrics": {"prompt_tokens": 2000, "completion_tokens": 200},
                 "tool_calls": [{"tool_call_id": "e", "function_name": "edit_file"}]},
                {"source": "tool", "observation": {"results": [{"source_call_id": "e", "content": '{"status":"ok"}'}]}},
                {"source": "agent", "elapsed_seconds": 6, "metrics": {"prompt_tokens": 1800, "completion_tokens": 50},
                 "tool_calls": [{"tool_call_id": "p", "function_name": "submit_patch"}]},
            ],
        }
        (self.root / "traces" / "trace_trace.json").write_text(json.dumps(trace), encoding="utf-8")
        task = make_report(self.root, self.output)["tasks"][0]
        self.assertEqual(task["model_requests"], 3)
        self.assertEqual(task["traced_tool_calls"], 4)
        self.assertEqual(task["known_budgeted_tool_calls"], 1)
        self.assertEqual(task["analyzer_calls"], 1)
        self.assertEqual(task["tool_calls_by_name"]["edit_file"], 1)
        self.assertEqual(task["max_request_input_tokens"], 2000)
        self.assertEqual(task["input_tokens_sum_observed"], 4800)
        self.assertEqual(task["trace_cumulative_token_metrics"]["total_prompt_tokens"], 9000)
        self.assertEqual(task["first_successful_edit_trace_seconds"], 3)

    def test_cumulative_only_missing_malformed_unsupported_traces_are_not_zero_usage(self):
        self.write_results([{"id": identifier, "resolved": False} for identifier in ("cumulative", "missing", "malformed", "unsupported")])
        (self.root / "traces" / "trace_cumulative.json").write_text(json.dumps({"steps": [{"source": "agent", "message": "done"}], "final_metrics": {"total_prompt_tokens": 50000}}), encoding="utf-8")
        (self.root / "traces" / "trace_malformed.json").write_text("{broken", encoding="utf-8")
        (self.root / "traces" / "trace_unsupported.json").write_text(json.dumps({"steps": [{"mystery": "event"}]}), encoding="utf-8")
        tasks = {task["instance_id"]: task for task in make_report(self.root, self.output)["tasks"]}
        for identifier in tasks:
            if identifier != "cumulative":
                self.assertIsNone(tasks[identifier]["reviewer_calls"])
            self.assertIsNone(tasks[identifier]["max_request_input_tokens"])
            self.assertIsNone(tasks[identifier]["input_tokens_sum_observed"])
        self.assertIsNone(tasks["missing"]["model_requests"])
        self.assertFalse(tasks["malformed"]["trace_valid"])
        self.assertIsNone(tasks["unsupported"]["traced_tool_calls"])
        self.assertFalse(tasks["unsupported"]["trace_schema_supported"])
        self.assertEqual(tasks["cumulative"]["model_requests"], 1)
        markdown = (self.output / "diagnostics.md").read_text(encoding="utf-8")
        self.assertIn("Unrecognized step schema", markdown)

    def test_reviewer_delegation_is_distinct_from_analyzer_and_budgeted_builtins(self):
        self.write_results([{"instance_id": "review", "resolved": False}])
        trace = {"steps": [{"source": "agent", "tool_calls": [
            {"id": "a", "name": "code_analyzer_agent"},
            {"id": "r", "name": "patch_reviewer_agent"},
            {"id": "s", "name": "get_status"},
            {"id": "e", "name": "edit_file"},
        ]}]}
        (self.root / "traces" / "trace_review.json").write_text(json.dumps(trace), encoding="utf-8")
        task = make_report(self.root, self.output)["tasks"][0]
        self.assertEqual(task["analyzer_calls"], 1)
        self.assertEqual(task["reviewer_calls"], 1)
        self.assertEqual(task["known_budgeted_tool_calls"], 1)
        self.assertFalse(any("Unclassified tool names" in warning for warning in task["trace_warnings"]))
        self.assertIn("reviewer_calls", (self.output / "diagnostics.csv").read_text(encoding="utf-8").splitlines()[0])

    def test_locator_and_verifier_delegation_counts_use_trace_evidence(self):
        self.write_results([{"instance_id": "helpers", "resolved": False}, {"instance_id": "missing", "resolved": None}])
        trace = {"steps": [{"source": "agent", "tool_calls": [
            {"id": "l", "name": "repository_locator_agent"},
            {"id": "v", "name": "test_verifier_agent"},
            {"id": "r", "name": "run_command", "arguments": {"command": "pytest -q tests/test_case.py"}},
        ]}]}
        (self.root / "traces" / "trace_helpers.json").write_text(json.dumps(trace), encoding="utf-8")
        candidate = Path(__file__).resolve().parents[1] / "submission_4"
        report = make_report(self.root, self.output, candidate)
        tasks = {task["instance_id"]: task for task in report["tasks"]}
        task = tasks["helpers"]
        self.assertEqual(task["locator_calls"], 1)
        self.assertEqual(task["verifier_calls"], 1)
        self.assertEqual(task["helper_calls_by_name"]["repository_locator_agent"], 1)
        self.assertEqual(task["helper_calls_by_name"]["test_verifier_agent"], 1)
        self.assertEqual(task["analyzer_calls"], 0)
        self.assertEqual(task["reviewer_calls"], 0)
        self.assertEqual(task["known_budgeted_tool_calls"], 1)
        self.assertEqual(task["test_command_calls_observed"], 1)
        self.assertFalse(any("Unclassified tool names" in warning for warning in task["trace_warnings"]))
        for field in ["locator_calls", "verifier_calls", "helper_calls_by_name"]:
            self.assertIsNone(tasks["missing"][field])
        self.assertIn("verifier_calls", (self.output / "diagnostics.csv").read_text(encoding="utf-8").splitlines()[0])

    def test_failure_of_edit_not_success_and_missing_time_not_invented(self):
        path = self.root / "traces" / "trace_edit.json"
        trace = {"steps": [
            {"source": "agent", "elapsed_seconds": 5, "tool_calls": [{"id": "e", "name": "edit_file"}]},
            {"source": "tool", "tool_call_id": "e", "content": '{"status":"error","error_type":"FileEditError"}'},
            {"source": "agent", "tool_calls": [{"id": "w", "name": "write_file", "result": {"status": "ok"}}]},
        ]}
        path.write_text(json.dumps(trace), encoding="utf-8")
        metrics = trace_metrics(path, set())
        self.assertEqual(metrics["tool_errors"], 1)
        self.assertIsNone(metrics["first_successful_edit_trace_seconds"])

    def test_notebook_records_preferred_duplicate_and_invalid_lines_reported(self):
        self.write_results([{"id": "old", "resolved": False}])
        self.write_results([{"id": "new", "resolved": False}, {"id": "new", "resolved": True}], notebook=True)
        with (self.root / "notebook_task_results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("broken\n")
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["report", "--results-dir", str(self.root), "--output-dir", str(self.output)])
        self.assertEqual(status, 0)
        report = json.loads((self.output / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["finished_task_records"], 1)
        self.assertEqual(report["tasks"][0]["instance_id"], "new")
        self.assertTrue(report["tasks"][0]["resolved"])
        self.assertEqual(len(report["warnings"]), 2)

    def test_unknown_resolution_does_not_claim_an_overall_rate(self):
        self.write_results([{"id": "known", "repo": "project", "resolved": True}, {"id": "unknown", "repo": "project"}])
        summary = make_report(self.root, self.output)["summary"]
        self.assertIsNone(summary["resolution_rate"])
        self.assertEqual(summary["resolution_rate_known_records"], 1)
        self.assertIsNone(summary["per_repository"]["project"]["resolution_rate"])
        self.assertEqual(summary["per_repository"]["project"]["known_resolution_count"], 1)

    def test_terminal_metadata_beats_quoted_or_recovered_log_failure(self):
        self.write_results([{"id": "submitted", "resolved": False, "test_exit_code": 1, "stop_reason": "submitted"},
                            {"id": "quoted", "resolved": False, "test_exit_code": 1}])
        for identifier in ("submitted", "quoted"):
            self.write_patch(identifier)
            (self.root / "logs" / f"{identifier}.log").write_text("Model thought: avoid maximum context length.\nWARNING: earlier context window exceeded but recovered.\n", encoding="utf-8")
        # Submitted is explicit; unqualified model thoughts cannot establish a stop.
        (self.root / "logs" / "quoted.log").write_text("Model thought: avoid maximum context length.\n", encoding="utf-8")
        tasks = {task["instance_id"]: task for task in make_report(self.root, self.output)["tasks"]}
        self.assertEqual(tasks["submitted"]["stop_cause"], "submitted")
        self.assertEqual(tasks["submitted"]["failure_category"], "verification_failure")
        self.assertEqual(tasks["quoted"]["stop_cause"], "unknown")
        self.assertEqual(tasks["quoted"]["failure_category"], "verification_failure")

    def test_no_patch_sentinel_is_not_a_patch(self):
        self.write_results([{"id": "record", "resolved": False, "prediction": "NO_PATCH"},
                            {"id": "file", "resolved": False}])
        self.write_patch("file", "NO_PATCH\n")
        tasks = make_report(self.root, self.output)["tasks"]
        self.assertTrue(all(task["patch_present"] is False for task in tasks))
        self.assertTrue(all(task["failure_category"] == "empty_patch" for task in tasks))


if __name__ == "__main__":
    unittest.main()
