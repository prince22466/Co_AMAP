"""Offline integration checks for the portable evaluation workflow."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import nbformat


BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_evaluation_notebook", BASE / "prepare_evaluation_notebook.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class EvaluationWorkflowTests(unittest.TestCase):
    def comparison_namespace(self, root):
        output = root / "comparison.ipynb"
        workflow.generate(
            BASE / "submission_2", BASE / "tools" / "submission_tools.py",
            BASE.parent / "competition_material" / "getting-started-gemma-4-developer-agent.ipynb",
            output, baseline_dir=BASE / "submission_1",
        )
        notebook = nbformat.read(output, as_version=4)
        nbformat.validate(notebook)
        namespace = {}
        for cell in notebook.cells:
            if cell.cell_type == "code" and cell.source.startswith(
                ("from pathlib import Path", "# Curated text files", "def remove_agent_tools")
            ):
                exec(cell.source, namespace)
        namespace["WORKING_DIR"] = root / "working"
        return notebook, namespace

    def test_prompt_comparison_preserves_two_agents_and_only_changes_instructions(self):
        with tempfile.TemporaryDirectory(prefix="prompt_comparison_") as temporary:
            notebook, namespace = self.comparison_namespace(Path(temporary))
            self.assertEqual(notebook.metadata["comparison_mode"], "prompt_comparison")
            self.assertEqual(namespace["BENCHMARK_VARIANTS"], ["baseline", "candidate"])
            self.assertEqual(namespace["RELEASE_VARIANT"], "candidate")
            self.assertEqual(namespace["CANDIDATE_LABEL"], "submission_2")
            self.assertEqual(namespace["BASELINE_LABEL"], "submission_1")
            for files_key, hashes_key in [
                ("EMBEDDED_SUBMISSION", "EMBEDDED_SOURCE_HASHES"),
                ("EMBEDDED_BASELINE", "EMBEDDED_BASELINE_HASHES"),
            ]:
                for name, content in namespace[files_key].items():
                    self.assertEqual(hashlib.sha256(content.encode()).hexdigest(), namespace[hashes_key][name])
            changed = {
                name for name in namespace["EMBEDDED_BASELINE"]
                if namespace["EMBEDDED_BASELINE"][name] != namespace["EMBEDDED_SUBMISSION"][name]
            }
            self.assertEqual(changed, {"prompts/system.md", "sub_agents/analyzer_resources/instruction.md"})
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PREPARATION, namespace)
            manifests = namespace["VARIANT_MANIFESTS"]
            self.assertEqual(manifests["baseline"]["evaluation"], manifests["candidate"]["evaluation"])
            for variant in ["baseline", "candidate"]:
                self.assertEqual(manifests[variant]["agents"], 2)
                self.assertEqual(len(manifests[variant]["package_files"]), 7)
                self.assertEqual(manifests[variant]["agent_tool_names"], ["code_analyzer_agent"])
                self.assertIn("agent_tool", (namespace["VARIANT_DIRS"][variant] / "agent.yaml").read_text())

    def test_prompt_comparison_rejects_budget_changes_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, namespace = self.comparison_namespace(Path(temporary))
            namespace["EMBEDDED_SUBMISSION"]["eval_config.yaml"] = namespace["EMBEDDED_SUBMISSION"]["eval_config.yaml"].replace(
                "max_tool_calls: 40", "max_tool_calls: 41"
            )
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "identical evaluation budgets"):
                exec(workflow.PREPARATION, namespace)

    def test_prompt_comparison_rejects_sampling_or_tool_changes(self):
        changes = [
            ("configs/sampling.yaml", "temperature: 0.2", "temperature: 0.3"),
            ("agent.yaml", "  - get_code_subgraph\n", ""),
        ]
        for filename, before, after in changes:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                _, namespace = self.comparison_namespace(Path(temporary))
                namespace["EMBEDDED_SUBMISSION"][filename] = namespace["EMBEDDED_SUBMISSION"][filename].replace(before, after)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "matching YAML topology, sampling"):
                    exec(workflow.PREPARATION, namespace)

    def test_embedded_candidate_and_materialized_control_match_package(self):
        with tempfile.TemporaryDirectory(prefix="agent_workflow_") as temporary:
            root = Path(temporary)
            output = root / "evaluate.ipynb"
            hashes = workflow.generate(
                BASE / "submission_1", BASE / "tools" / "submission_tools.py",
                BASE.parent / "competition_material" / "getting-started-gemma-4-developer-agent.ipynb",
                output,
            )
            notebook = nbformat.read(output, as_version=4)
            nbformat.validate(notebook)
            self.assertTrue(all(not cell.get("outputs") for cell in notebook.cells))
            namespace = {}
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                if cell.source.startswith("from pathlib import Path"):
                    exec(cell.source, namespace)
                elif cell.source.startswith("# Curated text files") or cell.source.startswith("def remove_agent_tools"):
                    exec(cell.source, namespace)
            for name, content in namespace["EMBEDDED_SUBMISSION"].items():
                self.assertEqual(hashlib.sha256(content.encode()).hexdigest(), hashes[name])
                self.assertEqual(content, (BASE / "submission_1" / name).read_text(encoding="utf-8"))
            self.assertEqual(
                hashlib.sha256(namespace["EMBEDDED_TOOL_SOURCE"].encode()).hexdigest(),
                hashlib.sha256((BASE / "tools" / "submission_tools.py").read_bytes()).hexdigest(),
            )
            namespace["WORKING_DIR"] = root / "working"
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PREPARATION, namespace)
            manifests = namespace["VARIANT_MANIFESTS"]
            self.assertEqual(manifests["coder_only"]["agents"], 1)
            self.assertEqual(manifests["optional_analyzer"]["agents"], 2)
            self.assertEqual(manifests["coder_only"]["evaluation"], manifests["optional_analyzer"]["evaluation"])
            self.assertEqual(manifests["coder_only"]["adapters"], [])
            control_files = set(manifests["coder_only"]["package_files"])
            self.assertEqual(control_files, {"agent.yaml", "eval_config.yaml", "configs/sampling.yaml", "prompts/system.md"})
            control = namespace["VARIANT_DIRS"]["coder_only"]
            candidate = namespace["VARIANT_DIRS"]["optional_analyzer"]
            self.assertIn("!include", (control / "agent.yaml").read_text(encoding="utf-8"))
            self.assertNotIn("agent_tool", (control / "agent.yaml").read_text(encoding="utf-8"))
            for name in control_files - {"agent.yaml"}:
                self.assertEqual((control / name).read_bytes(), (candidate / name).read_bytes())
            package = root / "control.zip"
            namespace["tool_cli"]("build", "--submission-dir", control, "--output", package)
            with zipfile.ZipFile(package) as archive:
                self.assertEqual(set(archive.namelist()), control_files)
                self.assertFalse(any(name.startswith(("tools/", "logs/", "reference_adapters/")) for name in archive.namelist()))

    def test_selection_uses_matching_commits_and_spans_repositories(self):
        tasks = [
            {"instance_id": "alpha_1", "repo": "org/alpha", "base_commit": "aaa"},
            {"instance_id": "alpha_2", "repo": "org/alpha", "base_commit": "aab"},
            {"instance_id": "beta_1", "repo": "org/beta", "base_commit": "bbb"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            for task in tasks:
                files = {
                    "snapshots": task["instance_id"] + ".tgz",
                    "graphs": task["repo"].split("/")[-1] + "_" + task["base_commit"] + ".json",
                    "embeddings": task["repo"].split("/")[-1] + "_" + task["base_commit"] + ".npz",
                }
                for folder, name in files.items():
                    target = data / folder / name
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(b"fixture" * 30)
            selected, _ = workflow.select_tasks(tasks, data, limit=2)
            self.assertEqual([task["instance_id"] for task in selected], ["alpha_1", "beta_1"])
            selected, _ = workflow.select_tasks(tasks, data, task_ids=["beta_1", "alpha_2"], limit=1)
            self.assertEqual([task["instance_id"] for task in selected], ["beta_1", "alpha_2"])
            (data / "graphs" / "alpha_aaa.json").rename(data / "graphs" / "alpha_wrong.json")
            with self.assertRaisesRegex(ValueError, "complete matching resources"):
                workflow.select_tasks(tasks, data, task_ids=["alpha_1"])

    def test_current_partial_data_fails_before_gpu_inference(self):
        data = BASE.parent / "competition_material" / "competition_data"
        tasks = [json.loads(line) for line in (data / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        coverage = [workflow.task_resources(task, data) for task in tasks]
        # This check remains valid when the user downloads more matching files.
        if any(all(resources.values()) for resources in coverage):
            selected, _ = workflow.select_tasks(tasks, data, limit=1)
            self.assertEqual(len(selected), 1)
        else:
            with self.assertRaisesRegex(ValueError, "No task has matching"):
                workflow.select_tasks(tasks, data, limit=1)


if __name__ == "__main__":
    unittest.main()
