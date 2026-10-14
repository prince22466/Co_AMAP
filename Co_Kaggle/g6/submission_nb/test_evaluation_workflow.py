"""Offline integration checks for the portable evaluation workflow."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import nbformat


BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_evaluation_notebook", BASE / "prepare_evaluation_notebook.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class EvaluationWorkflowTests(unittest.TestCase):
    def test_smallest_model_candidate_preserves_submission_2_and_routes_its_notebook(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "smallest.ipynb"
            workflow.generate(
                BASE / "submission_6", BASE / "tools" / "submission_tools.py",
                BASE.parent / "competition_material" / "getting-started-gemma-4-developer-agent.ipynb", output,
            )
            notebook = nbformat.read(output, as_version=4)
            nbformat.validate(notebook)
            namespace = {}
            for cell in notebook.cells:
                if cell.cell_type == "code" and cell.source.startswith(
                    ("from pathlib import Path", "# Curated text files", "def remove_agent_tools")
                ):
                    exec(cell.source, namespace)
            smallest = "gemma-4-e2b-it-qat-w4a16-ct"
            self.assertEqual(notebook.metadata["base_model"], smallest)
            self.assertEqual(namespace["MODEL_PATH"].as_posix(), '/kaggle/input/models/google/gemma-4/other/' + smallest + '/2')
            updated = nbformat.read(BASE.parent / "competition_material" / "getting-started-gemma-4-developer-agent_update1.ipynb", as_version=4)
            install = next(cell.source for cell in updated.cells if cell.cell_type == "code")
            self.assertEqual(notebook.cells[4].source.strip(), install.strip())
            changed = {"agent.yaml", "sub_agents/code_analyzer.yaml"}
            for name, content in namespace["EMBEDDED_SUBMISSION"].items():
                expected = (BASE / "submission_2" / name).read_text(encoding="utf-8")
                if name in changed:
                    expected = expected.replace("gemma-4-31b-it-qat-w4a16-ct", smallest)
                self.assertEqual(content, expected)
            namespace["WORKING_DIR"] = root / "working"
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PREPARATION, namespace)
            self.assertEqual(namespace["VARIANT_MANIFESTS"]["coder_only"]["agents"], 1)
            self.assertEqual(namespace["VARIANT_MANIFESTS"]["optional_analyzer"]["agents"], 2)
            self.assertTrue(all(manifest["model"] == smallest for manifest in namespace["VARIANT_MANIFESTS"].values()))
            self.assertEqual(namespace["VARIANT_MANIFESTS"]["optional_analyzer"]["evaluation"], {
                "timeout_seconds": 60, "max_tool_calls": 40, "max_time_minutes": 4, "max_turns": 80,
            })
            # Exercise model routing without loading weights or launching vLLM.
            captured = {}
            fake_server = types.SimpleNamespace(
                start=lambda: None, base_url="http://localhost:8000",
                create_model_registry=lambda **kwargs: captured.update(registry=kwargs),
            )
            modules = {
                "litellm": types.SimpleNamespace(),
                "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(
                    is_available=lambda: True, device_count=lambda: 4, is_bf16_supported=lambda: True,
                )),
                "adk_submission": types.SimpleNamespace(
                    VllmConfig=lambda **kwargs: captured.update(config=kwargs),
                    VllmServer=lambda *args, **kwargs: fake_server,
                    discover_adapters=lambda *args, **kwargs: {},
                ),
                "adk_submission.discovery": types.SimpleNamespace(discover_declared_models=lambda directory: {smallest}),
                "swegemma.config": types.SimpleNamespace(ALLOWED_ADAPTER_EXTENSIONS={".safetensors"}),
            }
            with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.SERVER, namespace)
            self.assertEqual(captured["config"]["model"], str(namespace["MODEL_PATH"]))
            self.assertFalse(captured["config"]["scheduler_reserve_full_isl"])
            self.assertEqual(captured["registry"]["aliases"], [smallest])
            self.assertEqual(namespace["TARGET_MODEL_NAME"], smallest)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PACKAGE, namespace)
            with zipfile.ZipFile(namespace["WORKING_DIR"] / "submission.zip") as archive:
                self.assertEqual(len(archive.namelist()), 7)
                self.assertTrue(all(archive.read(name).decode('utf-8').replace('\r\n', '\n') == namespace['EMBEDDED_SUBMISSION'][name]
                                    for name in archive.namelist()))

    def runtime_namespace(self, root):
        output = root / "runtime.ipynb"
        workflow.generate(
            BASE / "submission_5", BASE / "tools" / "submission_tools.py",
            BASE.parent / "competition_material" / "getting-started-gemma-4-developer-agent.ipynb",
            output, runtime_baseline_dir=BASE / "submission_4",
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

    def test_runtime_controls_isolate_timeout_and_thinking_without_changing_agents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook, namespace = self.runtime_namespace(root)
            self.assertEqual(notebook.metadata["comparison_mode"], "runtime_comparison")
            self.assertEqual(namespace["BENCHMARK_VARIANTS"], ["baseline", "timeout_only", "thinking_off_only", "candidate"])
            self.assertEqual(namespace["RELEASE_VARIANT"], "candidate")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PREPARATION, namespace)
            manifests = namespace["VARIANT_MANIFESTS"]
            baseline_dir = namespace["VARIANT_DIRS"]["baseline"]
            candidate_dir = namespace["VARIANT_DIRS"]["candidate"]
            baseline_sampling = {"configs/sampling.yaml", "sub_agents/locator_resources/sampling.yaml",
                                 "sub_agents/verifier_resources/sampling.yaml"}
            for variant, timeout, thinking, changed_files in [
                ("baseline", 60, True, set()),
                ("timeout_only", 300, True, {"eval_config.yaml"}),
                ("thinking_off_only", 60, False, baseline_sampling),
                ("candidate", 300, False, baseline_sampling | {"eval_config.yaml"}),
            ]:
                manifest = manifests[variant]
                directory = namespace["VARIANT_DIRS"][variant]
                self.assertEqual(manifest["agents"], 3)
                self.assertEqual(manifest["agent_tool_names"], ["repository_locator_agent", "test_verifier_agent"])
                self.assertEqual(manifest["evaluation"], {
                    "timeout_seconds": timeout, "max_tool_calls": 40, "max_time_minutes": 4, "max_turns": 80,
                })
                actual_changes = {name for name in manifest["package_files"]
                                  if (directory / name).read_bytes() != (baseline_dir / name).read_bytes()}
                self.assertEqual(actual_changes, changed_files)
                for name in baseline_sampling:
                    config = workflow.yaml.safe_load((directory / name).read_text(encoding="utf-8"))
                    self.assertEqual(config["thinking_config"]["include_thoughts"], thinking)
                    if not thinking:
                        self.assertEqual(config["thinking_config"], {"include_thoughts": False})
                    self.assertEqual(config["temperature"], 0.2)
                    self.assertEqual(config["top_p"], 0.95)
                    self.assertEqual(config["max_output_tokens"], 16384 if name == "configs/sampling.yaml" else 2048)
                self.assertNotIn("README.md", manifest["package_files"])
            for name, content in namespace["EMBEDDED_SUBMISSION"].items():
                self.assertEqual(content, (BASE / "submission_5" / name).read_text(encoding="utf-8"))
                self.assertEqual(hashlib.sha256(content.encode()).hexdigest(), namespace["EMBEDDED_SOURCE_HASHES"][name])
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PACKAGE, namespace)
            with zipfile.ZipFile(namespace["WORKING_DIR"] / "submission.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), set(manifests["candidate"]["package_files"]))
                self.assertTrue(all(archive.read(name) == (candidate_dir / name).read_bytes() for name in archive.namelist()))

    def test_runtime_preflight_rejects_unrelated_or_incomplete_changes(self):
        for filename, before, after, error in [
            ("eval_config.yaml", "max_time_minutes: 4", "max_time_minutes: 5", "only command/grading timeout"),
            ("configs/sampling.yaml", "temperature: 0.2", "temperature: 0.7", "only disabling thinking"),
            ("sub_agents/verifier_resources/sampling.yaml", "include_thoughts: false", "include_thoughts: true", "only disabling thinking"),
            ("prompts/system.md", "Keep reasoning concise", "Spend more time reasoning", "identical agent configs, tools and prompts"),
        ]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                _, namespace = self.runtime_namespace(Path(temporary))
                self.assertIn(before, namespace["EMBEDDED_SUBMISSION"][filename])
                namespace["EMBEDDED_SUBMISSION"][filename] = namespace["EMBEDDED_SUBMISSION"][filename].replace(before, after)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, error):
                    exec(workflow.PREPARATION, namespace)

    def verification_namespace(self, root):
        output = root / "verification.ipynb"
        workflow.generate(
            BASE / "submission_4", BASE / "tools" / "submission_tools.py",
            BASE.parent / "competition_material" / "getting-started-gemma-4-developer-agent.ipynb",
            output, verification_control=True,
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

    def test_locator_verifier_variants_preserve_budget_and_package_active_helpers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook, namespace = self.verification_namespace(root)
            self.assertEqual(notebook.metadata["comparison_mode"], "verification_control")
            self.assertEqual(namespace["BENCHMARK_VARIANTS"], ["coder_only", "optional_locator", "optional_verifier", "locator_verifier"])
            self.assertEqual(namespace["RELEASE_VARIANT"], "locator_verifier")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PREPARATION, namespace)
            manifests = namespace["VARIANT_MANIFESTS"]
            candidate = namespace["VARIANT_DIRS"]["locator_verifier"]
            for variant, count, helpers in [
                ("coder_only", 1, []),
                ("optional_locator", 2, ["repository_locator_agent"]),
                ("optional_verifier", 2, ["test_verifier_agent"]),
                ("locator_verifier", 3, ["repository_locator_agent", "test_verifier_agent"]),
            ]:
                manifest = manifests[variant]
                self.assertEqual(manifest["agents"], count)
                self.assertEqual(manifest["agent_tool_names"], helpers)
                self.assertEqual(manifest["evaluation"], manifests["locator_verifier"]["evaluation"])
                self.assertEqual(manifest["adapters"], [])
                directory = namespace["VARIANT_DIRS"][variant]
                for name in set(manifest["package_files"]) - {"agent.yaml"}:
                    self.assertEqual((directory / name).read_bytes(), (candidate / name).read_bytes())
                self.assertEqual("sub_agents/repository_locator.yaml" in manifest["package_files"], "repository_locator_agent" in helpers)
                self.assertEqual("sub_agents/test_verifier.yaml" in manifest["package_files"], "test_verifier_agent" in helpers)
                self.assertNotIn("README.md", manifest["package_files"])
            for name, content in namespace["EMBEDDED_SUBMISSION"].items():
                self.assertEqual(content, (BASE / "submission_4" / name).read_text(encoding="utf-8"))
                self.assertEqual(hashlib.sha256(content.encode()).hexdigest(), namespace["EMBEDDED_SOURCE_HASHES"][name])
                if name.endswith("instruction.md") or name == "prompts/system.md":
                    self.assertIsNone(re.search(r"\{[^{}\n]+\}", content), "Unexpected ADK state interpolation")
            for name in ["eval_config.yaml", "configs/sampling.yaml"]:
                self.assertEqual((BASE / "submission_4" / name).read_bytes(), (BASE / "submission_2" / name).read_bytes())
            for name in ["sub_agents/repository_locator.yaml", "sub_agents/test_verifier.yaml"]:
                config = namespace["verification_loader"].load_yaml(candidate / name)
                self.assertEqual(config["tools"], ["run_command", "read_file"])
                self.assertEqual(config["include_contents"], "none")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PACKAGE, namespace)
            with zipfile.ZipFile(namespace["WORKING_DIR"] / "submission.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), set(manifests["locator_verifier"]["package_files"]))

    def test_locator_verifier_preflight_rejects_edit_tools_and_missing_helper(self):
        for filename, before, after in [
            ("sub_agents/test_verifier.yaml", "  - read_file\n", "  - read_file\n  - edit_file\n"),
            ("sub_agents/repository_locator.yaml", "include_contents: none", "include_contents: default"),
            ("agent.yaml", "  - agent_tool:\n      config_path: sub_agents/test_verifier.yaml\n      skip_summarization: true\n", ""),
        ]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                _, namespace = self.verification_namespace(Path(temporary))
                self.assertIn(before, namespace["EMBEDDED_SUBMISSION"][filename])
                namespace["EMBEDDED_SUBMISSION"][filename] = namespace["EMBEDDED_SUBMISSION"][filename].replace(before, after)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Verification comparison requires|Locator/verifier helpers must"):
                    exec(workflow.PREPARATION, namespace)

    def reviewer_namespace(self, root):
        output = root / "reviewer.ipynb"
        workflow.generate(
            BASE / "submission_3", BASE / "tools" / "submission_tools.py",
            BASE.parent / "competition_material" / "getting-started-gemma-4-developer-agent.ipynb",
            output, reviewer_control=True,
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

    def test_reviewer_controls_preserve_shared_files_and_package_only_reachable_helpers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook, namespace = self.reviewer_namespace(root)
            self.assertEqual(notebook.metadata["comparison_mode"], "reviewer_control")
            self.assertEqual(namespace["BENCHMARK_VARIANTS"], ["coder_only", "optional_analyzer", "optional_reviewer"])
            self.assertEqual(namespace["RELEASE_VARIANT"], "optional_reviewer")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PREPARATION, namespace)
            manifests = namespace["VARIANT_MANIFESTS"]
            candidate = namespace["VARIANT_DIRS"]["optional_reviewer"]
            for variant, count, helpers in [
                ("coder_only", 1, []),
                ("optional_analyzer", 2, ["code_analyzer_agent"]),
                ("optional_reviewer", 3, ["code_analyzer_agent", "patch_reviewer_agent"]),
            ]:
                manifest = manifests[variant]
                self.assertEqual(manifest["agents"], count)
                self.assertEqual(manifest["agent_tool_names"], helpers)
                self.assertEqual(manifest["evaluation"], manifests["optional_reviewer"]["evaluation"])
                directory = namespace["VARIANT_DIRS"][variant]
                for name in set(manifest["package_files"]) - {"agent.yaml"}:
                    self.assertEqual((directory / name).read_bytes(), (candidate / name).read_bytes())
                self.assertEqual(any("reviewer" in name for name in manifest["package_files"]), count == 3)
                self.assertEqual(any("analyzer" in name for name in manifest["package_files"]), count >= 2)
                for name, content in namespace["EMBEDDED_SUBMISSION"].items():
                    self.assertEqual(content, (BASE / "submission_3" / name).read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(workflow.PACKAGE, namespace)
            with zipfile.ZipFile(namespace["WORKING_DIR"] / "submission.zip") as archive:
                self.assertEqual(set(archive.namelist()), set(manifests["optional_reviewer"]["package_files"]))
                reviewer = namespace["reviewer_loader"].load_yaml(candidate / "sub_agents/patch_reviewer.yaml")
                self.assertEqual(reviewer["tools"], ["read_file"])
                self.assertEqual(reviewer["include_contents"], "none")
            # Existing model, sampling, analyzer and total budget stay byte-identical.
            for name in ["eval_config.yaml", "configs/sampling.yaml", "sub_agents/code_analyzer.yaml",
                         "sub_agents/analyzer_resources/instruction.md", "sub_agents/analyzer_resources/sampling.yaml"]:
                self.assertEqual((BASE / "submission_3" / name).read_bytes(), (BASE / "submission_2" / name).read_bytes())

    def test_reviewer_preflight_rejects_writable_tools_or_missing_reviewer(self):
        for filename, before, after in [
            ("sub_agents/patch_reviewer.yaml", "  - read_file\n", "  - read_file\n  - edit_file\n"),
            ("agent.yaml", "  - agent_tool:\n      config_path: sub_agents/patch_reviewer.yaml\n      skip_summarization: true\n", ""),
        ]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                _, namespace = self.reviewer_namespace(Path(temporary))
                self.assertIn(before, namespace["EMBEDDED_SUBMISSION"][filename])
                namespace["EMBEDDED_SUBMISSION"][filename] = namespace["EMBEDDED_SUBMISSION"][filename].replace(before, after)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Reviewer comparison requires|Patch reviewer must"):
                    exec(workflow.PREPARATION, namespace)

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
