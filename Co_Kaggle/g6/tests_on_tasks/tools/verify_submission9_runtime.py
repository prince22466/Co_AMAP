"""CPU smoke test of the release ZIP through the official ADK compiler/executor.

Install adk-submission 0.2.13, google-adk 1.36.1, adk-eval-core 0.1.0 and google-genai 2.11.0
from the competition wheelhouse. No model request or benchmark grading occurs.
Native tools/model are test doubles; skill tools and subprocess execution are real.
"""
import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from adk_submission import (compile_submission, GenerationConstraints, ModelRegistry,
                            NumericRange, SubmissionLimits, ToolRegistry)
from adk_eval_core.sandbox import AdkSandboxCodeExecutor, SubprocessSandbox
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.base_llm import BaseLlm
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.tools.tool_context import ToolContext

from build_submission9 import G6, MODEL, RUNTIME_SKILL


class NoInference(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        raise AssertionError("this smoke test must not request model inference")
        yield  # Keep the ADK async-generator interface without making a request.


def stub(name):
    def unused_native_tool(command: str = "") -> str:
        raise AssertionError("native tools are not executed by this smoke test")
    unused_native_tool.__name__ = name
    return unused_native_tool


async def verify(archive):
    if not __debug__:
        raise RuntimeError("run runtime verification without Python optimization")
    versions = {p: version(p) for p in ("adk-submission", "google-adk", "adk-eval-core",
                                       "google-genai", "pydantic", "PyYAML")}
    expected = {"adk-submission": "0.2.13", "google-adk": "1.36.1",
                "adk-eval-core": "0.1.0", "google-genai": "2.11.0"}
    for name, required in expected.items():
        if versions[name] != required:
            raise ValueError(f"expected {name} {required}, got {versions[name]}")
    checks = []
    with tempfile.TemporaryDirectory(prefix="submission9-runtime-") as directory:
        temp = Path(directory)
        package, root = temp / "package", temp / "workspace"
        with zipfile.ZipFile(archive) as z:
            if any(Path(n).is_absolute() or ".." in Path(n).parts for n in z.namelist()):
                raise ValueError("unsafe archive member")
            z.extractall(package)
        root.mkdir()
        sandbox = SubprocessSandbox(root, environment={"PYTHONDONTWRITEBYTECODE": "1"})
        sandbox.initialize({"src/example_pkg/__init__.py": "def answer():\n    return 42\n"})
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT)
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=Runtime smoke", "-c", "user.email=smoke@example.invalid",
            "commit", "-qm", "synthetic baseline")
        executor = AdkSandboxCodeExecutor(sandbox=sandbox, timeout_seconds=30)
        models, native = ModelRegistry(), ToolRegistry()
        models.register(MODEL, NoInference(model=MODEL))
        for name in ("run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"):
            native.register(name, stub(name))
        # The extension/token bounds below match swegemma 0.2.11 config.py.
        limits = SubmissionLimits(allowed_file_extensions=frozenset(
            {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}))
        constraints = GenerationConstraints(allowed_fields=None,
            max_output_tokens=NumericRange(1, 32768), thinking_budget=NumericRange(0, 32768))
        agent = compile_submission(package, native, models, limits=limits,
                                   generation_constraints=constraints,
                                   code_executor=executor, script_timeout=30)
        helpers = [tool for tool in agent.tools if isinstance(tool, AgentTool)]
        assert len(helpers) == 2 and all(not tool.agent.tools for tool in helpers)
        assert all(tool.agent.model.model == MODEL for tool in helpers)
        checks.append("official_compiler_accepts_archive_and_preserves_two_tool_free_helpers")
        skillsets = [tool for tool in agent.tools if isinstance(tool, SkillToolset)]
        assert len(skillsets) == 1
        context = ToolContext(InvocationContext(
            invocation_id="submission9-smoke", agent=agent,
            session=Session(id="smoke", app_name="smoke", user_id="smoke"),
            session_service=InMemorySessionService()))
        tools = {tool.name: tool for tool in await skillsets[0].get_tools()}
        skill_name = Path(RUNTIME_SKILL).name
        async def call(name, **args):
            result = await tools[name].run_async(args={"skill_name": skill_name, **args},
                                                 tool_context=context)
            assert "error" not in result, result
            return result
        loaded = await call("load_skill")
        assert loaded["frontmatter"]["name"] == skill_name
        checks.append("skill_activation")
        for name in ("contracts", "tool_recovery", "verification"):
            result = await call("load_skill_resource", file_path=f"references/{name}.md")
            content = (package / RUNTIME_SKILL / "resources" / f"{name}.md").read_text()
            assert result["content"] == content
        checks.append("all_three_resource_aliases_load_exact_content")
        async def script(name, **args):
            result = await call("run_skill_script", file_path=f"scripts/{name}.py",
                                args={"workspace": str(root), **args})
            assert result["status"] == "success", result
            assert len(result["stdout"]) < 5000, result
            return json.loads(result["stdout"])
        result = await script("inspect_source", path="src/example_pkg/__init__.py", symbol="answer")
        assert "return 42" in result["source"]
        checks.append("source_inspection_through_adk_runpy_wrapper")
        result = await script("check", module="example_pkg", code="import example_pkg\nassert example_pkg.answer() == 42")
        assert result["outcome"] == "passed" and result["assertions_passed"] == 1, result
        assert result["module_origin"].startswith(str(root / "src")), result
        checks.append("fresh_workspace_import_and_executed_assertion")
        result = await script("check", module="example_pkg",
                              code="try:\n    assert False\nexcept AssertionError:\n    pass")
        assert result["outcome"] == "failed" and result["assertions_failed"] == 1, result
        checks.append("caught_failure_stays_failed_through_tool_envelope")
        result = await script("audit_patch")
        assert result["empty_patch"], result
        assert result["diff_check_returncode"] == 0, result
        checks.append("clean_repository_audit_ignores_only_active_executor_artifacts")
        sandbox.write_file("src/example_pkg/__init__.py", "def answer():\n    return 43\n")
        result = await script("check", module="example_pkg", code="import example_pkg\nassert example_pkg.answer() == 43")
        assert result["outcome"] == "passed", result
        before = git("status", "--porcelain=v1")
        result = await script("audit_patch")
        assert result["changed_paths"] == ["src/example_pkg/__init__.py"], result
        assert git("status", "--porcelain=v1") == before and git("diff", "--cached") == b""
        checks.append("edited_source_seen_and_audit_leaves_real_diff_unchanged")
        sandbox.close()
    return {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "status": "passed", "versions": versions, "checks": checks,
            "scope": "Official compiler, skill tools and adk-eval-core subprocess executor; synthetic repo, native-tool/model test doubles. No inference, task grading, Docker or competition budget validation.",
            "gemma_evaluation": "not run", "public_score": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=G6 / "submission_nb/artifacts/submission_9.zip")
    parser.add_argument("--output", type=Path, default=G6 / "tests_on_tasks/diagnostics/submission_9_review/runtime_smoke.json")
    args = parser.parse_args()
    result = asyncio.run(verify(args.archive))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
