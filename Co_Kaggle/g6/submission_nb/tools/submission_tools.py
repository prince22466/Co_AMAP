"""Local-only validation, packaging, and evidence-based evaluation diagnostics.

Build packages the dependency closure of agent.yaml and eval_config.yaml, plus
whole referenced skill/adapter directories. Unreferenced documentation is omitted.
This checks the documented submission subset; Kaggle's compiler is authoritative.
Report never treats task duration or cumulative tokens as agent time/context size.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

MODEL = "gemma-4-31b-it-qat-w4a16-ct"
CORE_TOOLS = frozenset({
    "run_command", "read_file", "edit_file", "write_file", "get_status",
    "submit_patch", "get_code_neighbors", "search_similar_code", "get_code_subgraph",
})
FREE_TOOLS = frozenset({"get_status", "submit_patch"})
SKILL_TOOLS = frozenset({"load_skill_resource", "run_skill_script"})
ALLOWED_EXTENSIONS = frozenset({".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"})
MAX_BYTES = 3 * 1024 ** 3
MAX_YAML_BYTES = 50 * 1024 ** 2


class ValidationError(ValueError):
    pass


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class SubmissionValidator:
    """Resolve dependencies without importing competitor Python or executing YAML."""

    def __init__(self, directory: str | Path):
        requested = Path(directory).absolute()
        if requested.is_symlink():
            raise ValidationError("Submission directory must not be a symlink")
        self.root = requested.resolve()
        self.files: set[Path] = set()
        self.agent_paths: set[Path] = set()
        self.adapter_names: set[str] = set()
        self.agent_tool_names: set[str] = set()
        self.agent_stack: set[Path] = set()
        self.agents = 0
        self.warnings: list[str] = []

    def resolve(self, ref: Any, parent: Path, *, directory: bool = False) -> Path:
        if not isinstance(ref, str) or not ref or "\x00" in ref:
            raise ValidationError(f"Invalid path reference: {ref!r}")
        windows = PureWindowsPath(ref)
        if Path(ref).is_absolute() or windows.is_absolute() or windows.drive or ref.startswith(("/", "\\")):
            raise ValidationError(f"Absolute path is forbidden: {ref}")
        # Normalize separators so Windows-style traversal is also detected on Linux.
        raw = parent / Path(ref.replace("\\", "/"))
        resolved = raw.resolve()
        if not _inside(resolved, self.root):
            raise ValidationError(f"Path escapes submission root: {ref}")
        for part in [raw, *raw.parents]:
            if part.is_symlink():
                raise ValidationError(f"Symlink is forbidden: {ref}")
            if part == self.root:
                break
        if directory and not resolved.is_dir():
            raise ValidationError(f"Missing directory: {ref}")
        if not directory and not resolved.is_file():
            raise ValidationError(f"Missing file: {ref}")
        return resolved

    def load_yaml(self, path: Path, stack: tuple[Path, ...] = ()) -> Any:
        if path in stack:
            raise ValidationError(f"Include cycle: {path.relative_to(self.root)}")
        if len(stack) > 10:
            raise ValidationError("Include nesting exceeds 10")
        if path.stat().st_size > MAX_YAML_BYTES:
            raise ValidationError(f"YAML exceeds 50 MiB: {path.name}")
        self.files.add(path)
        owner = self

        class Loader(yaml.SafeLoader):
            pass

        def include(loader: Loader, node: yaml.Node) -> Any:
            target = owner.resolve(loader.construct_scalar(node), path.parent)
            owner.files.add(target)
            if target.suffix.lower() in {".yaml", ".yml"}:
                return owner.load_yaml(target, (*stack, path))
            if target.suffix.lower() not in {".md", ".txt"}:
                raise ValidationError(f"Unsupported include format: {target.name}")
            return target.read_text(encoding="utf-8")

        Loader.add_constructor("!include", include)
        try:
            return yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)
        except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
            raise ValidationError(f"Invalid YAML in {path.name}: {exc}") from exc

    def collect_directory(self, directory: Path) -> None:
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValidationError(f"Symlink is forbidden: {path.relative_to(self.root)}")
            if path.is_file():
                self.files.add(path)

    def generation_config(self, config: Any) -> None:
        if config is None:
            return
        if not isinstance(config, dict):
            raise ValidationError("generate_content_config must be a mapping")
        permitted = {
            "temperature", "top_p", "top_k", "max_output_tokens", "presence_penalty",
            "frequency_penalty", "stop_sequences", "response_mime_type", "seed", "thinking_config",
        }
        extra = set(config) - permitted
        if extra:
            raise ValidationError(f"Unsupported generation fields: {sorted(extra)}")
        for key, lower, upper in [("max_output_tokens", 1, 32768), ("top_p", 0, 1), ("temperature", 0, None), ("top_k", 1, None)]:
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < lower or (upper is not None and value > upper):
                    raise ValidationError(f"Invalid generation setting: {key}")
                if key in {"max_output_tokens", "top_k"} and not isinstance(value, int):
                    raise ValidationError(f"{key} must be an integer")
        thinking = config.get("thinking_config")
        if thinking is not None:
            if not isinstance(thinking, dict) or set(thinking) - {"thinking_budget", "thinking_level", "include_thoughts"}:
                raise ValidationError("Unsupported thinking_config")
            if "thinking_budget" in thinking:
                value = thinking["thinking_budget"]
                if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 32768:
                    raise ValidationError("thinking_budget must be an integer in 0..32768")
            if "include_thoughts" in thinking and not isinstance(thinking["include_thoughts"], bool):
                raise ValidationError("include_thoughts must be boolean")
            if "thinking_level" in thinking:
                if str(thinking["thinking_level"]).upper() not in {"MINIMAL", "LOW", "MEDIUM", "HIGH", "NONE"}:
                    raise ValidationError("Invalid thinking_level")
                self.warnings.append("thinking_level can map to unsupported reasoning_effort; prefer thinking_budget")

    def agent(self, config: Any, path: Path, depth: int = 0, *, as_tool: bool = False) -> None:
        if depth > 50 or self.agents >= 500:
            raise ValidationError("Agent tree exceeds documented depth/count limits")
        if not isinstance(config, dict):
            raise ValidationError(f"Agent config must be a mapping: {path.name}")
        self.agents += 1
        kind = config.get("agent_class", "LlmAgent")
        if not isinstance(kind, str) or kind not in {"LlmAgent", "SequentialAgent", "ParallelAgent", "LoopAgent"}:
            raise ValidationError(f"Unsupported agent_class: {kind}")
        name = config.get("name")
        if not isinstance(name, str) or not name:
            raise ValidationError(f"Missing agent name: {path.name}")
        common_fields = {"name", "description", "agent_class", "sub_agents"}
        llm_fields = {
            "model", "adapter", "instruction", "global_instruction", "tools", "skills",
            "output_key", "include_contents", "disallow_transfer_to_parent",
            "disallow_transfer_to_peers", "generate_content_config",
        }
        permitted = common_fields | (llm_fields if kind == "LlmAgent" else {"max_iterations"} if kind == "LoopAgent" else set())
        if set(config) - permitted:
            raise ValidationError(f"Unsupported agent fields in local documented subset: {sorted(set(config) - permitted)}")
        if as_tool:
            self.agent_tool_names.add(name)
        if kind == "LlmAgent":
            if config.get("model") != MODEL:
                raise ValidationError(f"Agent {name} must declare model {MODEL}")
            if not isinstance(config.get("instruction", ""), str):
                raise ValidationError(f"instruction must be text: {name}")
            if len(config.get("instruction", "")) > 1_000_000:
                raise ValidationError(f"instruction exceeds 1,000,000 characters: {name}")
            if config.get("include_contents", "default") not in {"default", "none"}:
                raise ValidationError(f"Invalid include_contents: {name}")
            self.generation_config(config.get("generate_content_config"))
            adapter = config.get("adapter")
            if adapter is not None:
                if not isinstance(adapter, str) or "/" in adapter or "\\" in adapter or adapter in {".", ".."}:
                    raise ValidationError("adapter must be a directory name under adapters/")
                directory = self.resolve(f"adapters/{adapter}", self.root, directory=True)
                self.resolve("adapter_config.json", directory)
                self.resolve("adapter_model.safetensors", directory)
                self.adapter_names.add(adapter)
                self.collect_directory(directory)
            skills = config.get("skills") or []
            if not isinstance(skills, list):
                raise ValidationError("skills must be a list")
            for ref in skills:
                directory = self.resolve(ref, self.root, directory=True)
                manifest = self.resolve("SKILL.md", directory)
                text = manifest.read_text(encoding="utf-8")
                match = re.match(r"\A---\s*\n(.*?)\n---(?:\s*\n|$)", text, re.S)
                metadata = yaml.safe_load(match.group(1)) if match else None
                if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str) or not metadata["name"]:
                    raise ValidationError(f"Skill manifest needs YAML frontmatter with name: {ref}")
                self.collect_directory(directory)
            tools = config.get("tools", [])
            if not isinstance(tools, list):
                raise ValidationError("tools must be a list")
            for tool in tools:
                if isinstance(tool, str):
                    if tool not in CORE_TOOLS and not (skills and tool in SKILL_TOOLS):
                        raise ValidationError(f"Unsupported tool {tool!r} in {name}")
                elif isinstance(tool, dict) and "agent_tool" in tool:
                    if set(tool) != {"agent_tool"} or not isinstance(tool["agent_tool"], dict):
                        raise ValidationError("agent_tool must be its own mapping")
                    spec = tool["agent_tool"]
                    if set(spec) - {"config_path", "skip_summarization"}:
                        raise ValidationError("Unsupported agent_tool fields")
                    if "skip_summarization" in spec and not isinstance(spec["skip_summarization"], bool):
                        raise ValidationError("skip_summarization must be boolean")
                    self.agent_reference(spec, path, depth, as_tool=True)
                elif isinstance(tool, dict) and "name" in tool:
                    self.agent(tool, path, depth + 1, as_tool=True)
                else:
                    raise ValidationError(f"Invalid tool declaration in {name}")
        if kind == "LoopAgent":
            iterations = config.get("max_iterations", 500)
            if not isinstance(iterations, int) or isinstance(iterations, bool) or not 1 <= iterations <= 500:
                raise ValidationError("max_iterations must be in 1..500")
        children = config.get("sub_agents", [])
        if not isinstance(children, list):
            raise ValidationError("sub_agents must be a list")
        for child in children:
            if isinstance(child, dict) and "config_path" in child:
                self.agent_reference(child, path, depth)
            elif isinstance(child, dict):
                self.agent(child, path, depth + 1)
            else:
                raise ValidationError("Invalid sub_agents declaration")

    def agent_reference(self, spec: dict, parent: Path, depth: int, *, as_tool: bool = False) -> None:
        target = self.resolve(spec.get("config_path"), self.root)
        if target in self.agent_stack:
            raise ValidationError(f"Agent reference cycle: {target.name}")
        self.agent_stack.add(target)
        self.agent_paths.add(target)
        try:
            self.agent(self.load_yaml(target), target, depth + 1, as_tool=as_tool)
        finally:
            self.agent_stack.remove(target)

    def validate(self) -> dict:
        if not self.root.is_dir():
            raise ValidationError(f"Missing submission directory: {self.root}")
        all_files = []
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValidationError(f"Symlink is forbidden: {path.relative_to(self.root)}")
            if path.is_file():
                all_files.append(path)
                if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                    raise ValidationError(f"Forbidden file extension: {path.relative_to(self.root)}")
        if len(all_files) > 10_000:
            raise ValidationError("Submission exceeds 10,000 files")
        # Runtime preflight also scans standalone agent YAML files, even if build
        # would omit them. Inspect declarations without resolving unused includes.
        class DeclarationLoader(yaml.SafeLoader):
            pass
        DeclarationLoader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))
        for path in all_files:
            if path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            if path.stat().st_size > MAX_YAML_BYTES:
                raise ValidationError(f"YAML exceeds 50 MiB: {path.name}")
            try:
                declaration = yaml.load(path.read_text(encoding="utf-8"), Loader=DeclarationLoader)
            except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
                raise ValidationError(f"Invalid source YAML: {path.name}: {exc}") from exc
            if isinstance(declaration, dict) and "name" in declaration and "model" in declaration and declaration["model"] != MODEL:
                raise ValidationError(f"Standalone agent {path.name} declares a different model")
        root_yaml = self.resolve("agent.yaml", self.root)
        self.agent_stack.add(root_yaml)
        self.agent(self.load_yaml(root_yaml), root_yaml)
        self.agent_stack.remove(root_yaml)
        evaluation = None
        eval_path = self.root / "eval_config.yaml"
        if eval_path.exists():
            evaluation = self.load_yaml(eval_path)
            if not isinstance(evaluation, dict) or not isinstance(evaluation.get("evaluation"), dict):
                raise ValidationError("eval_config.yaml needs an evaluation mapping")
            allowed = {"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"}
            for key, value in evaluation["evaluation"].items():
                if key not in allowed or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValidationError(f"Invalid evaluation setting: {key}")
                if key != "max_time_minutes" and not isinstance(value, int):
                    raise ValidationError(f"{key} must be an integer")
        stale = [p.relative_to(self.root).as_posix() for p in all_files if p.relative_to(self.root).parts[0] == "adapters" and p not in self.files]
        if stale:
            raise ValidationError("Unreferenced adapter assets: " + ", ".join(sorted(stale)))
        loose_weights = [p for p in all_files if p.suffix.lower() == ".safetensors" and p not in self.files]
        if loose_weights:
            raise ValidationError("Unreferenced weight assets are not allowed")
        total = sum(p.stat().st_size for p in self.files)
        if total >= MAX_BYTES:
            raise ValidationError("Unpacked package must be smaller than 3 GiB")
        packaged = sorted(p.relative_to(self.root).as_posix() for p in self.files)
        ignored = sorted(p.relative_to(self.root).as_posix() for p in all_files if p not in self.files)
        if ignored:
            self.warnings.append("Build omits unreferenced files; inspect omitted_files before packaging")
        return {
            "valid": True, "submission_dir": str(self.root), "model": MODEL,
            "agents": self.agents, "adapters": sorted(self.adapter_names),
            "agent_tool_names": sorted(self.agent_tool_names), "package_bytes": total,
            "package_files": packaged, "omitted_files": ignored,
            "evaluation": evaluation.get("evaluation") if evaluation else None,
            "warnings": self.warnings,
            "scope": "Documented local checks; the official runtime compiler remains authoritative",
        }


def validate_submission(directory: str | Path) -> dict:
    return SubmissionValidator(directory).validate()


def build_submission(directory: str | Path, output: str | Path) -> dict:
    validator = SubmissionValidator(directory)
    manifest = validator.validate()
    destination = Path(output).absolute().resolve()
    if _inside(destination, validator.root):
        raise ValidationError("Build output must be outside the submission directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in manifest["package_files"]:
            archive.write(validator.root / name, arcname=name)
    return {**manifest, "archive": str(destination), "archive_bytes": destination.stat().st_size}


def _number(value: Any) -> float | int | None:
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _pick(mapping: dict, *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _truth(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def _timestamp(value: Any) -> float | None:
    if _number(value) is not None:
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return None


def _artifact(root: Path, subdir: str, filename: str) -> Path | None:
    target = (root / subdir / filename).resolve()
    return target if _inside(target, root) and target.is_file() and not target.is_symlink() else None


def _structured_result(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def trace_metrics(path: Path | None, analyzer_names: set[str], reviewer_names: set[str] | None = None) -> dict:
    reviewer_names = reviewer_names or set()
    result = {
        "trace_present": path is not None, "trace_valid": None,
        "trace_schema_supported": None, "tool_calls_by_name": None,
        "test_command_calls_observed": None,
        "model_requests": None, "traced_tool_calls": None, "known_budgeted_tool_calls": None,
        "analyzer_calls": None, "reviewer_calls": None, "tool_errors": None, "first_successful_edit_trace_seconds": None,
        "input_tokens_sum_observed": None, "output_tokens_sum_observed": None,
        "max_request_input_tokens": None, "requests_with_input_tokens": None,
        "trace_cumulative_token_metrics": None, "trace_warnings": [],
    }
    if path is None:
        return result
    try:
        trace = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["trace_valid"] = False
        result["trace_warnings"].append(f"Unreadable trace: {type(exc).__name__}")
        return result
    if not isinstance(trace, dict) or not isinstance(trace.get("steps"), list):
        result["trace_valid"] = False
        result["trace_warnings"].append("Unsupported trace shape; expected a mapping with steps[]")
        return result
    result["trace_valid"] = True
    cumulative = trace.get("final_metrics", trace.get("metrics"))
    if isinstance(cumulative, dict):
        result["trace_cumulative_token_metrics"] = {
            key: value for key, value in cumulative.items() if "token" in key.lower() and _number(value) is not None
        } or None
    calls: list[dict] = []
    observations: dict[str, dict] = {}
    inputs: list[int | float] = []
    outputs: list[int | float] = []
    requests = 0
    malformed_calls = 0
    recognized = not trace["steps"]
    canonical = all(isinstance(step, dict) and step.get("source", step.get("role")) in ("agent", "assistant", "tool", "user", "system") for step in trace["steps"])
    trace_start = _timestamp(_pick(trace, "start_time", "timestamp", "task_start_time"))
    for step in trace["steps"]:
        if not isinstance(step, dict):
            result["trace_warnings"].append("Non-mapping step ignored")
            continue
        source = step.get("source", step.get("role"))
        kind = step.get("type", step.get("step_type"))
        model_step = source in ("agent", "assistant") or kind in ("model_response", "llm_response", "model_call", "llm_call")
        if model_step:
            recognized = True
            requests += 1
            usage = next((step[key] for key in ("metrics", "usage", "token_usage", "usage_metadata") if isinstance(step.get(key), dict)), {})
            input_tokens = _number(_pick(usage, "prompt_tokens", "input_tokens", "prompt_token_count"))
            output_tokens = _number(_pick(usage, "completion_tokens", "output_tokens", "candidates_token_count"))
            if input_tokens is not None:
                inputs.append(input_tokens)
            if output_tokens is not None:
                outputs.append(output_tokens)
        elapsed = _number(_pick(step, "elapsed_seconds", "relative_seconds"))
        if elapsed is None and trace_start is not None:
            stamp = _timestamp(step.get("timestamp"))
            elapsed = stamp - trace_start if stamp is not None and stamp >= trace_start else None
        declared = step.get("tool_calls")
        if isinstance(declared, list):
            recognized = True
        elif isinstance(step.get("tool_call"), dict):
            declared = [step["tool_call"]]
            recognized = True
        elif kind == "tool_call" and isinstance(step.get("tool_name"), str):
            declared = [step]
            recognized = True
        else:
            declared = []
        for call in declared:
            if not isinstance(call, dict):
                result["trace_warnings"].append("Non-mapping tool call ignored")
                malformed_calls += 1
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = _pick(call, "function_name", "tool_name", "name") or function.get("name")
            if not isinstance(name, str):
                result["trace_warnings"].append("Tool call with unknown name ignored")
                malformed_calls += 1
                continue
            arguments = _structured_result(call.get("arguments", function.get("arguments"))) or {}
            calls.append({"name": name, "id": str(_pick(call, "tool_call_id", "call_id", "id") or ""),
                          "elapsed": elapsed, "result": _structured_result(_pick(call, "result", "output")),
                          "command": arguments.get("command")})
        observation = step.get("observation")
        if isinstance(observation, dict) and isinstance(observation.get("results"), list):
            for observed in observation["results"]:
                if isinstance(observed, dict):
                    identifier = _pick(observed, "source_call_id", "tool_call_id", "call_id")
                    value = _structured_result(_pick(observed, "content", "result", "output"))
                    if identifier is not None and value is not None:
                        observations[str(identifier)] = value
        if source == "tool" or kind in ("tool_result", "tool_response"):
            identifier = _pick(step, "tool_call_id", "source_call_id", "call_id")
            value = _structured_result(_pick(step, "content", "message", "result", "output"))
            if identifier is not None and value is not None:
                observations[str(identifier)] = value
    if not recognized:
        result["trace_schema_supported"] = False
        result["trace_warnings"].append("Unrecognized step schema; request/tool metrics unavailable")
        return result
    result["trace_schema_supported"] = True
    frequency = dict(sorted(Counter(call["name"] for call in calls).items()))
    unknown_names = set(frequency) - CORE_TOOLS - SKILL_TOOLS - analyzer_names - reviewer_names
    if unknown_names:
        result["trace_warnings"].append("Unclassified tool names (may be other AgentTools): " + ", ".join(sorted(unknown_names)))
    if malformed_calls:
        result["trace_warnings"].append("Tool-call counts are partial because some calls could not be parsed")
    result.update({
        "model_requests": requests if requests or canonical else None, "traced_tool_calls": len(calls),
        "tool_calls_by_name": frequency,
        "test_command_calls_observed": sum(call["name"] == "run_command" and isinstance(call["command"], str) and bool(re.search(r"\b(?:pytest|unittest|nosetests|tox)\b", call["command"])) for call in calls),
        "known_budgeted_tool_calls": sum(c["name"] in ((CORE_TOOLS - FREE_TOOLS) | SKILL_TOOLS) for c in calls),
        "analyzer_calls": sum(c["name"] in analyzer_names for c in calls),
        "reviewer_calls": sum(c["name"] in reviewer_names for c in calls),
        "requests_with_input_tokens": len(inputs),
        "input_tokens_sum_observed": sum(inputs) if inputs else None,
        "output_tokens_sum_observed": sum(outputs) if outputs else None,
        "max_request_input_tokens": max(inputs) if inputs else None,
    })
    statuses = []
    edit_times = []
    for call in calls:
        observed = call["result"] or observations.get(call["id"])
        if observed is None:
            continue
        status = observed.get("status")
        if status in ("ok", "success", "error"):
            statuses.append(status)
        if call["name"] in {"edit_file", "write_file"} and status in ("ok", "success") and call["elapsed"] is not None:
            edit_times.append(call["elapsed"])
    result["tool_errors"] = sum(status == "error" for status in statuses) if statuses else None
    result["first_successful_edit_trace_seconds"] = min(edit_times) if edit_times else None
    if requests and len(inputs) != requests:
        result["trace_warnings"].append("Per-request input token metrics are incomplete")
    if calls and len(statuses) != len(calls):
        result["trace_warnings"].append("Some tool outcomes are unavailable; successful edit time may be incomplete")
    return result


STOP_PATTERNS = [
    ("time", r"TimeExceeded|AgentTimedOut|agent (?:session )?timed? ?out|(?:agent|session) timeout|time (?:budget|limit|allowance).{0,35}(?:exhaust|exceed|reached)|(?:exhaust|exceed).{0,25}time budget"),
    ("tool_budget", r"(?:tool.?calls? (?:budget|limit)|max_tool_calls).{0,35}(?:exhaust|exceed|reached)|(?:exhaust|exceed).{0,25}tool.?call"),
    ("turn_budget", r"(?:turn (?:budget|limit)|max_turns|max_loop_iterations).{0,35}(?:exhaust|exceed|reached)"),
    ("context", r"ContextWindowExceeded|context (?:window|length).{0,30}(?:exceed|overflow)|maximum context length"),
    ("setup", r"(?:failed|unable) to (?:create|start|prepare).{0,20}(?:sandbox|container)|(?:snapshot|wheel|dependency|container) (?:setup|installation|extraction) failed|ContainerSetupError|SandboxSetupError"),
]


def classify(record: dict, patch_present: bool | None, agent_log: str, test_log: str) -> tuple[str, str, list[str]]:
    error = record.get("error")
    metadata = " ".join(str(record.get(k) or "") for k in ("stop_reason", "exit_status", "error_type", "error", "stage"))
    stop = "unknown"
    evidence: list[str] = []
    explicit_stop = _pick(record, "stop_reason", "exit_status")
    if isinstance(explicit_stop, str) and explicit_stop.lower() in {"submitted", "complete", "completed", "finished"}:
        stop = "submitted"
        evidence.append(f"Recorded stop: {explicit_stop}")
    # Metadata is authoritative. Transcript fallback requires a harness logging
    # severity prefix, rather than a matching phrase in model thoughts/tool output.
    qualified_log = "\n".join(line for line in agent_log.splitlines() if re.search(r"(?:^|\s|\[)(?:ERROR|WARNING|CRITICAL)(?:\s|:|\])", line))
    for name, pattern in STOP_PATTERNS:
        match = re.search(pattern, metadata, re.I)
        if match is None and stop == "unknown":
            match = re.search(pattern, qualified_log, re.I)
        if match:
            stop = name
            evidence.append(match.group(0)[:200])
            break
    if stop == "unknown" and error:
        if re.search(r"FileEditError|ToolExecutionError|invalid tool call|malformed tool|unknown tool|tool.call.{0,20}truncat", metadata, re.I):
            stop = "tool_error"
        elif re.search(r"timeout|timed out", metadata, re.I):
            stop = "timeout_unspecified"
        evidence.append(str(error)[:400])
    resolved = _truth(record.get("resolved"))
    if resolved is True:
        return "solved", stop, evidence
    if re.search(r"failed to apply (?:agent )?patch|PatchApplyError", metadata + "\n" + test_log, re.I):
        return "patch_application_failure", stop, evidence + ["Patch application error recorded"]
    if stop in {"setup", "context", "tool_budget", "turn_budget", "tool_error", "timeout_unspecified"}:
        return stop, stop, evidence
    if stop == "time":
        return ("timeout_with_patch" if patch_present is True else "timeout_before_patch" if patch_present is False else "timeout_patch_unknown"), stop, evidence
    if patch_present is False:
        return "empty_patch", stop, evidence + ["Recorded patch is empty"]
    exit_code = _number(_pick(record, "test_exit_code", "verification_exit_code"))
    if resolved is False and exit_code is not None and exit_code >= 0:
        if re.search(r"ImportError|ModuleNotFoundError|collected 0 items", test_log):
            return "verification_environment_failure", stop, evidence + ["Verification output reports import/collection failure"]
        return "verification_failure", stop, evidence + [f"Unresolved with verification exit code {exit_code}"]
    if error:
        return "reported_error", stop, evidence
    return "unknown", stop, evidence


def verification_status(record: dict, patch_present: bool | None) -> str:
    if _truth(record.get("resolved")) is True:
        return "passed"
    error = str(record.get("error") or "")
    if re.search(r"failed to apply (?:agent )?patch|PatchApplyError", error, re.I):
        return "not_run"
    code = _number(_pick(record, "test_exit_code", "verification_exit_code"))
    if code is not None and code >= 0:
        return "failed" if code != 0 or _truth(record.get("resolved")) is False else "unknown"
    if patch_present is False or code == -1:
        return "not_run"
    return "unknown"


SUGGESTIONS = {
    "solved": "Retain as regression coverage; compare runtime and calls with the control.",
    "timeout_before_patch": "Inspect localization and model latency; compare a larger agent allowance with the same task.",
    "timeout_with_patch": "Check verification and completion discipline before changing adapters or architecture.",
    "timeout_patch_unknown": "Recover the patch artifact and session trace before diagnosing the timeout.",
    "timeout_unspecified": "Determine whether the timeout affected a command, inference request, setup, or whole session.",
    "tool_budget": "Inspect repeated navigation and unsuccessful edits; reserve calls for verification.",
    "turn_budget": "Inspect repeated responses and incomplete tool calls before increasing turn limits.",
    "context": "Reduce broad reads and preserve a concise task state; compare selective analyzer delegation.",
    "setup": "Check snapshots, wheels, sandbox backend, and environment; this is not a patch-quality result.",
    "tool_error": "Inspect tool arguments and call truncation; use smaller uniquely targeted edits.",
    "empty_patch": "Check premature completion, repeated searches, and whether any edit succeeded.",
    "patch_application_failure": "Inspect patch paths and formatting against the task baseline.",
    "verification_failure": "Inspect targeted test output and task requirements; exit zero alone is insufficient.",
    "verification_environment_failure": "Separate dependency/collection problems from incorrect task logic.",
    "reported_error": "Read the recorded error and task artifacts; the failure is not classified further.",
    "unknown": "Capture the final stop reason, agent elapsed time, trace, patch, and verification output.",
}


def make_report(results_dir: str | Path, output_dir: str | Path, submission_dir: str | Path | None = None) -> dict:
    root = Path(results_dir).resolve()
    source = root / "notebook_task_results.jsonl"
    if not source.is_file():
        source = root / "task_results.jsonl"
    if not source.is_file():
        raise ValueError(f"Missing task_results.jsonl: {root}")
    warnings: list[str] = []
    records: dict[str, dict] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row is not a mapping")
            identifier = _pick(row, "instance_id", "task_id", "id")
            if not isinstance(identifier, (str, int)) or not str(identifier):
                raise ValueError("row has no task identifier")
            identifier = str(identifier)
            if identifier in records:
                warnings.append(f"Duplicate task {identifier}: report uses the last finished record")
            records[identifier] = row
        except (json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"task_results.jsonl line {line_number}: {exc}")
    configuration = validate_submission(submission_dir) if submission_dir else None
    analyzer_names = {"code_analyzer_agent"}
    reviewer_names = {"patch_reviewer_agent"}
    if configuration:
        analyzer_names |= {name for name in configuration["agent_tool_names"] if "analy" in name.lower()}
        reviewer_names |= {name for name in configuration["agent_tool_names"] if "review" in name.lower()}
    tasks = []
    for identifier, original in records.items():
        row = dict(original)
        raw = row.get("raw_result", row.get("metadata"))
        if isinstance(raw, dict):
            for key in ("error", "error_type", "exit_status", "stop_reason", "stage", "agent_elapsed_seconds"):
                if row.get(key) is None and raw.get(key) is not None:
                    row[key] = raw[key]
        patch_path = _artifact(root, "patches", f"{identifier}.patch")
        agent_paths = [path for filename in (f"{identifier}.log", f"observer_{identifier}.log", f"exception_{identifier}.log") if (path := _artifact(root, "logs", filename)) is not None]
        test_path = _artifact(root, "test_outputs", f"{identifier}.log")
        trace_path = _artifact(root, "traces", f"trace_{identifier}.json")
        patch = row.get("agent_patch", row.get("prediction"))
        sentinel = isinstance(patch, str) and patch.strip() == "NO_PATCH"
        patch_chars = (0 if sentinel else len(patch)) if isinstance(patch, str) else _number(_pick(row, "patch_chars", "patch_size"))
        patch_bytes = patch_path.stat().st_size if patch_path else None
        file_sentinel = patch_path is not None and patch_bytes is not None and patch_bytes < 100 and patch_path.read_text(encoding="utf-8", errors="replace").strip() == "NO_PATCH"
        file_patch_present = False if file_sentinel else patch_bytes > 0 if patch_bytes is not None else None
        patch_present = patch_chars > 0 if patch_chars is not None else file_patch_present
        if patch_chars is not None and file_patch_present is not None and patch_present != file_patch_present:
            warnings.append(f"{identifier}: record and patch artifact disagree on patch presence; record used")
        agent_log = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in agent_paths)
        test_log = test_path.read_text(encoding="utf-8", errors="replace") if test_path else ""
        category, stop, evidence = classify(row, patch_present, agent_log, test_log)
        observed = trace_metrics(trace_path, analyzer_names, reviewer_names)
        tasks.append({
            "instance_id": identifier, "repo": row.get("repo"), "resolved": _truth(row.get("resolved")),
            "test_exit_code": _number(_pick(row, "test_exit_code", "verification_exit_code")),
            "tool_calls": _number(_pick(row, "tool_calls", "tool_calls_used")),
            "duration_seconds": _number(_pick(row, "duration_seconds", "duration")),
            "agent_elapsed_seconds": _number(row.get("agent_elapsed_seconds")),
            "observer_wall_seconds": _number(row.get("observer_wall_seconds")),
            "patch_present": patch_present, "patch_chars": patch_chars, "patch_bytes": patch_bytes,
            "failure_category": category, "stop_cause": stop, "evidence": evidence,
            "verification_status": verification_status(row, patch_present),
            "suggestion": SUGGESTIONS[category],
            "error": str(row["error"])[:600] if row.get("error") else None,
            "agent_log_present": bool(agent_paths), "test_output_present": test_path is not None,
            **observed,
        })
    count = len(tasks)
    known = sum(task["resolved"] is not None for task in tasks)
    solved = sum(task["resolved"] is True for task in tasks)
    per_repository = {}
    for repo in sorted({str(task["repo"]) if task["repo"] is not None else "unknown" for task in tasks}):
        subset = [task for task in tasks if (str(task["repo"]) if task["repo"] is not None else "unknown") == repo]
        known_count = sum(task["resolved"] is not None for task in subset)
        solved_count = sum(task["resolved"] is True for task in subset)
        per_repository[repo] = {
            "count": len(subset), "known_resolution_count": known_count, "resolved": solved_count,
            "resolution_rate": solved_count / len(subset) if known_count == len(subset) else None,
            "resolution_rate_known_records": solved_count / known_count if known_count else None,
        }
    summary = {
        "finished_task_records": count, "known_resolution_records": known,
        "resolved": solved, "unresolved": sum(task["resolved"] is False for task in tasks),
        "resolution_rate": solved / count if count and known == count else None,
        "resolution_rate_known_records": solved / known if known else None,
        "trace_present": sum(task["trace_present"] for task in tasks),
        "trace_valid": sum(task["trace_valid"] is True for task in tasks),
        "trace_schema_supported": sum(task["trace_schema_supported"] is True for task in tasks),
        "nonempty_patch_records": sum(task["patch_present"] is True for task in tasks),
        "patch_presence_unknown": sum(task["patch_present"] is None for task in tasks),
        "failure_categories": dict(sorted(Counter(task["failure_category"] for task in tasks).items())),
        "per_repository": per_repository,
        "tool_calls_by_name_observed": dict(sorted(sum((Counter(task["tool_calls_by_name"] or {}) for task in tasks), Counter()).items())),
    }
    report = {
        "results_dir": str(root), "source": str(source), "summary": summary,
        "configuration": configuration, "tasks": tasks, "warnings": warnings,
        "interpretation": [
            "Rates cover finished records only; missing/unstarted tasks are not in this denominator.",
            "duration_seconds is the recorded total task duration, not inferred agent-loop time.",
            "Missing metrics remain null; unsupported or incomplete traces cannot establish zero usage.",
            "max_request_input_tokens uses per-agent-step input usage only; cumulative tokens are separate.",
            "first_successful_edit_trace_seconds is relative to the trace clock, not necessarily the agent-session clock.",
            "known_budgeted_tool_calls counts recognized chargeable built-ins/skill tools; get_status and submit_patch are free.",
            "Analyzer calls are identified by known configured analyzer names; delegation does not imply a separate model.",
            "Reviewer calls are identified by known configured reviewer names; a call count does not establish review benefit or successful repair.",
            "Tool frequencies cover parsed trace calls; test_command_calls_observed recognizes named pytest/unittest/nosetests/tox commands only.",
            "Stop causes require explicit recorded error/exhaustion evidence, not duration or budget comparisons.",
        ],
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "instance_id", "repo", "resolved", "failure_category", "stop_cause", "verification_status", "test_exit_code",
        "patch_present", "patch_chars", "patch_bytes", "duration_seconds", "agent_elapsed_seconds", "observer_wall_seconds",
        "tool_calls", "model_requests", "traced_tool_calls", "known_budgeted_tool_calls", "analyzer_calls", "reviewer_calls",
        "first_successful_edit_trace_seconds", "max_request_input_tokens", "input_tokens_sum_observed",
        "output_tokens_sum_observed", "requests_with_input_tokens", "trace_present", "trace_valid",
        "trace_schema_supported", "test_command_calls_observed",
    ]
    with (destination / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tasks)
    rate = f"{summary['resolution_rate']:.1%}" if summary["resolution_rate"] is not None else "unknown"
    markdown = [
        "# Evaluation diagnostics", "", f"Resolved **{solved}/{count}** finished records; recorded resolution rate: **{rate}**.",
        f"Traces present: {summary['trace_present']}/{count}; valid envelopes: {summary['trace_valid']}/{count}; recognized step schemas: {summary['trace_schema_supported']}/{count}.", "",
        "| Task | Resolved | Failure category | Verification | Stop evidence | Total task seconds | Agent seconds |", "|---|---|---|---|---|---:|---:|",
    ]
    for task in tasks:
        evidence = "; ".join(task["evidence"]) or "Not recorded"
        values = [task["instance_id"], task["resolved"], task["failure_category"], task["verification_status"], evidence, task["duration_seconds"], task["agent_elapsed_seconds"]]
        markdown.append("| " + " | ".join("unknown" if value is None else str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ") for value in values) + " |")
    markdown.extend(["", "## Suggested checks", ""])
    for category in summary["failure_categories"]:
        markdown.append(f"- **{category}**: {SUGGESTIONS[category]}")
    markdown.extend(["", "## Observed tool calls", ""])
    frequency = summary["tool_calls_by_name_observed"]
    markdown.extend([f"- `{name}`: {frequency[name]}" for name in frequency] or ["No parsed tool-call records available."])
    markdown.extend(["", "## Repository results", "", "| Repository | Resolved | Known outcomes | Finished records |", "|---|---:|---:|---:|"])
    for repo, stats in per_repository.items():
        markdown.append(f"| {repo.replace('|', '/')} | {stats['resolved']} | {stats['known_resolution_count']} | {stats['count']} |")
    trace_warnings = [f"{task['instance_id']}: {warning}" for task in tasks for warning in task["trace_warnings"]]
    if trace_warnings:
        markdown.extend(["", "## Trace warnings", "", *[f"- {warning}" for warning in trace_warnings]])
    markdown.extend(["", "## Limits of these observations", "", *[f"- {item}" for item in report["interpretation"]]])
    if warnings:
        markdown.extend(["", "## Input warnings", "", *[f"- {item}" for item in warnings]])
    (destination / "diagnostics.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "build"):
        child = subparsers.add_parser(name)
        child.add_argument("--submission-dir", required=True)
        if name == "build":
            child.add_argument("--output", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--results-dir", required=True)
    report.add_argument("--output-dir", required=True)
    report.add_argument("--submission-dir")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_submission(args.submission_dir)
        elif args.command == "build":
            result = build_submission(args.submission_dir, args.output)
        else:
            result = make_report(args.results_dir, args.output_dir, args.submission_dir)["summary"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValidationError, ValueError, OSError, yaml.YAMLError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
