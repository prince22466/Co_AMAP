"""Regenerate the Kaggle companion from the current curated submission and tools.

Run: python submission_nb/prepare_evaluation_notebook.py
Prompt comparison: add --submission-dir submission_nb/submission_2
--baseline-dir submission_nb/submission_1 --output submission_nb/evaluate_submission_2.ipynb
The embedded candidate contains text configuration only. Future trained weights
should be supplied through the notebook's SUBMISSION_SOURCE_DIR setting.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from collections import defaultdict
from pathlib import Path

import nbformat
import yaml


def remove_agent_tools(text):
    """Remove delegation from tools while preserving !include YAML nodes."""
    node = yaml.compose(text)
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("agent.yaml must be a mapping")
    removed = 0
    for key, value in node.value:
        if key.value == "tools":
            if not isinstance(value, yaml.SequenceNode):
                raise ValueError("tools must be a sequence")
            keep = []
            for item in value.value:
                is_delegate = isinstance(item, yaml.MappingNode) and any(
                    k.value == "agent_tool" for k, _ in item.value
                )
                removed += int(is_delegate)
                if not is_delegate:
                    keep.append(item)
            value.value = keep
    if not removed:
        raise ValueError("Candidate has no agent_tool to remove for coder_only")
    return yaml.serialize(node)


def task_resources(task, data_dir):
    """Check task-named/commit-named files described by the public dataset."""
    def field(name):
        return task[name] if isinstance(task, dict) else getattr(task, name)

    identifier = field("instance_id")
    repo_short = field("repo").split("/")[-1]
    commit_stem = repo_short + "_" + field("base_commit")
    root = Path(data_dir)
    choices = {
        "snapshot": [root / "snapshots" / (identifier + ".tgz")],
        "graph": [root / "graphs" / (commit_stem + ".json"), root / "graphs" / (identifier + ".json")],
        "embedding": [root / "embeddings" / (commit_stem + ".npz"), root / "embeddings" / (identifier + ".npz")],
    }
    return {kind: next((str(p) for p in paths if p.is_file() and p.stat().st_size > (0 if kind == 'snapshot' else 100)), None)
            for kind, paths in choices.items()}


def select_tasks(tasks, data_dir, task_ids=None, limit=2):
    """Select complete tasks deterministically, round robin by repository."""
    def field(task, name):
        return task[name] if isinstance(task, dict) else getattr(task, name)

    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError("TASK_LIMIT must be a positive integer or None")
    indexed = {field(t, "instance_id"): t for t in tasks}
    if len(indexed) != len(tasks):
        raise ValueError("Duplicate instance_id in tasks.jsonl")
    coverage = {identifier: task_resources(t, data_dir) for identifier, t in indexed.items()}
    complete = {identifier for identifier, resources in coverage.items() if all(resources.values())}
    if task_ids:
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("TASK_IDS contains duplicates")
        missing = {identifier: coverage.get(identifier, "unknown task") for identifier in task_ids
                   if identifier not in complete}
        if missing:
            raise ValueError("TASK_IDS lack complete matching resources: " + json.dumps(missing))
        selected = [indexed[identifier] for identifier in task_ids]
    else:
        by_repo = defaultdict(list)
        for identifier in sorted(complete):
            task = indexed[identifier]
            by_repo[field(task, "repo")].append(task)
        selected = []
        while any(by_repo.values()):
            for repo in sorted(by_repo):
                if by_repo[repo]:
                    selected.append(by_repo[repo].pop(0))
        if limit is not None:
            selected = selected[:limit]
    if not selected:
        raise ValueError("No task has matching snapshot, graph and embedding files. Attach the full competition dataset.")
    return selected, coverage


def extract_curated(archive_path, destination):
    """Extract a curated archive to a fresh directory, validating member paths."""
    import shutil

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (resolved / info.filename).resolve()
            if not target.is_relative_to(resolved) or "\\" in info.filename:
                raise ValueError("Unsafe archive member: " + info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)


def file_sha256(path):
    """Hash large adapter assets without materializing the whole file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


CONFIG = '''
from pathlib import Path

DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
# Set to a mounted copy of YOUR candidate to test later configurations/adapters.
# None uses the curated candidate embedded when this notebook was generated.
SUBMISSION_SOURCE_DIR = None
BENCHMARK_VARIANTS = ['coder_only', 'optional_analyzer']
RELEASE_VARIANT = 'optional_analyzer'
TASK_IDS = []  # Explicit IDs take precedence over TASK_LIMIT.
TASK_LIMIT = 2  # Smoke test only; None evaluates all tasks with complete resources.
SANDBOX = 'subprocess'  # Kaggle starter mode. Use 'docker' where Docker/image are available.
WORKING_DIR = Path('/kaggle/working')
'''

PREPARATION = '''
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
import yaml

if SANDBOX not in {'subprocess', 'docker'}:
    raise ValueError('SANDBOX must be subprocess or docker')
if not BENCHMARK_VARIANTS or len(set(BENCHMARK_VARIANTS)) != len(BENCHMARK_VARIANTS):
    raise ValueError('Choose unique benchmark variants')
allowed_variants = {'baseline', 'candidate'} if COMPARISON_MODE == 'prompt_comparison' else {'coder_only', 'optional_analyzer'}
if set(BENCHMARK_VARIANTS) - allowed_variants:
    raise ValueError('Unknown benchmark variant')
if RELEASE_VARIANT not in BENCHMARK_VARIANTS:
    raise ValueError('RELEASE_VARIANT must be evaluated in this run')
run_parent = WORKING_DIR / 'agent_runs'
run_parent.mkdir(parents=True, exist_ok=True)
run_prefix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ_')
RUN_DIR = Path(tempfile.mkdtemp(prefix=run_prefix, dir=run_parent))
AUX_DIR = RUN_DIR / 'tools'
AUX_DIR.mkdir()
TOOL_SCRIPT = AUX_DIR / 'submission_tools.py'
TOOL_SCRIPT.write_text(EMBEDDED_TOOL_SOURCE, encoding='utf-8')

def tool_cli(*args):
    process = subprocess.run([sys.executable, str(TOOL_SCRIPT), *map(str, args)],
                             capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(process.stderr or process.stdout)
    return json.loads(process.stdout)

candidate_variant = 'candidate' if COMPARISON_MODE == 'prompt_comparison' else 'optional_analyzer'
candidate_dir = RUN_DIR / 'variants' / candidate_variant
candidate_zip = RUN_DIR / 'candidate_source.zip'
if SUBMISSION_SOURCE_DIR is not None:
    source = Path(SUBMISSION_SOURCE_DIR)
    if not (source / 'agent.yaml').is_file():
        raise ValueError('SUBMISSION_SOURCE_DIR must contain your candidate agent.yaml')
    tool_cli('build', '--submission-dir', source, '--output', candidate_zip)
    extract_curated(candidate_zip, candidate_dir)
else:
    candidate_dir.mkdir(parents=True, exist_ok=False)
    for name, content in EMBEDDED_SUBMISSION.items():
        target = candidate_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')

VARIANT_DIRS = {candidate_variant: candidate_dir}
if COMPARISON_MODE == 'prompt_comparison':
    baseline_dir = RUN_DIR / 'variants' / 'baseline'
    baseline_dir.mkdir(parents=True, exist_ok=False)
    for name, content in EMBEDDED_BASELINE.items():
        target = baseline_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
    VARIANT_DIRS['baseline'] = baseline_dir
elif 'coder_only' in BENCHMARK_VARIANTS:
    control_source = RUN_DIR / 'control_source'
    shutil.copytree(candidate_dir, control_source)
    control_yaml = control_source / 'agent.yaml'
    control_yaml.write_text(remove_agent_tools(control_yaml.read_text(encoding='utf-8')), encoding='utf-8')
    # Resolve the changed config before validation to stage only reachable files.
    # This retains coder adapters and excludes analyzer-only adapter directories.
    import importlib.util
    spec = importlib.util.spec_from_file_location('notebook_submission_tools', TOOL_SCRIPT)
    curator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(curator)
    closure = curator.SubmissionValidator(control_source)
    closure.agent(closure.load_yaml(control_yaml), control_yaml)
    control_eval = control_source / 'eval_config.yaml'
    if control_eval.is_file():
        closure.load_yaml(control_eval)
    control_stage = RUN_DIR / 'control_curated'
    control_stage.mkdir()
    for file in sorted(closure.files):
        target = control_stage / file.relative_to(closure.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    control_zip = RUN_DIR / 'coder_only_source.zip'
    tool_cli('build', '--submission-dir', control_stage, '--output', control_zip)
    control_dir = RUN_DIR / 'variants' / 'coder_only'
    extract_curated(control_zip, control_dir)
    VARIANT_DIRS['coder_only'] = control_dir

VARIANT_MANIFESTS = {name: tool_cli('validate', '--submission-dir', VARIANT_DIRS[name])
                     for name in BENCHMARK_VARIANTS}
budgets = [manifest['evaluation'] for manifest in VARIANT_MANIFESTS.values()]
if any(budget != budgets[0] for budget in budgets):
    raise ValueError('Benchmark variants must use identical evaluation budgets')
if COMPARISON_MODE == 'prompt_comparison':
    # Check both source variants even when BENCHMARK_VARIANTS selects only one.
    comparison_manifests = {name: tool_cli('validate', '--submission-dir', VARIANT_DIRS[name])
                            for name in ['baseline', 'candidate']}
    if any(manifest['agents'] != 2 for manifest in comparison_manifests.values()):
        raise ValueError('This prompt comparison requires the same two-agent system in both variants')
    if any(manifest['adapters'] for manifest in comparison_manifests.values()):
        raise ValueError('This prompt comparison uses unchanged base weights. Compare adapter changes in a separate mounted-weight experiment.')
    if comparison_manifests['baseline']['model'] != comparison_manifests['candidate']['model']:
        raise ValueError('Baseline and candidate must use the same base model')
    if comparison_manifests['baseline']['evaluation'] != comparison_manifests['candidate']['evaluation']:
        raise ValueError('Baseline and candidate must use identical evaluation budgets')
    spec = importlib.util.spec_from_file_location('notebook_comparison_tools', TOOL_SCRIPT)
    curator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(curator)

    def without_instruction(value):
        if isinstance(value, dict):
            return {key: without_instruction(item) for key, item in value.items() if key != 'instruction'}
        if isinstance(value, list):
            return [without_instruction(item) for item in value]
        return value

    def resolved_configuration(name):
        directory = VARIANT_DIRS[name]
        loader = curator.SubmissionValidator(directory)
        return {relative: without_instruction(loader.load_yaml(directory / relative))
                for relative in comparison_manifests[name]['package_files']
                if Path(relative).suffix.lower() in {'.yaml', '.yml'}}

    if resolved_configuration('baseline') != resolved_configuration('candidate'):
        raise ValueError('Prompt comparison requires matching YAML topology, sampling and budget settings; only instruction text may differ')
print('Run:', RUN_DIR)
print('Comparison mode:', COMPARISON_MODE)
print('Variants:', BENCHMARK_VARIANTS)
print('Budgets:', budgets[0])
print('Sandbox:', SANDBOX, '(Docker scoring parity requires a separate Docker run)' if SANDBOX == 'subprocess' else '')
'''

PREFLIGHT = '''
from swegemma.models import load_tasks

TASKS_PATH = DATA_DIR / 'tasks.jsonl'
for required in [TASKS_PATH, DATA_DIR / 'wheels', DATA_DIR / 'sandbox' / 'setup.py', MODEL_PATH]:
    if not required.exists():
        raise FileNotFoundError(f'Missing required dataset/model resource: {required}')
if not any((DATA_DIR / 'wheels').glob('*.whl')):
    raise ValueError('Task dependency wheels directory is empty')
tasks = load_tasks(TASKS_PATH)
SELECTED_TASKS, COVERAGE = select_tasks(tasks, DATA_DIR, TASK_IDS, TASK_LIMIT)
GRAPH_DIR = str(DATA_DIR / 'graphs')
EMBEDDINGS_DIR = str(DATA_DIR / 'embeddings')
selected_ids = [task.instance_id for task in SELECTED_TASKS]
(RUN_DIR / 'task_selection.json').write_text(json.dumps({
    'task_ids': selected_ids, 'selection': 'explicit' if TASK_IDS else 'stable_round_robin_by_repo',
    'available_tasks': len(tasks), 'complete_tasks': sum(all(row.values()) for row in COVERAGE.values()),
    'resources': {identifier: COVERAGE[identifier] for identifier in selected_ids},
}, indent=2), encoding='utf-8')
print(f'Selected {len(SELECTED_TASKS)} identical tasks for each variant:', selected_ids)
if len(SELECTED_TASKS) <= 2:
    print('This is a smoke test, not reliable evidence of general performance.')
'''

SERVER = '''
import litellm
import torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
from swegemma.models.discovery import validate_single_declared_model

litellm.drop_params = True
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
INFERENCE_API_KEY = 'EMPTY'  # Local vLLM routing placeholder, no external API.
declared_model = validate_single_declared_model(candidate_dir)
if declared_model != TARGET_MODEL_NAME:
    raise ValueError('Use the competition-required Gemma base model')
adapters = discover_adapters(str(candidate_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
if not gpu_count:
    raise RuntimeError('Enable Kaggle GPU acceleration before starting the model')
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
vllm_cfg = VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1',
    tool_call_parser='gemma4', reasoning_parser='gemma4', max_model_len=32768,
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
    gpu_memory_utilization=0.80, enable_auto_tool_choice=True,
    enable_lora=bool(adapters), max_loras=8, max_lora_rank=128,
    tensor_parallel_size=tp_size, startup_timeout=60 * 20,
)
server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)
server_instance.start()
models = server_instance.create_model_registry(
    aliases=[declared_model, TARGET_MODEL_NAME], model_prefix='openai/', api_key=INFERENCE_API_KEY,
)
print(f'Local vLLM started: {server_instance.base_url}; tensor parallel size={tp_size}')
'''

EVALUATION = '''
import asyncio
import concurrent.futures
import dataclasses
import importlib.metadata
import time
import traceback
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator

def run_sync(coro_or_fn, *args, **kwargs):
    fn = (lambda: coro_or_fn(*args, **kwargs)) if callable(coro_or_fn) else (lambda: coro_or_fn)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(fn())).result()
    return asyncio.run(fn())

def serializable_result(result):
    try:
        if isinstance(result, dict):
            value = result
        elif hasattr(result, 'model_dump'):
            value = result.model_dump(mode='json')
        elif dataclasses.is_dataclass(result):
            value = dataclasses.asdict(result)
        elif hasattr(result, '__dict__'):
            value = vars(result)
        else:
            return None
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return None

def result_field(result, name):
    return result.get(name) if isinstance(result, dict) else getattr(result, name, None)

def runtime_versions():
    versions = {}
    for package in ['swegemma', 'adk-submission', 'adk-eval-core', 'google-adk', 'vllm', 'litellm', 'torch']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions

versions = runtime_versions()
SUMMARY_ROWS = []
for variant in BENCHMARK_VARIANTS:
    agent_dir = VARIANT_DIRS[variant]
    results_dir = RUN_DIR / 'results' / variant
    results_dir.mkdir(parents=True, exist_ok=False)
    for directory in ['logs', 'patches', 'raw_results']:
        (results_dir / directory).mkdir(exist_ok=True)
    budget = VARIANT_MANIFESTS[variant]['evaluation']
    limits, gen_constraints = build_submission_limits()
    variant_adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    config = EvalConfig(
        tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots',
        results_dir=results_dir, submission_dir=agent_dir, models=models, sandbox=SANDBOX,
        timeout_seconds=int(budget.get('timeout_seconds', 300)),
        max_time_minutes=float(budget.get('max_time_minutes', 60)),
        max_tool_calls=int(budget.get('max_tool_calls', 100)),
        max_turns=int(budget['max_turns']) if budget.get('max_turns') is not None else None,
        limits=limits, generation_constraints=gen_constraints, adapter_manifest=variant_adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=5, overlap_size=2, token_threshold=14336, event_retention_size=5,
        ),
        graph_dir=GRAPH_DIR, embeddings_dir=EMBEDDINGS_DIR, wheels_dir=DATA_DIR / 'wheels', verbose=True,
    )
    file_hashes = {name: file_sha256(agent_dir / name)
                   for name in VARIANT_MANIFESTS[variant]['package_files']}
    (results_dir / 'run_manifest.json').write_text(json.dumps({
        'variant': variant, 'task_ids': selected_ids, 'budgets': budget, 'sandbox': SANDBOX,
        'file_sha256': file_hashes, 'runtime_versions': versions, 'base_model': TARGET_MODEL_NAME,
        'model_path': str(MODEL_PATH), 'max_model_len': 32768, 'gpu_memory_utilization': 0.80,
        'compaction_interval': 5, 'benchmark_order': BENCHMARK_VARIANTS,
        'embedded_source_sha256': EMBEDDED_SOURCE_HASHES,
        'embedded_baseline_sha256': EMBEDDED_BASELINE_HASHES,
        'candidate_source_label': CANDIDATE_LABEL, 'baseline_source_label': BASELINE_LABEL,
        'comparison_mode': COMPARISON_MODE,
    }, indent=2), encoding='utf-8')
    evaluator = Evaluator(config)
    records = []
    normalized_path = results_dir / 'notebook_task_results.jsonl'
    for idx, task in enumerate(SELECTED_TASKS, start=1):
        print(f'[{variant} {idx}/{len(SELECTED_TASKS)}] {task.instance_id} ({task.repo})')
        identifier = task.instance_id
        if not identifier or any(c in identifier for c in '/\\\\') or identifier in {'.', '..'}:
            raise ValueError('Unsafe task identifier')
        handler = logging.FileHandler(results_dir / 'logs' / f'observer_{identifier}.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
        started = time.monotonic()
        record = {'instance_id': identifier, 'repo': task.repo, 'resolved': None,
                  'test_exit_code': None, 'agent_patch': None, 'tool_calls': None,
                  'duration_seconds': None, 'agent_elapsed_seconds': None, 'error': None}
        try:
            result = run_sync(evaluator.evaluate_task, task=task, task_index=idx, total_tasks=len(SELECTED_TASKS))
            for field in ['resolved', 'test_exit_code', 'agent_patch', 'tool_calls', 'duration_seconds',
                          'agent_elapsed_seconds', 'error', 'error_type', 'stop_reason', 'exit_status', 'stage']:
                value = result_field(result, field)
                if value is not None:
                    record[field] = value
            raw = serializable_result(result)
            if raw is not None:
                record['raw_result'] = raw
                (results_dir / 'raw_results' / f'{identifier}.json').write_text(json.dumps(raw, indent=2), encoding='utf-8')
            patch = record['agent_patch']
            if isinstance(patch, str):
                record['patch_chars'] = len(patch)
                patch_path = results_dir / 'patches' / f'{identifier}.patch'
                if not patch_path.exists():
                    patch_path.write_text(patch, encoding='utf-8')
        except Exception as error:
            record.update(error=str(error), error_type=type(error).__name__)
            (results_dir / 'logs' / f'exception_{identifier}.log').write_text(traceback.format_exc(), encoding='utf-8')
            logging.exception('Evaluation failed for %s', identifier)
        finally:
            record['observer_wall_seconds'] = time.monotonic() - started
            root_logger.removeHandler(handler)
            handler.close()
            root_logger.setLevel(old_level)
        with normalized_path.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record) + '\\n')
        records.append(record)
        print({key: record.get(key) for key in ['resolved', 'test_exit_code', 'tool_calls', 'duration_seconds', 'error']})
    report_summary = tool_cli('report', '--results-dir', results_dir,
                              '--output-dir', results_dir / 'report', '--submission-dir', agent_dir)
    SUMMARY_ROWS.append({'variant': variant, 'selected_tasks': len(selected_ids), **report_summary})
(RUN_DIR / 'comparison.json').write_text(json.dumps(SUMMARY_ROWS, indent=2), encoding='utf-8')
try:
    import pandas as pd
    display(pd.DataFrame(SUMMARY_ROWS))
except ImportError:
    print(json.dumps(SUMMARY_ROWS, indent=2))
print('Full logs, patches, traces when emitted by the runtime, and reports:', RUN_DIR)
'''

PACKAGE = '''
# Only the selected variant's active dependency closure goes into the ZIP.
# Reports, notebook code, tasks and traces remain in RUN_DIR.
release_dir = VARIANT_DIRS[RELEASE_VARIANT]
tool_cli('validate', '--submission-dir', release_dir)
zip_path = WORKING_DIR / 'submission.zip'
if zip_path.exists():
    # Preserve a previous release before replacing this conventional Kaggle output.
    shutil.copy2(zip_path, RUN_DIR / 'previous_submission.zip')
package_manifest = tool_cli('build', '--submission-dir', release_dir, '--output', zip_path)
(RUN_DIR / 'release_manifest.json').write_text(json.dumps(package_manifest, indent=2), encoding='utf-8')
print('Release variant:', RELEASE_VARIANT)
print('Created:', zip_path, 'bytes:', zip_path.stat().st_size)
'''


def embed_submission(submission_dir: Path, tool_path: Path):
    """Read only the curator's active text dependency closure."""
    with tempfile.TemporaryDirectory(prefix="submission_notebook_") as temp:
        archive = Path(temp) / "candidate.zip"
        subprocess.run([sys.executable, str(tool_path), "build", "--submission-dir", str(submission_dir),
                        "--output", str(archive)], check=True, capture_output=True, text=True)
        with zipfile.ZipFile(archive) as bundle:
            embedded = {}
            for name in sorted(bundle.namelist()):
                if name.endswith(".safetensors") or name.startswith("adapters/"):
                    raise ValueError("Notebook embedding supports text-only unchanged-model submissions. Mount weight assets for a separate adapter experiment rather than embedding baseline/candidate binaries.")
                embedded[name] = bundle.read(name).decode("utf-8")
    return embedded


def generate(submission_dir: Path, tool_path: Path, starter_path: Path, output: Path,
             baseline_dir: Path | None = None):
    """Generate a control comparison, or a two-agent prompt comparison with baseline."""
    submission_dir = Path(submission_dir)
    baseline_dir = Path(baseline_dir) if baseline_dir is not None else None
    tool_source = tool_path.read_text(encoding="utf-8")
    embedded = embed_submission(submission_dir, tool_path)
    baseline = embed_submission(baseline_dir, tool_path) if baseline_dir is not None else {}
    hashes = {name: hashlib.sha256(content.encode("utf-8")).hexdigest() for name, content in embedded.items()}
    hashes["notebook_tools/submission_tools.py"] = hashlib.sha256(tool_source.encode("utf-8")).hexdigest()
    baseline_hashes = {name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                       for name, content in baseline.items()}
    comparison_mode = "prompt_comparison" if baseline_dir is not None else "architecture_control"
    candidate_label = submission_dir.name
    baseline_label = baseline_dir.name if baseline_dir is not None else None
    config_source = CONFIG
    if baseline_dir is not None:
        config_source = config_source.replace(
            "BENCHMARK_VARIANTS = ['coder_only', 'optional_analyzer']",
            "BENCHMARK_VARIANTS = ['baseline', 'candidate']",
        ).replace("RELEASE_VARIANT = 'optional_analyzer'", "RELEASE_VARIANT = 'candidate'")
        config_source = config_source.replace(
            "# Set to a mounted copy of YOUR candidate to test later configurations/adapters.",
            "# Set to a mounted copy of YOUR candidate to test later prompt revisions.",
        )
    starter = nbformat.read(starter_path, as_version=4)
    install = next(cell.source for cell in starter.cells if cell.cell_type == "code")
    helpers = "\n\n".join(inspect.getsource(fn) for fn in [remove_agent_tools, task_resources, select_tasks, extract_curated, file_sha256])
    payload = "\n".join([
        "# Curated text files; regenerate this notebook after editing the candidate or tools.",
        "EMBEDDED_SUBMISSION = " + repr(embedded),
        "EMBEDDED_BASELINE = " + repr(baseline),
        "EMBEDDED_TOOL_SOURCE = " + repr(tool_source),
        "EMBEDDED_SOURCE_HASHES = " + repr(hashes),
        "EMBEDDED_BASELINE_HASHES = " + repr(baseline_hashes),
        "COMPARISON_MODE = " + repr(comparison_mode),
        "CANDIDATE_LABEL = " + repr(candidate_label),
        "BASELINE_LABEL = " + repr(baseline_label),
    ])
    if baseline_dir is not None:
        comparison_description = (
            f"Compare `{baseline_label}` (baseline) with `{candidate_label}` (candidate) as a "
            "two-agent prompt comparison on identical public tasks, model, sampling and budgets. "
            "Both retain the coding agent and optional analyzer. This notebook makes no claim "
            "that the candidate improves performance. Both curated configurations are embedded; "
            "neither is copied from the competition sample submission."
        )
        preparation_description = (
            "## 3. Materialize the baseline and candidate\n\nBoth preserve their two-agent YAML. "
            "The baseline stays embedded; `SUBMISSION_SOURCE_DIR` can override the candidate with "
            "a mounted text configuration. Preflight requires the same model, agent count, "
            "resolved YAML topology, sampling and budgets. Only instruction text may differ. "
            "Tooling stays outside both submission directories."
        )
    else:
        comparison_description = (
            f"Compare `{candidate_label}` with a coder-only control on identical public tasks "
            "and budgets. The embedded candidate is curated from your selected submission "
            "directory; it is never copied from the competition sample submission."
        )
        preparation_description = (
            "## 3. Materialize the candidate and control\n\nThe control removes only the analyzer "
            "`agent_tool`; YAML includes retain their tags. It shares prompts, generation settings "
            "and budgets with the selected candidate. The curator omits detached analyzer resources "
            "from its package. Tooling stays outside both submission directories."
        )
    md = nbformat.v4.new_markdown_cell
    code = lambda source: nbformat.v4.new_code_cell(textwrap.dedent(source).strip() + "\n")
    cells = [
        md(f"# Evaluate {candidate_label} with unchanged Gemma weights\n\n" + comparison_description + "\n\nAttach the competition data, Gemma model, and Gemma 4 Developer Agent Wheelhouse. Enable GPU acceleration and run cells in order. The default two tasks provide a smoke test. Increase task coverage for comparisons, keep a held-out set for later tuning, and repeat runs to assess generation variation. This notebook evaluates and packages; it performs no training."),
        md("## 1. Choose data, candidate, tasks and variants\n\nSet `SUBMISSION_SOURCE_DIR` only for your mounted candidate. Explicit `TASK_IDS` preserve a fixed validation set across revisions. Runs use unique directories and preserve earlier artifacts."),
        code(config_source),
        md("## 2. Install the provided offline runtime\n\nThis installation cell comes from the starter notebook. The task `wheels/` directory is a separate dependency collection."),
        code(install),
        md(preparation_description),
        code(payload), code(helpers), code(PREPARATION),
        md("## 4. Preflight data coverage and choose the shared task set\n\nA task requires its snapshot and matching task-named or commit-named graph and embedding. Automatic selection uses a stable repository round robin; explicit IDs fail early if resources are incomplete. Missing local resources are not a performance result."),
        code(PREFLIGHT),
        md("## 5. Start the required Gemma model locally\n\nUses the starter vLLM API, scoring context length 32,768, GPU memory fraction 0.80, and LoRA only when a manifest exists. `EMPTY` is a local routing placeholder. No external API is called."),
        code(SERVER),
        md("## 6. Generate patches, verify and report\n\nBoth variants get the same tasks and budgets. Subprocess isolation matches the starter; Docker is the closer scoring environment when available. Variants run in the configured order, so repeat/reverse the order when checking timing.\n\nThe notebook writes `notebook_task_results.jsonl` and leaves harness `task_results.jsonl`, traces and logs intact. Each variant receives JSON, CSV and Markdown diagnostics. Absent agent time, request usage or context metrics stay unknown; total task duration is not agent-loop time."),
        code(EVALUATION),
        md("## 7. Build the chosen release\n\nInspect reports first. `RELEASE_VARIANT` chooses which evaluated directory becomes `/kaggle/working/submission.zip`. The ZIP contains only active agent assets. Packaging is not proof of task success."),
        code(PACKAGE),
        md("## 8. Stop the server when finished\n\nRun after all evaluation or further experiments are complete."),
        code("server_instance.stop()\nprint('Local model server stopped.')"),
    ]
    # Stable cell IDs make regenerations reviewable; no outputs are embedded.
    for index, cell in enumerate(cells):
        cell.id = "".join(char if char.isalnum() or char in "_-" else "-" for char in candidate_label)[:40] + "-" + str(index).zfill(2)
        if cell.cell_type == "code":
            ast.parse(cell.source)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "candidate_source_sha256": hashes,
        "baseline_source_sha256": baseline_hashes,
        "candidate_label": candidate_label, "baseline_label": baseline_label,
        "comparison_mode": comparison_mode,
    })
    nbformat.validate(notebook)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    return hashes


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=base / "submission_1")
    parser.add_argument("--baseline-dir", type=Path, help="Embed a second two-agent baseline for prompt comparison")
    parser.add_argument("--output", type=Path, default=base / "evaluate_submission_1.ipynb")
    args = parser.parse_args()
    hashes = generate(args.submission_dir, base / "tools" / "submission_tools.py",
                      base.parent / "competition_material" / "getting-started-gemma-4-developer-agent.ipynb", args.output,
                      baseline_dir=args.baseline_dir)
    print(json.dumps({"notebook": str(args.output), "embedded_file_sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
