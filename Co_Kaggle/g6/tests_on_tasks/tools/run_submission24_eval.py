"""Run a fixed pilot/S1 evaluation with the official evaluator and an existing E4B server.

Never installs model weights, changes the harness, or supplies reference fixes to agents.
The caller must start the competition's normal vLLM server first (e.g. notebook v11).
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen
import zipfile

from build_submission24 import ARCHIVE, MODEL, build
from compare_submission24 import PILOT_IDS, S1_IDS, normalize


def baseline_ready(row):
    output = row.get('test_output') or ''
    return (row.get('resolved') is False and row.get('test_exit_code') == 1
            and not (row.get('error') or row.get('error_message')) and bool(output.strip())
            and not re.search(r'ModuleNotFoundError|ImportError|ERROR collecting|ERROR at setup|'
                              r'recursive dependency involving fixture|no tests ran', output, re.I))


def serialize_result(result):
    # SessionTrace is an arbitrary runtime object, not a Pydantic JSON field.
    # Serialize it using the official ATIF exporter, preserving the raw outcome.
    row = result.model_dump(mode='json', exclude={'trace'})
    trace = result.trace
    row['trace'] = trace.to_dict() if hasattr(trace, 'to_dict') else trace
    json.dumps(row)  # Fail before writing a partial JSONL record.
    return row


def extract_package(archive, target):
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = Path(info.filename)
            if path.is_absolute() or '..' in path.parts or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe archive member: ' + info.filename)
        z.extractall(target)
    if not (target / 'agent.yaml').is_file():
        raise ValueError('Archive needs agent.yaml at root')


async def evaluate(args):
    from adk_submission import ModelRegistry
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.budget import EvaluationBudget
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new output directory; previous evidence is never overwritten')
    if args.archive is None:
        build()
    archive = args.archive or ARCHIVE
    task_ids = S1_IDS if args.stage == 's1' else PILOT_IDS
    tasks = {t.instance_id: t for t in load_tasks(args.tasks)}
    if set(task_ids) - tasks.keys():
        raise ValueError('Missing required task IDs: ' + str(sorted(set(task_ids) - tasks.keys())))
    served = args.served_model
    if not args.baseline_only:
        with urlopen(args.model_url.rstrip('/') + '/models', timeout=15) as response:
            available = [m['id'] for m in json.load(response)['data']]
        if served not in available or MODEL not in served:
            raise ValueError('Server must expose the exact E4B QAT model; available IDs: ' + str(available))
    args.output.mkdir(parents=True, exist_ok=True)
    time_limit = args.time_limit or None
    manifest = {
        'scope': 'real E4B inference with official phase-2 verification',
        'task_ids': list(task_ids), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'tasks_sha256': hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        'model_alias': MODEL, 'served_model': served, 'repeats': args.repeats,
        'sandbox': args.sandbox,
        'runtime_versions': {n: importlib.metadata.version(n) for n in
                             ['adk-submission', 'adk-eval-core', 'swegemma', 'google-adk', 'google-genai', 'litellm']},
        'effective_limits': {'max_tool_calls': 40, 'max_turns': 48, 'max_time_minutes': time_limit,
                             'timeout_seconds': 30, 'verification_timeout_seconds': 300},
    }
    if args.baseline_only:
        manifest['scope'] = 'baseline-only environment preflight; no model calls'
    reports = []
    with tempfile.TemporaryDirectory() as temp:
        package = Path(temp) / 'package'; package.mkdir()
        extract_package(archive, package)
        manifest['package_files'] = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(package.rglob('*')) if p.is_file()}
        (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        limits, constraints = build_submission_limits()
        async def run(name, baseline=False, seed=0):
            models = ModelRegistry()
            models.register(MODEL, LiteLlm(model='openai/' + served, api_base=args.model_url,
                                           api_key='EMPTY', seed=seed))
            folder = args.output / name; folder.mkdir()
            config = EvalConfig(tasks_path=args.tasks, snapshots_dir=args.snapshots,
                results_dir=folder, submission_dir=package, models=models,
                sandbox=args.sandbox, wheels_dir=args.wheels,
                budget=EvaluationBudget(tool_calls=40, turns=48, time_minutes=time_limit),
                timeout_seconds=30, verification_timeout_seconds=300,
                skip_agent_patch=baseline, display_mode='quiet',
                limits=limits, generation_constraints=constraints)
            evaluator = Evaluator(config)
            rows = []
            try:
                for task_id in task_ids:
                    result = await evaluator.evaluate_task(tasks[task_id])
                    row = serialize_result(result)
                    rows.append(row)
                    # Separate from the evaluator's own task_results.jsonl; retain every outcome.
                    with (folder / 'notebook_task_results.jsonl').open('a') as file:
                        file.write(json.dumps(row) + '\n')
                    print(name, task_id, 'resolved=', row.get('resolved'), flush=True)
            finally:
                evaluator.sandbox.close()
            return rows
        baseline = await run('baseline', baseline=True)
        blocked = [r.get('instance_id') or r.get('task_id') for r in baseline if not baseline_ready(r)]
        if blocked or args.baseline_only:
            report = {'stage': 'baseline_blocked' if blocked else 'baseline_ready',
                      'blocked_ids': blocked, 'real_inference_runs': 0}
        else:
            for repeat in range(args.repeats):
                name = f'candidate_{repeat + 1}'
                rows = await run(name, seed=repeat + 1)
                indexed = normalize(rows, expected=task_ids)
                reports.append({'run': name, 'seed': repeat + 1,
                    'resolved': sum(r['resolved'] for r in indexed.values()),
                    'empty_patches': sum(not r['agent_patch'].strip() for r in indexed.values()),
                    'solved_ids': [n for n, r in indexed.items() if r['resolved']]})
            report = {'stage': 'evaluation_complete', 'runs': reports,
                      'note': 'Use compare_submission24.py per complete S1 run. Do not select only the best repeat.'}
    (args.output / 'evaluation_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Optional S21/S22 archive for a fresh matched control')
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--wheels', type=Path)
    parser.add_argument('--model-url', default='http://localhost:8000/v1')
    parser.add_argument('--served-model', default=MODEL, help='Exact /v1/models ID; may be an absolute model path')
    parser.add_argument('--sandbox', choices=['docker', 'subprocess'], default='subprocess')
    parser.add_argument('--stage', choices=['pilot', 's1'], default='pilot')
    parser.add_argument('--repeats', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--time-limit', type=float, default=0,
                        help='Minutes; 0 matches historical S1 (disabled); 4 matches submission scoring config')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.time_limit < 0:
        parser.error('--time-limit must be nonnegative')
    report = asyncio.run(evaluate(args))
    print(json.dumps(report, indent=2))
    if report['stage'] == 'baseline_blocked':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
