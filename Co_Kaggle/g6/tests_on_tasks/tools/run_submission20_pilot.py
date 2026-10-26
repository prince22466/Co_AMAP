"""Opt-in real E4B pilot. Baselines first, then smoke, then five fixed tasks; never launches a full run."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile
from build_submission20 import build, ARCHIVE, MODEL
from smoke_submission20 import run_smoke

PILOT_IDS=('fastapi_15588','fastapi_14986','requests_7427','requests_7433','fastapi_15023')


def baseline_ready(result):
    """Behavioral baseline failures are expected; setup/collection failures need repair/review."""
    output=result.get('test_output','')
    blocked=bool(result.get('error') or result.get('error_message')) or result.get('test_exit_code') not in (0,1)
    blocked=blocked or bool(re.search(
        r'ModuleNotFoundError|recursive dependency involving fixture|ERROR collecting|ERROR at setup|no tests ran',output,re.I))
    return not blocked and bool(output.strip())


def readiness(smoke, results):
    by_id={r.get('instance_id') or r.get('task_id'):r for r in results}
    # Conservative promotion gate. A patch or successful tool call is never a solved task.
    return (smoke.get('passed') is True and smoke.get('scope')=='real E4B synthetic smoke; not benchmark'
            and len(results)==len(PILOT_IDS) and set(by_id)==set(PILOT_IDS)
            and all(r.get('resolved') is True and not (r.get('error') or r.get('error_message')) and bool(r.get('agent_patch'))
                    for r in by_id.values()))


async def pilot(args):
    from adk_submission import ModelRegistry
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.evaluate import EvalConfig, Evaluator
    from swegemma.models.task import load_tasks
    from swegemma.config import build_submission_limits
    # Loading evaluation data remains outside the submission package.
    available={t.instance_id:t for t in load_tasks(args.tasks)}
    missing=set(PILOT_IDS)-set(available)
    if missing:raise ValueError('Missing pilot tasks: '+', '.join(sorted(missing)))
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=build()
    (args.output/'package.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    models=ModelRegistry()
    models.register(MODEL,LiteLlm(model='openai/'+MODEL,api_base=args.model_url,api_key='EMPTY'))
    limits,constraints=build_submission_limits()
    baselines=[];results=[];smoke={};evaluators=[]
    with tempfile.TemporaryDirectory() as temp:
        package=Path(temp)/'package'
        with zipfile.ZipFile(ARCHIVE) as z:z.extractall(package)
        def evaluator(name,baseline=False):
            config=EvalConfig(tasks_path=args.tasks,snapshots_dir=args.snapshots,results_dir=args.output/name,
                submission_dir=package,models=models,sandbox=args.sandbox,wheels_dir=args.wheels,
                max_tool_calls=40,max_turns=48,max_time_minutes=4,timeout_seconds=30,
                verification_timeout_seconds=120,skip_agent_patch=baseline,display_mode='quiet',
                limits=limits,generation_constraints=constraints)
            config.results_dir.mkdir(parents=True,exist_ok=True)
            e=Evaluator(config);evaluators.append(e);return e
        def save(name,result):
            row=result.model_dump(mode='json',exclude={'trace'})
            # Preserve all validation stdout/stderr and the returned patch; no cherry-picking.
            (args.output/name).write_text(json.dumps(row,indent=2)+'\n')
            return row
        try:
            base=evaluator('baseline',True)
            for task_id in PILOT_IDS:
                row=save('baseline_'+task_id+'.json',await base.evaluate_task(available[task_id]))
                baselines.append(row)
            if not all(baseline_ready(r) for r in baselines):
                outcome=dict(ready_for_larger_run=False,stage='baseline_blocked',
                    blocked_tasks=[r.get('instance_id',r.get('task_id')) for r in baselines if not baseline_ready(r)],
                    note='No model calls made. Inspect logs and fix the external evaluation environment; do not change tests or omit tasks.')
            else:
                smoke=await run_smoke(args.output/'real_smoke',models=models,submission_dir=package)
                if not smoke['passed']:
                    outcome=dict(ready_for_larger_run=False,stage='real_smoke_failed',note='Pilot task inference was not started.')
                else:
                    agent=evaluator('candidate')
                    for task_id in PILOT_IDS:
                        results.append(save('candidate_'+task_id+'.json',await agent.evaluate_task(available[task_id])))
                    outcome=dict(ready_for_larger_run=readiness(smoke,results),stage='pilot_complete',
                        solved=sum(r.get('resolved') is True for r in results),total=len(PILOT_IDS),
                        note='Five fixed development tasks, not a generalization estimate. No full evaluation is launched.')
        finally:
            for e in evaluators:e.sandbox.close()
    outcome.update(model=MODEL,package_sha256=manifest['archive_sha256'],task_ids=list(PILOT_IDS),
                   tasks_file_sha256=hashlib.sha256(args.tasks.read_bytes()).hexdigest())
    (args.output/'readiness.json').write_text(json.dumps(outcome,indent=2)+'\n')
    return outcome


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--wheels',type=Path)
    p.add_argument('--model-url',required=True,help='Existing local E4B server /v1 URL; this script does not start or download a model')
    p.add_argument('--sandbox',choices=['docker','subprocess'],default='subprocess')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=asyncio.run(pilot(a));print(json.dumps(result,indent=2))
    if not result['ready_for_larger_run']:raise SystemExit(1)
