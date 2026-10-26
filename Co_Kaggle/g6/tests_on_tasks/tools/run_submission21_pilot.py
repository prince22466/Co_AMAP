"""Opt-in fixed five-task official E4B pilot; no large-run automation."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile
from build_submission21 import build, ARCHIVE, MODEL

PILOT_IDS=('fastapi_15588','fastapi_14986','requests_7427','requests_7433','fastapi_15023')
TOOL_FAILURE=r'SKILL_NOT_FOUND|SCRIPT_NOT_FOUND|Tool [\\x27][^\\x27]+[\\x27] not found|Argument [\\x27](?:skill_name|file_path)[\\x27] is required'

def baseline_ready(row):
    output=row.get('test_output') or ''
    blocked=bool(row.get('error') or row.get('error_message'))
    blocked=blocked or row.get('test_exit_code') not in (0,1)
    blocked=blocked or bool(re.search(
        r'ModuleNotFoundError|recursive dependency involving fixture|ERROR collecting|ERROR at setup|no tests ran',
        output,re.I))
    return bool(output.strip()) and not blocked

def pilot_metrics(rows):
    by_id={row.get('instance_id') or row.get('task_id'):row for row in rows}
    if len(rows)!=len(PILOT_IDS) or len(by_id)!=len(PILOT_IDS) or set(by_id)!=set(PILOT_IDS):
        return {'valid_task_set':False,'patched':0,'solved':0,'tool_failure_events':0,'trace_available':False}
    traces=[json.dumps(row.get('trace',{}),ensure_ascii=False) for row in rows]
    return {
        'valid_task_set':True,
        'patched':sum(bool(row.get('agent_patch')) for row in rows),
        'solved':sum(row.get('resolved') is True for row in rows),
        'agent_errors':sum(bool(row.get('error') or row.get('error_message')) for row in rows),
        'tool_failure_events':sum(len(re.findall(TOOL_FAILURE,t)) for t in traces),
        'trace_available':all(bool(row.get('trace')) for row in rows),
    }

def readiness(rows):
    m=pilot_metrics(rows)
    # Real trace inspection is part of promotion; no scripted-model proxy.
    return (m['valid_task_set'] and m['patched']==len(PILOT_IDS)
            and m['solved']>=1 and m['agent_errors']==0
            and m['tool_failure_events']==0 and m['trace_available'])

async def pilot(args):
    from adk_submission import ModelRegistry
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.evaluate import EvalConfig, Evaluator
    from swegemma.models.task import load_tasks
    from swegemma.config import build_submission_limits

    available={t.instance_id:t for t in load_tasks(args.tasks)}
    missing=set(PILOT_IDS)-set(available)
    if missing:raise ValueError('Missing five fixed pilot tasks: '+', '.join(sorted(missing)))
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=build()
    (args.output/'package.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    models=ModelRegistry()
    models.register(MODEL,LiteLlm(model='openai/'+MODEL,api_base=args.model_url,api_key='EMPTY'))
    limits,constraints=build_submission_limits()
    baselines=[];results=[];evaluators=[]
    with tempfile.TemporaryDirectory() as temp:
        package=Path(temp)/'submission'
        with zipfile.ZipFile(ARCHIVE) as z:z.extractall(package)
        def evaluator(name,baseline=False):
            config=EvalConfig(
                tasks_path=args.tasks,snapshots_dir=args.snapshots,results_dir=args.output/name,
                submission_dir=package,models=models,sandbox=args.sandbox,wheels_dir=args.wheels,
                max_tool_calls=40,max_turns=48,max_time_minutes=4,timeout_seconds=30,
                verification_timeout_seconds=120,skip_agent_patch=baseline,display_mode='quiet',
                limits=limits,generation_constraints=constraints)
            config.results_dir.mkdir(parents=True,exist_ok=True)
            e=Evaluator(config);evaluators.append(e);return e
        def save(name,r):
            row=r.model_dump(mode='json') # Include trace for human/mechanical tool checks.
            (args.output/name).write_text(json.dumps(row,indent=2)+'\n')
            return row
        try:
            baseline=evaluator('baseline',True)
            for task_id in PILOT_IDS:
                baselines.append(save('baseline_'+task_id+'.json',
                    await baseline.evaluate_task(available[task_id])))
            blocked=[r.get('instance_id',r.get('task_id')) for r in baselines if not baseline_ready(r)]
            if blocked:
                report={'ready_for_15_tasks':False,'stage':'baseline_blocked','blocked_tasks':blocked,
                        'note':'No E4B inference started. Fix baseline environment, do not omit task IDs.'}
            else:
                agent=evaluator('candidate')
                for task_id in PILOT_IDS:
                    results.append(save('candidate_'+task_id+'.json',
                        await agent.evaluate_task(available[task_id])))
                report={'ready_for_15_tasks':readiness(results),'stage':'pilot_complete',
                        **pilot_metrics(results),
                        'note':'Manual review must confirm relevant checks and valid edits; pilot is not a leaderboard estimate.'}
        finally:
            for e in evaluators:e.sandbox.close()
    report.update(model=MODEL,package_sha256=manifest['archive_sha256'],
                  task_ids=list(PILOT_IDS),tasks_file_sha256=hashlib.sha256(args.tasks.read_bytes()).hexdigest())
    (args.output/'readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--wheels',type=Path)
    p.add_argument('--model-url',required=True)
    p.add_argument('--sandbox',choices=['docker','subprocess'],default='subprocess')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    r=asyncio.run(pilot(a))
    print(json.dumps(r,indent=2))
    if not r['ready_for_15_tasks']:raise SystemExit(1)
