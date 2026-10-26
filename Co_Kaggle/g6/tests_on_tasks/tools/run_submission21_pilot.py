"""Opt-in E4B pilot: baseline -> real native-tool smoke -> five tasks -> manual review."""
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
# Check actual S20 error text, including cases without structured error_code.
TOOL_FAILURE=re.compile(
    r"""SKILL_NOT_FOUND|SCRIPT_NOT_FOUND|(?:Tool|Skill)\s+['"][^'"]+['"]\s+not\s+found|
    Script\s+['"][^'"]+['"]\s+not\s+found|
    Argument\s+['"](?:skill_name|file_path)['"]\s+(?:is\s+required|missing)""",
    re.X|re.I,
)
NATIVE={'get_status','read_file','edit_file','write_file','run_command','submit_patch','run_skill_script'}


def baseline_ready(row):
    output=row.get('test_output') or ''
    blocked=bool(row.get('error') or row.get('error_message'))
    blocked=blocked or row.get('test_exit_code') not in (0,1)
    blocked=blocked or bool(re.search(
        r'ModuleNotFoundError|recursive dependency involving fixture|ERROR collecting|ERROR at setup|no tests ran',output,re.I))
    return bool(output.strip()) and not blocked


def trace_calls(trace):
    """Read common evaluator trace/tool JSON structures; unknown formats fail closed."""
    calls=[]
    def scan(node):
        if isinstance(node,list):
            for item in node:scan(item)
        elif isinstance(node,dict):
            name=node.get('tool') or node.get('tool_name') or node.get('name')
            if name in NATIVE and any(k in node for k in ('args','arguments','result','response')):
                args=node.get('arguments') or node.get('args') or {}
                if isinstance(args,str):
                    try:args=json.loads(args)
                    except (ValueError,TypeError):args={}
                result=node.get('result') or node.get('response') or {}
                calls.append((name,args if isinstance(args,dict) else {},
                              result if isinstance(result,dict) else {}))
                return
            for value in node.values():
                if isinstance(value,(list,dict)):scan(value)
    scan(trace)
    return calls


def focused_command(command):
    """Candidate test evidence only; correctness/relevance requires external review."""
    if not isinstance(command,str):return False
    if re.search(r'\bassert\s+(?:True|1)(?:\s|[;)]|$)',command):return False
    if re.search(r'git\s+diff\s+--check|ast\.parse|compileall',command,re.I):return False
    if re.search(r'\bpytest\b',command):
        return bool(re.search(r'(?:[\w./-]+/)?test_[\w.-]+\.py(?:::\w+)?',command))
    return bool(re.search(r'\bpython(?:\d+(?:\.\d+)?)?\s+-c\b',command)
                and re.search(r'\bassert\s+(?!True\b|1\b)\S+',command))


def successful_command(result):
    if result.get('status') not in ('ok','success') or result.get('exit_code')!=0:
        return False
    output=str(result.get('stdout') or '')+'\n'+str(result.get('stderr') or '')
    return re.search(r'\b(?:FAILED|ERROR at setup|ERROR collecting)\b',output,re.I) is None


def task_evidence(row):
    calls=trace_calls(row.get('trace'))
    last_edit=max((i for i,(name,_,response) in enumerate(calls)
                   if name in ('edit_file','write_file') and response.get('status') in ('ok','success')),
                  default=-1)
    checks=[i for i,(name,args,response) in enumerate(calls)
            if name=='run_command' and focused_command(args.get('command'))
            and successful_command(response)]
    submits=[i for i,(name,_,response) in enumerate(calls)
             if name=='submit_patch' and response.get('status') in ('ok','success')]
    after=last_edit>=0 and bool(checks) and checks[-1]>last_edit
    return {'tool_records':len(calls),'native_edit_observed':last_edit>=0,
            'focused_checks':len(checks),'post_edit_check_observed':after,
            'submit_after_check_observed':after and len(submits)==1 and submits[0]>checks[-1]}


def pilot_metrics(rows):
    ids=[r.get('instance_id') or r.get('task_id') for r in rows]
    valid=len(rows)==len(PILOT_IDS) and len(set(ids))==len(PILOT_IDS) and set(ids)==set(PILOT_IDS)
    evidence={r.get('instance_id') or r.get('task_id') or 'unknown':task_evidence(r) for r in rows}
    return {'valid_task_set':valid,
            'patched':sum(bool(r.get('agent_patch')) for r in rows),
            'solved':sum(r.get('resolved') is True for r in rows),
            'agent_errors':sum(bool(r.get('error') or r.get('error_message')) for r in rows),
            'tool_failure_events':sum(len(TOOL_FAILURE.findall(json.dumps(r.get('trace'),ensure_ascii=False)))
                                      for r in rows),
            'trace_available':valid and all(bool(r.get('trace')) for r in rows),
            'post_edit_checks':sum(x['post_edit_check_observed'] for x in evidence.values()),
            'submit_after_check':sum(x['submit_after_check_observed'] for x in evidence.values()),
            'per_task_evidence':evidence}


def promotion_candidate(rows,smoke):
    """An observed test command is NOT proof of semantic relevance."""
    m=pilot_metrics(rows)
    return (smoke.get('passed') is True
            and smoke.get('scope')=='real-model synthetic smoke; not benchmark'
            and m['valid_task_set'] and m['patched']==len(PILOT_IDS)
            and m['solved']>=1 and m['agent_errors']==0 and m['tool_failure_events']==0
            and m['trace_available'] and m['post_edit_checks']==len(PILOT_IDS)
            and m['submit_after_check']==len(PILOT_IDS))


def review_complete(review):
    if not isinstance(review,dict) or not isinstance(review.get('tasks'),list):return False
    tasks=review['tasks']
    if len(tasks)!=len(PILOT_IDS) or not all(isinstance(x,dict) for x in tasks):return False
    if {r.get('instance_id') for r in tasks}!=set(PILOT_IDS):return False
    return all(r.get('patch_relevant') is True and r.get('behavior_check_relevant') is True
               and r.get('final_diff_reviewed') is True for r in tasks)


def readiness(rows,smoke=None,review=None):
    return promotion_candidate(rows,smoke or {}) and review_complete(review)


def review_existing(output,review_file):
    """Post-pilot independent review; no model calls or runtime dependencies."""
    output=Path(output)
    rows=[json.loads((output/('candidate_'+id+'.json')).read_text()) for id in PILOT_IDS]
    smoke=json.loads((output/'real_smoke'/'smoke.json').read_text())
    review=json.loads(Path(review_file).read_text())
    report={'ready_for_15_tasks':readiness(rows,smoke,review),
            'candidate_for_review':promotion_candidate(rows,smoke),
            'manual_review_complete':review_complete(review),**pilot_metrics(rows)}
    (output/'manual_review_result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


async def pilot(args):
    from adk_submission import ModelRegistry
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.evaluate import EvalConfig, Evaluator
    from swegemma.models.task import load_tasks
    from swegemma.config import build_submission_limits
    from smoke_submission21 import run_smoke

    tasks={task.instance_id:task for task in load_tasks(args.tasks)}
    missing=set(PILOT_IDS)-set(tasks)
    if missing:raise ValueError('Missing five fixed pilot tasks: '+', '.join(sorted(missing)))
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=build()
    (args.output/'package.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    models=ModelRegistry()
    models.register(MODEL,LiteLlm(model='openai/'+MODEL,api_base=args.model_url,api_key='EMPTY'))
    limits,constraints=build_submission_limits()
    baselines=[];results=[];evaluators=[];smoke={}
    with tempfile.TemporaryDirectory() as temp:
        package=Path(temp)/'submission'
        with zipfile.ZipFile(ARCHIVE) as z:z.extractall(package)
        def evaluator(name,baseline=False):
            config=EvalConfig(tasks_path=args.tasks,snapshots_dir=args.snapshots,
                results_dir=args.output/name,submission_dir=package,models=models,
                sandbox=args.sandbox,wheels_dir=args.wheels,max_tool_calls=40,
                max_turns=48,max_time_minutes=4,timeout_seconds=30,
                verification_timeout_seconds=120,skip_agent_patch=baseline,
                display_mode='quiet',limits=limits,generation_constraints=constraints)
            config.results_dir.mkdir(parents=True,exist_ok=True)
            e=Evaluator(config);evaluators.append(e);return e
        def save(name,result):
            row=result.model_dump(mode='json')
            (args.output/name).write_text(json.dumps(row,indent=2)+'\n')
            return row
        try:
            baseline=evaluator('baseline',True)
            for id in PILOT_IDS:
                baselines.append(save('baseline_'+id+'.json',
                    await baseline.evaluate_task(tasks[id])))
            blocked=[r.get('instance_id',r.get('task_id')) for r in baselines if not baseline_ready(r)]
            if blocked:
                report={'ready_for_15_tasks':False,'candidate_for_review':False,
                        'stage':'baseline_blocked','blocked_tasks':blocked,
                        'note':'No real E4B calls started; repair environment and keep all IDs.'}
            else:
                try:
                    smoke=await run_smoke(args.output/'real_smoke',models=models,
                                          submission_dir=package,native_only=True)
                except Exception as exc:
                    smoke={'passed':False,'error':repr(exc)}
                    (args.output/'smoke_error.json').write_text(json.dumps(smoke,indent=2)+'\n')
                if not smoke.get('passed') or smoke.get('scope')!='real-model synthetic smoke; not benchmark':
                    report={'ready_for_15_tasks':False,'candidate_for_review':False,
                            'stage':'real_smoke_failed','smoke':smoke,
                            'note':'Five task-level E4B inferences were not started.'}
                else:
                    agent=evaluator('candidate')
                    for id in PILOT_IDS:
                        results.append(save('candidate_'+id+'.json',
                            await agent.evaluate_task(tasks[id])))
                    report={'ready_for_15_tasks':False,
                            'candidate_for_review':promotion_candidate(results,smoke),
                            'manual_review_complete':False,'stage':'pilot_complete',
                            **pilot_metrics(results),
                            'note':'Machine evidence only. Inspect EACH diff and test manually; '
                                   'use --review-existing for a separate review decision.'}
        finally:
            for e in evaluators:e.sandbox.close()
    report.update(model=MODEL,package_sha256=manifest['archive_sha256'],
                  task_ids=list(PILOT_IDS),
                  tasks_file_sha256=hashlib.sha256(args.tasks.read_bytes()).hexdigest())
    (args.output/'readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path)
    p.add_argument('--snapshots',type=Path)
    p.add_argument('--wheels',type=Path)
    p.add_argument('--model-url',help='Existing E4B server /v1 URL')
    p.add_argument('--sandbox',choices=['docker','subprocess'],default='subprocess')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review-existing',action='store_true')
    p.add_argument('--review-file',type=Path)
    args=p.parse_args()
    if args.review_existing:
        if not args.review_file:p.error('--review-existing requires --review-file')
        report=review_existing(args.output,args.review_file)
    else:
        if not args.tasks or not args.snapshots or not args.model_url:
            p.error('pilot requires --tasks, --snapshots and --model-url')
        report=asyncio.run(pilot(args))
    print(json.dumps(report,indent=2))
    if not report['ready_for_15_tasks'] and not report.get('candidate_for_review'):
        raise SystemExit(1)
