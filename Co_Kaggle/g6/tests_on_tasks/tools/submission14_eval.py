"""Local evaluation support only. NEVER packaged with the submission.

Preserve official tests and result flags. Add src-layout imports to subprocess
sandboxes, run empty-patch baselines before inference, and replay saved patches.
"""
import dataclasses
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def select_tasks(tasks, task_ids=(), limit=10):
    indexed = {t.instance_id:t for t in tasks}
    if task_ids:
        if len(set(task_ids)) != len(task_ids) or set(task_ids)-indexed.keys():
            raise ValueError('Duplicate or unknown task IDs')
        return [indexed[i] for i in task_ids]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('Task limit must be a positive integer')
    return tasks[:limit]


def enable_src_imports():
    """Opt-in local harness fix; no monkeypatch of tests, grading or agent code."""
    from swegemma.sandbox.subprocess import SubprocessManager
    if getattr(SubprocessManager.exec, '_submission14_src', False):
        return
    original = SubprocessManager.exec
    def execute(self, sandbox_id, command, *, timeout=None):
        # The original manager translates these virtual paths to the real sandbox.
        prefix = 'export PYTHONPATH=/workspace/src:/workspace; '
        if ' -m pytest ' in command:
            # Check provenance in the same interpreter/environment as grading.
            probe = """import importlib.util,json,pathlib,sys
root=pathlib.Path('/workspace').resolve()
origins={}
for name in ('fastapi','requests','rich','httpx'):
    if not any((base/name/'__init__.py').is_file() for base in (root/'src',root)):
        continue
    spec=importlib.util.find_spec(name)
    origin=pathlib.Path(spec.origin).resolve() if spec and spec.origin else None
    origins[name]=str(origin)
    if origin is None or not origin.is_relative_to(root):
        print('IMPORT_ORIGIN_ERROR',name,origin);sys.exit(1)
print('Workspace import origins:',json.dumps(origins))
"""
            prefix += 'python3 -s -c '+shlex.quote(probe)+' && '
        return original(self, sandbox_id, prefix + command, timeout=timeout)
    execute._submission14_src = True
    SubprocessManager.exec = execute


def ensure_test_dependencies(tasks, wheel_dirs):
    """Install only absent test dependencies from attached offline wheels.

    SubprocessManager inherits host packages and skips harness dependency setup.
    Missing packages therefore need to exist before sandboxes are created.
    """
    modules = {'pytest':'pytest'}
    if any(t.repo.split('/')[-1]=='fastapi' for t in tasks):
        modules.update({'inline_snapshot':'inline-snapshot', 'dirty_equals':'dirty-equals',
                        'httpx':'httpx', 'multipart':'python-multipart', 'email_validator':'email-validator'})
    if any(t.repo.split('/')[-1]=='requests' for t in tasks):
        modules.update({'pytest_httpbin':'pytest-httpbin','pytest_mock':'pytest-mock'})
    dirs = [Path(d) for d in wheel_dirs if Path(d).is_dir() and any(Path(d).glob('*.whl'))]
    missing = [distribution for module, distribution in modules.items() if importlib.util.find_spec(module) is None]
    if missing:
        if not dirs:
            raise RuntimeError(f'Missing {missing}; attach a compatible test-dependency wheelhouse before enabling GPU.')
        command = [sys.executable,'-m','pip','install','--no-index']
        for d in dirs:command += ['--find-links',str(d)]
        subprocess.run(command+missing,check=True,timeout=180)
        importlib.invalidate_caches()
    return {dist:importlib.metadata.version(dist) for dist in modules.values()}


def infrastructure_reason(record):
    """Known environment signals; a missing new API is not an environment failure."""
    text = str(record.get('test_output',''))+'\n'+str(record.get('error',record.get('error_message','')) or '')
    project=str(record.get('repo','')).rsplit('/',1)[-1].replace('-','_')
    for missing in re.findall(r"No module named ['\"]([^'\"]+)['\"]", text):
        if project and missing.startswith(project+'.'):
            continue  # A missing module inside the project may be the requested feature.
        return 'Missing import: '+missing
    for pattern in (r"recursive dependency involving fixture ['\"]httpbin(?:_secure)?['\"] detected",
                    r'Snapshot file not found:', r'No module named [\'\"](?:inline_snapshot|dirty_equals|pytest_httpbin|pytest_mock|email_validator|multipart)[\'\"]',
                    r'No module named pytest', r'No tests ran', r'no tests ran',
                    r'could not load conftest', r'ImportPathMismatchError', r'Zero tests passed', r'Missing JUnit XML',
                    r'IMPORT_ORIGIN_ERROR',
                    r'pytest: reading from stdin while output is captured',
                    r'pytest: reading from stdin while output is captured!'):
        found = re.search(pattern,text,re.I)
        if found:return found.group(0)
    if record.get('test_exit_code') in {-1,3,4,5,124}:
        return 'Verification did not complete a usable test run'
    return None


def result_dict(result):
    if hasattr(result,'model_dump'):
        d=result.model_dump(mode='json',exclude={'trace'})
    elif dataclasses.is_dataclass(result):
        d=dataclasses.asdict(result)
    else:
        d=dict(result)
    d.setdefault('instance_id',d.get('task_id'))
    d['infrastructure_error']=infrastructure_reason(d)
    return d


async def cpu_evaluate(config, tasks, output, patches=None):
    """Use official Phase 2 verification, with no registered inference model."""
    from adk_eval_core.tracing import SessionTrace
    from swegemma.evaluate import Evaluator
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    baseline=patches is None
    cpu_config=dataclasses.replace(config,results_dir=output,skip_agent_patch=baseline)
    evaluator=Evaluator(cpu_config)
    if not baseline:
        async def saved_patch(task,*args,**kwargs):
            if task.instance_id not in patches:
                raise ValueError(f'No saved patch for {task.instance_id}')
            return patches[task.instance_id],None,SessionTrace()
        evaluator._run_agent_sandbox=saved_patch
    rows=[]
    try:
        for i,task in enumerate(tasks,1):
            result=await evaluator.evaluate_task(task,task_index=i,total_tasks=len(tasks))
            row=result_dict(result);rows.append(row)
            (output/f'{task.instance_id}.json').write_text(json.dumps(row,indent=2,default=str)+'\n')
            print(f'CPU {i}/{len(tasks)} {task.instance_id}: resolved={row.get("resolved")} infra={row["infrastructure_error"]}',flush=True)
    finally:
        evaluator.docker.close()
    (output/'summary.json').write_text(json.dumps(rows,indent=2,default=str)+'\n')
    return rows


def require_usable_baseline(rows):
    errors = {r['instance_id']:r['infrastructure_error'] for r in rows if r['infrastructure_error']}
    if errors:
        raise RuntimeError('CPU preflight stopped before model startup. Fix these environment failures and rerun: '+json.dumps(errors))


def acceptance_report(candidate, baseline, expected_ids):
    before={r['instance_id']:r for r in baseline}
    after={r['instance_id']:r for r in candidate}
    rows=[]
    for identifier in expected_ids:
        a=after.get(identifier,{})
        b=before.get(identifier,{})
        infra=infrastructure_reason(a) if a else 'task not run'
        nonempty=bool(str(a.get('agent_patch','') or '').strip())
        resolved=a.get('resolved') is True
        new=resolved and nonempty and b.get('resolved') is False and not infra and not b.get('infrastructure_error')
        rows.append({'instance_id':identifier,'raw_resolved':resolved,'nonempty_patch':nonempty,
                     'baseline_resolved':b.get('resolved'),'newly_resolved':new,'infrastructure_error':infra})
    count=sum(r['newly_resolved'] for r in rows)
    return {'tasks':rows,'raw_resolved':sum(r['raw_resolved'] for r in rows),'newly_resolved':count,
            'denominator':len(expected_ids),'target_at_least_2_of_10_met':len(expected_ids)==10 and count>=2,
            'note':'Keeps the original denominator. Empty patches, already-passing baselines and infrastructure failures do not count as newly resolved.'}


def submission_metadata(agent_dir):
    """Resolve nested workflow configs like the compiler, and hash skill code too."""
    import hashlib
    import yaml
    root=Path(agent_dir).resolve()
    hashes={};agents={};helpers=set();visited=set()
    def load(path,stack=()):
        path=path.resolve()
        if not path.is_relative_to(root) or path in stack:
            raise ValueError('Unsafe or cyclic include')
        data=path.read_bytes();hashes[path.relative_to(root).as_posix()]=hashlib.sha256(data).hexdigest()
        if path.suffix not in {'.yaml','.yml'}:return data.decode()
        class Loader(yaml.SafeLoader):pass
        Loader.add_constructor('!include',lambda loader,node:load(path.parent/loader.construct_scalar(node),(*stack,path)))
        return yaml.load(data,Loader=Loader)
    def visit(path):
        path=path.resolve()
        if path in visited:raise ValueError('Repeated/cyclic agent config')
        visited.add(path);cfg=load(path)
        agents[cfg['name']]={k:cfg.get(k) for k in ('model','agent_class','generate_content_config','include_contents','max_iterations')}
        refs=list(cfg.get('sub_agents',[]))
        for tool in cfg.get('tools',[]):
            if isinstance(tool,dict) and 'agent_tool' in tool:refs.append(tool['agent_tool'])
        for ref in refs:
            relative=ref['config_path'];candidate=path.parent/relative
            visit(candidate if candidate.exists() else root/relative)
    visit(root/'agent.yaml')
    # Include executable skills in the snapshot, rather than just YAML/prompts.
    for path in root.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.md','.yaml','.yml','.json'}:
            if path.is_symlink():raise ValueError('Symlink in submission')
            hashes[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'root_agent':next(iter(agents)), 'agents':agents,'helper_names':sorted(helpers),
            'declared_models':sorted({v['model'] for v in agents.values() if v.get('model')}),
            'file_sha256':hashes, 'config_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}
