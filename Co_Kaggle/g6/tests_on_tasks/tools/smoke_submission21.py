"""Official-runtime regression and optional real-model smoke. No benchmark data enters agent."""
import asyncio
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from build_submission21 import MODEL, SOURCE

# A fixed behavioral oracle for a synthetic formatting task. It observes output,
# not an assertion supplied by the model. It is never packaged with the agent.
PROBE = "python -c 'import io; from contextlib import redirect_stdout; import formatter; b=io.StringIO(); exec(\"with redirect_stdout(b): formatter.render()\"); assert b.getvalue() == \"first\\nsecond\\n\", repr(b.getvalue())'"
FIXTURE = 'def render():\n    print("first second")\n'
WRONG = 'def render(:\n    print("first")\n'
FIXED = 'def render():\n    print("first\\nsecond")\n'


def evaluate_events(events, final_hash, oracle_passed, patch):
    checks=[(i,e) for i,e in enumerate(events) if e['tool']=='run_command' and e['args'].get('command')==PROBE]
    failed=[i for i,e in checks if e['result'].get('status')=='error']
    passed=[(i,e) for i,e in checks if e['result'].get('status')=='ok' and e['result'].get('exit_code')==0]
    edits=[i for i,e in enumerate(events) if e['tool'] in {'edit_file','write_file'} and e['result'].get('status')=='ok']
    submits=[(i,e) for i,e in enumerate(events) if e['tool']=='submit_patch' and e['result'].get('status')=='ok']
    conditions={
        'baseline_failure':bool(failed and edits) and failed[0]<edits[0],
        'failed_candidate_repaired':bool(edits and failed and passed) and any(edits[0]<i<edits[-1] for i in failed),
        'final_behavior_passed':oracle_passed,
        'nonempty_patch':bool(patch.strip()),
        'fresh_final_check':bool(passed) and passed[-1][1]['patch_hash']==final_hash,
        'submit_once_after_check':len(submits)==1 and bool(passed) and passed[-1][0]<submits[0][0]
            and submits[0][1]['result'].get('patch_size',0)>0,
        'native_tools_and_audit_only':all(e['tool'] in {'get_status','read_file','edit_file','write_file','run_command','submit_patch','run_skill_script'} for e in events),
    }
    audits=[e for e in events if e['tool']=='run_skill_script']
    def decoded(e):
        try:return json.loads(e['result'].get('stdout',''))
        except (ValueError,TypeError):return {}
    conditions['syntax_failure_detected']=any(decoded(e).get('status')=='blocked' for e in audits)
    conditions['final_audit_clean']=bool(audits) and decoded(audits[-1]).get('status')=='structurally_clean' and audits[-1]['patch_hash']==final_hash
    conditions['audit_does_not_claim_behavior']=bool(audits) and all(decoded(e).get('behavior_verified') is False for e in audits)
    return {'passed':all(conditions.values()),'conditions':conditions}


async def run_smoke(output_dir, models=None, submission_dir=SOURCE):
    from adk_submission import compile_submission, ModelRegistry, ToolRegistry
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.agents.run_config import RunConfig
    from google.genai import types
    from swegemma.context import SwegemmaContext
    from swegemma.sandbox.subprocess import SubprocessManager
    from swegemma.tools import create_tools
    from swegemma.config import build_submission_limits
    expected={'adk-submission':'0.2.13','adk-eval-core':'0.1.0','google-adk':'1.36.1',
              'google-genai':'2.11.0','swegemma':'0.2.11'}
    versions={name:importlib.metadata.version(name) for name in expected}
    if versions!=expected:raise RuntimeError(f'Runtime mismatch: {versions}; expected {expected}')
    events=[];scripted=models is None;index=0
    def call(name,**args):return types.FunctionCall(name=name,args=args)
    plan=[call('get_status'),call('read_file',filepath='formatter.py'),
          call('run_command',command=PROBE),
          call('edit_file',filepath='formatter.py',old_string=FIXTURE,new_string=WRONG),
          call('run_command',command=PROBE),
          call('run_skill_script',skill_name='patch-audit',file_path='scripts/audit.py'),
          call('edit_file',filepath='formatter.py',old_string=WRONG,new_string=FIXED),
          call('run_command',command=PROBE),call('run_command',command='git diff --check'),
          call('run_skill_script',skill_name='patch-audit',file_path='scripts/audit.py'),
          call('submit_patch'),'Submitted the checked repair.']
    class ScriptedModel(BaseLlm):
        async def generate_content_async(self,llm_request,stream=False):
            nonlocal index
            item=plan[index];index+=1
            yield LlmResponse(content=types.Content(role='model',parts=[
                types.Part(text=item) if isinstance(item,str) else types.Part(function_call=item)]))
    if scripted:
        models=ModelRegistry();models.register(MODEL,ScriptedModel(model=MODEL))
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        sandbox=SubprocessManager(base_dir=Path(tmp))
        try:
            sandbox.initialize({'formatter.py':FIXTURE, '.gitignore':'__pycache__/\n'})
            init=sandbox.run_command('git init -q && git add . && git -c user.name=Smoke -c user.email=smoke@example.invalid commit -qm baseline')
            assert init.exit_code==0,init.stderr
            ctx=SwegemmaContext(docker_manager=sandbox,container_id=next(iter(sandbox.sandboxes)),
                repo='synthetic/formatter',max_tool_calls=40,max_time_minutes=4,max_exec_seconds=30)
            def patch():return sandbox.run_command('git diff --binary HEAD').stdout
            def digest():return hashlib.sha256(patch().encode()).hexdigest()
            native=create_tools(ctx);registry=ToolRegistry()
            def wrap(name,fn):
                @functools.wraps(fn)
                def invoke(*args,**kwargs):
                    result=fn(*args,**kwargs)
                    events.append({'tool':name,'args':kwargs,'result':result,'patch_hash':digest()})
                    return result
                return invoke
            for name in ['get_status','read_file','edit_file','write_file','run_command','submit_patch']:
                registry.register(name,wrap(name,native[name]))
            limits,constraints=build_submission_limits()
            from adk_eval_core.sandbox import AdkSandboxCodeExecutor
            executor=AdkSandboxCodeExecutor(sandbox=sandbox,timeout_seconds=30,budget_check_fn=ctx.check_budget)
            agent=compile_submission(Path(submission_dir),registry,models,limits=limits,generation_constraints=constraints,code_executor=executor,script_timeout=30)
            assert not agent.sub_agents
            if not scripted:
                assert agent.model._additional_args['extra_body']['chat_template_kwargs']['enable_thinking'] is False
            sessions=InMemorySessionService()
            problem=('Synthetic smoke only: formatter.render() must print exactly first, newline, second, newline. '
                'Read formatter.py and run this exact behavioral command before editing: '+PROBE+
                '. For fault injection, first replace def render(): with def render(: (an intentional syntax error), rerun the same command '
                'and run patch-audit scripts/audit.py without args to observe a blocker. Then fix the syntax and output, rerun the exact command, run the audit again, inspect the diff and submit once. '
                'Do not change the expected output or create another verification command.')
            session=await sessions.create_session(app_name='smoke21',user_id='test',state={
                'problem_description':problem,'hints_text':''})
            runner=Runner(agent=agent,app_name='smoke21',session_service=sessions)
            error=None
            async def consume():
                async for event in runner.run_async(user_id='test',session_id=session.id,
                    new_message=types.Content(role='user',parts=[types.Part(text='Complete the task.')]),
                    run_config=RunConfig(max_llm_calls=48)):
                    if event.content:
                        for part in event.content.parts or []:
                            if part.function_response and part.function_response.name=='run_skill_script':
                                events.append({'tool':'run_skill_script','args':{},'result':part.function_response.response,'patch_hash':digest()})
                            if part.text and part.text.count('<|channel>')>=4:raise RuntimeError('Repeated channel markers')
            try:await asyncio.wait_for(consume(),timeout=240)
            except Exception as exc:error=repr(exc)
            oracle=sandbox.run_command(PROBE)
            report=evaluate_events(events,digest(),oracle.exit_code==0,ctx.submitted_patch or '')
            if error:report.update(passed=False,error=error)
            report.update(scope='scripted mechanics; no inference' if scripted else 'real-model synthetic smoke; not benchmark',
                          versions=versions, tool_calls_used=ctx.tool_calls_used,model_calls=index if scripted else None,
                          oracle_exit_code=oracle.exit_code)
            (output_dir/'events.json').write_text(json.dumps(events,indent=2)+'\n')
            (output_dir/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
            (output_dir/'synthetic.patch').write_text(ctx.submitted_patch or '')
            return report
        finally:sandbox.close()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    from build_submission21 import build, ARCHIVE
    import zipfile
    build()
    with tempfile.TemporaryDirectory() as package:
        with zipfile.ZipFile(ARCHIVE) as z:z.extractall(package)
        result=asyncio.run(run_smoke(args.output_dir,submission_dir=Path(package)))
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
