"""Packaged ADK integration test or opt-in real E4B smoke. Never a benchmark score."""
import asyncio
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import zipfile
from build_submission20 import MODEL, SOURCE, build, ARCHIVE

FIXTURE='def render():\n    return "first second"\n'
FIXED='def render():\n    return "first\\nsecond\\n"\n'
PROBE='import formatter\nassert formatter.render() == "first\\nsecond\\n"'

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
    from adk_eval_core.sandbox import AdkSandboxCodeExecutor
    expected={'adk-submission':'0.2.13','adk-eval-core':'0.1.0','google-adk':'1.36.1',
              'google-genai':'2.11.0','swegemma':'0.2.11'}
    versions={name:importlib.metadata.version(name) for name in expected}
    if versions!=expected:raise RuntimeError(f'Runtime mismatch: {versions}')
    events=[];scripted=models is None;index=0
    def call(name,**args):return types.FunctionCall(name=name,args=args)
    def skill(**args):return call('run_skill_script',skill_name='candidate',file_path='scripts/gate.py',args=args)
    plan=[call('get_status'),call('read_file',filepath='formatter.py'),
          # Deliberately bad wrapper: dispatcher must return an error; corrected next.
          call('run_skill_script',skill_name='candidate'),skill(),
          skill(op='edit',path='formatter.py',old=FIXTURE,new='def render(:\n pass\n'),
          skill(op='edit',path='formatter.py',old=FIXTURE,new=FIXED),
          call('submit_patch'),  # early capture must be empty
          skill(op='check',module='formatter',probe='assert True'),
          skill(op='publish'),  # cannot publish using a tautology
          skill(op='check',module='formatter',probe=PROBE),
          skill(op='publish'),call('submit_patch'),'Submitted the validated repair.']
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
            sandbox.initialize({'formatter.py':FIXTURE,'.gitignore':'__pycache__/\n'})
            init=sandbox.run_command('git init -q && git add . && git -c user.name=Smoke -c user.email=smoke@example.invalid commit -qm baseline')
            assert init.exit_code==0,init.stderr
            ctx=SwegemmaContext(docker_manager=sandbox,container_id=next(iter(sandbox.sandboxes)),
                repo='synthetic/formatter',max_tool_calls=40,max_time_minutes=4,max_exec_seconds=30)
            def patch():return sandbox.run_command('git diff --binary HEAD').stdout
            native=create_tools(ctx);registry=ToolRegistry()
            def wrap(name,fn):
                @functools.wraps(fn)
                def invoke(*args,**kwargs):
                    result=fn(*args,**kwargs)
                    events.append({'tool':name,'args':kwargs,'result':result,'live_patch':patch()})
                    return result
                return invoke
            for name in ['get_status','read_file','submit_patch']:
                registry.register(name,wrap(name,native[name]))
            limits,constraints=build_submission_limits()
            executor=AdkSandboxCodeExecutor(sandbox=sandbox,timeout_seconds=30,budget_check_fn=ctx.check_budget)
            agent=compile_submission(Path(submission_dir),registry,models,limits=limits,
                generation_constraints=constraints,code_executor=executor,script_timeout=30)
            sessions=InMemorySessionService()
            # No prescribed tool sequence for real-model mode: model must follow packaged instructions.
            session=await sessions.create_session(app_name='smoke20',user_id='test',state={
                'problem_description':'Change formatter.render() to RETURN exactly first, newline, second, newline. Preserve its public name.',
                'hints_text':''})
            runner=Runner(agent=agent,app_name='smoke20',session_service=sessions)
            error=None
            async def consume():
                async for event in runner.run_async(user_id='test',session_id=session.id,
                    new_message=types.Content(role='user',parts=[types.Part(text='Complete the task.')]),
                    run_config=RunConfig(max_llm_calls=48)):
                    if event.content:
                        for part in event.content.parts or []:
                            if part.function_response and part.function_response.name=='run_skill_script':
                                events.append({'tool':'run_skill_script','result':part.function_response.response,'live_patch':patch()})
            try:await asyncio.wait_for(consume(),timeout=240)
            except Exception as exc:error=repr(exc)
            # Independent oracle, not the model's chosen check.
            oracle=sandbox.run_command("python -c 'import formatter; assert formatter.render() == \"first\\nsecond\\n\"'")
            decoded=[]
            for event in events:
                if event['tool']=='run_skill_script':
                    try:decoded.append(json.loads(event['result'].get('stdout','')))
                    except (ValueError,TypeError):pass
            submits=[e for e in events if e['tool']=='submit_patch']
            conditions={'oracle_passed':oracle.exit_code==0,'nonempty_submission':bool(ctx.submitted_patch),
                        'gate_published':any(d.get('status')=='published' for d in decoded),'no_runtime_error':error is None}
            if scripted:
                conditions.update(early_submission_empty=bool(submits) and submits[0]['result']['patch_size']==0,
                    syntax_edit_rolled_back=any(d.get('status')=='edit_rejected' for d in decoded),
                    tautology_blocked=sum(d.get('status')=='blocked' for d in decoded)>=2,
                    wrapper_error_recovered=any(e['result'].get('error_code')=='INVALID_ARGUMENTS' for e in events if e['tool']=='run_skill_script'))
            report=dict(passed=all(conditions.values()),conditions=conditions,error=error,versions=versions,
                scope='scripted mechanics; no inference' if scripted else 'real E4B synthetic smoke; not benchmark',
                model=MODEL,tool_calls_used=ctx.tool_calls_used,
                package_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())
            (output_dir/'events.json').write_text(json.dumps(events,indent=2)+'\n')
            (output_dir/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
            (output_dir/'synthetic.patch').write_text(ctx.submitted_patch or '')
            return report
        finally:sandbox.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--real-model-url',help='Existing local E4B OpenAI-compatible /v1 endpoint; no server is launched')
    p.add_argument('--served-model',default=MODEL)
    a=p.parse_args();models=None
    if a.real_model_url:
        from adk_submission import ModelRegistry
        from google.adk.models.lite_llm import LiteLlm
        models=ModelRegistry();models.register(MODEL,LiteLlm(model='openai/'+a.served_model,api_base=a.real_model_url,api_key='EMPTY'))
    build()
    with tempfile.TemporaryDirectory() as package:
        with zipfile.ZipFile(ARCHIVE) as z:z.extractall(package)
        result=asyncio.run(run_smoke(a.output_dir,models=models,submission_dir=Path(package)))
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
