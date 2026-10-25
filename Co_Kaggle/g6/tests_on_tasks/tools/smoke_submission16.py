"""Official-runtime mechanics smoke; optional real Gemma registry from the notebook.

No competition task/test/solution data is used. Scripted mode is NOT inference.
"""
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time

MODEL = 'gemma-4-12b-it-qat-w4a16-ct'
SKILLS = {'coder':'repo-navigation','verifier':'repo-verification','finalizer':'repo-finalization'}


def summarize_events(events):
    """Fail closed on missing evidence or repeated visible generation markers."""
    results=[e['result'] for e in events if e['kind']=='skill_result']
    answers=[e.get('text','') for e in events if e['kind']=='answer']
    edits=[r for r in results if r.get('outcome')=='edited']
    verifier_checks=[e['result'] for e in events if e['kind']=='skill_result' and e['agent']=='verifier'
                     and e['result'].get('action') in {'verify','check'}]
    final_audits=[e['result'] for e in events if e['kind']=='skill_result' and e['agent']=='finalizer'
                  and e['result'].get('action')=='audit']
    marker_failure=any(t.count('<|channel>') >= 4 or t.count('<channel|>') >= 4 for t in answers)
    stage_order=[]
    for event in events:
        role=event.get('agent')
        if role in SKILLS and (not stage_order or role != stage_order[-1]):stage_order.append(role)
    conditions={
        'two_round_stage_order':stage_order == ['coder','verifier','coder','verifier','finalizer'],
        'edit_observed':bool(edits),
        'failed_check_observed':any(r.get('verification')=='failed' for r in edits),
        'passing_edit_check':any(r.get('verification')=='passed' for r in edits),
        'verifier_executed_check':any(r.get('outcome')=='passed' for r in verifier_checks),
        'final_audit_verified':bool(final_audits) and final_audits[-1].get('outcome')=='clean'
            and final_audits[-1].get('verification')=='passed' and final_audits[-1].get('evidence_fresh') is True,
        'submit_once':sum(e['kind']=='submit' for e in events)==1,
        'no_marker_repetition':not marker_failure,
    }
    def positions(predicate):
        return [i for i,e in enumerate(events) if predicate(e)]
    failed=positions(lambda e:e['kind']=='skill_result' and e['result'].get('outcome')=='edited'
                     and e['result'].get('verification')=='failed')
    passed=positions(lambda e:e['kind']=='skill_result' and e['result'].get('outcome')=='edited'
                     and e['result'].get('verification')=='passed')
    checked=positions(lambda e:e['kind']=='skill_result' and e['agent']=='verifier'
                      and e['result'].get('action') in {'verify','check'} and e['result'].get('outcome')=='passed')
    audited=positions(lambda e:e['kind']=='skill_result' and e['agent']=='finalizer'
                      and e['result'].get('action')=='audit' and e['result'].get('verification')=='passed')
    submitted=positions(lambda e:e['kind']=='submit')
    conditions['ordered_repair_verify_submit']=bool(failed and passed and checked and audited and submitted) and (
        failed[0] < passed[-1] < checked[-1] < audited[-1] < submitted[0])
    return {'passed':all(conditions.values()),'conditions':conditions}


async def run_smoke(submission_dir, output_dir, models=None):
    from adk_submission import compile_submission, ModelRegistry, ToolRegistry
    from adk_eval_core.sandbox import AdkSandboxCodeExecutor
    from google.adk.agents.run_config import RunConfig
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from swegemma.config import build_submission_limits
    from swegemma.sandbox.subprocess import SubprocessManager

    submission_dir=Path(submission_dir);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    events=[];observed=[];counts=dict.fromkeys(SKILLS,0);scripted=models is None
    start=time.perf_counter();script_calls=0
    versions={n:importlib.metadata.version(n) for n in
              ['adk-submission','adk-eval-core','google-adk','google-genai','swegemma']}
    expected={'adk-submission':'0.2.13','adk-eval-core':'0.1.0','google-adk':'1.36.1',
              'google-genai':'2.11.0','swegemma':'0.2.11'}
    if versions != expected:raise RuntimeError(f'Runtime mismatch: {versions}; expected {expected}')
    def call(role,action=None,**args):
        if action is not None:args={'action':action,**args}
        return types.FunctionCall(name='run_skill_script',args={
            'skill_name':SKILLS[role],'file_path':'scripts/workbench.py','args':args})
    class ScriptedModel(BaseLlm):
        async def generate_content_async(self,llm_request,stream=False):
            instruction=str(llm_request.config.system_instruction)
            role=('coder' if 'You implement a minimal' in instruction else
                  'verifier' if 'You independently verify' in instruction else 'finalizer')
            i=counts[role];counts[role]+=1;observed.append(role)
            assert '{"action":"audit"}' in instruction
            assert 'UNTRUSTED_HANDOFF_SENTINEL' not in instruction
            probe={'module':'pkg','code':'import pkg\nassert pkg.answer() == 42'}
            plans={
                'coder':[call('coder',query='answer'),call('coder','inspect',path='src/pkg/__init__.py'),
                         call('coder','replace',path='src/pkg/__init__.py',old='return 41',new='return 40',**probe),
                         'UNTRUSTED_HANDOFF_SENTINEL: edit failed its check',
                         call('coder','audit'),
                         call('coder','replace',path='src/pkg/__init__.py',old='return 40',new='return 42',**probe),
                         'Repair applied; consult stored audit'],
                'verifier':[call('verifier','audit'),call('verifier','verify'),'FAIL: answer must be 42',
                            call('verifier','audit'),call('verifier','verify'),call('verifier','audit'),
                            'PASS: fresh local assertion'],
                'finalizer':[call('finalizer','audit'),types.FunctionCall(name='submit_patch',args={}),
                             'Captured tested patch.'],
            }
            item=plans[role][i]
            part=types.Part(function_call=item) if isinstance(item,types.FunctionCall) else types.Part(text=item)
            yield LlmResponse(content=types.Content(role='model',parts=[part]))
    if scripted:
        models=ModelRegistry();models.register(MODEL,ScriptedModel(model=MODEL))
    report={'scope':'scripted official-runtime mechanics; no inference' if scripted else
            'real-model synthetic smoke; not a competition benchmark', 'versions':versions}
    try:
        with tempfile.TemporaryDirectory(prefix='submission16-smoke-') as temp:
            sandbox=SubprocessManager(base_dir=Path(temp))
            try:
                sandbox.initialize({'src/pkg/__init__.py':'def answer():\n    return 41\n'})
                init=sandbox.run_command('git init -q && git add . && git -c user.name=Smoke -c user.email=smoke@example.invalid commit -qm baseline')
                if init.exit_code:raise RuntimeError(init.stderr)
                def budget_check():
                    nonlocal script_calls
                    if script_calls >= 40:return 'Smoke tool budget exhausted (40 calls)'
                    script_calls+=1
                    return None
                def get_status() -> dict:
                    return {'time_remaining_seconds':max(0,240-(time.perf_counter()-start)),
                            'tool_calls_remaining':max(0,40-script_calls)}
                def submit_patch() -> str:
                    events.append({'kind':'submit'})
                    return 'Synthetic patch captured'
                registry=ToolRegistry();registry.register('get_status',get_status);registry.register('submit_patch',submit_patch)
                limits,constraints=build_submission_limits()
                agent=compile_submission(submission_dir,registry,models,limits=limits,generation_constraints=constraints,
                    code_executor=AdkSandboxCodeExecutor(sandbox=sandbox,timeout_seconds=30,
                        start_time=start,max_time_minutes=4,budget_check_fn=budget_check),script_timeout=30)
                loop,finalizer=agent.sub_agents
                assert loop.__class__.__name__=='LoopAgent' and loop.max_iterations==2
                assert [a.name for a in loop.sub_agents]==['coder','verifier']
                for a in [*loop.sub_agents,finalizer]:
                    assert a.generate_content_config.thinking_config.thinking_budget==0
                    if not scripted:
                        extra=a.model._additional_args.get('extra_body',{})
                        assert extra.get('chat_template_kwargs',{}).get('enable_thinking') is False, extra
                        assert 'thinking_token_budget' not in extra, extra
                if scripted:
                    # Inspect the actual compiler's LiteLlm request bridge without inference.
                    from google.adk.models.lite_llm import LiteLlm
                    wire_models=ModelRegistry()
                    wire_models.register(MODEL,LiteLlm(model='openai/local-gemma',
                        api_base='http://127.0.0.1:8000/v1',api_key='EMPTY'))
                    wire_agent=compile_submission(submission_dir,registry,wire_models,
                        limits=limits,generation_constraints=constraints,
                        code_executor=AdkSandboxCodeExecutor(sandbox=sandbox,timeout_seconds=30))
                    for a in [*wire_agent.sub_agents[0].sub_agents,wire_agent.sub_agents[1]]:
                        extra=a.model._additional_args['extra_body']
                        assert extra['chat_template_kwargs']['enable_thinking'] is False
                        assert 'thinking_token_budget' not in extra
                    report['compiler_disables_thinking_on_all_stages']=True
                sessions=InMemorySessionService()
                problem=('Synthetic repair exercise: pkg.answer() must return 42. On the FIRST coder stage, '
                         'deliberately replace return 41 with return 40 using the assertion '
                         'import pkg; assert pkg.answer() == 42, then RETURN a short handoff. '
                         'The verifier must execute verify and report failure. On the SECOND coder stage, '
                         'repair return 40 to return 42 using the same assertion. The verifier must verify again. '
                         'The finalizer must audit and submit. Do not edit tests. All code is under src/pkg. '
                         'Use get_status and the stage-specific skill scripts. This is intentional fault injection.')
                await sessions.create_session(app_name='smoke',user_id='test',session_id='one',state={'problem_description':problem})
                runner=Runner(app_name='smoke',agent=agent,session_service=sessions)
                async with asyncio.timeout(240):
                    async for event in runner.run_async(user_id='test',session_id='one',
                            new_message=types.Content(role='user',parts=[types.Part(text=problem)]),
                            run_config=RunConfig(max_llm_calls=48)):
                        if event.content:
                            for part in event.content.parts or []:
                                if part.function_call:
                                    events.append({'kind':'call','agent':event.author,
                                        'name':part.function_call.name,'args':part.function_call.args})
                                if part.function_response and part.function_response.name=='run_skill_script':
                                    outer=part.function_response.response
                                    try:result=json.loads(outer.get('stdout',''))
                                    except (ValueError,TypeError):result={'outcome':'transport_error','response':outer}
                                    events.append({'kind':'skill_result','agent':event.author,'result':result})
                                if part.text and not part.thought:
                                    events.append({'kind':'answer','agent':event.author,'text':part.text})
                                    if part.text.count('<|channel>')>=4 or part.text.count('<channel|>')>=4:
                                        raise RuntimeError('Repeated channel markers: inspect the saved model/ADK events before running tasks')
                report.update(summarize_events(events))
                patch=sandbox.run_command('git diff --binary HEAD').stdout
                (output_dir/'synthetic.patch').write_text(patch)
                report['nonempty_patch']=bool(patch.strip())
                report['passed']=report['passed'] and report['nonempty_patch']
                if scripted:
                    assert observed==['coder']*4+['verifier']*3+['coder']*3+['verifier']*4+['finalizer']*3, observed
                    assert any('Missing action' in str(e) for e in events)
            finally:sandbox.close()
    except Exception as exc:
        report.update(passed=False,error=f'{type(exc).__name__}: {exc}')
    report.update(script_calls=script_calls,elapsed_seconds=time.perf_counter()-start,
                  files={str(p.relative_to(submission_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in submission_dir.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    (output_dir/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    (output_dir/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
