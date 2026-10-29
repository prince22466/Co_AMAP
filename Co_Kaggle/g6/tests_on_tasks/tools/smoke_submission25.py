"""Run the actual packaged single agent and sandbox skills with scripted responses.

This exercises the official runtime, not Gemma's ability to produce these calls.
"""
import asyncio
import functools
import importlib.metadata
import json
from pathlib import Path
import tempfile
import zipfile

from build_submission25 import ARCHIVE, MODEL, build
from run_submission24_eval import COMPACTION, CONTEXT_CACHE
from smoke_submission24 import VERSIONS

SOURCE = 'def render(disabled=False):\n    return "\\n"\n'
PROBE = 'import sample\nassert sample.render(True) == ""\nassert sample.render() == "\\n"'


async def run_smoke(output=None, partial=False):
    from adk_submission import ModelRegistry, ToolRegistry, compile_submission
    from adk_eval_core.sandbox import AdkSandboxCodeExecutor
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.agents.run_config import RunConfig
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps.app import App, EventsCompactionConfig
    from google.genai import types
    from swegemma.config import build_submission_limits
    from swegemma.context import SwegemmaContext
    from swegemma.sandbox.subprocess import SubprocessManager
    from swegemma.tools import create_tools

    versions = {n: importlib.metadata.version(n) for n in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError(f'Expected {VERSIONS}; found {versions}')
    manifest = build()
    def call(name, **args):
        return types.FunctionCall(name=name, args=args)
    def step(op, **args):
        return call('run_skill_script', skill_name='repair', file_path='scripts/work.py', args=dict(op=op, **args))
    plans = [
        call('run_skill_script', skill_name='repair', file_path='scripts/work.py', args='bad wrapper'),
        step('search', query='render'),
        step('probe', code=PROBE),
        step('read', path='sample.py', start=1, end=2),
        step('edit', path='sample.py', start=2, end=2, text='    if disabled:\n'),
        step('edit', path='sample.py', start=2, end=2, text='    return "" if disabled else "\\n"\n'),
        step('check'),
        step('read', path='sample.py', start=1, end=2),
        step('edit', path='sample.py', start=2, end=2, text='    return "wrong"\n'),
        step('diff'), call('submit_patch'), 'Submitted the retained repair.',
    ]
    if partial:
        plans = [step('read', path='sample.py'),
                 step('edit', path='sample.py', start=2, end=2, text='    return "" if disabled else "\\n"\n'),
                 step('test', path='does_not_exist.py'), step('diff'), call('submit_patch'),
                 'Submitted valid source; test path unavailable.']
    requests, events, responses = [], [], []
    class ScriptedModel(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            index = len(requests)
            requests.append(llm_request.model_dump(mode='json'))
            if index >= len(plans):
                raise AssertionError('Unexpected additional model call')
            item = plans[index]
            part = types.Part(text=item) if isinstance(item, str) else types.Part(function_call=item)
            yield LlmResponse(content=types.Content(role='model', parts=[part]))

    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        package = folder / 'package'
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(package)
        sandbox = SubprocessManager(base_dir=folder / 'sandbox')
        try:
            sandbox.initialize({'sample.py': SOURCE, '.gitignore': '__pycache__/\n.adk_exec_*\n'})
            init = sandbox.run_command('git init -q && git add . && git -c user.name=Test -c user.email=test@example.invalid commit -qm baseline')
            assert init.exit_code == 0, init.stderr
            ctx = SwegemmaContext(docker_manager=sandbox, container_id=next(iter(sandbox.sandboxes)),
                repo='synthetic/sample', max_tool_calls=40, max_time_minutes=4, max_exec_seconds=30)
            registry = ToolRegistry()
            for name, fn in create_tools(ctx).items():
                def wrap(name, fn):
                    @functools.wraps(fn)
                    def invoke(*args, **kwargs):
                        result = fn(*args, **kwargs)
                        events.append(dict(tool=name, args=kwargs, result=result))
                        return result
                    return invoke
                registry.register(name, wrap(name, fn))
            models = ModelRegistry()
            models.register(MODEL, ScriptedModel(model=MODEL))
            limits, constraints = build_submission_limits()
            executor = AdkSandboxCodeExecutor(sandbox=sandbox, timeout_seconds=30, budget_check_fn=ctx.check_budget)
            agent = compile_submission(package, registry, models, limits=limits,
                generation_constraints=constraints, code_executor=executor, script_timeout=30)
            native = [getattr(t, '__name__', type(t).__name__) for t in agent.tools]
            agent.model = ScriptedModel(model=MODEL)
            sessions = InMemorySessionService()
            session = await sessions.create_session(app_name='smoke25', user_id='test', state={
                'problem_description': 'Return no newline when disabled, preserve normal output.',
                'hints': 'HINT_SENTINEL: retain the public argument.'})
            app = App(name='smoke25', root_agent=agent,
                context_cache_config=ContextCacheConfig(**CONTEXT_CACHE),
                events_compaction_config=EventsCompactionConfig(**COMPACTION))
            runner = Runner(app=app, session_service=sessions)
            async def consume():
                async for event in runner.run_async(user_id='test', session_id=session.id,
                    new_message=types.Content(role='user', parts=[types.Part(text='Complete the task.')]),
                    run_config=RunConfig(max_llm_calls=48)):
                    for part in (event.content.parts if event.content else []) or []:
                        if part.function_response and part.function_response.name == 'run_skill_script':
                            responses.append(part.function_response.response)
            await asyncio.wait_for(consume(), timeout=90)
            results = [json.loads(r['stdout']) for r in responses if r.get('stdout')]
            oracle = sandbox.run_command('python -c \'import sample; assert sample.render(True) == ""; assert sample.render() == "\\n"\'')
            conditions = dict(
                one_agent_no_reentry=not agent.sub_agents and len(requests) == len(plans),
                raw_mutation_tools_unavailable=not ({'edit_file', 'write_file', 'run_command'} & set(native)),
                native_submission_available='submit_patch' in native,
                task_and_hints_in_prompt='HINT_SENTINEL' in json.dumps(requests[0]['config']),
                source_behavior=oracle.exit_code == 0,
                patch_preserved=bool((ctx.submitted_patch or '').strip()),
                no_internal_files_in_patch='s25-workbench' not in (ctx.submitted_patch or ''),
                one_submission=len([e for e in events if e['tool'] == 'submit_patch']) == 1,
                bounded_execution=len(requests) <= 12 and ctx.tool_calls_used <= 15,
            )
            if partial:
                conditions['invalid_test_does_not_discard_source'] = results[2]['status'] == 'BLOCKED'
            else:
                conditions.update(
                    malformed_dispatch_recovered=responses[0].get('error_code') == 'INVALID_ARGUMENTS',
                    baseline_failure_observed=results[1]['status'] == 'FAIL',
                    syntax_rejected_before_writing=results[3]['status'] == 'REJECTED' and any(
                        name in results[3]['reason'] for name in ['SyntaxError', 'IndentationError']),
                    same_check_fail_to_pass=results[5]['status'] == 'PASS' and results[5]['fail_to_pass'],
                    subsequent_regression_rolled_back=results[7]['status'] == 'REJECTED' and 'rolled back' in results[7]['reason'],
                )
            report = dict(passed=all(conditions.values()), conditions=conditions,
                scope='scripted official-runtime mechanics; NOT E4B inference or task score', versions=versions,
                model_calls=len(requests), tool_calls_used=ctx.tool_calls_used,
                package_sha256=manifest['archive_sha256'], context_cache=CONTEXT_CACHE, compaction=COMPACTION)
            if output:
                output = Path(output); output.mkdir(parents=True, exist_ok=True)
                (output / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
                (output / 'events.json').write_text(json.dumps(responses, indent=2) + '\n')
            return report
        finally:
            sandbox.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = asyncio.run(run_smoke(args.output))
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
