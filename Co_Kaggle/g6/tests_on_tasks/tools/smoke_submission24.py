"""Exercise the packaged graph with official ADK and real native/skill execution.

The model is scripted. This tests mechanics and fault recovery, NOT E4B ability.
"""
import asyncio
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import zipfile

from build_submission24 import ARCHIVE, MODEL, build
from run_submission24_eval import COMPACTION, CONTEXT_CACHE

FIXTURE = '# LOCATOR_RAW_READ_SENTINEL\ndef render():\n    return "first second"\n'
BROKEN = 'def render(:\n    return "first second"\n'
FIXED = 'def render():\n    return "first\\nsecond"\n'
PROBE = "python -c 'import formatter; assert formatter.render() == \"first\\nsecond\"'"
VERSIONS = {'adk-submission': '0.2.13', 'adk-eval-core': '0.1.0',
            'swegemma': '0.2.11', 'google-adk': '1.36.1', 'google-genai': '2.11.0'}


def leaves(agent):
    if not agent.sub_agents:
        yield agent
    for child in agent.sub_agents:
        yield from leaves(child)


async def run_smoke(output=None, empty=False, malformed_locator=False):
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
    events, requests, role_order = [], [], []
    def call(name, **args):
        return types.FunctionCall(name=name, args=args)
    audit = call('run_skill_script', skill_name='patch-audit', file_path='scripts/audit.py')
    plans = {
        'locator': [call('read_file', filepath='formatter.py'),
                    call('run_command', command=PROBE), 'LOCATION_HANDOFF formatter.py; baseline failed'],
        'editor': [call('edit_file', filepath='formatter.py', old_string=FIXTURE, new_string=BROKEN),
                   'CANDIDATE_ONE formatter.py changed; review needed',
                   call('edit_file', filepath='formatter.py', old_string=BROKEN, new_string=FIXED),
                   'CANDIDATE_TWO formatter.py repaired'],
        'reviewer': [audit, call('run_command', command=PROBE),
                     'REPAIR syntax error in formatter.py',
                     audit, call('run_command', command=PROBE),
                     'READY final candidate has relevant passing check'],
        'finalizer': [call('run_command', command='git diff --check && git diff HEAD --stat'),
                      call('submit_patch'), 'Submitted existing patch.'],
    }
    if malformed_locator:
        plans['locator'] = ['Printed tool syntax is not an executed call. No location found.']
    if empty:
        plans['editor'] = ['No edit made.', 'No edit made.']
        plans['reviewer'] = [audit, 'EMPTY no patch', audit, 'EMPTY no patch']
    positions = {name: 0 for name in plans}

    class ScriptedModel(BaseLlm):
        role: str = ''
        async def generate_content_async(self, llm_request, stream=False):
            role = self.role
            if not role_order or role_order[-1] != role:
                role_order.append(role)
            requests.append({'role': role, 'request': llm_request.model_dump(mode='json')})
            index = positions[role]
            if index >= len(plans[role]):
                raise AssertionError(f'Unexpected extra model call for {role}')
            positions[role] += 1
            item = plans[role][index]
            part = types.Part(text=item) if isinstance(item, str) else types.Part(function_call=item)
            yield LlmResponse(content=types.Content(role='model', parts=[part]))

    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        package = folder / 'package'
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(package)
        sandbox = SubprocessManager(base_dir=folder / 'sandbox')
        try:
            sandbox.initialize({'formatter.py': FIXTURE, '.gitignore': '__pycache__/\n.adk_exec_*\n'})
            init = sandbox.run_command('git init -q && git add . && git -c user.name=Test -c user.email=test@example.invalid commit -qm baseline')
            assert init.exit_code == 0, init.stderr
            ctx = SwegemmaContext(docker_manager=sandbox, container_id=next(iter(sandbox.sandboxes)),
                repo='synthetic/formatter', max_tool_calls=40, max_time_minutes=4, max_exec_seconds=30)
            registry = ToolRegistry()
            for name, fn in create_tools(ctx).items():
                def wrap(name, fn):
                    @functools.wraps(fn)
                    def invoke(*args, **kwargs):
                        result = fn(*args, **kwargs)
                        events.append({'tool': name, 'args': kwargs, 'result': result})
                        return result
                    return invoke
                registry.register(name, wrap(name, fn))
            models = ModelRegistry()
            models.register(MODEL, ScriptedModel(model=MODEL))
            limits, constraints = build_submission_limits()
            executor = AdkSandboxCodeExecutor(sandbox=sandbox, timeout_seconds=30, budget_check_fn=ctx.check_budget)
            agent = compile_submission(package, registry, models, limits=limits,
                generation_constraints=constraints, code_executor=executor, script_timeout=30)
            permissions = {}
            for leaf in leaves(agent):
                permissions[leaf.name] = [getattr(t, '__name__', type(t).__name__) for t in leaf.tools]
                leaf.model = ScriptedModel(model=MODEL, role=leaf.name)
            sessions = InMemorySessionService()
            session = await sessions.create_session(app_name='smoke24', user_id='test', state={
                'problem_description': 'Make formatter.render return first, newline, second.',
                'hints': 'HARNESS_HINT_SENTINEL: preserve the return type.'})
            app = App(name='smoke24', root_agent=agent,
                context_cache_config=ContextCacheConfig(**CONTEXT_CACHE),
                events_compaction_config=EventsCompactionConfig(**COMPACTION))
            runner = Runner(app=app, session_service=sessions)
            async def consume():
                async for event in runner.run_async(user_id='test', session_id=session.id,
                    new_message=types.Content(role='user', parts=[types.Part(text='Complete the task.')]),
                    run_config=RunConfig(max_llm_calls=48)):
                    for part in (event.content.parts if event.content else []) or []:
                        if part.function_response and part.function_response.name == 'run_skill_script':
                            events.append({'tool': 'run_skill_script', 'result': part.function_response.response})
            await asyncio.wait_for(consume(), timeout=90)
            final_session = await sessions.get_session(app_name='smoke24', user_id='test', session_id=session.id)
            oracle = sandbox.run_command(PROBE)
            checks = [e['result'] for e in events if e['tool'] == 'run_command' and e['args']['command'] == PROBE]
            audits = [json.loads(e['result']['stdout']) for e in events if e['tool'] == 'run_skill_script']
            submitted = [e for e in events if e['tool'] == 'submit_patch']
            editor_requests = [r['request'] for r in requests if r['role'] == 'editor']
            # Second editor round receives reviewer output through ADK state.
            repaired_request = json.dumps(editor_requests[-1])
            first_editor_contents = json.dumps(editor_requests[0].get('contents', []))
            second_editor_contents = json.dumps(editor_requests[2 if not empty else 1].get('contents', []))
            conditions = {
                'stage_order': role_order == ['locator', 'editor', 'reviewer', 'editor', 'reviewer', 'finalizer'],
                'only_finalizer_can_submit': [n for n, ts in permissions.items() if 'submit_patch' in ts] == ['finalizer'],
                'native_editor_available': {'read_file', 'edit_file', 'write_file', 'run_command'} <= set(permissions['editor']),
                'two_audits_executed': len(audits) == 2,
                'single_final_submission': len(submitted) == 1 and events[-1]['tool'] == 'submit_patch',
                'output_keys_written': all(k in final_session.state for k in ['location', 'candidate', 'review', 'completion']),
                'prior_raw_reads_not_replayed': 'LOCATOR_RAW_READ_SENTINEL' not in first_editor_contents,
                'harness_hints_reach_fresh_editor': 'HARNESS_HINT_SENTINEL' in json.dumps(editor_requests[0]['config']),
                'source_behavior': (oracle.exit_code != 0) if empty else oracle.exit_code == 0,
                'patch_preserved': not (ctx.submitted_patch or '').strip() if empty else bool((ctx.submitted_patch or '').strip()),
            }
            if not empty:
                conditions.update(
                    syntax_error_detected=audits[0]['status'] == 'blocked',
                    syntax_repaired=audits[1]['status'] == 'structurally_clean',
                    reviewer_feedback_in_editor_prompt='REPAIR syntax error' in repaired_request,
                    old_tool_calls_not_replayed='old_string' not in second_editor_contents,
                    same_probe_fail_to_pass=bool(checks) and checks[0].get('exit_code') != 0 and checks[-1].get('exit_code') == 0)
            else:
                conditions['empty_patch_detected'] = all('empty patch' in a['blockers'] for a in audits)
            report = {'passed': all(conditions.values()), 'conditions': conditions,
                'scope': 'scripted mechanics; no E4B inference', 'versions': versions,
                'context_cache': CONTEXT_CACHE, 'compaction': COMPACTION,
                'model_calls': sum(positions.values()), 'tool_calls_used': ctx.tool_calls_used,
                'package_sha256': manifest['archive_sha256'], 'stage_order': role_order}
            if output:
                output = Path(output); output.mkdir(parents=True, exist_ok=True)
                (output / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
                (output / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
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
