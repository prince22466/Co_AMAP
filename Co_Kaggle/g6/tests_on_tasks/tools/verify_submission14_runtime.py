"""Exercise the ZIP through the competition compiler, ADK Runner and real skill executor.

Scripted model responses test mechanics, not Gemma reasoning or task accuracy.
"""
import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import zipfile

from adk_submission import compile_submission, ModelRegistry, ToolRegistry
from adk_eval_core.sandbox import AdkSandboxCodeExecutor
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from swegemma.config import build_submission_limits
from swegemma.sandbox.subprocess import SubprocessManager

from build_submission14 import ARCHIVE, G6, MODEL

OBSERVED = []
COUNTS = {'coder': 0, 'verifier': 0, 'finalizer': 0}
SUBMITTED = []


def skill(action, **args):
    return types.FunctionCall(name='run_skill_script', args={
        'skill_name': 'repo-navigation', 'file_path': 'scripts/workbench.py',
        'args': {'action': action, **args}})


class ScriptedModel(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        instruction = str(llm_request.config.system_instruction)
        role = ('coder' if 'You implement one minimal' in instruction else
                'verifier' if 'You verify the current patch' in instruction else 'finalizer')
        index = COUNTS[role]
        COUNTS[role] += 1
        OBSERVED.append(role)
        if role == 'coder' and index == 3:
            assert 'PASS: tested answer is 42' in instruction, 'verifier state not handed to second coder'
        plans = {
            'coder': [skill('inspect', path='src/pkg/__init__.py', symbol='answer'),
                      skill('replace', path='src/pkg/__init__.py', old='return 41', new='return 42'),
                      'candidate: changed answer to 42', 'candidate: retained verified patch'],
            'verifier': [skill('audit'),
                         skill('check', module='pkg', code='import pkg\nassert pkg.answer() == 42'),
                         skill('audit'), 'PASS: tested answer is 42',
                         skill('audit'), 'PASS: tested answer is 42'],
            'finalizer': [skill('audit'), types.FunctionCall(name='submit_patch',args={}), 'Captured patch.'],
        }
        item = plans[role][index]
        part = types.Part(function_call=item) if isinstance(item, types.FunctionCall) else types.Part(text=item)
        yield LlmResponse(content=types.Content(role='model', parts=[part]))


def get_status() -> dict:
    return {'time_remaining_seconds': 220, 'tool_calls_remaining': 38}


def submit_patch() -> str:
    SUBMITTED.append(True)
    return 'Synthetic patch captured'


async def verify(archive):
    OBSERVED.clear(); SUBMITTED.clear()
    COUNTS.update(dict.fromkeys(COUNTS, 0))
    expected = {'adk-submission': '0.2.13', 'adk-eval-core': '0.1.0',
                'google-adk': '1.36.1', 'google-genai': '2.11.0', 'swegemma': '0.2.11'}
    versions = {name: version(name) for name in expected}
    assert versions == expected, versions
    with tempfile.TemporaryDirectory(prefix='submission14-runtime-') as directory:
        package = Path(directory)/'package'
        with zipfile.ZipFile(archive) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(package)
        sandbox = SubprocessManager(base_dir=Path(directory))
        try:
            sandbox.initialize({'src/pkg/__init__.py':'def answer():\n    return 41\n'})
            init = sandbox.run_command('git init -q && git add . && git -c user.name=Smoke -c user.email=smoke@example.invalid commit -qm baseline')
            assert init.exit_code == 0, init
            registry, models = ToolRegistry(), ModelRegistry()
            registry.register('get_status', get_status)
            registry.register('submit_patch', submit_patch)
            models.register(MODEL, ScriptedModel(model=MODEL))
            limits, constraints = build_submission_limits()
            agent = compile_submission(package, registry, models, limits=limits,
                generation_constraints=constraints,
                code_executor=AdkSandboxCodeExecutor(sandbox=sandbox, timeout_seconds=25), script_timeout=25)
            loop, finalizer = agent.sub_agents
            coder, verifier = loop.sub_agents
            assert agent.__class__.__name__ == 'SequentialAgent'
            assert loop.__class__.__name__ == 'LoopAgent' and loop.max_iterations == 2
            assert all(a.model.model == MODEL for a in [coder,verifier,finalizer])
            for a in [coder,verifier]:
                assert not any(getattr(t,'name',None)=='submit_patch' for t in a.tools)
            sessions = InMemorySessionService()
            await sessions.create_session(app_name='smoke', user_id='cpu', session_id='one',
                                          state={'problem_description': 'Make answer return 42.'})
            runner = Runner(app_name='smoke', agent=agent, session_service=sessions)
            evidence = []
            async for event in runner.run_async(user_id='cpu', session_id='one',
                    new_message=types.Content(role='user',parts=[types.Part(text='Make answer return 42.')])):
                if event.content:
                    for part in event.content.parts or []:
                        response = part.function_response
                        if response and response.name=='run_skill_script':
                            result = response.response
                            assert result.get('status')=='success', result
                            parsed = json.loads(result['stdout'])
                            assert parsed['outcome'] not in {'error','failed'}, parsed
                            evidence.append(parsed)
            assert OBSERVED == ['coder']*3+['verifier']*4+['coder']+['verifier']*2+['finalizer']*3, OBSERVED
            assert len(SUBMITTED)==1
            assert 'return 42' in sandbox.read_file('src/pkg/__init__.py')
            assert evidence[-1]['verification']=='passed' and evidence[-1]['evidence_fresh']
            assert evidence[-1]['changed_paths']==['src/pkg/__init__.py'], evidence[-1]
            assert any(e.get('module_origin','').startswith(sandbox.work_dir+'/src/') for e in evidence)
        finally:
            sandbox.close()
    return {'status':'passed','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'versions':versions,'model_calls':len(OBSERVED),'model':'scripted test double (no inference)',
            'stage_order':['coder','verifier','coder','verifier','finalizer'],
            'checks':['official compiler accepts complete ZIP', 'same 12B model on all LLM stages',
                      'real ADK LoopAgent runs exactly twice', 'state handoff reaches second coder',
                      'only finalizer can submit; called once', 'real swegemma subprocess skill execution',
                      'workspace detected without an argument after ADK changes cwd',
                      'src-layout import provenance and fresh assertion evidence',
                      'only intended source file appears in audit'],
            'benchmark_score':None,'scope':'Synthetic CPU mechanics check. Does not establish Gemma accuracy, GPU latency or competition score.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ARCHIVE)
    p.add_argument('--output',type=Path,default=G6/'tests_on_tasks/reviews/submission_14/runtime_smoke.json')
    args=p.parse_args()
    result=asyncio.run(verify(args.archive))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
