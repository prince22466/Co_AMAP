"""Build Submission 16 notebook without modifying historical notebooks."""
import ast
import copy
import json
from pathlib import Path
from build_submission16 import entries

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def code(source):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
            'source':source.splitlines(keepends=True)}


def build():
    notebook=json.loads((ROOT/'gemma-4-developer-agent-testing_v8.ipynb').read_text())
    cells=notebook['cells']
    for cell in cells:
        text=''.join(cell['source'])
        for old,new in [('submission_14','submission_16'),('submission14','submission16'),
                        ('submission-14','submission-16'),('Submission 14','Submission 16'),
                        ('3.0-bounded-repair','4.0-checked-edits')]:text=text.replace(old,new)
        cell['source']=text.splitlines(keepends=True)
        if cell['cell_type']=='code':cell['execution_count']=None;cell['outputs']=[]
    cells[0]['source']=['# Submission 16: checked edits and real-model smoke gate\n',
        '\nStart with `RUN_MODE="preflight"` on CPU. Attach `submission_16.zip` as the ',
        '`submission-16-zip` dataset. The notebook verifies every release file hash.\n',
        '\nUse `RUN_MODE="smoke"` on GPU to exercise an intentional failed-check/repair cycle ',
        'with real Gemma 12B and the same compiled agent. `evaluate` repeats that smoke first ',
        'and stops before the same first ten competition tasks if it fails.\n',
        '\nThe submission timer stays at four minutes and history compaction stays disabled. ',
        'CPU baseline/test grading is unchanged from v8; synthetic smoke is not a competition score. ',
        'No gold solutions or competition verification output are supplied to the agent.\n']
    cells[1]=code("RUN_MODE = 'preflight'  # preflight, smoke, evaluate, replay\n"
        "REPLAY_PATCH_DIR = '/kaggle/input/saved-agent-patches/patches'\n"
        "if RUN_MODE not in {'preflight','smoke','evaluate','replay'}:raise ValueError('Unknown RUN_MODE')\n")
    # Pin the uploaded package, including executable skill files, before GPU initialization.
    import hashlib
    hashes={n:hashlib.sha256(d).hexdigest() for n,d in entries().items()}
    model=''.join(cells[5]['source'])
    model += '\nEXPECTED_RELEASE_FILES = '+repr(hashes)+'\n'
    model += '''import hashlib
for name, digest in EXPECTED_RELEASE_FILES.items():
    target=Submission_DIR/name
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
        raise ValueError(f'Submission 16 file mismatch: {name}; attach the matching release ZIP')
actual={str(p.relative_to(Submission_DIR)) for p in Submission_DIR.rglob('*') if p.is_file()}
if actual != set(EXPECTED_RELEASE_FILES):
    raise ValueError(f'Unexpected package files: {sorted(actual-set(EXPECTED_RELEASE_FILES))}')
'''
    cells[5]=code(model)
    serving=''.join(cells[13]['source']).replace("if RUN_MODE=='evaluate':", "if RUN_MODE in {'smoke','evaluate'}:",1)
    cells[13]=code(serving)
    evaluation=''.join(cells[15]['source'])
    evaluation=evaluation.replace("    budget = EvaluationBudget(",
        "    if effective_limits['max_time_minutes'] != 4 or effective_limits['max_tool_calls'] != 40 or effective_limits['max_turns'] != 48:\n"
        "        raise ValueError('Use the released 4-minute / 40-call / 48-turn settings for the acceptance run')\n"
        "    if COMPACTION_SETTINGS is not None:raise ValueError('Compaction must be disabled')\n"
        "    budget = EvaluationBudget(")
    evaluation=evaluation.replace("    evaluator = Evaluator(eval_config)",
        "    write_json(RESULTS_DIR / 'real_model_smoke.json', SMOKE_REPORT)\n"
        "    evaluator = Evaluator(eval_config)")
    # Prevent stale records from a prior notebook execution entering the finally report.
    evaluation=evaluation.replace("    try:\n        TEST_RECORDS", "    TEST_RECORDS = []\n    try:\n        TEST_RECORDS")
    cells[15]=code(evaluation)
    smoke=(HERE/'smoke_submission16.py').read_text()
    gate='''if RUN_MODE in {'smoke','evaluate'}:
    SMOKE_REPORT = None
    try:
        SMOKE_REPORT = await run_smoke(AGENT_DIR, WORKING_DIR/'submission16_real_model_smoke', models=models)
        print(json.dumps(SMOKE_REPORT, indent=2))
        if not SMOKE_REPORT['passed']:
            raise RuntimeError('Real-model repair smoke failed. Inspect submission16_real_model_smoke/events.json before spending GPU on tasks.')
    finally:
        if RUN_MODE=='smoke' or not SMOKE_REPORT or not SMOKE_REPORT.get('passed'):
            server_instance.stop()
'''
    # Cell indices above refer to the immutable v8 layout.
    cells[14:14]=[code(smoke),code(gate)]
    for cell in cells:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),'<notebook>','exec',ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    path=ROOT/'gemma-4-developer-agent-testing_v9.ipynb'
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
    return path


if __name__=='__main__':print(build())
