"""Generate the self-contained v8 notebook from v7 diagnostics plus local CPU helpers."""
import ast
import copy
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def code(source):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':source.splitlines(keepends=True)}
def md(source):return {'cell_type':'markdown','metadata':{},'source':source.splitlines(keepends=True)}
def source(cell):return ''.join(cell['source'])

def build():
    old=json.loads((ROOT/'gemma-4-developer-agent-testing_v7.ipynb').read_text())
    cells=old['cells']
    controls=source(cells[12]).replace("'max_time_minutes': None,           # None disables the per-task session timer.",
        "'max_time_minutes': INHERIT,        # Keep submission_14's four-minute limit.")
    controls=controls.replace('TASK_LIMIT = 7','TASK_LIMIT = 10')
    controls=controls.replace("COMPACTION_SETTINGS = {'compaction_interval': 15, 'overlap_size': 2,\n                       'token_threshold': 14336, 'event_retention_size': 5}",
        'COMPACTION_SETTINGS = None  # Stage handoffs already bound history; no uncapped summarizer calls.')
    controls=controls.replace("DIAGNOSTICS_VERSION = '2.2-result-serialization'", "DIAGNOSTICS_VERSION = '3.0-bounded-repair'")
    controls=controls.replace("        elif name == 'submit_patch':", """        elif name == 'run_skill_script':
            action = value(value(args, 'args', {}), 'action', '')
            stages.add({'inspect':'localization', 'replace':'implementation', 'create':'implementation',
                        'check':'verification', 'audit':'verification'}.get(action, 'unknown'))
        elif name == 'submit_patch':""")
    model='''# Attach the NEW 12B submission_14 ZIP as a Kaggle dataset (extracted or raw ZIP).
import zipfile
from adk_submission.discovery import discover_declared_models
submission_x = 'submission-14-zip'
Submission_DIR = Path(f'/kaggle/input/datasets/prince22466/{submission_x}')
if not (Submission_DIR/'agent.yaml').is_file():
    archives=list(Submission_DIR.glob('*.zip'))
    if len(archives)!=1:
        raise ValueError('Expected extracted agent.yaml or exactly one submission ZIP')
    unpack=Path('/kaggle/working/submission14_input')
    if unpack.exists():shutil.rmtree(unpack)
    with zipfile.ZipFile(archives[0]) as archive:
        if any(Path(n).is_absolute() or '..' in Path(n).parts for n in archive.namelist()):
            raise ValueError('Unsafe ZIP path')
        archive.extractall(unpack)
    Submission_DIR=unpack
names=discover_declared_models(Submission_DIR)
if names != {'gemma-4-12b-it-qat-w4a16-ct'}:
    raise ValueError(f'Expected revised submission_14 on 12B, got {names}. Replace the old E4B dataset.')
MODEL_NAME=next(iter(names))
print('Submission model:',MODEL_NAME)
'''
    preflight='''from adk_submission import ModelRegistry
from swegemma.config import EvalConfig

SELECTED_TASKS=select_tasks(tasks,TASK_IDS,TASK_LIMIT)
GRAPH_DIR=str(DATA_DIR/'graphs')
EMBEDDINGS_DIR=str(DATA_DIR/'embeddings')
TEST_WHEEL_DIRS=[DATA_DIR/'wheels', DATA_DIR/'sandbox/wheels', tmp_whl,
    Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-metric-data/sandbox/wheels')]
DEPENDENCY_VERSIONS=ensure_test_dependencies(SELECTED_TASKS,TEST_WHEEL_DIRS)
enable_src_imports()
CPU_DIR=WORKING_DIR/'submission14_cpu'
CPU_CONFIG=EvalConfig(tasks_path=TASKS_PATH,snapshots_dir=DATA_DIR/'snapshots',
    results_dir=CPU_DIR,submission_dir=AGENT_DIR,models=ModelRegistry(),
    sandbox='subprocess',skip_agent_patch=True,verification_timeout_seconds=120)
BASELINE_RECORDS=await cpu_evaluate(CPU_CONFIG,SELECTED_TASKS,CPU_DIR/'baseline')
require_usable_baseline(BASELINE_RECORDS)
print('Empty-patch baseline complete:',sum(r['resolved'] for r in BASELINE_RECORDS),'/',len(BASELINE_RECORDS))
if RUN_MODE=='replay':
    replay_dir=Path(REPLAY_PATCH_DIR)
    patches={t.instance_id:next(p for p in (replay_dir/f'{t.instance_id}.patch',replay_dir/f'{t.instance_id}_returned.patch') if p.is_file()).read_text() for t in SELECTED_TASKS}
    replay=await cpu_evaluate(CPU_CONFIG,SELECTED_TASKS,CPU_DIR/'replay',patches=patches)
    report=acceptance_report(replay,BASELINE_RECORDS,[t.instance_id for t in SELECTED_TASKS])
    (CPU_DIR/'replay_acceptance.json').write_text(json.dumps(report,indent=2)+'\\n')
    print(json.dumps(report,indent=2))
'''
    serving=source(cells[9])
    evaluation=source(cells[13])
    evaluation=evaluation.replace('events_compaction_config=EventsCompactionConfig(**COMPACTION_SETTINGS)',
                                  'events_compaction_config=None')
    evaluation=evaluation.replace("wheels_dir=DATA_DIR / 'wheels'", 'wheels_dir=None')
    evaluation=evaluation.replace("evaluator = Evaluator(eval_config)",'''# Recheck selection consistency before spending model time.
assert [t.instance_id for t in SAMPLE_TASKS] == [t.instance_id for t in SELECTED_TASKS]
write_json(RESULTS_DIR / 'cpu_baseline.json', BASELINE_RECORDS)
write_json(RESULTS_DIR / 'local_environment.json', {
    'src_import_path_fix': True, 'test_dependencies': DEPENDENCY_VERSIONS,
    'grading_and_tests_modified_by_v8': False,
    'note': 'Local subprocess environment; competition Docker environment may differ.'})
evaluator = Evaluator(eval_config)''')
    evaluation=evaluation.replace('finally:\n    DIAGNOSTICS_ZIP', '''finally:
    if 'TEST_RECORDS' in globals():
        raw_candidates=[json.loads((RESULTS_DIR/'raw_results'/f"{t.instance_id}.json").read_text())
                        for t in SAMPLE_TASKS if (RESULTS_DIR/'raw_results'/f"{t.instance_id}.json").exists()]
        for r,t in zip(raw_candidates,[t for t in SAMPLE_TASKS if (RESULTS_DIR/'raw_results'/f"{t.instance_id}.json").exists()]):
            r.setdefault('instance_id',t.instance_id)
        acceptance=acceptance_report(raw_candidates,BASELINE_RECORDS,[t.instance_id for t in SAMPLE_TASKS])
        write_json(RESULTS_DIR/'acceptance.json',acceptance)
        print('Newly resolved:',acceptance['newly_resolved'],'/',acceptance['denominator'],
              '2/10 target met:',acceptance['target_at_least_2_of_10_met'])
    evaluator.docker.close()
    if 'server_instance' in globals() and server_instance is not None:
        server_instance.stop()
    DIAGNOSTICS_ZIP''')
    intro='''# Submission 14: bounded 12B evaluation

Use the **revised 12B** ZIP from the accompanying PR. The previous E4B ZIP is a separate baseline.
Start with `RUN_MODE='preflight'` on a CPU notebook (Accelerator: None). This runs official
empty-patch verification and catches missing dependencies/fixtures before model startup.
Then use `RUN_MODE='evaluate'` on the intended GPU and the same first ten tasks. `replay` grades
saved `.patch` files without inference. All raw outcomes and the original denominator are retained.

The 2/10 target is **not an established score**. A new success needs a nonempty patch and a baseline
that failed under the same test environment. Model setup, final verification and notebook overhead
are outside the four-minute agent timer. v8 adds src/ to local subprocess imports; official tests
and pass/fail flags are unchanged. Known infrastructure errors stop preflight and require a real
environment fix; do not bypass them by editing tests. v7 and historical diagnostics are preserved.
'''
    mode="""RUN_MODE = 'preflight'  # 'preflight', 'replay', or 'evaluate'
REPLAY_PATCH_DIR = '/kaggle/input/saved-agent-patches/patches'
if RUN_MODE not in {'preflight','replay','evaluate'}:raise ValueError('Unknown RUN_MODE')
"""
    new=[md(intro),code(mode),md('## Runtime installation'),copy.deepcopy(cells[2]),
         md('## Submission and tasks'),code(model),copy.deepcopy(cells[5]),
         md('## Bounded evaluation settings and diagnostics'),code(controls),
         md('## CPU baseline and saved-patch replay'),code((HERE/'submission14_eval.py').read_text()),code(preflight),
         md('## Model inference (evaluate mode only)'),
         code("if RUN_MODE=='evaluate':\n"+''.join('    '+l if l.strip() else l for l in serving.splitlines(keepends=True))),
         md('## Evaluation and acceptance report'),
         code("if RUN_MODE=='evaluate':\n"+''.join('    '+l if l.strip() else l for l in evaluation.splitlines(keepends=True))),
         md('Outputs: `/kaggle/working/submission14_cpu` and `/kaggle/working/diagnostic_runs` (including `acceptance.json`).')]
    for c in new:
        if c['cell_type']=='code':
            c['outputs']=[];c['execution_count']=None
            # ast.parse accepts top-level await used by notebook kernels.
            ast.parse(source(c))
    notebook={'cells':new,'metadata':old['metadata'],'nbformat':4,'nbformat_minor':5}
    path=ROOT/'gemma-4-developer-agent-testing_v8.ipynb'
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
    return path


if __name__=='__main__':print(build())
