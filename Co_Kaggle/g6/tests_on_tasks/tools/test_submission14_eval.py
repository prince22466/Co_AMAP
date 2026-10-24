"""Regression tests for local evaluation controls and official CPU patch replay."""
import asyncio
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

from submission14_eval import (acceptance_report, cpu_evaluate, enable_src_imports,
                               infrastructure_reason, require_usable_baseline, select_tasks)


class EvaluationTests(unittest.TestCase):
    def test_acceptance_keeps_denominator_and_excludes_empty_or_baseline_pass(self):
        ids=[f'case_{i}' for i in range(10)]
        baseline=[{'instance_id':i,'resolved':False} for i in ids]
        baseline[1]['resolved']=True
        rows=[{'instance_id':i,'resolved':True,'agent_patch':'diff','test_exit_code':0} for i in ids[:3]]
        rows[0]['agent_patch']=''
        r=acceptance_report(rows,baseline,ids)
        self.assertEqual(r['raw_resolved'],3)
        self.assertEqual(r['newly_resolved'],1)
        self.assertFalse(r['target_at_least_2_of_10_met'])
        self.assertEqual(r['denominator'],10)
    def test_environment_errors_do_not_hide_missing_feature_failures(self):
        self.assertIsNotNone(infrastructure_reason({'test_output':"ModuleNotFoundError: No module named 'inline_snapshot'"}))
        self.assertIsNotNone(infrastructure_reason({'test_output':"recursive dependency involving fixture 'httpbin' detected"}))
        self.assertIsNone(infrastructure_reason({'test_output':"ImportError: cannot import name 'NewResponse' from fastapi.responses"}))
        with self.assertRaises(RuntimeError):
            require_usable_baseline([{'instance_id':'one','infrastructure_error':'missing dependency'}])
    def test_notebook_and_workflow_metadata(self):
        import ast
        from build_submission14 import G6, SOURCE, MODEL
        from submission14_eval import submission_metadata
        notebook=json.loads((G6/'tests_on_tasks/gemma-4-developer-agent-testing_v8.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),'<notebook>','exec',ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        ns={}
        exec(''.join(notebook['cells'][8]['source']),ns)
        exec(''.join(notebook['cells'][10]['source']),ns)
        metadata=ns['submission_metadata'](SOURCE)
        self.assertEqual(metadata['declared_models'],[MODEL])
        self.assertEqual(set(metadata['agents']),{'repair_pipeline','bounded_repair','coder','verifier','finalizer'})
        self.assertIn('skills/repo-navigation/scripts/workbench.py',metadata['file_sha256'])
        self.assertEqual(ns['TASK_LIMIT'],10)
        self.assertEqual(ns['EVAL_OVERRIDES']['max_time_minutes'],'submission')
        self.assertIsNone(ns['COMPACTION_SETTINGS'])
        self.assertEqual(''.join(notebook['cells'][10]['source']),Path(__file__).with_name('submission14_eval.py').read_text())

    def test_official_baseline_and_saved_patch_replay_without_model(self):
        from adk_submission import ModelRegistry
        from swegemma.config import EvalConfig
        from swegemma.models import Task
        enable_src_imports()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);repo=root/'repo';repo.mkdir()
            (repo/'src/pkg').mkdir(parents=True)
            (repo/'src/pkg/__init__.py').write_text('def answer():\n    return 41\n')
            subprocess.run(['git','init','-q',str(repo)],check=True)
            subprocess.run(['git','-C',str(repo),'add','.'],check=True)
            subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','baseline'],check=True)
            snapshots=root/'snapshots';snapshots.mkdir()
            with tarfile.open(snapshots/'synthetic_1.tar.gz','w:gz') as archive:
                archive.add(repo,arcname='.')
            task=Task(instance_id='synthetic_1',repo='example/pkg',base_commit='abc123',problem_statement='Return 42',
                test_patch='diff --git a/test_answer.py b/test_answer.py\nnew file mode 100644\n--- /dev/null\n+++ b/test_answer.py\n@@ -0,0 +1,3 @@\n+import pkg\n+def test_answer():\n+    assert pkg.answer() == 42\n')
            tasks_file=root/'tasks.jsonl';tasks_file.write_text('')
            config=EvalConfig(tasks_path=tasks_file,snapshots_dir=snapshots,results_dir=root/'results',
                submission_dir=root,models=ModelRegistry(),sandbox='subprocess',verification_timeout_seconds=30)
            before=asyncio.run(cpu_evaluate(config,[task],root/'baseline'))
            self.assertFalse(before[0]['resolved'],before)
            self.assertIsNone(before[0]['infrastructure_error'],before)
            patch='diff --git a/src/pkg/__init__.py b/src/pkg/__init__.py\n--- a/src/pkg/__init__.py\n+++ b/src/pkg/__init__.py\n@@ -1,2 +1,2 @@\n def answer():\n-    return 41\n+    return 42\n'
            after=asyncio.run(cpu_evaluate(config,[task],root/'replay',patches={'synthetic_1':patch}))
            self.assertTrue(after[0]['resolved'],after)
            self.assertEqual(acceptance_report(after,before,['synthetic_1'])['newly_resolved'],1)


if __name__=='__main__':unittest.main()
