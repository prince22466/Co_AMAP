"""Notebook integrity and smoke acceptance regressions; no GPU inference."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from build_submission16 import G6, SOURCE, entries
from build_testing_v9 import build
from smoke_submission16 import summarize_events
from submission14_eval import acceptance_report, submission_metadata


class EvaluationTests(unittest.TestCase):
    def test_notebook_defaults_and_embedded_helpers(self):
        path=build();before=path.read_bytes();build();self.assertEqual(path.read_bytes(),before)
        nb=json.loads(path.read_text());code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        for text in code:compile(text,'<notebook>','exec',ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        ns={};exec(code[0],ns)
        self.assertEqual(ns['RUN_MODE'],'preflight')
        controls=next(t for t in code if 'EVAL_OVERRIDES =' in t);exec(controls,ns)
        self.assertEqual(ns['TASK_LIMIT'],10)
        self.assertEqual(ns['EVAL_OVERRIDES']['max_time_minutes'],'submission')
        self.assertIsNone(ns['COMPACTION_SETTINGS'])
        self.assertEqual(ns['DIAGNOSTICS_VERSION'],'4.0-checked-edits')
        helper=Path(__file__).with_name('smoke_submission16.py').read_text()
        self.assertIn(helper,code)
        model=next(t for t in code if 'EXPECTED_RELEASE_FILES =' in t)
        tree=ast.parse(model)
        expected=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='EXPECTED_RELEASE_FILES' for t in n.targets))
        import hashlib
        self.assertEqual(expected,{n:hashlib.sha256(d).hexdigest() for n,d in entries().items()})
        self.assertIn("submission_x = 'submission-16-zip'",model)
        gate=next(t for t in code if "SMOKE_REPORT = None" in t)
        self.assertIn("if not SMOKE_REPORT['passed']",gate)
        self.assertLess(code.index(gate),next(i for i,t in enumerate(code) if 'TEST_RECORDS = evaluate_with_diagnostics' in t))
        self.assertNotIn('submission_14', ''.join(code))
    def test_metadata_covers_all_role_scripts(self):
        m=submission_metadata(SOURCE)
        for name in ['repo-navigation','repo-verification','repo-finalization']:
            self.assertIn(f'skills/{name}/scripts/workbench.py',m['file_sha256'])
        self.assertEqual(m['declared_models'],['gemma-4-12b-it-qat-w4a16-ct'])
        for name in ['coder','verifier','finalizer']:
            self.assertEqual(m['agents'][name]['generate_content_config']['thinking_config']['thinking_budget'],0)
    def test_smoke_rejects_empty_false_or_corrupt_evidence(self):
        def result(agent,**r):return {'kind':'skill_result','agent':agent,'result':r}
        events=[result('coder',outcome='edited',verification='failed'),
                result('verifier',action='verify',outcome='failed'),
                result('coder',outcome='edited',verification='passed'),
                result('verifier',action='verify',outcome='passed'),
                result('finalizer',action='audit',outcome='clean',verification='passed',evidence_fresh=True),
                {'kind':'submit'}]
        self.assertTrue(summarize_events(events)['passed'])
        self.assertFalse(summarize_events([])['passed'])
        for i in range(len(events)):
            self.assertFalse(summarize_events(events[:i]+events[i+1:])['passed'])
        self.assertFalse(summarize_events(list(reversed(events)))['passed'])
        self.assertFalse(summarize_events(events+[{'kind':'answer','text':'<|channel>thought '*20}])['passed'])
        bad=copy.deepcopy(events);bad[4]['result']['evidence_fresh']=False
        self.assertFalse(summarize_events(bad)['passed'])
    def test_acceptance_still_excludes_baseline_and_empty_patch_successes(self):
        ids=['a','b','c']
        baseline=[{'instance_id':i,'resolved':i=='a'} for i in ids]
        rows=[{'instance_id':i,'resolved':True,'agent_patch':'' if i=='b' else 'patch','test_exit_code':0} for i in ids]
        r=acceptance_report(rows,baseline,ids)
        self.assertEqual(r['newly_resolved'],1);self.assertEqual(r['denominator'],3)
        self.assertFalse(r['target_at_least_2_of_10_met'])


if __name__=='__main__':unittest.main()
