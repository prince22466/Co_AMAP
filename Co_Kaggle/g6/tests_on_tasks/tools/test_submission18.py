"""CPU checks tied to archived submission_17 failures; no inference claim."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import zipfile
from build_submission18 import build, ARCHIVE, SOURCE, G6
from smoke_submission18 import evaluate_events, PROBE

class Regressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=json.loads((G6/'tests_on_tasks/reviews/submission_18/submission17_failures.json').read_text())['cases']

    def test_failure_excerpts_match_archived_real_trace(self):
        archive=G6/'tests_on_tasks/diagnostics/submission-17-zip_First30Tasks.zip'
        with zipfile.ZipFile(archive) as z:
            for case in self.cases.values():
                name=next(n for n in z.namelist() if n.endswith('/report/'+case['source']))
                raw=z.read(name)
                self.assertEqual(hashlib.sha256(raw).hexdigest(),case['source_sha256'])
                self.assertTrue(any(t.get('arguments')==case['arguments'] and t.get('result')==case['result']
                                    for t in json.loads(raw)))

    def test_missing_wrapper_and_script_path_failure_surface_removed(self):
        self.assertEqual(self.cases['missing_wrapper']['result']['error_code'],'INVALID_ARGUMENTS')
        self.assertEqual(self.cases['repository_path_as_skill_script']['result']['error_code'],'SCRIPT_NOT_FOUND')
        build()
        config=(SOURCE/'agent.yaml').read_text()
        self.assertNotIn('skills:',config);self.assertNotIn('run_skill_script',config)
        self.assertIn('  - read_file',config);self.assertIn('  - edit_file',config)

    def test_no_separate_finalizer_can_be_starved(self):
        config=(SOURCE/'agent.yaml').read_text()
        self.assertNotIn('sub_agents:',config)
        self.assertIn('  - submit_patch',config)

    def test_package_deterministic_and_contains_no_diagnostics(self):
        a=build();original=ARCHIVE.read_bytes();self.assertEqual(a,build())
        self.assertEqual(original,ARCHIVE.read_bytes())
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertEqual(set(z.namelist()),{'agent.yaml','instruction.md','eval_config.yaml'})
            self.assertFalse(any(b'fastapi_15661' in z.read(n) for n in z.namelist()))

    def events(self):
        def event(tool,result,args=None,h='final'):
            return dict(tool=tool,args=args or {},result=result,patch_hash=h)
        return [event('run_command',{'status':'error'},{'command':PROBE},'baseline'),
            event('edit_file',{'status':'ok'},h='wrong'),
            event('run_command',{'status':'error'},{'command':PROBE},'wrong'),
            event('edit_file',{'status':'ok'}),
            event('run_command',{'status':'ok','exit_code':0},{'command':PROBE}),
            event('submit_patch',{'status':'ok','patch_size':100})]

    def test_real_model_gate_requires_fail_repair_pass_submit(self):
        events=self.events()
        self.assertTrue(evaluate_events(events,'final',True,'diff')['passed'])
        for i in range(len(events)):
            self.assertFalse(evaluate_events(events[:i]+events[i+1:],'final',True,'diff')['passed'])

    def test_archived_assert_true_does_not_satisfy_behavioral_gate(self):
        bad=self.cases['vacuous_check']['arguments']['args']['code']
        self.assertIn('assert True',bad)
        events=self.events();events[4]['args']['command']='python -c '+repr(bad)
        self.assertFalse(evaluate_events(events,'final',True,'diff')['passed'])
        self.assertFalse(evaluate_events(self.events(),'final',False,'diff')['passed'])

    def test_empty_submission_never_passes_gate(self):
        self.assertEqual(self.cases['empty_submission']['result']['patch_size'],0)
        self.assertFalse(evaluate_events(self.events(),'final',True,'')['passed'])
        events=self.events();events[-1]['result']['patch_size']=0
        self.assertFalse(evaluate_events(events,'final',True,'diff')['passed'])

    def test_stale_evidence_and_duplicate_submit_rejected(self):
        self.assertFalse(evaluate_events(self.events(),'changed-after-check',True,'diff')['passed'])
        events=self.events();events.append(copy.deepcopy(events[-1]))
        self.assertFalse(evaluate_events(events,'final',True,'diff')['passed'])

if __name__=='__main__':unittest.main()
