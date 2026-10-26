"""CPU fault-injection regressions. No claim of real Gemma task success."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from build_submission20 import SOURCE, G6, build, ARCHIVE
spec=importlib.util.spec_from_file_location('gate20',SOURCE/'skills/candidate/scripts/gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
BASE='class Public:\n    value = 1\n\ndef render():\n    return 1\n'
FIX=BASE.replace('return 1','return 2')
CHECK={'module':'pkg','probe':'import pkg\nassert pkg.render() == 2'}

class Regressions(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'pkg.py').write_text(BASE)
        (self.root/'tests').mkdir();(self.root/'tests/test_pkg.py').write_text('from pkg import Public\ndef test_public(): assert Public.value == 1\n')
        (self.root/'guide.md').write_text('old documentation\n')
        self.git('init','-q');self.git('add','.');self.git('-c','user.name=T','-c','user.email=t@x','commit','-qm','baseline')
        self.g=gate.Gate(self.root)
    def tearDown(self):
        shutil.rmtree(self.g.home,ignore_errors=True);self.tmp.cleanup()
    def git(self,*args):return subprocess.run(['git','-C',str(self.root),*args],check=True,capture_output=True)
    def stage(self):self.assertEqual(self.g.edit('pkg.py',content=FIX)['status'],'staged')
    def test_edits_need_no_probe_and_do_not_touch_submittable_workspace(self):
        self.stage();self.assertEqual((self.root/'pkg.py').read_text(),BASE)
        self.assertEqual(self.git('diff','HEAD').stdout,b'')
        self.assertEqual(self.g.check(publish=True)['status'],'blocked')
    def test_publish_requires_baseline_failure_candidate_pass_and_local_import(self):
        self.stage();r=self.g.check(CHECK);self.assertEqual(r['status'],'verified',r)
        self.assertEqual(r['reports']['baseline']['status'],'assertion_failed')
        self.assertTrue(r['changed_python_executed'])
        self.assertEqual(self.g.check(publish=True)['status'],'published')
        self.assertEqual((self.root/'pkg.py').read_text(),FIX)
    def test_repaired_after_failed_behavior(self):
        self.g.edit('pkg.py',content=BASE.replace('return 1','return 3'))
        self.assertEqual(self.g.check(CHECK)['status'],'blocked')
        self.stage();self.assertEqual(self.g.check(publish=True)['status'],'published')
    def test_stale_verification_cannot_publish_new_bad_edit(self):
        self.stage();self.assertEqual(self.g.check(CHECK)['status'],'verified')
        self.g.edit('pkg.py',content=BASE.replace('return 1','return 3'))
        self.assertEqual(self.g.check(publish=True)['status'],'blocked')
        self.assertEqual((self.root/'pkg.py').read_text(),BASE)
    def test_previous_published_patch_survives_failed_new_candidate(self):
        self.stage();self.g.check(CHECK);self.g.check(publish=True)
        self.g.edit('pkg.py',content=BASE.replace('return 1','return 3'))
        self.assertEqual(self.g.check(publish=True)['status'],'blocked')
        self.assertEqual((self.root/'pkg.py').read_text(),FIX)
    def test_syntax_and_valid_class_erasure_rollback(self):
        for text in ['def render(:\n pass\n','def render():\n return 2\n']:
            self.assertEqual(self.g.edit('pkg.py',content=text)['status'],'edit_rejected')
            self.assertEqual((self.g.stage/'pkg.py').read_text(),BASE)
    def test_protected_paths_and_symlink_escape(self):
        for name in ['tests/test_pkg.py','pytest.ini','scripts/test.sh','pyproject.toml','../outside.py']:
            with self.assertRaises(ValueError):self.g.edit(name,content='x=1\n')
        (self.g.stage/'link.py').symlink_to(self.root/'pkg.py')
        with self.assertRaises(ValueError):self.g.edit('link.py',content='x=1\n')
    def test_two_identical_edit_failures_force_different_request(self):
        for _ in range(2):self.assertEqual(self.g.edit('pkg.py',old='missing',new='x')['status'],'edit_rejected')
        self.assertEqual(self.g.edit('pkg.py',old='missing',new='x')['status'],'retry_blocked')
        self.assertEqual(self.g.state['edits'],2)
        self.stage()
    def test_missing_edit_arguments_return_actionable_error_not_gate_on_assertions(self):
        result=self.g.dispatch({'op':'edit','path':'pkg.py'})
        self.assertEqual(result['status'],'edit_rejected');self.assertIn('old',result['error'])
        self.assertNotIn('assertion',result['error'])
    def test_edit_budget_does_not_block_check_and_publish(self):
        self.stage();self.g.state['edits']=12;self.g.save()
        with self.assertRaises(ValueError):self.g.edit('pkg.py',content=FIX)
        self.g.check(CHECK);self.assertEqual(self.g.check(publish=True)['status'],'published')
    def test_assert_true_and_unrelated_passing_checks_do_not_qualify(self):
        self.stage()
        for probe in ['assert True','import pkg\nassert pkg.Public.value == 1']:
            self.assertEqual(self.g.check(dict(module='pkg',probe=probe))['status'],'blocked')
    def test_unrelated_failing_to_passing_probe_without_changed_line_execution_blocked(self):
        self.stage()
        p="from pathlib import Path\nassert 'return 2' in Path('pkg.py').read_text()"
        r=self.g.check(dict(module='pkg',probe=p));self.assertEqual(r['status'],'blocked')
        self.assertFalse(r['changed_python_executed'])
    def test_printed_failure_with_zero_exit_is_blocked(self):
        self.stage();r=self.g.check(dict(module='pkg',probe=CHECK['probe']+"\nprint('ORJSON test FAILED')"))
        self.assertEqual(r['status'],'blocked');self.assertEqual(r['reports']['candidate']['status'],'reported_failure')
    def test_missing_dependencies_and_installed_import_are_not_passes(self):
        self.stage()
        for module in ['json','dependency_that_does_not_exist_s20']:
            r=self.g.check(dict(CHECK,module=module));self.assertEqual(r['status'],'blocked')
            self.assertEqual(r['reports']['baseline']['status'],'setup_error')
    def test_test_discovery_and_baseline_regression(self):
        self.stage()
        with self.assertRaisesRegex(ValueError,'discovered'):self.g.check(dict(CHECK,test='tests/test_models.py'))
        r=self.g.check(dict(CHECK,test='tests/test_pkg.py'));self.assertEqual(r['status'],'verified',r)
        self.assertEqual(r['reports']['baseline']['regression']['exit_code'],0)
    def test_probe_side_effects_do_not_publish(self):
        self.stage();p=CHECK['probe']+"\nfrom pathlib import Path\nPath('pkg.py').write_text('corrupted')"
        self.assertEqual(self.g.check(dict(module='pkg',probe=p))['status'],'blocked')
        self.assertEqual((self.g.stage/'pkg.py').read_text(),FIX)
    def test_docs_can_use_content_assertions_without_python_import(self):
        self.g.edit('guide.md',content='Required new documentation\n')
        r=self.g.check(dict(module='',probe="from pathlib import Path\nassert 'Required new' in Path('guide.md').read_text()"))
        self.assertEqual(r['status'],'verified',r)
    def test_changed_live_workspace_is_never_overwritten(self):
        self.stage();(self.root/'pkg.py').write_text('external change\n')
        self.assertEqual(self.g.check(CHECK,publish=True)['status'],'blocked')
        self.assertEqual((self.root/'pkg.py').read_text(),'external change\n')
    def test_official_executor_temp_file_not_treated_as_task_edit(self):
        (self.root/'.adk_exec_example.py').write_text('# official executor wrapper\n')
        self.stage();self.assertEqual(self.g.check(CHECK,publish=True)['status'],'published')
    def test_exact_archived_s18_syntax_breaking_replacement_is_rolled_back(self):
        case=json.loads((G6/'tests_on_tasks/reviews/submission_19/failures.json').read_text())['requests_7433']['arguments']
        baseline='def prepare_body(data):\n'+case['old_string']+'\n        ):\n            pass\n'
        self.assertEqual(self.g.edit('source.py',content=baseline)['status'],'staged')
        r=self.g.edit('source.py',old=case['old_string'],new=case['new_string'])
        self.assertEqual(r['status'],'edit_rejected');self.assertEqual((self.g.stage/'source.py').read_text(),baseline)
    def test_preinstalled_ignored_build_artifacts_do_not_block_initialization(self):
        shutil.rmtree(self.g.home)
        (self.root/'.git/info/exclude').write_text('*.egg-info/\n')
        (self.root/'pkg.egg-info').mkdir()
        (self.root/'pkg.egg-info/PKG-INFO').write_text('installed baseline metadata\n')
        self.g=gate.Gate(self.root)
        self.stage();self.g.check(CHECK)
        self.assertEqual(self.g.check(publish=True)['status'],'published')
        self.assertEqual((self.root/'pkg.egg-info/PKG-INFO').read_text(),'installed baseline metadata\n')
    def test_focused_edit_supports_large_existing_files(self):
        text=BASE+'# padding\n'*12000
        (self.g.stage/'large.py').write_text(text)
        r=self.g.edit('large.py',old='return 1',new='return 2')
        self.assertEqual(r['status'],'staged',r)
        self.assertEqual(self.g.edit('large.py',content=text)['status'],'edit_rejected')
    def test_s19_failure_excerpts_match_archived_tool_results(self):
        cases=json.loads((G6/'tests_on_tasks/reviews/submission_20/failures.json').read_text())
        for case in cases.values():
            with zipfile.ZipFile(G6/'tests_on_tasks/diagnostics'/case['archive']) as z:
                raw=z.read(case['source'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(),case['sha256'])
                self.assertTrue(any(t.get('arguments')==case['arguments'] and t.get('result')==case['result'] for t in json.loads(raw)))

    def test_reproducible_package_and_no_direct_mutation_tool(self):
        a=build();raw=ARCHIVE.read_bytes();self.assertEqual(a,build());self.assertEqual(raw,ARCHIVE.read_bytes())
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertEqual(len(z.namelist()),5)
            agent=z.read('agent.yaml').decode()
            for name in ['edit_file','write_file','run_command','callbacks','sub_agents']:
                self.assertNotIn(name,agent)

if __name__=='__main__':unittest.main()
