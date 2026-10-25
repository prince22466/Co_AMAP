"""CPU regressions from Submission 15 traces; no model accuracy claims."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from build_submission16 import SOURCE, build


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'checkout'
        (self.root/'src/pkg').mkdir(parents=True)
        (self.root/'src/pkg/__init__.py').write_text('def answer():\n    return 41\n')
        (self.root/'tests').mkdir()
        (self.root/'tests/test_api.py').write_text('assert True\n')
        self.git('init','-q');self.git('add','.')
        self.git('-c','user.name=CPU tests','-c','user.email=test@example.invalid','commit','-qm','baseline')
    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.root),*args])
    def call(self,role='coder',**args):
        skill={'coder':'repo-navigation','verifier':'repo-verification','finalizer':'repo-finalization'}[role]
        command=[sys.executable,str(SOURCE/'skills'/skill/'scripts/workbench.py')]
        for k,v in args.items():command += ['--'+k,str(v)]
        p=subprocess.run(command,cwd=self.temp.name,env={**os.environ,'PYTHONPATH':str(self.root)},
                         capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertLess(len(p.stdout),4900)
        return json.loads(p.stdout)
    def edit(self,value=42,old=41,**overrides):
        args=dict(action='replace',path='src/pkg/__init__.py',old=f'return {old}',new=f'return {value}',
                  module='pkg',code='import pkg\nassert pkg.answer() == 42')
        args.update(overrides)
        return self.call(**args)
    def state(self):return json.loads((self.root/'.git/submission16_workbench.json').read_text())
    def test_missing_action_never_audits_and_recovery_works(self):
        for args in ({},{'query':'answer'},{'path':'src/pkg/__init__.py'}):
            result=self.call(**args)
            self.assertEqual(result['outcome'],'error')
            self.assertIn('Missing action',result['error'])
            self.assertNotIn('empty patch',result['error'])
        self.assertEqual(self.state()['attempts'],3)
        self.assertTrue(self.call(action='inspect',query='answer')['matches'])
    def test_rejected_repetitions_are_counted_and_audit_remains_available(self):
        for i in range(32):
            r=self.call(query='answer')
            self.assertEqual(r['attempts'],i+1)
        self.assertEqual(self.state()['work_attempts'],32)
        self.assertIn('budget exhausted',self.call(action='inspect')['error'])
        for _ in range(3):self.assertEqual(self.call(action='audit')['errors'],['empty patch'])
        self.assertEqual(self.state()['attempts'],36)
    def test_unknown_argument_is_structured_and_counted(self):
        r=self.call(inspect_path='pkg')
        self.assertEqual(r['outcome'],'error')
        self.assertIn('unrecognized arguments',r['error'])
        self.assertEqual(self.state()['attempts'],1)
    def test_edit_requires_valid_assertion_before_write(self):
        before=(self.root/'src/pkg/__init__.py').read_bytes()
        for kwargs in ({'code':''},{'module':''},{'code':'print(42)'},{'code':'assert ('},{'timeout':'0'}):
            self.assertEqual(self.edit(**kwargs)['outcome'],'error')
            self.assertEqual((self.root/'src/pkg/__init__.py').read_bytes(),before)
    def test_failed_edit_check_retained_then_repaired_and_verified(self):
        bad=self.edit(value=40)
        self.assertEqual((bad['outcome'],bad['verification']),('edited','failed'))
        self.assertIn('return 40',(self.root/'src/pkg/__init__.py').read_text())
        self.assertEqual(self.call(role='verifier',action='verify')['outcome'],'failed')
        good=self.edit(value=42,old=40)
        self.assertEqual(good['verification'],'passed')
        self.assertEqual(self.call(role='verifier',action='verify')['outcome'],'passed')
        a=self.call(role='finalizer',action='audit')
        self.assertEqual((a['outcome'],a['verification'],a['evidence_fresh']),('clean','passed',True))
        self.assertEqual(a['edits'],2);self.assertEqual(a['checks'],4)
    def test_out_of_band_edits_invalidate_evidence(self):
        self.edit()
        (self.root/'src/pkg/__init__.py').write_text('def answer():\n    return 43\n')
        self.assertFalse(self.call(action='audit')['evidence_fresh'])
        self.assertEqual(self.call(role='verifier',action='verify')['outcome'],'failed')
    def test_repeat_audit_does_not_blacklist_unverified_state(self):
        for _ in range(3):self.assertEqual(self.call(action='audit')['outcome'],'failed')
        self.edit()
        for _ in range(3):self.assertEqual(self.call(action='audit')['verification'],'passed')
    def test_invalid_edit_is_atomic_and_duplicate_error_is_actionable(self):
        before=(self.root/'src/pkg/__init__.py').read_bytes()
        self.assertEqual(self.edit(new='return (')['outcome'],'error')
        self.assertIn('Repeated failed edit',self.edit(new='return (')['error'])
        self.assertEqual(self.state()['attempts'],2)
        self.assertEqual((self.root/'src/pkg/__init__.py').read_bytes(),before)
    def test_role_permissions(self):
        for role in ['verifier','finalizer']:
            r=self.call(role=role,action='replace',path='src/pkg/__init__.py',old='return 41',new='return 42',
                        module='pkg',code='import pkg\nassert pkg.answer() == 42')
            self.assertIn('Action not allowed',r['error'])
        self.assertIn('Action not allowed',self.call(role='finalizer',action='check')['error'])
        self.assertIn('return 41',(self.root/'src/pkg/__init__.py').read_text())
    def test_workflows_allowed_but_test_configuration_protected(self):
        r=self.call(action='create',path='.github/workflows/release.yml',new='name: Release\n',module='-',
                    code="from pathlib import Path\nassert 'name: Release' in Path('.github/workflows/release.yml').read_text()")
        self.assertEqual(r['verification'],'passed')
        for name in ['.git/config','tests/new.py','pyproject.toml','conftest.py','../outside']:
            self.assertEqual(self.call(action='create',path=name,new='x=1',module='-',code='assert True')['outcome'],'error')
        self.assertEqual(self.call(action='audit')['verification'],'passed')
    def test_scoped_literal_search_and_directory_inspection(self):
        self.assertEqual(self.call(action='inspect',path='src')['entries'],['pkg/'])
        self.assertTrue(self.call(action='inspect',path='src/pkg',query='return 41')['matches'])
        self.assertFalse(self.call(action='inspect',path='tests',query='return 41')['matches'])
        self.assertFalse(self.call(action='inspect',query='return .*')['matches'])
        self.assertIn('return 41',self.call(action='inspect',path='src/pkg/__init__.py',symbol='answer')['source'])
    def test_false_evidence_cannot_pass(self):
        self.edit()
        cases=[('try:\n assert False\nexcept AssertionError:\n pass','failed'),
               ('if False:\n assert True','no_assertions_executed'),
               ('import sys\nsys.exit(0)\nassert True','incomplete')]
        for code,outcome in cases:
            self.assertEqual(self.call(action='check',module='pkg',code=code)['outcome'],outcome)
            self.assertEqual(self.call(action='audit')['verification'],'unverified')
        self.assertEqual(self.call(action='check',module='json',code='assert True')['outcome'],'import_origin_error')
    def test_mutation_timeout_and_protected_changes_fail_audit(self):
        self.edit()
        r=self.call(action='check',module='pkg',timeout='1',code='import time\ntime.sleep(5)\nassert True')
        self.assertEqual(r['outcome'],'timeout')
        code="from pathlib import Path\nPath('src/pkg/__init__.py').write_text('answer = 1\\n')\nassert True"
        self.assertEqual(self.call(action='check',module='pkg',code=code)['outcome'],'failed')
        (self.root/'tests/test_api.py').write_text('assert False\n')
        self.assertEqual(self.call(action='audit')['outcome'],'failed')
    def test_verify_without_check_fails(self):
        self.assertIn('No recorded check',self.call(role='verifier',action='verify')['error'])
    def test_reproducible_archive_and_role_copies(self):
        a=Path(self.temp.name)/'a.zip';b=Path(self.temp.name)/'b.zip'
        build(a);build(b)
        self.assertEqual(a.read_bytes(),b.read_bytes())


if __name__=='__main__':unittest.main()
