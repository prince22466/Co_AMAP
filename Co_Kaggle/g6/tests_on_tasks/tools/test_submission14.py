"""CPU regression tests for failure modes observed in submissions 13/14."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from build_submission14 import SOURCE, build

SCRIPT = SOURCE / 'skills/repo-navigation/scripts/workbench.py'


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'checkout'
        self.root.mkdir()
        (self.root/'src/pkg').mkdir(parents=True)
        (self.root/'src/pkg/__init__.py').write_text('def answer():\n    return 41\n')
        (self.root/'tests').mkdir()
        (self.root/'tests/test_api.py').write_text('assert True\n')
        self.git('init','-q')
        self.git('add','.')
        self.git('-c','user.name=CPU tests','-c','user.email=test@example.invalid','commit','-qm','baseline')
    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.root),*args])
    def call(self,**args):
        # Deliberately execute from a different directory, as ADK does.
        cmd=[sys.executable,str(SCRIPT)]
        for k,v in args.items():cmd += ['--'+k,str(v)]
        p=subprocess.run(cmd,cwd=self.temp.name,env={**os.environ,'PYTHONPATH':str(self.root)},
                         capture_output=True,text=True,timeout=25)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertLess(len(p.stdout),4900)
        return json.loads(p.stdout)
    def fix(self):
        return self.call(action='replace',path='src/pkg/__init__.py',old='return 41',new='return 42')
    def test_real_workspace_src_import_and_patch_bound_evidence(self):
        result=self.call(action='check',module='pkg',code='import pkg\nassert pkg.answer() == 42')
        self.assertEqual(result['outcome'],'failed')
        self.assertEqual(self.fix()['outcome'],'edited')
        result=self.call(action='check',module='pkg',code='import pkg\nassert pkg.answer() == 42')
        self.assertEqual(result['outcome'],'passed')
        self.assertTrue(result['module_origin'].startswith(str(self.root/'src')))
        audit=self.call(action='audit')
        self.assertEqual((audit['outcome'],audit['verification']),('clean','passed'))
        self.call(action='replace',path='src/pkg/__init__.py',old='return 42',new='return 43')
        self.assertFalse(self.call(action='audit')['evidence_fresh'])
    def test_invalid_edit_is_atomic_and_repeated_failure_blocked(self):
        args=dict(action='replace',path='src/pkg/__init__.py',old='return 41',new='return (')
        before=(self.root/'src/pkg/__init__.py').read_bytes()
        self.assertEqual(self.call(**args)['outcome'],'error')
        self.assertEqual((self.root/'src/pkg/__init__.py').read_bytes(),before)
        self.assertIn('Repeated',self.call(**args)['error'])
    def test_protected_ambiguous_traversal_and_new_files(self):
        self.assertEqual(self.call(action='replace',path='tests/test_api.py',old='True',new='False')['outcome'],'error')
        self.assertEqual(self.call(action='replace',path='../outside',old='x',new='y')['outcome'],'error')
        self.assertEqual(self.call(action='replace',path='src/pkg/__init__.py',old=' ',new='x')['outcome'],'error')
        self.assertEqual(self.call(action='create',path='src/pkg/new.py',new='value = 2\n')['outcome'],'edited')
        self.assertIn('src/pkg/new.py',self.call(action='audit')['changed_paths'])
        self.assertEqual(self.git('diff','--cached'),b'')
    def test_errors_never_become_passing_evidence(self):
        self.fix()
        for code,expected in [('try:\n assert False\nexcept AssertionError:\n pass','failed'),
                              ('if False:\n assert True','no_assertions_executed'),
                              ('import sys\nsys.exit(0)','incomplete')]:
            self.assertEqual(self.call(action='check',module='pkg',code=code)['outcome'],expected)
            self.assertEqual(self.call(action='audit')['verification'],'unverified')
        self.assertEqual(self.call(action='check',module='json',code='assert True')['outcome'],'import_origin_error')
    def test_check_timeout_and_mutation_are_not_accepted(self):
        r=self.call(action='check',module='pkg',timeout=1,code='import time\ntime.sleep(10)\nassert True')
        self.assertEqual(r['outcome'],'timeout')
        code="from pathlib import Path\nPath('src/pkg/__init__.py').write_text('value = 3\\n')\nassert True"
        self.assertEqual(self.call(action='check',module='pkg',code=code)['outcome'],'failed')
    def test_empty_and_out_of_band_invalid_patches_fail_audit(self):
        self.assertEqual(self.call(action='audit')['outcome'],'failed')
        (self.root/'src/pkg/__init__.py').write_text('def broken(:\n')
        self.assertEqual(self.call(action='audit')['outcome'],'failed')
    def test_symbol_inspection_and_literal_search(self):
        r=self.call(action='inspect',path='src/pkg/__init__.py',symbol='answer')
        self.assertIn('return 41',r['source'])
        self.assertTrue(self.call(action='inspect',query='return 41')['matches'])
    def test_reproducible_archive(self):
        a=Path(self.temp.name)/'a.zip'; b=Path(self.temp.name)/'b.zip'
        build(a);build(b)
        self.assertEqual(a.read_bytes(),b.read_bytes())


if __name__ == '__main__':unittest.main()
