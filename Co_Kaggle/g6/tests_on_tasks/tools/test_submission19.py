"""Regression coverage for Submission 18 structural and evaluation failures."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
from build_submission19 import SOURCE, G6, build, ARCHIVE
from preflight_submission19 import preflight
spec=importlib.util.spec_from_file_location('audit19',SOURCE/'skills/patch-audit/scripts/audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class Regressions(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.write('pkg.py','class Public:\n    value = 1\n\ndef render():\n    return 1\n')
        self.write('pytest.ini','[pytest]\n');self.write('scripts/test.sh','python -m pytest tests\n')
        self.write('.gitignore','__pycache__/\n.pytest_cache/\n')
        self.git('init','-q');self.git('add','.');self.git('-c','user.name=Test','-c','user.email=t@example.invalid','commit','-qm','baseline')
    def tearDown(self):self.tmp.cleanup()
    def git(self,*args):return subprocess.run(['git','-C',str(self.root),*args],check=True,capture_output=True)
    def write(self,name,data):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
    def test_failure_excerpts_are_actual_submission18_calls(self):
        cases=json.loads((G6/'tests_on_tasks/reviews/submission_19/failures.json').read_text())
        with zipfile.ZipFile(G6/'tests_on_tasks/diagnostics/submission-18-zip_First15Tasks.zip') as z:
            for case in cases.values():
                raw=z.read(case['source']);self.assertEqual(hashlib.sha256(raw).hexdigest(),case['sha256'])
                self.assertTrue(any(t.get('arguments')==case['arguments'] and t.get('result')==case['result'] for t in json.loads(raw)))

    def test_exact_archived_requests_replacement_triggers_audit(self):
        case=json.loads((G6/'tests_on_tasks/reviews/submission_19/failures.json').read_text())['requests_7433']['arguments']
        baseline='def prepare_body(data):\n'+case['old_string']+'\n        ):\n            pass\n'
        ast.parse(baseline)
        self.write('source.py',baseline);self.git('add','.');self.git('-c','user.name=T','-c','user.email=t@x','commit','-qm','fixture')
        self.write('source.py',baseline.replace(case['old_string'],case['new_string']))
        self.assertEqual(audit.audit(self.root)['status'],'blocked')

    def test_extra_closing_parenthesis_is_blocked(self):
        self.write('pkg.py','if True:\n):\n    pass\n')
        self.assertEqual(audit.audit(self.root)['status'],'blocked')
    def test_decorator_inside_parameters_is_blocked(self):
        self.write('pkg.py','class App:\n    def __init__(self,\n    @property\n    def vibe(self): return "default"\n')
        self.assertEqual(audit.audit(self.root)['status'],'blocked')
    def test_class_erasure_detected_even_when_syntax_valid(self):
        self.write('pkg.py','def render():\n    return 1\n    value = 1\n')
        r=audit.audit(self.root);self.assertIn('pkg.py:Public',r['removed_definitions']);self.assertEqual(r['status'],'review_required')
    def test_test_configuration_and_runner_changes_blocked(self):
        for n in ['pytest.ini','scripts/test.sh','scripts/coverage.sh','tests/test_api.py','pyproject.toml']:
            self.write(n,'# changed\n')
        r=audit.audit(self.root);self.assertEqual(len([b for b in r['blockers'] if 'protected' in b]),5)
    def test_workflow_changes_allowed_when_task_requires_them(self):
        self.write('.github/workflows/release.yml','name: release\n');self.assertEqual(audit.audit(self.root)['status'],'structurally_clean')
    def test_clean_audit_never_claims_behavior_verified(self):
        self.write('pkg.py','class Public:\n    value = 2\n\ndef render():\n    return 1\n')
        r=audit.audit(self.root);self.assertEqual(r['status'],'structurally_clean');self.assertIs(r['behavior_verified'],False)
        h=r['patch_sha256'];self.write('pkg.py','class Public:\n    value = 3\n\ndef render():\n    return 1\n')
        self.assertNotEqual(h,audit.audit(self.root)['patch_sha256'])
    def test_empty_patch_cannot_pass(self):self.assertEqual(audit.audit(self.root)['status'],'blocked')
    def test_symlink_and_oversized_files_fail_closed(self):
        (self.root/'link.py').symlink_to('/etc/hosts');self.write('huge.py','x'*1_000_001)
        self.assertEqual(audit.audit(self.root)['status'],'blocked')
    def test_preflight_rejects_invented_test_path(self):
        with self.assertRaisesRegex(ValueError,'Discover'):preflight(self.root,'pkg','tests/test_models.py')
    def test_preflight_requires_clean_baseline(self):
        self.write('pkg.py','x=2\n')
        with self.assertRaisesRegex(ValueError,'clean'):preflight(self.root,'pkg','tests/test_models.py')
    def test_preflight_rejects_installed_module_as_local_evidence(self):
        self.write('tests/test_api.py','def test_ok(): assert True\n');self.git('add','.');self.git('-c','user.name=T','-c','user.email=t@x','commit','-qm','tests')
        r=preflight(self.root,'json','tests/test_api.py');self.assertFalse(r['local_import']);self.assertFalse(r['ready'])
    def test_preflight_validates_local_src_import_and_selected_test(self):
        self.write('src/demo19/__init__.py','VALUE=42\n');self.write('tests/test_api.py','from demo19 import VALUE\ndef test_value(): assert VALUE == 42\n')
        self.git('add','.');self.git('-c','user.name=T','-c','user.email=t@x','commit','-qm','tests')
        r=preflight(self.root,'demo19','tests/test_api.py');self.assertTrue(r['ready'],r)
    def test_reproducible_package_excludes_test_evidence(self):
        a=build();b=ARCHIVE.read_bytes();self.assertEqual(a,build());self.assertEqual(b,ARCHIVE.read_bytes())
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertEqual(len(z.namelist()),5)
            self.assertFalse(any('reviews' in n or 'test_submission' in n for n in z.namelist()))

if __name__=='__main__':unittest.main()
