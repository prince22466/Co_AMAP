"""CPU regressions for S21 native editing, optional structural audit, and S20 failures."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
import yaml
from build_submission21 import SOURCE, G6, ARCHIVE, build

spec=importlib.util.spec_from_file_location('audit21', SOURCE/'skills/patch-audit/scripts/audit.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
BASE='class Public:\n    value = 1\n\ndef render():\n    return 1\n'

class NativeRegressions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.write('pkg.py', BASE)
        self.write('guide.md','old docs\n')
        self.write('.gitignore','__pycache__/\n')
        self.git('init','-q')
        self.git('add','.')
        self.git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','baseline')
    def tearDown(self):
        self.temp.cleanup()
    def write(self,name,data):
        p=self.root/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(data)
    def git(self,*args):
        return subprocess.run(['git','-C',str(self.root),*args],capture_output=True,check=True)
    def test_native_editing_never_depends_on_gate_skill(self):
        agent=yaml.safe_load((SOURCE/'agent.yaml').read_text().replace('!include instruction.md','null'))
        tools=agent['tools']
        self.assertEqual(tools,['get_status','read_file','edit_file','write_file','run_command','submit_patch'])
        self.assertEqual(agent['skills'],['skills/patch-audit'])
        self.assertNotIn('candidate',str(agent))
    def test_packaged_instruction_uses_native_only_for_edits(self):
        text=(SOURCE/'instruction.md').read_text()
        self.assertIn('edit_file(filepath, old_string, new_string)',text)
        self.assertIn('run_command(command)',text)
        self.assertIn('baseline',text.lower())
        self.assertNotIn('skill_name="candidate"',text)
        self.assertNotIn('scripts/gate.py',text)
    def test_replay_s20_fatal_gateway_regression_evidence(self):
        log=(G6/'tests_on_tasks/diagnostics/submission-20-zip_First15TasksPrintouts.txt').read_text()
        self.assertIn("Skill 'patch_applier' not found.",log)
        self.assertIn("Tool 'edit' not found.",log)
        self.assertIn("Script 'scripts/candidate' not found",log)
        self.assertIn('patch_chars\tduration_seconds',log)
        # S21 cannot require those nonexistent edit wrappers.
        tools=yaml.safe_load((SOURCE/'agent.yaml').read_text().replace('!include instruction.md','null'))['tools']
        self.assertIn('edit_file',tools)
    def test_broken_syntax_is_a_blocker(self):
        self.write('pkg.py','def render(:\n    pass\n')
        self.assertEqual(audit.audit(self.root)['status'],'blocked')
    def test_removed_definition_needs_review(self):
        self.write('pkg.py','def render():\n    return 2\n')
        result=audit.audit(self.root)
        self.assertEqual(result['status'],'review_required')
        self.assertIn('pkg.py:Public',result['removed_definitions'])
    def test_protected_tests_config_and_harness_paths_flagged(self):
        for name in ['Tests/test_pkg.py','.pytest.ini','_swegemma_stubs.py',
                     'scripts/test.sh','pyproject.toml']:
            self.write(name,'new\n')
        result=audit.audit(self.root)
        self.assertEqual(result['status'],'blocked')
        self.assertEqual(len([v for v in result['blockers'] if 'protected' in v]),5)
    def test_clean_syntax_does_not_claim_behavior(self):
        self.write('pkg.py',BASE.replace('return 1','return 2'))
        result=audit.audit(self.root)
        self.assertEqual(result['status'],'structurally_clean')
        self.assertFalse(result['behavior_verified'])
        self.assertTrue(result['patch_sha256'])
        self.write('pkg.py',BASE.replace('return 1','return 3'))
        self.assertNotEqual(result['patch_sha256'],audit.audit(self.root)['patch_sha256'])
    def test_empty_patch_rejected(self):
        self.assertEqual(audit.audit(self.root)['status'],'blocked')
    def test_archive_package_fields(self):
        manifest=build()
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertEqual(set(z.namelist()),set(manifest['files']))
            for n in z.namelist():
                if n.endswith('.py'):ast.parse(z.read(n).decode())
        self.assertIn('archive_sha256',manifest)

if __name__=='__main__':
    unittest.main()
