"""Submission 22 CPU regressions. No benchmark tests are changed or mocked."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
import yaml
from build_submission22 import ARCHIVE, G6, SOURCE, FILES, build

def load(name):
    path = SOURCE / 'skills/patch-audit/scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location('submission22_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

audit = load('audit')
recover = load('recover')
verify = load('verify')
BASE = 'class Public:\n    value = 1\n\ndef render():\n    return 1\n'

class TestNativeRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.write('pkg.py', BASE)
        self.write('guide.md', 'old guide\n')
        self.write('.gitignore', '__pycache__/\n')
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c',
                 'user.email=tests@example.invalid', 'commit', '-qm', 'baseline')

    def tearDown(self):
        self.temp.cleanup()
        try:
            recover.failure_state(self.root).unlink()
        except FileNotFoundError:
            pass

    def write(self, path, data):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(data)
        return file

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args],
                              check=True, capture_output=True, text=True)

    def test_keeps_native_tools_and_single_agent(self):
        agent = yaml.safe_load((SOURCE / 'agent.yaml').read_text()
                               .replace('!include instruction.md', 'null'))
        self.assertEqual(agent['tools'], [
            'get_status', 'read_file', 'edit_file', 'write_file', 'run_command', 'submit_patch'
        ])
        self.assertEqual(agent['skills'], ['skills/patch-audit'])
        self.assertNotIn('sub_agents', agent)
        text = (SOURCE / 'instruction.md').read_text()
        self.assertIn('On a SECOND failed edit', text)
        self.assertIn('git diff HEAD --stat', text)
        self.assertIn('Rerun the SAME', text)
        self.assertNotIn('file_path="scripts/gate.py"', text)

    def test_unique_recovery_edits_live_file(self):
        r = recover.attempt(self.root, 'pkg.py', 'return 1', 'return 2')
        self.assertEqual(r['status'], 'edited')
        self.assertIn('return 2', (self.root / 'pkg.py').read_text())
        self.assertFalse(r['behavior_verified'])
        self.assertEqual(audit.audit(self.root)['status'], 'structurally_clean')

    def test_anchor_mismatch_then_deterministic_retry_block(self):
        a = recover.attempt(self.root, 'pkg.py', 'return 999', 'return 2')
        b = recover.attempt(self.root, 'pkg.py', 'return 999', 'return 2')
        self.assertEqual(a['status'], 'anchor_mismatch')
        self.assertEqual(b['status'], 'retry_blocked')
        self.assertTrue(a['nearby']['context'])
        self.assertEqual((self.root / 'pkg.py').read_text(), BASE)

    def test_ambiguous_anchor_must_not_modify_file(self):
        r = recover.attempt(self.root, 'pkg.py', '    ', '  ')
        self.assertEqual(r['matches'], BASE.count('    '))
        self.assertEqual(r['status'], 'anchor_mismatch')
        self.assertEqual((self.root / 'pkg.py').read_text(), BASE)

    def test_syntax_failure_rolls_back(self):
        r = recover.attempt(self.root, 'pkg.py', 'def render():', 'def render(:')
        self.assertEqual(r['status'], 'blocked')
        self.assertEqual((self.root / 'pkg.py').read_text(), BASE)

    def test_recover_from_preexisting_syntax_error(self):
        self.write('pkg.py', BASE.replace('def render():', 'def render(:'))
        result = recover.attempt(self.root, 'pkg.py', 'def render(:', 'def render():\n    # valid repair')
        self.assertEqual(result['status'], 'edited')
        self.assertTrue(result['baseline_had_syntax_error'])
        ast.parse((self.root / 'pkg.py').read_text())

    def test_undo_broken_edit_without_inventing_nonempty_patch(self):
        self.write('pkg.py', BASE.replace('def render():', 'def render(:'))
        result = recover.attempt(self.root, 'pkg.py', 'def render(:', 'def render():')
        self.assertEqual(result['status'], 'restored_baseline')
        self.assertFalse(result['submittable'])
        self.assertEqual((self.root / 'pkg.py').read_text(), BASE)
        self.assertFalse(self.git('status', '--porcelain').stdout.strip())

    def test_ignored_file_cannot_be_reported_as_submittable(self):
        self.write('.gitignore', '__pycache__/\\ngenerated.tmp\\n'.replace('\\n', '\n'))
        self.write('generated.tmp', 'alpha\n')
        result = recover.attempt(self.root, 'generated.tmp', 'alpha', 'beta')
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('not visible', result['reason'])
        self.assertEqual((self.root / 'generated.tmp').read_text(), 'alpha\n')

    def test_public_api_removal_rolls_back(self):
        r = recover.attempt(self.root, 'pkg.py', 'class Public:', 'if False:')
        self.assertEqual(r['status'], 'blocked')
        self.assertEqual((self.root / 'pkg.py').read_text(), BASE)

    def test_protected_tests_are_refused(self):
        self.write('tests/test_pkg.py', 'def test_x(): pass\n')
        with self.assertRaises(ValueError):
            recover.attempt(self.root, 'tests/test_pkg.py', 'pass', 'return 1')
        with self.assertRaises(ValueError):
            recover.attempt(self.root, '../pkg.py', 'return 1', 'return 2')

    def test_legitimate_pyproject_changes_are_advisory_not_blocked(self):
        self.write('pyproject.toml', 'requires-python = ">=3.8"\n')
        self.git('add', 'pyproject.toml')
        self.git('-c', 'user.name=Test', '-c',
                 'user.email=tests@example.invalid', 'commit', '-qm', 'config')
        self.write('pyproject.toml', 'requires-python = ">=3.9"\n')
        result = audit.audit(self.root)
        self.assertEqual(result['status'], 'structurally_clean')
        self.assertTrue(result['advisory'])

    def test_empty_patch_and_protected_paths_are_blocked(self):
        self.assertEqual(audit.audit(self.root)['status'], 'blocked')
        self.write('tests/test_bug.py', 'pass\n')
        result = audit.audit(self.root)
        self.assertEqual(result['status'], 'blocked')
        self.assertTrue(any('protected' in s for s in result['blockers']))

    def test_audit_never_claims_submission_or_behavior_verified(self):
        self.write('pkg.py', BASE.replace('return 1', 'return 2'))
        one = audit.audit(self.root)
        self.assertFalse(one['behavior_verified'])
        self.assertFalse(one['submission_verified'])
        self.assertTrue(one['patch_sha256'])
        self.write('pkg.py', BASE.replace('return 1', 'return 3'))
        self.assertNotEqual(one['patch_sha256'], audit.audit(self.root)['patch_sha256'])

    def test_candidate_assertion_and_local_import_without_baseline_claim(self):
        self.write('pkg.py', BASE.replace('return 1', 'return 2'))
        result = verify.verify(self.root, 'pkg', 'import pkg\nassert pkg.render() == 2')
        self.assertEqual(result['status'], 'candidate_passed')
        self.assertTrue(result['candidate_passed'])
        self.assertFalse(result['baseline_verified'])
        self.assertFalse(result['behavior_verified'])
        self.assertTrue(result['origin'].startswith(str(self.root)))

    def test_candidate_failure_and_unrelated_green_test_rejected(self):
        self.write('pkg.py', BASE.replace('return 1', 'return 2'))
        self.assertEqual(verify.verify(self.root, 'pkg', 'assert True')['status'], 'unverified')
        self.assertEqual(verify.verify(self.root, 'pkg', 'assert 1 == 1')['status'], 'unverified')
        wrong = verify.verify(self.root, 'pkg', 'import pkg\nassert pkg.render() == 3')
        self.assertEqual(wrong['status'], 'environment_or_probe_failure')

    def test_unrelated_green_check_does_not_execute_changed_code(self):
        self.write('pkg.py', BASE.replace('return 1', 'return 2'))
        result = verify.verify(self.root, 'pkg',
                               'import pkg\nassert pkg.__name__ == "pkg"')
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertFalse(result['candidate_passed'])
        self.assertFalse(result['changed_code_executed'])

    def test_import_time_changed_line_is_traced(self):
        self.write('pkg.py', 'FLAG = 2\n')
        result = verify.verify(self.root, 'pkg', 'import pkg\nassert pkg.FLAG == 2')
        self.assertEqual(result['status'], 'candidate_passed')
        self.assertTrue(result['changed_code_executed'])

    def test_generated_adk_scratch_is_not_a_candidate_patch(self):
        self.write('.adk_exec_probe.py', 'VALUE = 2\n')
        result = verify.verify(self.root, '', 'import os\nassert bool(os.getcwd())')
        self.assertEqual(result['status'], 'unverified')
        self.assertIn('No candidate patch', result['reason'])

    def test_import_provenance_and_no_patch(self):
        self.assertEqual(verify.verify(self.root, 'pkg', 'import pkg\nassert pkg.render() == 1')['status'],
                         'unverified')
        self.write('pkg.py', BASE.replace('return 1', 'return 2'))
        r = verify.verify(self.root, 'json', 'import json\nassert json.dumps([1]) == "[1]"')
        self.assertEqual(r['status'], 'environment_or_probe_failure')

    def test_mutating_probe_is_not_evidence(self):
        self.write('pkg.py', BASE.replace('return 1', 'return 2'))
        p = ("from pathlib import Path\nPath('pkg.py').write_text('def hacked(): pass')\n"
             "assert Path('pkg.py').exists()")
        self.assertEqual(verify.verify(self.root, '', p)['status'], 'blocked')

    def test_package_exact_manifest_and_scripts_parse(self):
        manifest = build()
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertEqual(set(z.namelist()), set(FILES))
            for name in FILES:
                data = z.read(name)
                self.assertEqual(__import__('hashlib').sha256(data).hexdigest(),
                                 manifest['files'][name])
                if name.endswith('.py'):
                    ast.parse(data.decode())
        self.assertIn('archive_sha256', manifest)

if __name__ == '__main__':
    unittest.main()
