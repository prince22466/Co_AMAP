"""S27 reproducibility and proxy-analysis tests; NOT model-performance tests."""
import hashlib
import unittest
import zipfile

import yaml
from build_submission27 import ARCHIVE, FILES, IncludeLoader, SOURCE, TOOLS, build
from analyze_submission27_traces import analyze


class Submission27Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build()

    def test_package_identical_on_rebuild(self):
        first = ARCHIVE.read_bytes()
        self.assertEqual(build()['archive_sha256'], hashlib.sha256(first).hexdigest())
        self.assertEqual(ARCHIVE.read_bytes(), first)
        with zipfile.ZipFile(ARCHIVE) as archive:
            self.assertEqual(sorted(archive.namelist()), sorted(FILES))
            self.assertIsNone(archive.testzip())
            for path in FILES:
                self.assertEqual(archive.read(path), (SOURCE / path).read_bytes())

    def test_native_workflow_has_no_new_runtime_dependency(self):
        with zipfile.ZipFile(ARCHIVE) as archive:
            config = yaml.load(archive.read('agent.yaml'), Loader=IncludeLoader)
        self.assertEqual(tuple(config['tools']), TOOLS)
        self.assertNotIn('skills', config)
        self.assertNotIn('sub_agents', config)
        self.assertEqual(config['include_contents'], 'default')

    def test_contract_addresses_failure_modes(self):
        instruction = (SOURCE / 'instruction.md').read_text()
        for criterion in ('R =', 'L =', 'H =', 'F =', 'SECOND anchor failure',
                          'same file+line range', 'STOP new discovery', 'submit_patch',
                          'UNVERIFIED', 'run_skill_script', 'bench', 'diff --check'):
            with self.subTest(criterion=criterion):
                self.assertIn(criterion, instruction)

    def test_printed_trace_proxies_do_not_invent_arg_identity(self):
        trace = """[candidate 1/2] Evaluating sample_1 (repo)
  [  1.0s] 📖 read_file (p.py)
  [  1.1s]   ✓ ok
  [  2.0s] 📖 read_file (p.py)
  [  2.1s]   ✓ ok
  [  3.0s] ✏️ edit_file (p.py)
  [  3.1s]   ❌ old_string not found
  [  4.0s] ✏️ edit_file (p.py)
  [  4.1s]   ❌ old_string not found
  [  5.0s] 🤖 submit_patch
  -> resolved=False, patch_chars=0, tools=40
[candidate 2/2] Evaluating sample_2 (repo)
  [  1.0s] ✏️ edit_file (x.py)
  [  1.1s]   ✓ ok
  -> resolved=True, patch_chars=73, tools=2
"""
        result = analyze(trace)
        self.assertEqual(result['tasks_with_result'], 2)
        self.assertEqual(result['resolved'], 1)
        self.assertEqual(result['empty_patches'], 1)
        self.assertEqual(result['charged_call_exhaustions'], 1)
        self.assertEqual(result['failed_edits_observed'], 2)
        self.assertEqual(result['same_path_failed_edit_streaks'], 1)
        self.assertEqual(result['same_path_read_streaks'], 1)
        self.assertIn('proxy', result['scope'].lower())

    def test_no_result_not_silently_counted_as_failure(self):
        result = analyze('[candidate 1/2] Evaluating missing (repo)')
        self.assertEqual(result['tasks_seen'], 1)
        self.assertEqual(result['tasks_with_result'], 0)
        self.assertIsNone(result['tasks'][0]['resolved'])


if __name__ == '__main__':
    unittest.main()
