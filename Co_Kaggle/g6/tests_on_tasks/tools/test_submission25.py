"""Executable regressions derived from S24 failure classes, not E4B score tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from build_submission25 import ARCHIVE, FILES, SOURCE, build

spec = importlib.util.spec_from_file_location('s25_work', SOURCE / 'skills/repair/scripts/work.py')
work = importlib.util.module_from_spec(spec)
spec.loader.exec_module(work)

BASE = 'def render(disabled=False):\n    return "\\n"\n'
PROBE = 'import sample\nassert sample.render(True) == ""\nassert sample.render(False) == "\\n"\n'


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'sample.py').write_text(BASE)
        (self.root / 'tests').mkdir()
        (self.root / 'tests/test_sample.py').write_text(
            'import sample\ndef test_render():\n    assert sample.render(True) == ""\n'
            'def test_control():\n    assert sample.render() == "\\n"\n')
        (self.root / '.gitignore').write_text('__pycache__/\n.pytest_cache/\n')
        for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'baseline')]:
            work.git(self.root, *args)
        self.w = work.Workbench(self.root)

    def fix(self):
        self.w.read('sample.py')
        return self.w.edit('sample.py', 2, 2, '    return "" if disabled else "\\n"\n')

    def test_s24_repeated_guard_cannot_corrupt_first_repair(self):
        baseline = self.w.check(dict(op='probe', code=PROBE))
        self.assertEqual(baseline['status'], 'FAIL')
        self.fix()
        self.assertTrue(self.w.check()['fail_to_pass'])
        retained = (self.root / 'sample.py').read_bytes()
        self.w.read('sample.py')
        with self.assertRaises(SyntaxError):
            self.w.edit('sample.py', 2, 2, '    if not disabled:\n    if not disabled:\n        return "\\n"\n')
        self.assertEqual((self.root / 'sample.py').read_bytes(), retained)
        self.w.read('sample.py')
        with self.assertRaisesRegex(ValueError, 'saved passing check'):
            self.w.edit('sample.py', 2, 2, '    return "broken"\n')
        self.assertEqual((self.root / 'sample.py').read_bytes(), retained)
        self.assertEqual(self.w.check()['status'], 'PASS')

    def test_stale_ranges_and_identical_failures_are_blocked(self):
        self.w.read('sample.py')
        with self.assertRaises(SyntaxError):
            self.w.edit('sample.py', 2, 2, '    if disabled:\n')
        with self.assertRaisesRegex(ValueError, 'Identical failed edit'):
            self.w.edit('sample.py', 2, 2, '    if disabled:\n')
        self.fix()
        with self.assertRaisesRegex(ValueError, 'Stale/missing read'):
            self.w.edit('sample.py', 2, 2, '    return "different"\n')
        # A read fixes the precondition; the earlier missing-read rejection must
        # not permanently blacklist a potentially legitimate edit.
        self.w.read('sample.py')
        self.assertEqual(self.w.edit('sample.py', 2, 2, '    return "different"\n')['status'], 'EDITED')

    def test_actual_source_change_after_read_rejects_stale_edit(self):
        self.w.read('sample.py')
        (self.root / 'sample.py').write_text('# another edit\n' + BASE)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            self.w.edit('sample.py', 2, 2, '    return 42\n')
        self.assertTrue((self.root / 'sample.py').read_text().startswith('# another edit'))

    def test_compile_rejects_invalid_python_that_ast_parse_accepts(self):
        self.w.read('sample.py')
        with self.assertRaises(SyntaxError):
            self.w.edit('sample.py', 1, 2, 'return 42\n')
        self.assertEqual((self.root / 'sample.py').read_text(), BASE)

    def test_missing_text_argument_never_means_delete(self):
        import os, sys
        self.w.read('sample.py')
        result = subprocess.run([sys.executable, str(SOURCE / 'skills/repair/scripts/work.py'),
            '--op', 'edit', '--path', 'sample.py', '--start', '1', '--end', '2'],
            cwd=self.root, env=dict(os.environ, PYTHONPATH=str(self.root)),
            text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'REJECTED')
        self.assertEqual((self.root / 'sample.py').read_text(), BASE)

    def test_replacement_does_not_copy_ambiguous_anchor_or_double_escape(self):
        source = 'def one():\n    return 1\ndef two():\n    return 1\n'
        (self.root / 'sample.py').write_text(source)
        self.w.read('sample.py')
        self.w.edit('sample.py', 4, 4, '    return 2\n')
        self.assertEqual((self.root / 'sample.py').read_text(), source[:-13] + '    return 2\n')
        self.w.read('sample.py')
        before = (self.root / 'sample.py').read_bytes()
        with self.assertRaises(SyntaxError):
            self.w.edit('sample.py', 4, 4, r'    if True:\n        return 3')
        self.assertEqual((self.root / 'sample.py').read_bytes(), before)

    def test_missing_api_and_empty_checks_never_pass(self):
        for code in ['print("all passed")', 'assert True',
                     'import sample\nassert True',
                     'import sample\nassert 1 == 1',
                     'import sample\nassert sample.missing_api() == 1',
                     'import sample\\nassert sample.render() == ""']:
            with self.subTest(code=code):
                try:
                    result = self.w.check(dict(op='probe', code=code))
                except SyntaxError:
                    continue
                self.assertNotEqual(result['status'], 'PASS')
        (self.root / 'tests/test_empty.py').write_text('value = 1\n')
        self.assertEqual(self.w.check(dict(op='test', path='tests/test_empty.py'))['status'], 'BLOCKED')
        self.assertEqual(self.w.check(dict(op='test', path='wrong/tests/test_sample.py'))['status'], 'BLOCKED')

    def test_pytest_fail_to_pass_uses_same_existing_file(self):
        self.assertEqual(self.w.check(dict(op='test', path='tests/test_sample.py'))['status'], 'FAIL')
        self.fix()
        checked = self.w.check()
        self.assertEqual((checked['status'], checked['executed_tests'], checked['fail_to_pass']), ('PASS', 2, True))
        self.assertEqual(self.w.diff()['substantive_python'], ['sample.py'])

    def test_probe_reads_new_source_despite_valid_old_bytecode_cache(self):
        import os, py_compile
        file = self.root / 'sample.py'
        file.write_text('def render():\n    return 1\n')
        before = file.stat()
        py_compile.compile(str(file), doraise=True)
        self.w.read('sample.py')
        self.w.edit('sample.py', 2, 2, '    return 2\n')
        os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))
        result = self.w.check(dict(op='probe', code='import sample\nassert sample.render() == 2'))
        self.assertEqual(result['status'], 'PASS', result)

    def test_docstring_change_is_not_reported_as_behavior_change(self):
        self.w.read('sample.py')
        self.w.edit('sample.py', 1, 1, 'def render(disabled=False):\n    """An invented field is only documented."""\n')
        self.assertEqual(self.w.diff()['substantive_python'], [])
        self.assertEqual(self.w.check(dict(op='probe', code=PROBE))['status'], 'FAIL')

    def test_partial_source_fix_survives_bad_check_and_finalization(self):
        self.fix()
        self.w.check(dict(op='test', path='missing.py'))
        self.assertIn('return "" if disabled', self.w.diff()['patch'])
        self.assertEqual(self.w.diff()['checks'][-1]['status'], 'BLOCKED')

    def test_only_visible_lines_are_authorized_and_attempts_bounded(self):
        self.w.read('sample.py', 1, 1)
        with self.assertRaisesRegex(ValueError, 'range'):
            self.w.edit('sample.py', 2, 2, '    return ""')
        self.w.state['attempts'] = work.MAX_EDITS
        with self.assertRaisesRegex(ValueError, 'limit'):
            self.fix()
        self.assertEqual((self.root / 'sample.py').read_text(), BASE)

    def test_paths_tests_setup_and_symlinks_are_protected(self):
        for name in ['../outside.py', '/tmp/outside.py', '.git/config']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.w.read(name)
        for name in ['tests/test_sample.py', 'setup.py', 'new.py']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.w.edit(name, 1, 1, 'pass\n')
        (self.root / 'link.py').symlink_to(self.root / 'sample.py')
        with self.assertRaises(ValueError):
            self.w.read('link.py')

    def test_interrupted_edit_journal_recovers_on_next_operation(self):
        (self.w.store / 'before.bin').write_bytes(BASE.encode())
        (self.w.store / 'pending.json').write_text(json.dumps({'path': 'sample.py'}))
        (self.root / 'sample.py').write_text('def render():\n    return "bad"\n')
        recovered = work.Workbench(self.root)
        self.assertEqual((self.root / 'sample.py').read_text(), BASE)
        self.assertFalse((recovered.store / 'pending.json').exists())

    def test_assertion_can_use_saved_repository_result(self):
        self.assertEqual(self.w.check(dict(
            op='probe', code='import sample\\nactual = sample.render(True)\\nassert actual == ""'
        ))['status'], 'FAIL')
        self.fix()
        result = self.w.check()
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['assertion_observes_repository'])
        self.assertTrue(result['fail_to_pass'])

    def test_untracked_python_edits_are_rejected_before_mutation(self):
        path = self.root / 'untracked.py'
        path.write_text('def answer():\\n    return 1\\n')
        before = path.read_bytes()
        self.w.read('untracked.py')
        with self.assertRaisesRegex(ValueError, 'Untracked files'):
            self.w.edit('untracked.py', 2, 2, '    return 2\\n')
        self.assertEqual(path.read_bytes(), before)
        self.assertNotIn('untracked.py', self.w.diff()['changed'])

    def test_timeout_is_blocked_and_preserves_patch(self):
        self.fix()
        original = work.execute
        def short(argv, root, **kw):
            return original(argv, root, timeout=0.2, **kw)
        with patch.object(work, 'execute', short):
            result = self.w.check(dict(op='probe', code='import sample, time\nassert sample.render(True) == ""\ntime.sleep(5)'))
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertTrue(result['timed_out'])
        self.assertIn('return "" if disabled', self.w.diff()['patch'])

    def test_probe_cannot_accidentally_edit_source_or_tests(self):
        self.fix()
        retained = (self.root / 'sample.py').read_bytes()
        test_before = (self.root / 'tests/test_sample.py').read_bytes()
        result = self.w.check(dict(op='probe', code='''import sample
from pathlib import Path
assert sample.render(True) == ""
Path('sample.py').write_text('broken')
Path('tests/test_sample.py').write_text('pass')
Path('invented.py').write_text('pass')
'''))
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertEqual((self.root / 'sample.py').read_bytes(), retained)
        self.assertEqual((self.root / 'tests/test_sample.py').read_bytes(), test_before)
        self.assertFalse((self.root / 'invented.py').exists())

    def test_escaped_read_output_only_authorizes_visible_lines(self):
        (self.root / 'sample.py').write_text(''.join('# ' + '\\' * 120 + '\n' for _ in range(100)))
        result = self.w.read('sample.py', 1, 100)
        self.assertLess(len(json.dumps(result)), 4000)
        self.assertLess(result['end'], 100)
        with self.assertRaisesRegex(ValueError, 'range'):
            self.w.edit('sample.py', result['end'] + 1, result['end'] + 1, '# unseen\n')

    def test_search_returns_real_paths_and_script_defaults_to_status(self):
        result = self.w.search('render')
        self.assertTrue(result['source_matches'][0].startswith('sample.py:'))
        self.assertTrue(result['test_matches'][0].startswith('tests/test_sample.py:'))
        import os, sys
        result = subprocess.run([sys.executable, str(SOURCE / 'skills/repair/scripts/work.py')],
                                cwd=self.root, env=dict(os.environ, PYTHONPATH=str(self.root)),
                                text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'OK')

    def test_search_prioritizes_implementation_over_docs_and_examples(self):
        for name in ['docs/aaa.md', 'examples/aaa.py']:
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('# render documentation\n' * 20)
        work.git(self.root, 'add', 'docs', 'examples')
        result = self.w.search('render')
        self.assertTrue(result['source_matches'][0].startswith('sample.py:'), result)
        self.assertTrue(result['test_matches'][0].startswith('tests/test_sample.py:'), result)


class PackageTests(unittest.TestCase):
    def test_deterministic_allowlisted_artifact(self):
        first = build()
        self.assertEqual(first, build())
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertEqual(set(z.namelist()), set(FILES))
            self.assertFalse(any('adapter' in n or 'tasks.jsonl' in n for n in z.namelist()))

    def test_s24_result_does_not_qualify_as_s25_improvement(self):
        from compare_submission24 import S1_IDS
        from compare_submission25 import compare
        def rows(wins):
            return [dict(instance_id=n, resolved=n in wins, agent_patch='synthetic patch', error=None) for n in S1_IDS]
        result = compare(rows({'rich_3905', 'rich_4077'}), rows({'rich_3518', 'rich_3454'}),
                         rows({'rich_4077'}), rows({'rich_4077'}))
        self.assertFalse(result['meets_single_run_target'])
        self.assertEqual(result['paired']['s24']['net_gain'], 0)
        candidate = rows(set(S1_IDS[:3]) | {'rich_4077', 'rich_3518', 'rich_3454'})
        result = compare(rows({'rich_3905', 'rich_4077'}), rows({'rich_3518', 'rich_3454'}), rows({'rich_4077'}), candidate)
        self.assertTrue(result['meets_single_run_target'])
        with self.assertRaises(ValueError):
            compare(rows(set()), rows(set()), rows(set()), candidate[:-1])


class EvaluationWrapperTests(unittest.TestCase):
    def test_s25_archive_is_passed_explicitly_to_shared_runner(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from run_submission25_eval import evaluate
        report = dict(stage='evaluation_complete', runs=[dict(
            resolved=3, solved_ids=['rich_4077', 'rich_3518', 'rich_3454'], empty_patches=0, agent_error_ids=[])])
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(archive=None, stage='pilot', output=Path(folder))
            with patch('run_submission25_eval.evaluate_official', new=AsyncMock(return_value=report)) as run:
                result = asyncio.run(evaluate(args))
            self.assertEqual(run.call_args.args[0].archive, ARCHIVE)
            self.assertTrue(result['ready_for_full_s1'])
            self.assertIn('compare_submission25.py', result['note'])

    def test_pilot_requires_every_repeat_and_preserves_baseline_block(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from run_submission25_eval import evaluate
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(archive=ARCHIVE, stage='pilot', output=Path(folder))
            for report in [dict(stage='baseline_blocked', blocked_ids=['x'], real_inference_runs=0),
                           dict(stage='evaluation_complete', runs=[
                               dict(resolved=5, solved_ids=['rich_4077'], empty_patches=0, agent_error_ids=[]),
                               dict(resolved=1, solved_ids=['rich_4077'], empty_patches=0, agent_error_ids=[])])]:
                with patch('run_submission25_eval.evaluate_official', new=AsyncMock(return_value=report)):
                    result = asyncio.run(evaluate(args))
                if result['stage'] == 'baseline_blocked':
                    self.assertNotIn('ready_for_full_s1', result)
                else:
                    self.assertFalse(result['ready_for_full_s1'])


if __name__ == '__main__':
    unittest.main()
