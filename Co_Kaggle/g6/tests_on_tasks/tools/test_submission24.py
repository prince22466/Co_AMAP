"""CPU packaging/comparison regressions. These do not run E4B."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from build_submission24 import ARCHIVE, FILES, SOURCE, build
from compare_submission24 import S1_IDS, compare, normalize
from run_submission24_eval import baseline_ready, extract_package, nonnegative_minutes


class Packaging(unittest.TestCase):
    def test_exact_deterministic_archive_and_no_adapters_or_task_data(self):
        first = build()
        self.assertEqual(first, build())
        with zipfile.ZipFile(ARCHIVE) as z:
            self.assertEqual(set(z.namelist()), set(FILES))
            for name in z.namelist():
                self.assertNotIn('adapters/', name)
                data = z.read(name).decode()
                for task_id in S1_IDS:
                    self.assertNotIn(task_id, data)

    def test_audit_checks_real_invalid_empty_and_valid_patch(self):
        spec = importlib.util.spec_from_file_location('audit24', SOURCE / 'skills/patch-audit/scripts/audit.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'app.py'
            source.write_text('def f():\n    return 1\n')
            def git(*args):
                subprocess.run(['git', '-C', folder, *args], check=True, capture_output=True)
            git('init', '-q'); git('add', '.')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'base')
            self.assertIn('empty patch', module.audit(root)['blockers'])
            source.write_text('def f(:\n    return 1\n')
            self.assertEqual(module.audit(root)['status'], 'blocked')
            source.write_text('def f():\n    return 2\n')
            result = module.audit(root)
            self.assertEqual(result['status'], 'structurally_clean')
            self.assertFalse(result['behavior_verified'])
            self.assertEqual(source.read_text(), 'def f():\n    return 2\n')


def rows(solved=(), empty=()):
    return [{'instance_id': n, 'resolved': n in solved, 'agent_patch': '' if n in empty else 'patch'} for n in S1_IDS]


class Comparison(unittest.TestCase):
    def test_incomplete_duplicate_and_nonboolean_outcomes_rejected(self):
        with self.assertRaises(ValueError): normalize(rows()[:-1])
        with self.assertRaises(ValueError): normalize(rows() + rows()[:1])
        bad = rows(); bad[0]['resolved'] = 'true'
        with self.assertRaises(ValueError): normalize(bad)
        bad = rows(); del bad[0]['agent_patch']
        with self.assertRaises(ValueError): normalize(bad)
        with self.assertRaises(ValueError): normalize(rows(S1_IDS[:1], S1_IDS[:1]))

    def test_two_different_wins_is_not_improvement(self):
        report = compare(rows(S1_IDS[:2]), rows(S1_IDS[2:4]), rows(S1_IDS[4:6]))
        self.assertFalse(report['meets_single_run_target'])
        self.assertEqual(report['paired']['s21']['net_gain'], 0)
        self.assertEqual(len(report['paired']['s21']['lost']), 2)

    def test_predeclared_target_and_empty_patch_penalty(self):
        base1, base2 = rows(S1_IDS[:2]), rows(S1_IDS[2:4])
        self.assertTrue(compare(base1, base2, rows(S1_IDS[:6]))['meets_single_run_target'])
        self.assertFalse(compare(base1, base2, rows(S1_IDS[:6], S1_IDS[-3:]))['meets_single_run_target'])
        self.assertFalse(compare(base1, base2, rows(S1_IDS[4:10]))['meets_single_run_target'])


class EvaluationPreflight(unittest.TestCase):
    def test_nonfinite_time_limits_are_rejected(self):
        for value in ['nan', 'inf', '-inf', '-1']:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                nonnegative_minutes(value)
        self.assertEqual(nonnegative_minutes('0'), 0)
        self.assertEqual(nonnegative_minutes('4.5'), 4.5)

    def test_baseline_failure_is_distinguished_from_environment_error(self):
        healthy = dict(resolved=False, test_exit_code=1, test_output='FAILED test_behavior - AssertionError')
        self.assertTrue(baseline_ready(healthy))
        for output in ['ERROR at setup', 'ModuleNotFoundError: inline_snapshot',
                       'recursive dependency involving fixture httpbin', 'no tests ran', '']:
            self.assertFalse(baseline_ready(dict(healthy, test_output=output)))
        self.assertFalse(baseline_ready(dict(healthy, resolved=True, test_exit_code=0)))
        self.assertFalse(baseline_ready(dict(healthy, test_exit_code=2)))
        self.assertFalse(baseline_ready(dict(healthy, error_message='timeout')))

    def test_package_extraction_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with zipfile.ZipFile(root / 'bad.zip', 'w') as z:
                z.writestr('../outside.py', 'bad')
            with self.assertRaises(ValueError):
                extract_package(root / 'bad.zip', root / 'package')
            self.assertFalse((root / 'outside.py').exists())


if __name__ == '__main__':
    unittest.main()
