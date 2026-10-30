"""S26 static regression tests. Model-emitted tool calls require GPU pilot."""
import hashlib
import unittest
import zipfile

import yaml
from build_submission26 import (ARCHIVE, EXPECTED_TOOLS, FILES, SOURCE,
                                 IncludeLoader, MODEL, build)


class Submission26PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build()

    def test_zip_is_minimal_safe_and_reproducible(self):
        first = ARCHIVE.read_bytes()
        self.assertEqual(build()['archive_sha256'], hashlib.sha256(first).hexdigest())
        self.assertEqual(ARCHIVE.read_bytes(), first)
        with zipfile.ZipFile(ARCHIVE) as archive:
            self.assertEqual(sorted(archive.namelist()), sorted(FILES))
            self.assertEqual(archive.testzip(), None)
            for path in FILES:
                self.assertEqual(archive.read(path), (SOURCE / path).read_bytes())
            self.assertFalse(any('tasks.jsonl' in p or 'adapter' in p for p in archive.namelist()))

    def test_native_tools_remain_direct(self):
        with zipfile.ZipFile(ARCHIVE) as archive:
            cfg = yaml.load(archive.read('agent.yaml'), Loader=IncludeLoader)
        self.assertEqual(cfg['model'], MODEL)
        self.assertEqual(tuple(cfg['tools']), EXPECTED_TOOLS)
        self.assertEqual(cfg['include_contents'], 'default')
        self.assertNotIn('skills', cfg)
        self.assertNotIn('sub_agents', cfg)
        self.assertEqual(cfg['generate_content_config']['thinking_config']['thinking_budget'], 512)

    def test_instruction_addresses_recurring_failures(self):
        text = (SOURCE / 'instruction.md').read_text()
        for criterion in ('old_string', 'new_string', 'head -n 35',
                          'git diff --check', 'submit_patch', 'UNVERIFIED',
                          '.toml', 'run_skill_script', 'Never make three identical'):
            with self.subTest(criterion=criterion):
                self.assertIn(criterion, text)
        self.assertNotIn('file_path="scripts/work.py"', text)
        self.assertLess(len(text), 3500)

    def test_pilot_gate_never_equates_nonempty_patch_with_resolution(self):
        from run_submission26_eval import objective_pilot_pass
        good = dict(resolved=3, solved_ids=['rich_4077', 'rich_3518', 'rich_3454'],
                    empty_patches=1, agent_error_ids=[])
        self.assertTrue(objective_pilot_pass(good))
        for change in (dict(resolved=2), dict(solved_ids=['rich_3518']),
                       dict(empty_patches=2), dict(agent_error_ids=['rich_3882'])):
            with self.subTest(change=change):
                self.assertFalse(objective_pilot_pass(dict(good, **change)))

    def test_budget_and_scope(self):
        data = yaml.safe_load((SOURCE / 'eval_config.yaml').read_text())
        self.assertEqual(data['evaluation']['max_tool_calls'], 40)
        self.assertEqual(data['evaluation']['max_turns'], 48)
        self.assertEqual(data['evaluation']['max_time_minutes'], 4)
        self.assertIn('Unscored', (SOURCE / 'README.md').read_text())


if __name__ == '__main__':
    unittest.main()
