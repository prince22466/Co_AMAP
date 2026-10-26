"""Run explicitly with the competition wheelhouse; actual tools, not model inference."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from build_submission18 import build, ARCHIVE, MODEL, SOURCE
from smoke_submission18 import run_smoke

class NativeRegressions(unittest.TestCase):
    def setUp(self):
        from swegemma.sandbox.subprocess import SubprocessManager
        from swegemma.context import SwegemmaContext
        from swegemma.tools import create_tools
        self.temp=tempfile.TemporaryDirectory()
        self.sandbox=SubprocessManager(base_dir=Path(self.temp.name))
        self.sandbox.initialize({'scripts/add_latest_release_date.py':'VALUE = 41\n', '.gitignore':'__pycache__/\n'})
        self.sandbox.run_command('git init -q && git add . && git -c user.name=Test -c user.email=test@example.invalid commit -qm baseline')
        self.ctx=SwegemmaContext(docker_manager=self.sandbox,container_id=next(iter(self.sandbox.sandboxes)),
            repo='synthetic/regression',max_tool_calls=40,max_time_minutes=4)
        self.tools=create_tools(self.ctx)
    def tearDown(self):self.sandbox.close();self.temp.cleanup()

    def test_repository_path_is_read_without_skill_wrapper(self):
        r=self.tools['read_file'](filepath='scripts/add_latest_release_date.py')
        self.assertEqual(r['status'],'ok',r)
        self.assertIn('41',str(r))

    def test_edit_does_not_require_module_or_assertion(self):
        r=self.tools['edit_file'](filepath='scripts/add_latest_release_date.py',old_string='41',new_string='42')
        self.assertEqual(r['status'],'ok',r)
        self.assertIn('42',self.sandbox.read_file('scripts/add_latest_release_date.py'))
        submitted=self.tools['submit_patch']();self.assertGreater(submitted['patch_size'],0)

    def test_failed_reads_count_and_reach_hard_native_budget(self):
        from dataclasses import replace
        self.ctx.budget=replace(self.ctx.budget,tool_calls=3)
        for _ in range(3):
            r=self.tools['read_file'](filepath='missing.py');self.assertEqual(r['status'],'error')
        self.assertEqual(self.ctx.tool_calls_used,3)
        r=self.tools['read_file'](filepath='scripts/add_latest_release_date.py')
        self.assertEqual(r['status'],'error',r)

    def test_failed_exact_match_does_not_destroy_candidate(self):
        r=self.tools['edit_file'](filepath='scripts/add_latest_release_date.py',old_string='absent',new_string='42')
        self.assertEqual(r['status'],'error')
        self.assertEqual(self.sandbox.read_file('scripts/add_latest_release_date.py'),'VALUE = 41\n')

    def test_compiler_disables_thinking_without_skills_or_handoffs(self):
        from adk_submission import ModelRegistry, ToolRegistry, compile_submission
        from google.adk.models.lite_llm import LiteLlm
        models=ModelRegistry();models.register(MODEL,LiteLlm(model='openai/local',api_base='http://localhost:8000/v1',api_key='EMPTY'))
        registry=ToolRegistry()
        for name,fn in self.tools.items():registry.register(name,fn)
        agent=compile_submission(SOURCE,registry,models)
        self.assertFalse(agent.sub_agents)
        self.assertEqual(len(agent.tools),6)
        self.assertFalse(agent.model._additional_args['extra_body']['chat_template_kwargs']['enable_thinking'])

    def test_packaged_runner_failure_repair_verification_submission(self):
        build()
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(ARCHIVE) as z:z.extractall(Path(d)/'package')
            r=asyncio.run(run_smoke(Path(d)/'evidence',submission_dir=Path(d)/'package'))
            self.assertTrue(r['passed'],r)
            self.assertEqual(r['tool_calls_used'],7)

if __name__=='__main__':unittest.main()
