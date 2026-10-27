"""Optional pinned ADK compiler smoke for S22; requires official runtime packages.

This uses the S21 synthetic scripted agent smoke on the *packaged S22 files*.
It does not run real E4B inference or establish a Kaggle benchmark score.
"""
import asyncio
from pathlib import Path
import tempfile
import unittest
import zipfile
from build_submission22 import build, ARCHIVE, MODEL
from smoke_submission21 import run_smoke

class OfficialRuntime(unittest.TestCase):
    def test_s22_archive_compiles_and_executes_native_repair_and_audit(self):
        build()
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'package'
            with zipfile.ZipFile(ARCHIVE) as z:
                z.extractall(folder)
            result = asyncio.run(run_smoke(Path(d) / 'output', submission_dir=folder))
            self.assertTrue(result['passed'], result)

    def test_tool_registration_model_and_disabled_thinking(self):
        from adk_submission import ModelRegistry, ToolRegistry, compile_submission
        from google.adk.models.lite_llm import LiteLlm
        from swegemma.tools import create_tools
        from swegemma.context import SwegemmaContext
        models = ModelRegistry()
        models.register(MODEL, LiteLlm(model='openai/local',
                        api_base='http://localhost:8000/v1', api_key='EMPTY'))
        registry = ToolRegistry()
        for name, fn in create_tools(SwegemmaContext()).items():
            registry.register(name, fn)
        build()
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(ARCHIVE) as z:
                z.extractall(d)
            agent = compile_submission(Path(d), registry, models)
            self.assertFalse(agent.sub_agents)
            self.assertFalse(agent.model._additional_args['extra_body']
                             ['chat_template_kwargs']['enable_thinking'])
            self.assertIn('edit_file', [getattr(t, '__name__', '') for t in agent.tools])

if __name__ == '__main__':
    unittest.main()
