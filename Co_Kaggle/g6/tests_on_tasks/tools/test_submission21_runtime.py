"""Pinned official runtime tests; scripted model, not E4B inference."""
import asyncio
from pathlib import Path
import tempfile
import unittest
import zipfile
from build_submission21 import build, ARCHIVE, MODEL
from smoke_submission21 import run_smoke

class RuntimeTests(unittest.TestCase):
    def test_packaged_audit_detects_bad_edit_then_repair_is_checked_and_submitted(self):
        build()
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(ARCHIVE) as z:z.extractall(Path(d)/'package')
            result=asyncio.run(run_smoke(Path(d)/'output',submission_dir=Path(d)/'package'))
            self.assertTrue(result['passed'],result)
            self.assertEqual(result['tool_calls_used'],9)
    def test_official_compiler_thinking_and_native_tools(self):
        from adk_submission import ModelRegistry,ToolRegistry,compile_submission
        from google.adk.models.lite_llm import LiteLlm
        from swegemma.tools import create_tools
        from swegemma.context import SwegemmaContext
        build();models=ModelRegistry();models.register(MODEL,LiteLlm(model='openai/local',api_base='http://localhost:8000/v1',api_key='EMPTY'))
        registry=ToolRegistry()
        for name,fn in create_tools(SwegemmaContext()).items():registry.register(name,fn)
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(ARCHIVE) as z:z.extractall(d)
            agent=compile_submission(Path(d),registry,models)
            self.assertFalse(agent.sub_agents)
            self.assertFalse(agent.model._additional_args['extra_body']['chat_template_kwargs']['enable_thinking'])
            self.assertIn('edit_file',[getattr(t,'__name__','') for t in agent.tools])

if __name__=='__main__':unittest.main()
