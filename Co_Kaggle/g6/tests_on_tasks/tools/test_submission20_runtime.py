"""Pinned official runtime integration, with scripted fault injection, not E4B inference."""
import asyncio
from pathlib import Path
import tempfile
import unittest
import zipfile
from build_submission20 import build, ARCHIVE, MODEL
from smoke_submission20 import run_smoke

class RuntimeTests(unittest.TestCase):
    def test_packaged_gate_enforces_publication_and_recovers_wrapper_error(self):
        build()
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(ARCHIVE) as z:z.extractall(Path(d)/'package')
            result=asyncio.run(run_smoke(Path(d)/'output',submission_dir=Path(d)/'package'))
            self.assertTrue(result['passed'],result)
    def test_gate_covers_official_static_protected_paths(self):
        from swegemma.harness.verification import _is_protected_test_or_config_path
        from test_submission20 import gate
        names=['conftest.py','pytest.ini','pyproject.toml','tox.ini','setup.cfg',
               '.pytest.ini','sitecustomize.py','usercustomize.py','_swegemma_stubs.py',
               'x.pth','test_api.py','api_test.py','Tests/helper.py','TEST/helper.py','testing/helper.py']
        for name in names:
            self.assertTrue(_is_protected_test_or_config_path(name),name)
            self.assertTrue(gate.protected(name),name)

    def test_compiler_has_no_mutation_bypass_or_finalizer(self):
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
            names={getattr(t,'__name__','') for t in agent.tools}
            self.assertTrue({'get_status','read_file','submit_patch'} <= names)
            self.assertFalse({'edit_file','write_file','run_command'} & names)

if __name__=='__main__':unittest.main()
