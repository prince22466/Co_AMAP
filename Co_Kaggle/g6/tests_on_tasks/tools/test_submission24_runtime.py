"""Official-runtime tests: deliberately require the competition wheels, no silent skips."""
import asyncio
import tempfile
import unittest
from pathlib import Path
import zipfile

from build_submission24 import ARCHIVE, MODEL, build
from smoke_submission24 import leaves, run_smoke


class Runtime(unittest.TestCase):
    def test_official_result_with_session_trace_serializes_without_losing_failure(self):
        import json
        from adk_eval_core.tracing.trace import SessionTrace
        from swegemma.models import TaskResult
        from run_submission24_eval import serialize_result
        trace = SessionTrace()
        trace.record_custom('task_prompt', 'Synthetic diagnostic event', author='harness')
        result = TaskResult(task_id='synthetic', resolved=False, agent_patch='',
                            test_exit_code=1, test_output='AssertionError', trace=trace)
        row = json.loads(json.dumps(serialize_result(result)))
        self.assertIs(row['resolved'], False)
        self.assertEqual(row['test_exit_code'], 1)
        self.assertTrue(row['trace']['steps'])

    def test_real_tools_and_skill_execute_repair_review_loop(self):
        result = asyncio.run(run_smoke())
        self.assertTrue(result['passed'], result)

    def test_plain_text_locator_does_not_end_entire_pipeline(self):
        result = asyncio.run(run_smoke(malformed_locator=True))
        self.assertTrue(result['passed'], result)

    def test_empty_patch_is_detected_and_reported_without_fabricated_success(self):
        result = asyncio.run(run_smoke(empty=True))
        self.assertTrue(result['passed'], result)

    def test_pinned_compiler_routes_thinking_budget_and_limits_submission_tools(self):
        from adk_submission import ModelRegistry, ToolRegistry, compile_submission
        from google.adk.models.lite_llm import LiteLlm
        from swegemma.config import build_submission_limits
        from swegemma.context import SwegemmaContext
        from swegemma.tools import create_tools
        registry = ToolRegistry()
        for name, fn in create_tools(SwegemmaContext()).items():
            registry.register(name, fn)
        models = ModelRegistry()
        models.register(MODEL, LiteLlm(model='openai/local', api_base='http://localhost:8000/v1', api_key='EMPTY'))
        limits, constraints = build_submission_limits()
        build()
        with tempfile.TemporaryDirectory() as folder:
            with zipfile.ZipFile(ARCHIVE) as z: z.extractall(folder)
            agent = compile_submission(Path(folder), registry, models, limits=limits, generation_constraints=constraints)
            self.assertEqual(agent.sub_agents[1].max_iterations, 2)
            for leaf in leaves(agent):
                self.assertTrue(leaf.disallow_transfer_to_parent)
                self.assertTrue(leaf.disallow_transfer_to_peers)
                self.assertEqual(leaf.include_contents, 'none')
                extra = leaf.model._additional_args['extra_body']
                self.assertEqual(extra['chat_template_kwargs']['enable_thinking'], leaf.name != 'finalizer')
                if leaf.name != 'finalizer':
                    self.assertGreater(extra['thinking_token_budget'], 0)
                    self.assertNotIn('submit_patch', [getattr(t, '__name__', '') for t in leaf.tools])
                else:
                    self.assertNotIn('thinking_token_budget', extra)


if __name__ == '__main__':
    unittest.main()
