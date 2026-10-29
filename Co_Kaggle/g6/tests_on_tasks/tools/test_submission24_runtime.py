"""Official-runtime tests: deliberately require the competition wheels, no silent skips."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import zipfile

from build_submission24 import ARCHIVE, MODEL, build
from smoke_submission24 import leaves, run_smoke


class EvaluationRunner(unittest.TestCase):
    """Exercise runner orchestration with real configs/results and injected task outcomes."""

    def arguments(self, folder, **updates):
        from compare_submission24 import PILOT_IDS
        root = Path(folder)
        tasks = root / 'tasks.jsonl'
        tasks.write_text(''.join(json.dumps(dict(instance_id=n, repo='synthetic/fixture',
            base_commit='base', problem_statement='Synthetic runner test')) + '\n' for n in PILOT_IDS))
        build()
        args = SimpleNamespace(archive=ARCHIVE, tasks=tasks, snapshots=root / 'snapshots',
            wheels=None, served_model=MODEL, model_url='http://localhost:8000/v1',
            baseline_only=False, output=root / 'results', time_limit=0, stage='pilot',
            sandbox='subprocess', repeats=2)
        vars(args).update(updates)
        return args

    def test_matched_configs_authenticated_server_and_all_repeat_outcomes(self):
        from compare_submission24 import PILOT_IDS, load_rows, normalize
        from run_submission24_eval import evaluate
        from swegemma.models import TaskResult
        observed = []
        async def task_result(evaluator, task, **kwargs):
            observed.append((evaluator.config, kwargs))
            if not evaluator.config.skip_agent_patch and task.instance_id == PILOT_IDS[1]:
                raise RuntimeError('synthetic sandbox failure')
            baseline = evaluator.config.skip_agent_patch
            return TaskResult(instance_id=task.instance_id, resolved=not baseline,
                agent_patch='' if baseline else 'synthetic patch',
                test_exit_code=1 if baseline else 0,
                test_output='FAILED test_behavior - AssertionError' if baseline else '1 passed')

        with tempfile.TemporaryDirectory() as folder:
            args = self.arguments(folder)
            response = io.BytesIO(json.dumps({'data': [{'id': MODEL}]}).encode())
            with patch('swegemma.evaluate.Evaluator.evaluate_task', task_result), \
                 patch('run_submission24_eval.urlopen', return_value=response) as request, \
                 patch.dict('os.environ', {'LOCAL_API_KEY': 'synthetic-test-token'}), \
                 redirect_stdout(io.StringIO()):
                report = asyncio.run(evaluate(args))
            self.assertEqual(report['stage'], 'evaluation_complete')
            self.assertEqual(len(observed), len(PILOT_IDS) * 3)
            self.assertEqual(request.call_args.args[0].get_header('Authorization'),
                             'Bearer synthetic-test-token')
            for config, kwargs in observed:
                self.assertEqual(config.context_cache_config.model_dump(),
                    dict(min_tokens=2048, ttl_seconds=1800, cache_intervals=10))
                self.assertEqual(config.events_compaction_config.model_dump(exclude={'summarizer'}),
                    dict(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5))
                self.assertIsNone(config.budget.time_minutes)
                self.assertEqual((config.budget.tool_calls, config.budget.turns), (40, 48))
                self.assertEqual(kwargs['total_tasks'], len(PILOT_IDS))
                self.assertEqual(config.models.get(MODEL)._additional_args['api_key'], 'synthetic-test-token')
            for repeat, summary in enumerate(report['runs'], start=1):
                self.assertEqual(summary['resolved'], len(PILOT_IDS) - 1)
                self.assertEqual(summary['agent_error_ids'], [PILOT_IDS[1]])
                rows = normalize(load_rows(args.output / f'candidate_{repeat}'), PILOT_IDS)
                self.assertIn('synthetic sandbox failure', rows[PILOT_IDS[1]]['error_message'])
                self.assertIn('RuntimeError', rows[PILOT_IDS[1]]['exception_traceback'])
                config = observed[len(PILOT_IDS) * repeat][0]
                self.assertEqual(config.models.get(MODEL)._additional_args['seed'], repeat)
            manifest = (args.output / 'run_manifest.json').read_text()
            self.assertNotIn('synthetic-test-token', manifest)
            self.assertIn('context_cache', json.loads(manifest))
            self.assertIn('compaction', json.loads(manifest))

    def test_setup_failure_records_complete_baseline_and_blocks_candidates(self):
        from compare_submission24 import PILOT_IDS, load_rows
        from run_submission24_eval import evaluate
        from swegemma.harness.container_setup import ContainerSetupError
        from swegemma.models import TaskResult
        async def task_result(evaluator, task, **kwargs):
            self.assertTrue(evaluator.config.skip_agent_patch)
            if task.instance_id == PILOT_IDS[1]:
                raise ContainerSetupError('synthetic setup failure')
            return TaskResult(instance_id=task.instance_id, resolved=False, agent_patch='',
                test_exit_code=1, test_output='FAILED test_behavior - AssertionError')
        with tempfile.TemporaryDirectory() as folder:
            args = self.arguments(folder)
            response = io.BytesIO(json.dumps({'data': [{'id': MODEL}]}).encode())
            with patch('swegemma.evaluate.Evaluator.evaluate_task', task_result), \
                 patch('run_submission24_eval.urlopen', return_value=response), \
                 redirect_stdout(io.StringIO()):
                report = asyncio.run(evaluate(args))
            self.assertEqual(report['stage'], 'baseline_blocked')
            self.assertEqual(report['blocked_ids'], [PILOT_IDS[1]])
            self.assertEqual(len(load_rows(args.output / 'baseline')), len(PILOT_IDS))
            self.assertEqual(report['real_inference_runs'], 0)
            self.assertFalse((args.output / 'candidate_1').exists())
            self.assertEqual(json.loads((args.output / 'evaluation_summary.json').read_text()), report)

    def test_baseline_only_never_contacts_model_server(self):
        from compare_submission24 import PILOT_IDS, load_rows
        from run_submission24_eval import evaluate
        from swegemma.models import TaskResult
        configs = []
        async def task_result(evaluator, task, **kwargs):
            configs.append(evaluator.config)
            self.assertTrue(evaluator.config.skip_agent_patch)
            return TaskResult(instance_id=task.instance_id, resolved=False, agent_patch='',
                test_exit_code=1, test_output='FAILED test_behavior - AssertionError')
        with tempfile.TemporaryDirectory() as folder:
            args = self.arguments(folder, baseline_only=True, time_limit=4)
            with patch('swegemma.evaluate.Evaluator.evaluate_task', task_result), \
                 patch('run_submission24_eval.urlopen') as request, redirect_stdout(io.StringIO()):
                report = asyncio.run(evaluate(args))
            request.assert_not_called()
            self.assertEqual(report['stage'], 'baseline_ready')
            self.assertIsNotNone(configs[0].context_cache_config)
            self.assertIsNotNone(configs[0].events_compaction_config)
            self.assertEqual(len(load_rows(args.output / 'baseline')), len(PILOT_IDS))
            manifest = json.loads((args.output / 'run_manifest.json').read_text())
            self.assertEqual(manifest['effective_limits']['max_time_minutes'], 4)

    def test_cancellation_propagates_and_closes_sandbox(self):
        from run_submission24_eval import evaluate
        async def cancelled(*args, **kwargs):
            raise asyncio.CancelledError()
        with tempfile.TemporaryDirectory() as folder:
            args = self.arguments(folder, baseline_only=True)
            with patch('swegemma.evaluate.Evaluator.evaluate_task', cancelled), \
                 patch('swegemma.sandbox.subprocess.SubprocessManager.close') as close, \
                 self.assertRaises(asyncio.CancelledError):
                asyncio.run(evaluate(args))
            close.assert_called_once()


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
