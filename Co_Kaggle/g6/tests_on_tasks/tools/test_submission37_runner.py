"""S37 research runner integration tests, no model/API/ADK wheels required."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import run_submission37_eval as runner
from compare_submission37 import compare, load_experiment
from compare_submission24 import PILOT_IDS
from submission37_repetition_guard import RepetitionGuard


def fake_experiment(variant: str, resolved: list[str]) -> dict:
    manifest = {
        "task_ids": ["rich_3905"], "archive_sha256": "exactzip",
        "tasks_sha256": "tasks", "model_alias": runner.MODEL,
        "served_model": runner.MODEL, "sandbox": "subprocess",
        "context_cache": {}, "compaction": {}, "runtime_versions": {},
        "effective_limits": {"max_tool_calls": 40}, "repeats": 1,
    }
    return {
        "manifest": manifest,
        "telemetry": {
            "variant": variant, "guard_installed": variant == "guard",
            "guard_counts": {
                "edit_attempts": 2,
                "duplicate_failures_blocked": (1 if variant == "guard" else 0),
            },
        },
        "candidates": {
            "candidate_1": {
                "resolved": resolved, "empty_patch_ids": [],
                "agent_error_ids": [], "duration_seconds": 5.0,
            },
        },
    }


class ComparatorTests(unittest.TestCase):
    def test_same_archives_count_real_wins_not_just_blocked_calls(self):
        a, b = fake_experiment("control", []), fake_experiment("guard", ["rich_3905"])
        report = compare(a, b)
        self.assertEqual(report["runs"][0]["delta_resolved"], 1)
        self.assertEqual(report["runs"][0]["gained_ids"], ["rich_3905"])
        self.assertTrue(report["intervention_confirmed"])
        self.assertFalse(report["promotion_approved"])

    def test_intervention_without_gain_not_promotion(self):
        report = compare(
            fake_experiment("control", ["rich_3905"]),
            fake_experiment("guard", ["rich_3905"]))
        self.assertEqual(report["runs"][0]["delta_resolved"], 0)
        self.assertFalse(report["promotion_approved"])

    def test_mismatched_package_or_runtime_is_rejected(self):
        a, b = fake_experiment("control", []), fake_experiment("guard", [])
        b["manifest"]["archive_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "matched"):
            compare(a, b)
        b["manifest"]["archive_sha256"] = "exactzip"
        b["manifest"]["effective_limits"]["max_tool_calls"] = 50
        with self.assertRaisesRegex(ValueError, "matched"):
            compare(a, b)


    def test_persisted_official_results_require_complete_task_and_hook_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in ("control", "guard"):
                folder = root / variant
                candidate = folder / "candidate_1"
                candidate.mkdir(parents=True)
                manifest = {
                    **fake_experiment(variant, [])["manifest"],
                    "task_ids": list(PILOT_IDS), "repeats": 1,
                }
                (folder / "run_manifest.json").write_text(json.dumps(manifest))
                telemetry = {
                    "variant": variant, "guard_installed": variant == "guard",
                    "archive_sha256": "exactzip", "research_only": True,
                    "app_instances": len(PILOT_IDS) if variant == "guard" else 0,
                    "expected_app_instances": len(PILOT_IDS),
                    "guard_counts": {"callback_tool_calls": 8,
                                     "missing_session": 0},
                }
                (folder / "submission37_research_telemetry.json").write_text(
                    json.dumps(telemetry))
                rows = [
                    {"instance_id": name, "resolved": False,
                     "agent_patch": "", "duration_seconds": 1.0}
                    for name in PILOT_IDS
                ]
                datafile = candidate / "notebook_task_results.jsonl"
                datafile.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))
                self.assertEqual(
                    len(load_experiment(folder, variant, "pilot")["candidates"]), 1)
                if variant == "guard":
                    telemetry["app_instances"] -= 1
                    (folder / "submission37_research_telemetry.json").write_text(
                        json.dumps(telemetry))
                    with self.assertRaisesRegex(ValueError, "coverage"):
                        load_experiment(folder, variant, "pilot")
                    telemetry["app_instances"] += 1
                    (folder / "submission37_research_telemetry.json").write_text(
                        json.dumps(telemetry))
                datafile.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows[:-1]))
                with self.assertRaises(ValueError):
                    load_experiment(folder, variant, "pilot")
                # Returning zero for a missing duration would be misleading.
                for invalid_duration in (None, -1, float("nan"), float("inf")):
                    invalid_rows = [dict(row) for row in rows]
                    invalid_rows[0]["duration_seconds"] = invalid_duration
                    datafile.write_text(
                        "".join(json.dumps(row) + "\n" for row in invalid_rows))
                    with self.subTest(invalid_duration=invalid_duration):
                        with self.assertRaisesRegex(ValueError, "duration"):
                            load_experiment(folder, variant, "pilot")


class RunnerTests(unittest.TestCase):
    def test_exact_s36_control_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "s36.zip"
            archive.write_bytes(b"test")
            async def run_fake(args):
                self.assertEqual(args.archive, archive)
                args.output.mkdir()
                return {"stage": "evaluation_complete", "runs": []}
            with patch.object(runner, "canonical_s36_archive",
                              return_value=archive), patch.object(
                                  runner.official, "evaluate", side_effect=run_fake):
                args = SimpleNamespace(
                    variant="control", archive=None,
                    output=Path(tmp) / "control", stage="pilot", repeats=1)
                result = asyncio.run(runner.evaluate(args))
            self.assertFalse(result["guard_installed"])
            telemetry = json.loads((
                args.output / "submission37_research_telemetry.json").read_text())
            self.assertEqual(
                telemetry["archive_sha256"],
                hashlib.sha256(b"test").hexdigest())
            self.assertEqual(telemetry["app_instances"], 0)

    def test_guard_runner_injects_tool_callback_before_native_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "s36.zip"
            archive.write_bytes(b"test")
            parent = ModuleType("google")
            adk = ModuleType("google.adk")
            apps = ModuleType("google.adk.apps")
            plugins = ModuleType("google.adk.plugins")
            base = ModuleType("google.adk.plugins.base_plugin")

            class FakeBasePlugin:
                def __init__(self, name):
                    self.name = name

                async def before_tool_callback(
                    self, *, tool, tool_args, tool_context
                ):
                    return None

                async def after_tool_callback(
                    self, *, tool, tool_args, tool_context, result
                ):
                    return None

            class FakeApp:
                def __init__(self, **kwargs):
                    self.plugins = kwargs["plugins"]

            apps.App = FakeApp
            base.BasePlugin = FakeBasePlugin

            async def run_fake(args):
                args.output.mkdir()
                app = FakeApp(plugins=[], name="swegemma_eval")
                self.assertEqual(len(app.plugins), 1)
                hook = app.plugins[0]
                context = SimpleNamespace(
                    _invocation_context=SimpleNamespace(
                        session=SimpleNamespace(id="task-1")))
                tool = SimpleNamespace(name="edit_file")
                edit = {"filepath": "a.py", "old_string": "x", "new_string": "y"}
                self.assertIsNone(await hook.before_tool_callback(
                    tool=tool, tool_args=edit, tool_context=context))
                await hook.after_tool_callback(
                    tool=tool, tool_args=edit, tool_context=context,
                    result={"status": "error",
                            "error_type": "FileEditError",
                            "error_message": "Failed to replace: old_string not found"})
                denial = await hook.before_tool_callback(
                    tool=tool, tool_args=edit, tool_context=context)
                self.assertEqual(denial["error_type"], "RepeatedFailedEdit")
                return {"stage": "evaluation_complete", "runs": []}

            overrides = {
                "google": parent,
                "google.adk": adk,
                "google.adk.apps": apps,
                "google.adk.plugins": plugins,
                "google.adk.plugins.base_plugin": base,
            }
            with patch.dict(sys.modules, overrides), patch.object(
                    runner, "canonical_s36_archive", return_value=archive), patch.object(
                    runner.official, "PILOT_IDS", ("rich_3905",)), patch.object(
                    runner.official, "evaluate", side_effect=run_fake):
                args = SimpleNamespace(
                    variant="guard", archive=None, output=Path(tmp) / "guard",
                    stage="pilot", repeats=1)
                result = asyncio.run(runner.evaluate(args))
            self.assertTrue(result["guard_installed"])
            self.assertEqual(result["guard_counts"]["duplicate_failures_blocked"], 1)
            telemetry = json.loads((
                args.output / "submission37_research_telemetry.json").read_text())
            self.assertEqual(telemetry["app_instances"], 1)
            self.assertEqual(telemetry["expected_app_instances"], 1)

    def test_unknown_variant_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Expected"):
            asyncio.run(runner.evaluate(SimpleNamespace(variant="invalid")))


if __name__ == "__main__":
    unittest.main()
