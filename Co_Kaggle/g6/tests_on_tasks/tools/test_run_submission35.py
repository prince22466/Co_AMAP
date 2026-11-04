"""S35 local runner checks; no real model server required."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import run_submission24_eval as official
from build_submission35 import MODEL, archive_path, build_variant
from gate_submission35 import PILOT_IDS
from run_submission35_eval import evaluate, verified_archive


class S35RunnerTests(unittest.TestCase):
    def test_archive_is_exact_and_variant_selection_cannot_swap(self):
        for variant in ("primary", "control"):
            build_variant(variant)
            self.assertEqual(verified_archive(variant, None),
                             archive_path(variant))
        with self.assertRaisesRegex(ValueError, "Archive mismatch"):
            verified_archive("primary", archive_path("control"))
        with self.assertRaisesRegex(ValueError, "Unknown S35"):
            verified_archive("not-a-variant", None)

    def test_official_e4b_model_and_pilot_scope(self):
        build_variant("primary")
        old_model, old_pilot = official.MODEL, official.PILOT_IDS
        self.assertEqual(old_model, MODEL)  # Both runners use the S34 model.

        async def fake_official(args):
            self.assertEqual(official.MODEL, MODEL)
            self.assertEqual(official.PILOT_IDS, PILOT_IDS)
            self.assertEqual(args.archive, archive_path("primary"))
            return {"stage": "evaluation_complete", "runs": []}

        with tempfile.TemporaryDirectory() as temp, patch.object(
                official, "evaluate", side_effect=fake_official):
            args = SimpleNamespace(
                variant="primary", archive=None, output=Path(temp),
            )
            result = asyncio.run(evaluate(args))
            self.assertEqual(result["model"], MODEL)
            self.assertFalse(result["promotion_approved"])
            self.assertTrue((Path(temp) / "submission35_summary.json").is_file())
        self.assertEqual(official.MODEL, old_model)
        self.assertEqual(official.PILOT_IDS, old_pilot)

    def test_restore_pilot_after_official_evaluator_error(self):
        build_variant("control")
        old_model, old_pilot = official.MODEL, official.PILOT_IDS

        async def broken(_args):
            raise RuntimeError("synthetic evaluator fault")

        with tempfile.TemporaryDirectory() as temp, patch.object(
                official, "evaluate", side_effect=broken):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                asyncio.run(evaluate(SimpleNamespace(
                    variant="control", archive=None, output=Path(temp))))
        self.assertEqual(official.MODEL, old_model)
        self.assertEqual(official.PILOT_IDS, old_pilot)


if __name__ == "__main__":
    unittest.main()
