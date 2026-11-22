"""P0 contract and scoring checks (the physical simulator is not needed)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from Co_Kaggle.g8.rl.p0.benchmark import (
    aggregate, default_data_root, discover_scenarios
)
from Co_Kaggle.g8.rl.p0.controllers import (
    LocalPolicy, VERSIONS, load_controller, validate_action
)
from Co_Kaggle.g8.rl.p0.metrics import score_one, weighted_losses


class ScoreTests(unittest.TestCase):
    def test_loss_decomposition(self):
        weights = {
            "productivity_weight": 0.6,
            "energy_cost_weight": 0.2,
            "carbon_emission_weight": 0.2,
        }
        loss = weighted_losses(-0.01, 0.02, 0.03, weights)
        self.assertAlmostEqual(loss["comfort"], 0.006)
        self.assertAlmostEqual(loss["energy"], 0.004)
        self.assertAlmostEqual(loss["carbon"], 0.006)
        result = score_one(
            [-0.016] * 3,
            {key: val * 3 for key, val in loss.items()},
            requested_steps=3,
            elapsed={"policy": 0.003, "simulator": 0.006, "diagnostics": 0.0015},
        )
        self.assertAlmostEqual(result["mean_loss"], 0.016)
        self.assertFalse(result["full_30_day_episode"])
        self.assertAlmostEqual(result["simulator_ms_per_step"], 2)

    def test_incomplete_episode_refused(self):
        with self.assertRaises(ValueError):
            score_one([-0.1], {"comfort": 0.1}, requested_steps=2,
                      elapsed={"policy": 0, "simulator": 0, "diagnostics": 0})

    def test_reward_mismatch_refused(self):
        with self.assertRaises(ValueError):
            score_one([-0.1], {"comfort": 0.2}, requested_steps=1,
                      elapsed={"policy": 0, "simulator": 0, "diagnostics": 0})

    def test_partial_not_labelled_leaderboard(self):
        row = {
            "version": 23, "steps": 20, "mean_loss": 0.04,
            "full_30_day_episode": False,
            "loss_components": {"comfort": 0.01, "energy": 0.02, "carbon": 0.01},
        }
        self.assertFalse(aggregate([row], full=True)["23"][
            "training_set_score_comparable"
        ])


class ControllerTests(unittest.TestCase):
    def test_actual_controller_cells_compile(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                cls = load_controller(version)
                self.assertTrue(issubclass(cls, LocalPolicy))

    def test_gateway_cell_not_executed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "toy.ipynb"
            path.write_text(json.dumps({
                "cells": [
                    {"cell_type": "code", "source": [
                        "from kaggle_evaluation import smart_buildings_inference_server\n",
                        "class TransferPolicy(smart_buildings_inference_server.Policy):\n",
                        "    pass\n",
                    ]},
                    {"cell_type": "code", "source": [
                        "raise RuntimeError('Gateway must never execute')\n"
                    ]},
                ]
            }), encoding="utf-8")
            self.assertTrue(issubclass(load_controller(7, path=path), LocalPolicy))

    def test_action_contract(self):
        spec = {
            "continuous_action": SimpleNamespace(shape=(2,)),
            "discrete_action": SimpleNamespace(shape=(1,)),
        }
        good = {"continuous_action": np.array([0, -1], dtype=np.float32),
                "discrete_action": np.array([1], dtype=np.int32)}
        self.assertEqual(validate_action(good, spec)["continuous_action"].shape, (2,))
        with self.assertRaises(ValueError):
            validate_action({**good, "continuous_action":
                             np.array([1.1, 0], dtype=np.float32)}, spec)
        with self.assertRaises(TypeError):
            validate_action({**good, "discrete_action": np.array([1])}, spec)

    def test_discover_uploaded_scenarios(self):
        scenarios = discover_scenarios(default_data_root())
        self.assertEqual(len(scenarios), 30)
        self.assertEqual(len(set(scenarios)), 30)


if __name__ == "__main__":
    unittest.main()
