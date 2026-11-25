"""Fast regression tests without simulator data."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ..p1.model import N_FEATURES
from .model import BOUND, SearchPolicy, candidates, PHYSICAL_FEATURES, signature
from . import train
from .train import choose_candidate, select_scenarios


def row(scenario, loss):
    return {"scenario": scenario, "steps": 12, "full_episode": False, "mean_loss": loss}


class P2Tests(unittest.TestCase):
    def test_initial_policy_is_v23(self):
        model = SearchPolicy()
        scores, value = model(torch.randn(64, N_FEATURES))
        self.assertTrue(torch.all(scores.argmax(-1) == 0))
        self.assertEqual(tuple(value.shape), (64,))
        self.assertEqual(signature()["baseline_version"], 23)

    def test_state_conditioning_and_no_previous_profile_lock(self):
        model = SearchPolicy()
        with torch.no_grad():
            model.coefficients[0, 0] = 1.
        x = torch.zeros(2, N_FEATURES)
        x[:, 0] = torch.tensor([-1., 1.])
        before = model(x)[0]
        x[:, PHYSICAL_FEATURES:] = 1.
        self.assertTrue(torch.equal(before, model(x)[0]))
        self.assertEqual(before.argmax(-1).tolist(), [0, 1])

    def test_antithetic_mutations_preserve_parent_and_other_profiles(self):
        model = SearchPolicy()
        before = model.coefficients.clone()
        plus, minus = candidates(model, 2, .5, np.random.default_rng(42))
        self.assertTrue(torch.equal(before, model.coefficients))
        self.assertTrue(torch.allclose(plus.coefficients + minus.coefficients, 2 * before))
        self.assertTrue(torch.equal(plus.coefficients[0], before[0]))
        self.assertFalse(torch.equal(plus.coefficients[1], before[1]))

    def test_candidate_acceptance_and_regression_guard(self):
        parent = [row("a", .02), row("b", .02)]
        good = [row("a", .019), row("b", .018)]
        unsafe = [row("a", .001), row("b", .026)]
        bad = [row("a", .021), row("b", .02)]
        self.assertEqual(choose_candidate(parent, [unsafe, good], 1e-6, .005), 1)
        self.assertIsNone(choose_candidate(parent, [parent, bad], 1e-6, .005))
        with self.assertRaises(ValueError):
            choose_candidate(parent, [list(reversed(good))], 1e-6, .005)

    def test_nonfinite_candidate_and_corrupt_checkpoint(self):
        parent = [row("a", .02), row("b", .02)]
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                choose_candidate(parent, [[row("a", .01), row("b", invalid)]], 0., .005)
        with self.assertRaises(ValueError):
            choose_candidate(parent, [], -1., .005)
        model = SearchPolicy()
        for invalid in (float("nan"), float("inf"), BOUND + 1.):
            with torch.no_grad():
                model.coefficients[0, 0] = invalid
            with self.assertRaises(ValueError):
                model.validate_coefficients()
        with self.assertRaises(ValueError):
            candidates(SearchPolicy(), 1, BOUND + 1., np.random.default_rng(42))

    def test_training_checkpoints_each_accepted_policy_without_test_leakage(self):
        # Exercise the actual CLI orchestration with cheap deterministic returns.
        scenarios = [f"building_train_{i}/weather_{j}"
                     for i in (1, 2, 4, 5, 7, 8) for j in range(5)]
        training_losses = {0.: .02, .5: .01, 1.: .03, 1.5: .009}
        validation_losses = {0.: .014, .5: .009, 1.: .03, 1.5: .012}
        validation_calls = []

        def fake_jobs(pool, root, selected, steps, interval, model, mode, seed):
            weight = float(model.coefficients[0, 0])
            validation = all(s.startswith("building_train_5/") for s in selected)
            self.assertFalse(any(s.startswith("building_train_8/") for s in selected))
            if validation:
                validation_calls.append((mode, weight))
            else:
                self.assertEqual(len(selected), 4)
                self.assertEqual({s.split("/")[0] for s in selected},
                                 {"building_train_1", "building_train_2",
                                  "building_train_4", "building_train_7"})
            losses = validation_losses if validation else training_losses
            loss = losses[0. if mode == "baseline" else weight]
            return [{**row(s, loss), "steps": steps,
                     "loss_components": {"comfort": loss, "energy": 0., "carbon": 0.},
                     "profile_steps": [steps, 0, 0, 0, 0, 0, 0]} for s in selected]

        def policy(weight):
            model = SearchPolicy()
            with torch.no_grad():
                model.coefficients[0, 0] = weight
            return model

        proposals = [[policy(.5), policy(1.)],
                     [policy(1.), policy(1.)],
                     [policy(1.5), policy(1.)]]
        with TemporaryDirectory() as tmp:
            argv = ["train", "--out", tmp, "--updates", "3", "--steps", "12"]
            with (patch("sys.argv", argv),
                  patch.object(train, "discover_scenarios", return_value=scenarios),
                  patch.object(train, "verify_training_manifest"),
                  patch.object(train, "run_jobs", side_effect=fake_jobs),
                  patch.object(train, "candidates", side_effect=proposals)):
                train.main()
            history = json.loads((Path(tmp) / "history.json").read_text())["history"]
            state = torch.load(Path(tmp) / "best.pt", weights_only=True)
        self.assertEqual([entry["accepted"] for entry in history[1:]], [0, None, 0])
        self.assertEqual([entry["validation_reused"] for entry in history[1:]],
                         [False, True, False])
        self.assertEqual(validation_calls,
                         [("baseline", 0.), ("eval", 0.), ("eval", .5), ("eval", 1.5)])
        self.assertEqual(state["update"], 1)
        self.assertAlmostEqual(state["validation_loss"], .009)
        self.assertEqual(float(state["model"]["coefficients"][0, 0]), .5)

    def test_balanced_deterministic_selection(self):
        training = [f"building_train_{i}/weather_{j}"
                    for i in (1, 2, 4, 7) for j in range(5)]
        first = select_scenarios(training, 2, np.random.default_rng(42))
        second = select_scenarios(training, 2, np.random.default_rng(42))
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 8)
        for building in (1, 2, 4, 7):
            self.assertEqual(sum(s.startswith(f"building_train_{building}/") for s in first), 2)


if __name__ == "__main__":
    unittest.main()
