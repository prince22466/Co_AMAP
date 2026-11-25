"""Fast regression tests without simulator data."""
import unittest

import numpy as np
import torch

from ..p1.model import N_FEATURES
from .model import BOUND, SearchPolicy, candidates, PHYSICAL_FEATURES, signature
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
