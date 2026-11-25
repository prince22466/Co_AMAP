"""Run without simulator assets: python -m unittest Co_Kaggle.g8.rl.p1.test_p1."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from .model import (ActorCritic, ProfileController, PARAMETERS, PROFILES,
                    N_FEATURES, advantages)
from .train import split_scenarios, update, comparison, assert_initial_parity, checkpoint_signature


class P1Tests(unittest.TestCase):
    def test_profiles_restore_original(self):
        b = SimpleNamespace(**{k: 2. for k in PARAMETERS})
        p = ProfileController(b)
        p.select(1)
        p.select(1)
        self.assertEqual(b.ACTIVE_THRESHOLD_K, 2. * PROFILES[1][1][0])
        p.select(0)
        self.assertTrue(all(getattr(b, k) == 2. for k in PARAMETERS))

    def test_initial_greedy_baseline(self):
        model = ActorCritic()
        logits, value = model(torch.randn(12, N_FEATURES))
        self.assertTrue(torch.all(logits.argmax(-1) == 0))
        self.assertEqual(tuple(value.shape), (12,))

    def test_variable_duration_gae(self):
        adv, returns = advantages([1., 2.], [0., 0.], [2, 1], 0., .5, 1.)
        np.testing.assert_allclose(adv, [1.5, 2.])
        np.testing.assert_allclose(returns, adv)
        adv, _ = advantages([1.], [0.], [2], 4., .5, 1.)
        np.testing.assert_allclose(adv, [2.])

    def test_split_disjoint_buildings(self):
        scenarios = [f"building_train_{i}/weather_{j}"
                     for i in (1, 2, 4, 5, 7, 8) for j in range(5)]
        split = split_scenarios(scenarios)
        self.assertEqual([len(split[k]) for k in split], [20, 5, 5])
        sets = [{s.split("/")[0] for s in group} for group in split.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_parity_detects_canceling_errors(self):
        def row(scenario, loss):
            return {"scenario": scenario, "steps": 1, "full_episode": False,
                    "mean_loss": loss, "loss_components": {"comfort": loss},
                    "profile_steps": [1, 0, 0, 0, 0, 0, 0]}
        baseline = [row("a", .1), row("b", .2)]
        learned = [row("a", .2), row("b", .1)]
        assert_initial_parity(comparison(baseline, baseline))
        with self.assertRaises(RuntimeError):
            assert_initial_parity(comparison(baseline, learned))

    def test_checkpoint_signature(self):
        self.assertEqual(checkpoint_signature()["features"], N_FEATURES)
        self.assertEqual(len(checkpoint_signature()["profiles"]), len(PROFILES))

    def test_ppo_update(self):
        torch.manual_seed(42)
        model = ActorCritic()
        x = torch.randn(8, N_FEATURES)
        with torch.no_grad():
            logits, v = model(x)
            d = torch.distributions.Categorical(logits=logits)
            a = d.sample()
            lp = d.log_prob(a)
        before = model.actor.weight.detach().clone()
        row = {"trajectory": {"x": x.numpy(), "a": a.numpy(), "logp": lp.numpy(),
                              "adv": np.arange(8, dtype=np.float32), "target": v.numpy() + 1.}}
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        loss = update(model, optimizer, [row], "cpu", 2, 4)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.actor.weight))


if __name__ == "__main__":
    unittest.main()
