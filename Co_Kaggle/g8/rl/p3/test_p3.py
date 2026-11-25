"""Fast regression tests; no simulator data required."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from ..common.profiles import N_ACTIONS, N_FEATURES
from .learning import Replay, bellman_target, exploration, optimize, transitions
from .model import QPolicy, signature
from .train import save_checkpoint, select_scenarios


def trajectory(terminal=False):
    return {"steps": 5, "terminated": terminal,
            "final_features": np.full(N_FEATURES, 2., dtype=np.float32),
            "trajectory": {"x": [np.zeros(N_FEATURES), np.ones(N_FEATURES)],
                           "a": [0, 2], "r": [-3., -2.], "duration": [3, 2]}}


class P3Tests(unittest.TestCase):
    def test_initial_policy_and_exploration(self):
        model = QPolicy()
        x = torch.randn(8, N_FEATURES)
        self.assertTrue((model(x)[0].argmax(-1) == 0).all())
        model.epsilon.fill_(.4)
        probs = model(x)[0].softmax(-1)
        self.assertTrue(torch.allclose(probs[:, 0], torch.full((8,), .6 + .4/N_ACTIONS)))
        self.assertTrue(torch.allclose(probs[:, 1:], torch.full((8, N_ACTIONS-1), .4/N_ACTIONS)))
        with torch.no_grad():
            model.network[-1].bias[2] = .02
        self.assertTrue((model.greedy_action(x) == 0).all())
        with torch.no_grad():
            model.network[-1].bias[2] = .04
        self.assertTrue((model.greedy_action(x) == 2).all())

    def test_duration_and_time_limit(self):
        partial = list(transitions(trajectory(), .9))
        self.assertEqual(len(partial), 2)
        self.assertAlmostEqual(partial[0][2], -.3)
        self.assertAlmostEqual(partial[0][4], .9**3)
        np.testing.assert_array_equal(partial[0][3], np.ones(N_FEATURES))
        np.testing.assert_array_equal(partial[1][3], np.full(N_FEATURES, 2.))
        self.assertAlmostEqual(partial[1][4], .9**2)
        broken = trajectory()
        broken["final_features"] = np.zeros(N_FEATURES + 1)
        with self.assertRaises(ValueError):
            list(transitions(broken, .9))
        terminal = list(transitions(trajectory(True), .9))
        self.assertEqual(len(terminal), 2)
        self.assertEqual(terminal[-1][4], 0.)
        self.assertAlmostEqual(terminal[-1][2], -.2)

    def test_replay_and_learning(self):
        memory = Replay(3)
        memory.extend(trajectory(True), .9)
        memory.extend(trajectory(True), .9)
        self.assertEqual(len(memory), 3)
        model, target = QPolicy(), QPolicy()
        target.requires_grad_(False)
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        before = model.network[-1].bias.detach().clone()
        batch = memory.sample(3, np.random.default_rng(42), "cpu")
        loss = optimize(model, target, opt, batch)
        self.assertGreater(loss, 0.)
        self.assertFalse(torch.equal(before, model.network[-1].bias))
        self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_double_dqn_not_target_max(self):
        online, target = QPolicy(), QPolicy()
        with torch.no_grad():
            online.network[-1].bias[1] = 2.
            target.network[-1].bias[1] = 3.
            target.network[-1].bias[2] = 8.
        actual = bellman_target(online, target, torch.tensor([-1., -2.]),
                                torch.zeros(2, N_FEATURES), torch.tensor([.5, 0.]))
        torch.testing.assert_close(actual, torch.tensor([.5, -2.]))

    def test_checkpoint(self):
        model = QPolicy()
        model.epsilon.fill_(.4)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "best.pt"
            save_checkpoint(path, model, 12, 0, .01, 864)
            state = torch.load(path, weights_only=True)
            self.assertEqual(state["signature"], signature())
            restored = QPolicy()
            restored.load_state_dict(state["model"])
            self.assertEqual(float(restored.epsilon), 0.)
            self.assertAlmostEqual(float(model.epsilon), .4)
            torch.testing.assert_close(restored.q_values(torch.zeros(N_FEATURES)),
                                       model.q_values(torch.zeros(N_FEATURES)))

    def test_sampling_and_schedule(self):
        scenarios = [f"building_train_{b}/city_{c}" for b in (1, 2, 4, 7) for c in range(5)]
        selected = select_scenarios(scenarios, np.random.default_rng(42))
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({s.split("/")[0] for s in selected}), 4)
        self.assertAlmostEqual(exploration(1, 20, .4, .1), .4)
        self.assertAlmostEqual(exploration(20, 20, .4, .1), .1)


if __name__ == "__main__":
    unittest.main()
