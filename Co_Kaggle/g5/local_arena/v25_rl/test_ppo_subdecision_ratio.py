"""Regression tests for the v25 per-worker PPO ratio update."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Categorical

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from worker_policy import ActorCritic, SubDecision, TurnRecord
from train_v25_worker_ppo import ppo_update


class SubdecisionPPOTest(unittest.TestCase):
    def test_joint_ratio_would_compound_but_subdecision_ratio_does_not(self):
        per_worker_ratio = 1.10
        workers = 12
        joint_ratio = math.exp(workers * math.log(per_worker_ratio))
        self.assertLess(per_worker_ratio, 1.2)
        self.assertGreater(joint_ratio, 1.2)

    def test_ppo_update_consumes_individual_worker_samples(self):
        torch.manual_seed(7)
        np.random.seed(7)

        candidate_dim = 6
        state_dim = 4
        model = ActorCritic(candidate_dim, state_dim, hidden=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)
        device = torch.device("cpu")

        records = []
        for turn, advantage in enumerate((1.0, -1.0)):
            state = np.random.randn(state_dim).astype(np.float32)
            st = torch.as_tensor(state, dtype=torch.float32)
            with torch.no_grad():
                old_value = float(model.value(st).item())

            subs = []
            for _ in range(4):
                candidates = np.random.randn(5, candidate_dim).astype(np.float32)
                ct = torch.as_tensor(candidates, dtype=torch.float32)
                with torch.no_grad():
                    dist = Categorical(logits=model.logits(ct))
                    action = int(torch.argmax(dist.logits).item())
                    old_log_prob = float(
                        dist.log_prob(torch.tensor(action)).item()
                    )
                subs.append(
                    SubDecision(
                        candidates=candidates.astype(np.float16),
                        action_index=action,
                        old_log_prob=old_log_prob,
                    )
                )

            record = TurnRecord(
                state=state,
                subdecisions=subs,
                old_value=old_value,
                turn=turn,
            )
            record.advantage = advantage
            record.return_target = old_value + advantage
            records.append(record)

        args = SimpleNamespace(
            ppo_epochs=2,
            minibatch_size=4,
            clip_ratio=0.2,
            target_kl=0.02,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
        )

        stats = ppo_update(model, opt, device, records, args)

        self.assertEqual(stats["actor_samples"], 8)
        self.assertAlmostEqual(stats["mean_subdecisions_per_turn"], 4.0)
        self.assertGreaterEqual(stats["ppo_epochs_completed"], 1)
        self.assertTrue(math.isfinite(stats["approx_kl"]))
        self.assertTrue(math.isfinite(stats["ratio_mean"]))
        self.assertGreaterEqual(stats["clip_fraction"], 0.0)
        self.assertLessEqual(stats["clip_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
