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
from train_v25_worker_ppo import (
    actor_samples_and_advantages,
    ppo_update,
    validation_checkpoint_decision,
)


class SubdecisionPPOTest(unittest.TestCase):
    def test_low_temperature_concentrates_probability_on_argmax(self):
        logits = torch.tensor([0.10, 0.08, 0.00, -0.05])
        normal = Categorical(logits=logits)
        cold = Categorical(logits=logits / 0.20)
        winner = int(torch.argmax(logits).item())
        self.assertGreater(
            float(cold.probs[winner].item()),
            float(normal.probs[winner].item()),
        )

    def test_joint_ratio_would_compound_but_subdecision_ratio_does_not(self):
        per_worker_ratio = 1.10
        workers = 12
        joint_ratio = math.exp(workers * math.log(per_worker_ratio))
        self.assertLess(per_worker_ratio, 1.2)
        self.assertGreater(joint_ratio, 1.2)

    def test_subdecision_bonus_breaks_same_turn_credit_smearing(self):
        state = np.zeros(4, dtype=np.float32)
        candidates = np.zeros((2, 6), dtype=np.float16)
        positive = SubDecision(candidates, 0, 0.0, actor_bonus=2.0)
        negative = SubDecision(candidates, 1, 0.0, actor_bonus=-1.0)
        record = TurnRecord(
            state=state,
            subdecisions=[positive, negative],
            old_value=0.0,
            turn=0,
        )
        record.advantage = 7.0
        record.return_target = 7.0

        (
            _turn_adv, samples, actor_adv, bonuses,
            advantage_scale, reward_equiv, reward_equiv_normalized,
        ) = actor_samples_and_advantages([record])
        self.assertEqual(len(samples), 2)
        np.testing.assert_allclose(bonuses, [2.0, -1.0], atol=1e-6)
        # A single turn normalizes to base advantage 0, so the two worker
        # choices now receive their own shaping rather than one shared signal.
        np.testing.assert_allclose(actor_adv, [2.0, -1.0], atol=1e-6)
        self.assertEqual(advantage_scale, 1.0)
        np.testing.assert_allclose(reward_equiv, [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            reward_equiv_normalized, [0.0, 0.0], atol=1e-6
        )

    def test_planned_plant_reward_equiv_uses_raw_gae_scale(self):
        state = np.zeros(4, dtype=np.float32)
        candidates = np.zeros((2, 6), dtype=np.float16)

        plant = SubDecision(
            candidates, 0, 0.0,
            actor_bonus=0.0,
            reward_equiv_bonus=8.0,
        )
        neutral = SubDecision(candidates, 1, 0.0)

        low = TurnRecord(state, [plant], 0.0, 0)
        high = TurnRecord(state, [neutral], 0.0, 1)
        low.advantage = 0.0
        high.advantage = 8.0
        low.return_target = 0.0
        high.return_target = 8.0

        (
            _turn_adv, _samples, actor_adv, bonuses,
            advantage_scale, reward_equiv, reward_equiv_normalized,
        ) = actor_samples_and_advantages([low, high])

        # Raw GAE advantages [0, 8] have std=4.  The +8 planned-plant
        # completion therefore contributes +2 normalized actor advantage.
        self.assertAlmostEqual(advantage_scale, 4.0)
        np.testing.assert_allclose(reward_equiv, [8.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            reward_equiv_normalized, [2.0, 0.0], atol=1e-6
        )
        np.testing.assert_allclose(bonuses, [2.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(actor_adv, [1.0, 1.0], atol=1e-6)

    def test_higher_validation_reward_selects_checkpoint(self):
        improved, rollback, ratio = validation_checkpoint_decision(
            score=40001.0, reference=40000.0, best=40000.0,
            collapse_restore_ratio=0.70,
        )
        self.assertTrue(improved)
        self.assertFalse(rollback)
        self.assertAlmostEqual(ratio, 40001.0 / 40000.0)

    def test_lower_reward_never_replaces_best(self):
        improved, rollback, ratio = validation_checkpoint_decision(
            score=39000.0, reference=40000.0, best=40000.0,
            collapse_restore_ratio=0.70,
        )
        self.assertFalse(improved)
        self.assertFalse(rollback)
        self.assertAlmostEqual(ratio, 0.975)

    def test_equal_reward_keeps_existing_checkpoint(self):
        improved, rollback, ratio = validation_checkpoint_decision(
            score=40000.0, reference=40000.0, best=40000.0,
            collapse_restore_ratio=0.70,
        )
        self.assertFalse(improved)
        self.assertFalse(rollback)
        self.assertEqual(ratio, 1.0)

    def test_reward_collapse_restores_best(self):
        improved, rollback, ratio = validation_checkpoint_decision(
            score=25000.0, reference=40000.0, best=40000.0,
            collapse_restore_ratio=0.70,
        )
        self.assertFalse(improved)
        self.assertTrue(rollback)
        self.assertAlmostEqual(ratio, 0.625)

    def test_nonfinite_reward_is_not_selected(self):
        improved, rollback, ratio = validation_checkpoint_decision(
            score=float("nan"), reference=40000.0, best=40000.0,
            collapse_restore_ratio=0.70,
        )
        self.assertFalse(improved)
        self.assertFalse(rollback)
        self.assertIsNone(ratio)

    def test_ppo_update_consumes_individual_worker_samples(self):
        torch.manual_seed(7)
        np.random.seed(7)

        candidate_dim = 6
        state_dim = 4
        model = ActorCritic(candidate_dim, state_dim, hidden=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)
        device = torch.device("cpu")

        rollout_temperature = 0.20
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
                    dist = Categorical(
                        logits=model.logits(ct) / rollout_temperature
                    )
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
            clip_ratio=0.10,
            target_kl=0.01,
            rollout_temperature=rollout_temperature,
            value_coef=0.5,
            entropy_coef=0.001,
            max_grad_norm=0.5,
        )

        stats = ppo_update(model, opt, device, records, args)

        self.assertEqual(stats["actor_samples"], 8)
        self.assertAlmostEqual(stats["mean_subdecisions_per_turn"], 4.0)
        self.assertGreaterEqual(stats["ppo_epochs_completed"], 1)
        self.assertGreaterEqual(stats["actor_minibatches_completed"], 1)
        self.assertTrue(math.isfinite(stats["actor_bonus_mean"]))
        self.assertGreaterEqual(stats["actor_bonus_positive_fraction"], 0.0)
        self.assertLessEqual(stats["actor_bonus_positive_fraction"], 1.0)
        self.assertTrue(math.isfinite(stats["approx_kl"]))
        self.assertTrue(math.isfinite(stats["ratio_mean"]))
        self.assertGreaterEqual(stats["clip_fraction"], 0.0)
        self.assertLessEqual(stats["clip_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
