"""Value network and conservative epsilon-greedy profile selection."""
from __future__ import annotations

import torch
from torch import nn

from ..common.profiles import N_ACTIONS, N_FEATURES, PROFILES

MARGIN = 0.03


class QPolicy(nn.Module):
    """Predict normalized discounted reward; retain V23 on small advantages."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(N_FEATURES, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        # Included in snapshots sent to simulator workers.
        self.register_buffer("epsilon", torch.tensor(0.0))

    def q_values(self, features):
        return self.network(features)

    def greedy_action(self, features):
        scores = self.q_values(features)
        # V23 wins ties; only switch when the predicted advantage is positive
        # by more than the margin (in normalized reward units).
        adjusted = torch.cat((scores[..., :1], scores[..., 1:] - MARGIN), dim=-1)
        return adjusted.argmax(dim=-1)

    def forward(self, features):
        """Adapt epsilon-greedy behavior to common.rollout's categorical API."""
        action = self.greedy_action(features)
        one_hot = torch.nn.functional.one_hot(action, N_ACTIONS).float()
        probs = (1 - self.epsilon) * one_hot + self.epsilon / N_ACTIONS
        logits = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
        return logits, torch.zeros_like(logits[..., 0])

    def validate(self):
        if not all(torch.isfinite(t).all() for t in self.state_dict().values()):
            raise ValueError("Nonfinite model state")
        if not 0 <= float(self.epsilon) < 1:
            raise ValueError("Invalid exploration rate")


def signature():
    return {"schema": 1, "algorithm": "baseline_gated_double_dqn",
            "features": N_FEATURES, "hidden": [64, 64],
            "margin": MARGIN, "baseline_version": 23,
            "profiles": [[name, list(factors)] for name, factors in PROFILES]}
