"""Building-independent PPO policy over bounded V23 operating profiles."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Re-export legacy names for existing P1 callers and tests.
from ..common.profiles import PARAMETERS, PROFILES, N_ACTIONS, N_FEATURES, ProfileController


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(N_FEATURES, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, N_ACTIONS)
        self.critic = nn.Linear(64, 1)
        # Initial greedy policy is exactly V23; stochastic training explores.
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        with torch.no_grad():
            self.actor.bias[0] = .1  # Small tie-breaking preference, not a learning barrier.

    def forward(self, features):
        hidden = self.encoder(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def advantages(rewards, values, durations, bootstrap, gamma=.999, lam=.95):
    """Variable-duration macro-step GAE, with per-simulator-step discount."""
    result = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.
    next_value = float(bootstrap)
    for i in reversed(range(len(rewards))):
        discount = gamma ** int(durations[i])
        delta = rewards[i] + discount * next_value - values[i]
        carry = delta + discount * lam * carry
        result[i] = carry
        next_value = values[i]
    return result, result + np.asarray(values, dtype=np.float32)
