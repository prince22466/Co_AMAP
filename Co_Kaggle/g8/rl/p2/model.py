"""Small, state-conditioned operating-profile policy; no critic or PPO."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..p1.model import N_FEATURES, N_ACTIONS, PROFILES

# Exclude previous profile one-hot to avoid self-reinforcing profile lock-in.
# Keep all 18 gateway-derived physical and clock features.
PHYSICAL_FEATURES = N_FEATURES - N_ACTIONS
MARGIN = .15
BOUND = 2.


class SearchPolicy(nn.Module):
    """Profile 0 has a fixed zero score; other profiles must beat it."""

    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(
            torch.zeros(N_ACTIONS - 1, PHYSICAL_FEATURES + 1),
            requires_grad=False,
        )
        with torch.no_grad():
            self.coefficients[:, -1] = -MARGIN

    def validate_coefficients(self):
        """Reject corrupted or out-of-range checkpoint parameters."""
        if (not torch.isfinite(self.coefficients).all()
                or (self.coefficients.abs() > BOUND).any()):
            raise ValueError("Nonfinite or out-of-range policy coefficients")

    def forward(self, features):
        physical = features[..., :PHYSICAL_FEATURES]
        scores = physical @ self.coefficients[:, :-1].T + self.coefficients[:, -1]
        baseline = torch.zeros_like(scores[..., :1])
        logits = torch.cat((baseline, scores), dim=-1)
        # Shared deterministic simulator interface; value is unused in P2.
        return logits, torch.zeros_like(logits[..., 0])


def candidates(parent, profile, sigma, rng):
    """Antithetic, single-profile mutations; the incumbent is never modified."""
    if not 1 <= profile < N_ACTIONS or not np.isfinite(sigma) or not 0 < sigma <= BOUND:
        raise ValueError("Invalid mutation")
    parent.validate_coefficients()
    direction = rng.normal(size=PHYSICAL_FEATURES)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Zero mutation direction")
    direction = np.r_[direction / norm, .5]
    result = []
    for sign in (1, -1):
        child = SearchPolicy()
        child.load_state_dict(parent.state_dict())
        with torch.no_grad():
            row = child.coefficients[profile - 1]
            step = torch.as_tensor(sign * sigma * direction, dtype=row.dtype)
            row.add_(step).clamp_(-BOUND, BOUND)
        result.append(child)
    return result


def signature():
    return {"schema": 1, "algorithm": "paired_profile_search",
            "physical_features": PHYSICAL_FEATURES, "margin": MARGIN,
            "coefficient_bound": BOUND, "baseline_version": 23,
            "profiles": [[name, list(factors)] for name, factors in PROFILES]}
