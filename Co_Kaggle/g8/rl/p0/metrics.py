"""Official normalized reward decomposition and strict episode scoring."""
from __future__ import annotations

import math
from typing import Mapping


def weighted_losses(
    normalized_productivity_regret: float,
    normalized_energy_cost: float,
    normalized_carbon_emission: float,
    weights: Mapping[str, float],
) -> dict[str, float]:
    wp = float(weights["productivity_weight"])
    we = float(weights["energy_cost_weight"])
    wc = float(weights["carbon_emission_weight"])
    total = wp + we + wc
    if total <= 0 or not all(map(math.isfinite, (wp, we, wc, total))):
        raise ValueError("Invalid reward weights")
    return {
        "comfort": -wp * float(normalized_productivity_regret) / total,
        "energy": we * float(normalized_energy_cost) / total,
        "carbon": wc * float(normalized_carbon_emission) / total,
    }


def score_one(
    rewards: list[float],
    component_sums: Mapping[str, float],
    *,
    requested_steps: int,
    elapsed: Mapping[str, float],
) -> dict:
    count = len(rewards)
    if count == 0 or count != requested_steps:
        raise ValueError(f"Incomplete episode: {count}/{requested_steps} steps")
    if not all(math.isfinite(float(r)) for r in rewards):
        raise ValueError("Nonfinite simulator reward")
    mean_reward = sum(rewards) / count
    contributions = {key: float(value) / count
                     for key, value in component_sums.items()}
    loss = sum(contributions.values())
    if not math.isclose(loss, -mean_reward, abs_tol=1e-5, rel_tol=1e-5):
        raise ValueError(f"Reward/decomposition disagree: {-mean_reward} vs {loss}")
    return {
        "steps": count,
        "mean_reward": mean_reward,
        "mean_loss": loss,
        "loss_components": contributions,
        "policy_ms_per_step": 1000 * elapsed["policy"] / count,
        "simulator_ms_per_step": 1000 * elapsed["simulator"] / count,
        "diagnostics_ms_per_step": 1000 * elapsed["diagnostics"] / count,
        "full_30_day_episode": count == 8640,
    }
