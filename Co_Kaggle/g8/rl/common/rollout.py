"""Shared profile-policy simulator and process dispatch; no model dependencies."""
from __future__ import annotations

import math
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
from torch.distributions import Categorical

from ..p0.controllers import load_controller, validate_action
from ..p0.metrics import weighted_losses
from ..p0.runner import simulator_working_directory
from .profiles import ProfileController, PROFILES


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rollout(task, *, model_factory, gamma=1., reward_scale=1.,
            return_final_features=False):
    """Collect one fresh simulator episode or partial window.

    Factories are supplied by the caller. Training returns raw macro-step
    transitions and a bootstrap value, not algorithm-specific advantages.
    Evaluation returns undiscounted raw loss and decomposed components.
    Off-policy callers can request the final observation features for
    bootstrapping at a nonterminal time limit.
    """
    root_text, scenario, steps, interval, weights, mode, seed = task
    torch.set_num_threads(1)
    if mode not in {"train", "eval", "baseline"}:
        raise ValueError(f"Unknown rollout mode: {mode}")
    if interval < 1 or (steps is not None and steps < 1):
        raise ValueError("Invalid rollout horizon or interval")
    if mode == "train" and steps is not None and steps % interval:
        raise ValueError("Training horizon must contain whole decision intervals")
    # Rollout sampling must not reset the optimizer's CPU/CUDA RNG stream.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    root = Path(root_text).resolve()
    cache = Path(tempfile.gettempdir()) / "g8_rl_common_cache" / str(os.getpid())
    for key, name in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "mpl"),
                      ("XDG_CACHE_HOME", "xdg")):
        path = cache / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Initializing a temporary model consumes CPU RNG; isolate that too.
    with torch.random.fork_rng(devices=[]):
        model = model_factory()
    model.load_state_dict(weights)
    model.eval()
    config = root / "smart_buildings" / "training_scenarios" / scenario / "config.gin"
    with simulator_working_directory(root):
        from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
        env = load_hybrid_environment(config)
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0  # Match gateway/P0 typo.
        try:
            spec = env.action_spec()
            metadata = create_building_meta_data(env, config.parent)
            baseline = load_controller(23)(metadata, spec)
            controller = ProfileController(baseline)
            obs, _ = env.reset()
            configured = int(env._num_timesteps_in_episode)
            if configured != 8640:
                raise ValueError("Unexpected simulator horizon")
            limit = steps if steps is not None else configured + 2
            count = 0
            raw_sum = 0.
            parts_sum = dict.fromkeys(("comfort", "energy", "carbon"), 0.)
            histogram = [0] * len(PROFILES)
            trajectory = {k: [] for k in ("x", "a", "logp", "v", "r", "duration")}
            done = terminated = truncated = False
            while count < limit and not done:
                if mode == "baseline":
                    # Independent, unmodified V23 path for a genuine parity check.
                    profile = 0
                else:
                    x = controller.features(obs)
                    with torch.no_grad():
                        logits, value = model(torch.from_numpy(x))
                        distribution = Categorical(logits=logits)
                        action = (torch.multinomial(distribution.probs, 1,
                                                    generator=generator).squeeze(0)
                                  if mode == "train" else logits.argmax())
                        logp = distribution.log_prob(action)
                    profile = int(action)
                    controller.select(profile)
                macro_reward = 0.
                duration = 0
                for j in range(min(interval, limit - count)):
                    policy = baseline if mode == "baseline" else controller
                    command = validate_action(policy.get_action(obs), spec)
                    obs, reward, terminated, truncated, _ = env.step(command)
                    reward = float(reward)
                    if not math.isfinite(reward):
                        raise ValueError(f"Nonfinite reward: {scenario}/{count}")
                    macro_reward += gamma ** j * reward_scale * reward
                    raw_sum += reward
                    if mode != "train":
                        _, response = env.get_reward_info_and_response()
                        parts = weighted_losses(
                            response.normalized_productivity_regret,
                            response.normalized_energy_cost,
                            response.normalized_carbon_emission,
                            env.reward_function.weights)
                        # Same per-step tolerance as P0 diagnostics_core: the
                        # simulator reward passes through float32 protos.
                        if not math.isclose(-reward, sum(parts.values()),
                                            abs_tol=1e-8, rel_tol=1e-7):
                            raise ValueError(
                                f"Reward decomposition mismatch {scenario}/{count}: "
                                f"{-reward!r} vs {sum(parts.values())!r}")
                        for k in parts_sum:
                            parts_sum[k] += parts[k]
                    count += 1
                    duration += 1
                    histogram[profile] += 1
                    done = bool(terminated or truncated)
                    if done:
                        break
                if mode == "train":
                    for key, val in zip(trajectory, (x, profile, float(logp), float(value),
                                                     macro_reward, duration)):
                        trajectory[key].append(val)
            if (steps is None and (not terminated or count < configured)) or (
                    steps is not None and count != steps) or (terminated and count < configured):
                raise RuntimeError(f"Incomplete rollout {scenario}: {count}")
            row = {"scenario": scenario, "steps": count, "mean_loss": -raw_sum / count,
                   "full_episode": bool(terminated and count >= configured),
                   "terminated": bool(terminated), "truncated": bool(truncated),
                   "profile_steps": histogram}
            if mode != "train":
                row["loss_components"] = {k: v / count for k, v in parts_sum.items()}
            else:
                # Partial windows are time limits, not simulator terminal states.
                final_features = controller.features(obs)
                with torch.no_grad():
                    _, next_value = model(torch.from_numpy(final_features))
                if return_final_features:
                    row["final_features"] = final_features.copy()
                row["trajectory"] = trajectory
                row["bootstrap"] = 0. if terminated else float(next_value)
            return row
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def run_jobs(pool, root, scenarios, steps, interval, model, mode, seed, *, worker):
    """Dispatch paired jobs in order; use a picklable worker for spawn."""
    weights = snapshot(model)
    tasks = [(str(root), s, steps, interval, weights, mode, seed + i)
             for i, s in enumerate(scenarios)]
    return list(map(worker, tasks)) if pool is None else list(pool.map(worker, tasks))
