"""Isolated official-simulator runner for one scenario and one controller.

Gin and working-directory state are process-global. Parallel workers MUST
use processes, never threads; a new environment and policy are created per job.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

from .controllers import load_controller, validate_action
from .metrics import score_one, weighted_losses


@contextmanager
def simulator_working_directory(data_root: Path):
    previous = Path.cwd()
    os.chdir(data_root)
    try:
        yield
    finally:
        os.chdir(previous)


def run_scenario(task: tuple[str, str, int, int | None]) -> dict:
    data_root_text, scenario_relative, version, steps = task
    data_root = Path(data_root_text).resolve()
    simulator_root = data_root / "smart_buildings"
    config_path = simulator_root / "training_scenarios" / scenario_relative / "config.gin"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    # The competition's data mount can be read-only. Keep all caches elsewhere.
    cache_root = Path(tempfile.gettempdir()) / "g8_rl_p0_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for key, subdir in (
        ("NUMBA_CACHE_DIR", "numba"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        path = cache_root / subdir
        path.mkdir(exist_ok=True)
        os.environ.setdefault(key, str(path))
    if str(data_root) not in sys.path:
        sys.path.insert(0, str(data_root))

    # The uploaded Gin configs contain relative asset paths.
    with simulator_working_directory(data_root):
        from smart_buildings.evaluate import (
            create_building_meta_data,
            load_hybrid_environment,
        )

        env = load_hybrid_environment(config_path)
        env.metrics_path = None  # Not env._metrics_path (incorrect attribute).
        # Preserve the official gateway's assignment, including its typo:
        # env.occupancy_normalization_constant (without _) is the real
        # attribute used by observations and remains 0.0 under shipped Gin.
        env._occupancy_normalization_constant = 125.0

        metadata = create_building_meta_data(env, config_path.parent)
        policy = load_controller(version)(metadata, env.action_spec())
        obs, _ = env.reset()
        weights = env.reward_function.weights
        configured_steps = int(env._num_timesteps_in_episode)
        if steps is None and configured_steps != 8640:
            raise ValueError(
                f"Unexpected full episode length: {configured_steps} steps"
            )
        # Official gateway runs until the environment signals termination.
        # Environment.step() checks _step_count BEFORE incrementing it,
        # so an 8640-step configuration currently terminates on step 8641.
        max_steps = configured_steps + 2 if steps is None else steps
        episode_terminated = False
        rewards: list[float] = []
        component_sums = {"comfort": 0.0, "energy": 0.0, "carbon": 0.0}
        elapsed = {"policy": 0.0, "simulator": 0.0, "diagnostics": 0.0}
        for step in range(max_steps):
            start = perf_counter()
            action = validate_action(policy.get_action(obs), env.action_spec())
            after_policy = perf_counter()
            obs, reward, terminated, truncated, _ = env.step(action)
            after_simulator = perf_counter()
            if not math.isfinite(float(reward)):
                raise ValueError(
                    f"Simulator rejected action or returned nonfinite reward "
                    f"at step {step + 1} of {scenario_relative}"
                )

            # Competition gateway only exposes the combined reward. Local
            # simulator lets us recompute its normalized parts post-step.
            _, response = env.get_reward_info_and_response()
            parts = weighted_losses(
                response.normalized_productivity_regret,
                response.normalized_energy_cost,
                response.normalized_carbon_emission,
                weights,
            )
            for key in component_sums:
                component_sums[key] += parts[key]
            rewards.append(float(reward))
            after_diagnostics = perf_counter()

            elapsed["policy"] += after_policy - start
            elapsed["simulator"] += after_simulator - after_policy
            elapsed["diagnostics"] += after_diagnostics - after_simulator
            if terminated or truncated:
                episode_terminated = bool(terminated or truncated)
                if steps is not None and step + 1 != steps:
                    raise RuntimeError(
                        f"{scenario_relative} terminated at {step + 1}/{steps}"
                    )
                break

        if steps is None and not episode_terminated:
            raise RuntimeError(
                f"{scenario_relative} failed to terminate within "
                f"{max_steps} steps"
            )
        result = score_one(
            rewards, component_sums, requested_steps=steps, elapsed=elapsed,
            episode_terminated=episode_terminated,
            configured_episode_steps=configured_steps,
        )
        result.update({"scenario": scenario_relative, "version": version})
        return result
