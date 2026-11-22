"""Isolated official-simulator runner for one scenario and one controller.

Gin and working-directory state are process-global. Parallel workers MUST
use processes, never threads; a new environment and policy are created per job.
"""
from __future__ import annotations

from contextlib import contextmanager
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


def run_scenario(task: tuple[str, str, int, int]) -> dict:
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
        env._occupancy_normalization_constant = 125.0

        metadata = create_building_meta_data(env, config_path.parent)
        policy = load_controller(version)(metadata, env.action_spec())
        obs, _ = env.reset()
        weights = env.reward_function.weights
        rewards: list[float] = []
        component_sums = {"comfort": 0.0, "energy": 0.0, "carbon": 0.0}
        elapsed = {"policy": 0.0, "simulator": 0.0, "diagnostics": 0.0}
        for step in range(steps):
            start = perf_counter()
            action = validate_action(policy.get_action(obs), env.action_spec())
            after_policy = perf_counter()
            obs, reward, terminated, truncated, _ = env.step(action)
            after_simulator = perf_counter()

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
                if step + 1 != steps:
                    raise RuntimeError(
                        f"{scenario_relative} terminated at {step + 1}/{steps}"
                    )
                break

        result = score_one(
            rewards, component_sums, requested_steps=steps, elapsed=elapsed
        )
        result.update({"scenario": scenario_relative, "version": version})
        return result
