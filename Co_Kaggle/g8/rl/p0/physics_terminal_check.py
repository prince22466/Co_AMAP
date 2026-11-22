"""Real-simulator regression test for gateway end-of-episode semantics.

The configured 30-day period has 8640 nominal intervals. In the uploaded
simulator, Environment.step() tests termination before incrementing its
counter, so the gateway collects 8641 rewards before the terminal step.
Keep this separate from dependency-light tests; it imports the full simulator.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from .benchmark import default_data_root
from .runner import simulator_working_directory


def main() -> None:
    data_root = default_data_root().resolve()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "g8_rl_p0_numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "g8_rl_p0_mpl"))
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(data_root))
    scenario = data_root / "smart_buildings/training_scenarios/building_train_1/atlanta_2"
    with simulator_working_directory(data_root):
        from smart_buildings.evaluate import load_hybrid_environment

        env = load_hybrid_environment(scenario / "config.gin")
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0
        env.reset()
        assert env._num_timesteps_in_episode == 8640
        # Shorten only the termination counter, not the physical scenario.
        # This reproduces the exact step/terminal order with three cheap steps.
        env._num_timesteps_in_episode = 2
        action = {
            "continuous_action": np.zeros(
                env.action_spec()["continuous_action"].shape, dtype=np.float32
            ),
            "discrete_action": np.ones(
                env.action_spec()["discrete_action"].shape, dtype=np.int32
            ),
        }
        flags = []
        for _ in range(3):
            _, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward), "Nonfinite simulator reward"
            assert not truncated, "Unexpected scenario truncation"
            flags.append(bool(terminated))
        assert flags == [False, False, True], f"Termination order changed: {flags}"
        print("Confirmed: configured=2 steps; gateway-style episode ends on step 3")


if __name__ == "__main__":
    main()
