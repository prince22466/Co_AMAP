#!/usr/bin/env python3
"""Load one scenario, inspect its spaces, and run a short rollout.

This is the quickest way to sanity-check that a scenario loads and to see the
exact shape of the observation and hybrid action spaces your policy must
produce for a given building. It loads a ``config.gin`` (by explicit path, or
by ``--building``/``--weather``), prints the action spec and the building
metadata your policy receives, then steps the environment with an "all on"
default action and reports the wall-clock cost per step.

Usage:
    # Inspect + smoke-test a scenario by building/weather
    python -m smart_buildings.run_environment \
        --building building_train_1 --weather fairbanks_1

    # Or point directly at a config.gin
    python -m smart_buildings.run_environment \
        --config smart_buildings/training_scenarios/building_train_2/phoenix_6/config.gin \
        --steps 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
from smart_buildings.smart_control.utils.scenario_generator import print_action_spec

DEFAULT_SCENARIOS_DIR = Path("smart_buildings/training_scenarios")


def make_all_on_action(env) -> dict:
    """A valid hybrid action that runs every AHU and the hot water system."""
    spec = env.action_spec()
    return {
        "continuous_action": np.ones(spec["continuous_action"].shape[0], dtype=np.float32),
        "discrete_action": np.ones(spec["discrete_action"].shape[0], dtype=np.int32),
    }


def run_smoke_test(env, num_steps: int) -> None:
    """Step the environment with a fixed action and report seconds/step."""
    env.reset()
    action = make_all_on_action(env)
    t0 = time.time()
    steps = 0
    for _ in range(num_steps):
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    elapsed = time.time() - t0
    print(f"\nRan {steps} step(s) in {elapsed:.1f}s "
          f"({elapsed / max(steps, 1):.2f} s/step).")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a scenario config.gin (overrides --building/--weather).",
    )
    parser.add_argument(
        "--building",
        default="building_train_1",
        help="Building stem, e.g. building_train_1.",
    )
    # The pair must be one that was actually generated: each building is paired
    # with one weather strip per station, so not every building/weather
    # combination exists on disk. List a building's strips with
    # `ls smart_buildings/training_scenarios/<building>`.
    parser.add_argument(
        "--weather",
        default="fairbanks_1",
        help=(
            "Weather strip, named <station>_<n> for stations atlanta, denver, "
            "fairbanks, phoenix and singapore."
        ),
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=DEFAULT_SCENARIOS_DIR,
        help="Base directory of generated scenarios.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps to run in the smoke test (default: 10).",
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    config_path = args.config or (
        args.scenarios_dir / args.building / args.weather / "config.gin"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.gin at {config_path}. Generate it first with "
            "`python -m smart_buildings.run_scenario_generator`."
        )

    print(f"Loading scenario: {config_path}")
    env = load_hybrid_environment(config_path)
    env._metrics_path = None
    env._occupancy_normalization_constant = 125.0

    print("\n=== Action spec ===")
    print_action_spec(env)

    meta = create_building_meta_data(env, config_path.parent)
    print("\n=== Building metadata (what your policy receives) ===")
    print(f"  discrete actions : {meta['n_discrete_action']}")
    print(f"  continuous actions: {meta['n_continuous_action']}")
    print(f"  AHUs             : {len(meta['ahu_zone_assignments'])}")
    print(f"  zone temp sensors: {len(meta['zone_temp_indices'])}")
    print(f"  observation size : {len(meta['observation_field_names'])}")
    print(f"  floors           : {meta['num_floors']}")

    run_smoke_test(env, args.steps)


if __name__ == "__main__":
    main()
