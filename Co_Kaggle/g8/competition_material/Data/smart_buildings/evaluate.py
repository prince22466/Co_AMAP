#!/usr/bin/env python3
"""Evaluate HVAC control policies on generated smart-building scenarios.

Participants use this script to test their own policy. It ships with a trivial
built-in ``default`` policy so it runs out of the box; replace it with your own.

Policies must implement:

    policy = MyPolicy(building_meta_data, action_spec)
    action = policy.get_action(obs)

The script discovers generated config.gin files under
``smart_buildings/training_scenarios``, runs selected policies, tracks rollouts
with SimulationTracker, and saves results under ``smart_buildings/results``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
)

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gin
import matplotlib.pyplot as plt
import numpy as np

from smart_buildings.baselines.all_off import AllOffPolicy
from smart_buildings.baselines.bang_bang import BangBangPolicy
from smart_buildings.baselines.fixed_setpoint import FixedSetpointPolicy
from smart_buildings.baselines.pid import PIDPolicy
# pid_plus.py also defines the intermediate classes PIDWaterResetPolicy is built
# from. They are base classes, not runnable policies, so only the leaf is
# imported here.
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
from smart_buildings.baselines.transfer_policy import TransferPolicy

# Gin registration imports. These are intentionally kept even if not directly
# referenced below, because gin needs the configurables to be imported before
# parsing generated config.gin files.
from smart_buildings.smart_control.reward import electricity_energy_cost  # noqa: F401
from smart_buildings.smart_control.reward import natural_gas_energy_cost  # noqa: F401
from smart_buildings.smart_control.reward import setpoint_energy_carbon_regret  # noqa: F401
from smart_buildings.smart_control.simulator import air_handler  # noqa: F401
from smart_buildings.smart_control.simulator import boiler  # noqa: F401
from smart_buildings.smart_control.simulator import building  # noqa: F401
from smart_buildings.smart_control.simulator import hot_water_system  # noqa: F401
from smart_buildings.smart_control.simulator import hvac_floorplan_based  # noqa: F401
from smart_buildings.smart_control.simulator import pump  # noqa: F401
from smart_buildings.smart_control.simulator import randomized_arrival_departure_occupancy  # noqa: F401
from smart_buildings.smart_control.simulator import setpoint_schedule  # noqa: F401
from smart_buildings.smart_control.simulator import simulator_building  # noqa: F401
from smart_buildings.smart_control.simulator import simulator_flexible_floor_plan  # noqa: F401
from smart_buildings.smart_control.simulator import stochastic_convection_simulator  # noqa: F401
from smart_buildings.smart_control.simulator import weather_controller  # noqa: F401
from smart_buildings.smart_control.utils import bounded_action_normalizer  # noqa: F401
from smart_buildings.smart_control.utils import environment_utils  # noqa: F401
from smart_buildings.smart_control.utils import observation_normalizer  # noqa: F401
from smart_buildings.smart_control.environment import hybrid_action_environment
from smart_buildings.smart_control.utils.scenario_generator import (
    SimulationTracker,
    generate_scenario_from_config,
    load_scenario_from_parts,
)


# Runnable policies, keyed by what you pass to --policies.
#
# "transfer" is the template you edit: it lives in baselines/transfer_policy.py,
# ships returning the zero action, and its module docstring is the interface
# contract. Register your own policy here.
#
# The other five are hand-written reference controllers. Four of them --
# bang_bang, fixed_setpoint, pid and pid_water -- form the fixed baseline set
# that defines the leaderboard scale; see BASELINES in score_leaderboard.py.
# `all_off` is provided so you can see what an unconditioned building costs,
# and is deliberately not part of that set.
#
# There is no learned baseline, and no policy here requires torch.
POLICY_MAP: dict[str, type[Any]] = {
    "transfer": TransferPolicy,
    # The baseline set.
    "bang_bang": BangBangPolicy,
    "fixed_setpoint": FixedSetpointPolicy,
    "pid": PIDPolicy,
    "pid_water": PIDWaterResetPolicy,
    # Runnable reference, not part of the baseline set.
    "all_off": AllOffPolicy,
}

DEFAULT_SCENARIOS_DIR = Path("smart_buildings/training_scenarios")
DEFAULT_RESULTS_DIR = Path("smart_buildings/results")
DEFAULT_BUILDINGS_DIR = Path("smart_buildings/buildings")
DEFAULT_WEATHER_DIR = Path("smart_buildings/weather")


def generate_config_dirs(
    buildings: list[str], weathers: list[str], training_scenarios_dir: Path
) -> list[Path]:
    """Generate config.gin files for a given list of buildings and weathers."""
    config_dirs: list[Path] = []
    for building_path in buildings:
        for weather_path in weathers:
            scenario_config = load_scenario_from_parts(
                building_path, weather_path, output_base_dir=str(training_scenarios_dir)
            )
            result = generate_scenario_from_config(scenario_config)
            config_dirs.append(Path(result["output_dir"]))
    return config_dirs


def discover_config_dirs(
    training_scenarios_dir: Path,
) -> list[Path]:
    """Return scenario output directories containing generated config.gin files."""
    config_dirs: list[Path] = []

    if not training_scenarios_dir.exists():
        raise FileNotFoundError(
            f"No scenario directory found: {training_scenarios_dir.resolve()}"
        )

    for building_dir in sorted(training_scenarios_dir.iterdir()):
        if not building_dir.is_dir():
            continue
        for weather_dir in sorted(building_dir.iterdir()):
            if not weather_dir.is_dir():
                continue
            if (weather_dir / "config.gin").exists():
                config_dirs.append(weather_dir)

    return config_dirs


def expand_building_patterns(
    building_patterns: Sequence[str] | None,
) -> set[str] | None:
    """Expand building globs/paths into a set of building folder stems."""
    if not building_patterns:
        return None

    stems: set[str] = set()
    for pattern in building_patterns:
        matches = glob.glob(pattern)
        paths = matches if matches else ([pattern] if Path(pattern).exists() else [])
        if not matches and not paths:
            stem = Path(pattern).stem
            if stem:
                stems.add(stem)
            continue
        for path in paths:
            stems.add(Path(path).stem)

    if not stems:
        raise FileNotFoundError(
            f"No building configs matched: {list(building_patterns)}"
        )
    return stems


def filter_config_dirs(
    config_dirs: Sequence[Path],
    building_stems: set[str] | None = None,
    weathers: Sequence[str] | None = None,
) -> list[Path]:
    """Filter discovered scenario dirs by building folder name and/or weather name."""
    filtered: list[Path] = []
    weather_set = set(weathers) if weathers else None

    for config_dir in config_dirs:
        building_name = config_dir.parent.name
        weather_name = config_dir.name

        if building_stems is not None and building_name not in building_stems:
            continue
        if weather_set is not None and weather_name not in weather_set:
            continue
        filtered.append(config_dir)

    return filtered


def resolve_config_dirs(
    scenarios_dir: Path,
    buildings: Sequence[str] | None = None,
    weathers: Sequence[str] | None = None,
    generate_scenarios: bool = False,
) -> list[Path]:
    """Discover (or generate) scenario config dirs with optional filters."""
    if buildings in (None, [], "all", ["all"]):
        buildings = None
    if weathers in (None, [], "all", ["all"]):
        weathers = None

    if generate_scenarios:
        building_paths: list[str] = []
        if buildings:
            for pattern in buildings:
                matches = glob.glob(pattern)
                if matches:
                    building_paths.extend(matches)
                elif Path(pattern).exists():
                    building_paths.append(pattern)
                else:
                    candidate = DEFAULT_BUILDINGS_DIR / f"{Path(pattern).stem}.yaml"
                    if candidate.exists():
                        building_paths.append(str(candidate))
        else:
            building_paths = [
                str(DEFAULT_BUILDINGS_DIR / f.name)
                for f in sorted(DEFAULT_BUILDINGS_DIR.glob("*.yaml"))
            ]
        if not building_paths:
            raise FileNotFoundError(
                f"No building configs matched: {list(buildings or [])}"
            )
        building_paths = sorted(set(building_paths))

        if weathers:
            weather_paths: list[str] = []
            for weather in weathers:
                if Path(weather).exists():
                    weather_paths.append(weather)
                else:
                    candidate = DEFAULT_WEATHER_DIR / f"{weather}.yaml"
                    if candidate.exists():
                        weather_paths.append(str(candidate))
                    else:
                        weather_paths.append(weather)
        else:
            weather_paths = [
                str(DEFAULT_WEATHER_DIR / f"{name}.yaml")
                for name in sorted(f.stem for f in DEFAULT_WEATHER_DIR.glob("*.yaml"))
            ]

        config_dirs = generate_config_dirs(building_paths, weather_paths, scenarios_dir)
    else:
        config_dirs = discover_config_dirs(scenarios_dir)

    building_stems = expand_building_patterns(buildings)
    return filter_config_dirs(config_dirs, building_stems, weathers)


def write_comparison_json(
    scenario_results_dir: Path, summaries: Mapping[str, Mapping[str, Any]]
) -> None:
    """Write comparison.json for one scenario results directory."""
    scenario_results_dir.mkdir(parents=True, exist_ok=True)
    with open(scenario_results_dir / "comparison.json", "w") as f:
        json.dump(dict(summaries), f, indent=2)


def plot_config_dir(output_dir: Path) -> None:
    """Plot floor_plan.npy and zone_map.npy for a generated scenario."""
    floor_plan = np.load(output_dir / "floor_plan.npy")
    zone_map = np.load(output_dir / "zone_map.npy")

    print("Scenario:", output_dir)
    print("Floor plan shape:", floor_plan.shape)
    print("Zone map shape:", zone_map.shape)

    plt.imshow(floor_plan)
    plt.title("Floor Plan")
    plt.colorbar()
    plt.show()

    plt.imshow(zone_map)
    plt.title("Zone Map")
    plt.colorbar()
    plt.show()


def load_hybrid_environment(gin_config_file: Path):
    """Load a HybridActionEnvironment from a generated config.gin file."""
    gin.clear_config()
    with gin.unlock_config():
        gin.parse_config_file(str(gin_config_file))
    return hybrid_action_environment.HybridActionEnvironment()


# Setpoints that belong to one zone rather than to the plant. These are indexed
# by device rather than pooled into a list, because a zone policy needs the
# entry for one particular VAV and the plant lists are positional.
ZONE_ACTION_FIELDS = (
    "supply_air_damper_percentage_command",
    "reheat_valve_percentage_command",
)


def get_action_indices(env) -> dict[str, Any]:
    """Return continuous-action indices grouped by the role of the setpoint.

    The plant roles are positional lists, in AHU order. The zone role is a
    mapping of device id to setpoint name to index, because a zone policy needs
    the entry for one particular VAV and has no order to count along.
    """
    temp_indices: list[int] = []
    pressure_indices: list[int] = []
    water_indices: list[int] = []
    recirculation_indices: list[int] = []
    zone_action_indices: dict[str, dict[str, int]] = {}

    continuous_idx = 0
    for device, action_name in env._device_action_tuples:
        if "run_command" in action_name:
            continue

        if "supply_air_temperature" in action_name:
            temp_indices.append(continuous_idx)
        elif "supply_water_temperature_setpoint" in action_name:
            water_indices.append(continuous_idx)
        elif action_name in ZONE_ACTION_FIELDS:
            zone_action_indices.setdefault(device, {})[action_name] = continuous_idx
        elif "recirculation" in action_name:
            recirculation_indices.append(continuous_idx)
        elif "pressure" in action_name:
            pressure_indices.append(continuous_idx)
        else:
            # Keep unknown continuous actions controllable as pressure-like values.
            pressure_indices.append(continuous_idx)

        continuous_idx += 1

    return {
        "temp_indices": temp_indices,
        "pressure_indices": pressure_indices,
        "water_indices": water_indices,
        "recirculation_indices": recirculation_indices,
        "zone_action_indices": zone_action_indices,
    }


def get_ahu_zone_assignments(
    env,
    rename_rooms_to_vavs: bool = False,
) -> dict[str, list[str]]:
    """Return AHU -> rooms/VAVs mapping.

    In the generated buildings, each room has one VAV, so room IDs and VAV IDs
    are equivalent for metadata purposes.

    If rename_rooms_to_vavs=False:
        {"ahu_1": ["room_1", "room_2", ...]}

    If rename_rooms_to_vavs=True:
        {"ahu_1": ["VAV_1", "VAV_2", ...]}
    """
    ahs = env.building.simulator._hvac.air_handler
    ahu_zone_assignments: dict[str, list[str]] = {}

    for ahu, rooms in ahs.ahu_zones_map.items():
        ahu_id = ahu.device_id()
        if rename_rooms_to_vavs:
            ahu_zone_assignments[ahu_id] = [
                room.replace("room_", "VAV_") for room in rooms
            ]
        else:
            ahu_zone_assignments[ahu_id] = list(rooms)

    return ahu_zone_assignments


def infer_num_floors(output_dir: Path) -> int:
    """Best-effort floor count from generated config.yaml, if available."""
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        return 1

    try:
        import yaml
    except ImportError:
        return 1

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return int(cfg.get("floor_plan", {}).get("num_floors", 1))
    except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError) as e:
        # Best effort, but do not swallow a broken config silently: a scenario
        # that reports one floor when it has several renders wrong.
        print(f"warning: could not read num_floors from {config_path}: {e}")
        return 1


def get_observation_normalization_metadata(env):
    constants = env.observation_normalizer._normalization_constants
    metadata = {}
    for key, value in constants.items():
        metadata[key] = {
            "sample_mean": value.sample_mean,
            "sample_variance": value.sample_variance,
            "sample_std": float(np.sqrt(value.sample_variance)),
        }
    return metadata


def get_action_normalization_metadata(env):
    metadata = {}
    for name, n in env.action_normalizers.items():
        metadata[str(name)] = {
            "min_native_value": float(n.min_native_value),
            "max_native_value": float(n.max_native_value),
            "min_normalized_value": float(n.min_normalized_value),
            "max_normalized_value": float(n.max_normalized_value),
        }
    return metadata


def create_building_meta_data(env, output_dir: Path) -> dict[str, Any]:
    """Create participant-facing static metadata for one loaded environment."""
    action_spec = env.action_spec()
    n_discrete = int(action_spec["discrete_action"].shape[0])
    n_continuous = int(action_spec["continuous_action"].shape[0])

    action_indices = get_action_indices(env)

    field_names = list(env.field_names)
    zone_temp_indices = [
        i for i, name in enumerate(field_names) if "zone_air_temperature" in name
    ]

    floor_plan_path = output_dir / "floor_plan.npy"
    zone_map_path = output_dir / "zone_map.npy"
    floor_plan = np.load(floor_plan_path) if floor_plan_path.exists() else None
    zone_map = np.load(zone_map_path) if zone_map_path.exists() else None

    ahu_zone_assignments = get_ahu_zone_assignments(
        env,
        rename_rooms_to_vavs=False,
    )

    num_floors = infer_num_floors(output_dir)
    schedule = env.building.simulator._hvac.schedule

    observation_normalization_metadata = get_observation_normalization_metadata(env)
    action_normalization_metadata = get_action_normalization_metadata(env)

    # Keep both names: `ahu_zone_assignments` is the correct name; `aha_*` is
    # included because some scratch code used that typo.
    return {
        "n_discrete_action": n_discrete,
        "n_continuous_action": n_continuous,
        **action_indices,
        "zone_temp_indices": zone_temp_indices,
        "observation_field_names": field_names,
        "floor_plan": floor_plan,
        "zone_map": zone_map,
        "ahu_zone_assignments": ahu_zone_assignments,
        "num_floors": num_floors,
        "morning_start_hour": schedule.morning_start_hour,
        "evening_start_hour": schedule.evening_start_hour,
        "comfort_temp_window": schedule.comfort_temp_window,
        "eco_temp_window": schedule.eco_temp_window,
        "time_zone": schedule._time_zone,
        "holidays": schedule.holidays,
        "observation_normalization_metadata": observation_normalization_metadata,
        "action_normalization_metadata": action_normalization_metadata,
    }


def initialize_policy(
    policy_class: type[Any], building_meta_data: Mapping[str, Any], action_spec
):
    """Initialize a policy class with the new competition-style interface.

    Preferred constructor:
        PolicyClass(building_meta_data, action_spec)

    Fallbacks let policies with a simpler constructor signature still load.
    """
    try:
        return policy_class(building_meta_data, action_spec)
    except TypeError:
        pass

    try:
        return policy_class(building_meta_data)
    except TypeError:
        pass

    return policy_class()


def get_policy_name(policy: Any, fallback: str) -> str:
    """Return a readable policy name."""
    name = getattr(policy, "name", fallback)
    if callable(name):
        return str(name())
    return str(name)


def get_policy_action(policy: Any, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Call a policy and return an action dict."""
    if hasattr(policy, "get_action"):
        return policy.get_action(obs)
    if callable(policy):
        return policy(obs)
    raise TypeError("Policy must define get_action(obs) or be callable.")


def validate_action(action: Mapping[str, Any], action_spec) -> dict[str, np.ndarray]:
    """Ensure a policy action exactly matches the hybrid action spec dtypes."""
    continuous = np.asarray(action["continuous_action"], dtype=np.float32)
    discrete = np.asarray(action["discrete_action"], dtype=np.int32)

    expected_continuous_shape = tuple(action_spec["continuous_action"].shape)
    expected_discrete_shape = tuple(action_spec["discrete_action"].shape)

    if tuple(continuous.shape) != expected_continuous_shape:
        raise ValueError(
            f"continuous_action has shape {continuous.shape}, "
            f"expected {expected_continuous_shape}"
        )
    if tuple(discrete.shape) != expected_discrete_shape:
        raise ValueError(
            f"discrete_action has shape {discrete.shape}, expected {expected_discrete_shape}"
        )

    return {
        "continuous_action": continuous,
        "discrete_action": discrete,
    }


def run_policy(
    env, tracker: SimulationTracker, policy: Any, num_steps: int
) -> dict[str, Any]:
    """Run one policy using SimulationTracker and return compact metrics."""
    tracker.reset()
    obs = tracker.last_obs
    step_times: list[float] = []

    for _ in range(num_steps):
        raw_action = get_policy_action(policy, obs)
        action = validate_action(raw_action, env.action_spec())

        t0 = time.time()
        obs, _, terminated, truncated, _ = tracker.step(action)
        step_times.append(time.time() - t0)

        if terminated or truncated:
            break

    rewards = tracker.rewards
    return {
        "num_steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "seconds_per_step": float(np.mean(step_times)) if step_times else 0.0,
    }


def save_data(
    tracker: SimulationTracker, scenario_results_dir: Path, policy_name: str
) -> dict[str, Any]:
    """Save tracker data to CSV/JSON files."""
    policy_dir = scenario_results_dir / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in tracker.timestamps],
            "reward": tracker.rewards,
            "occupancy": tracker.occupancy,
        }
    )

    zone_temps_arr = np.asarray(tracker.zone_temps)
    if zone_temps_arr.ndim == 2 and zone_temps_arr.shape[0] > 0:
        zone_temp_df = pd.DataFrame(
            zone_temps_arr,
            columns=[f"zone_{i}_temp" for i in range(zone_temps_arr.shape[1])],
        )
        df = pd.concat([df, zone_temp_df], axis=1)

    df.to_csv(policy_dir / "timeseries.csv", index=False)

    if tracker.reward_components:
        rc_df = pd.DataFrame(tracker.reward_components)
        rc_df["timestamp"] = [ts.isoformat() for ts in tracker.timestamps]
        rc_df.to_csv(policy_dir / "reward_components.csv", index=False)

    if tracker.ahu_log:
        ahu_df = pd.DataFrame(tracker.ahu_log)
        for col in list(ahu_df.columns):
            first_value = ahu_df[col].iloc[0]
            if isinstance(first_value, list):
                list_len = len(first_value)
                for i in range(list_len):
                    ahu_df[f"{col}_{i}"] = ahu_df[col].apply(
                        lambda x, idx=i: x[idx] if idx < len(x) else None
                    )
                ahu_df = ahu_df.drop(columns=[col])
        ahu_df.to_csv(policy_dir / "ahu_log.csv", index=False)

    rewards = [float(x) for x in tracker.rewards]
    occupancy = [float(x) for x in tracker.occupancy]

    summary: dict[str, Any] = {
        "policy": policy_name,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "total_steps": len(rewards),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_occupancy": float(np.mean(occupancy)) if occupancy else 0.0,
        "avg_zone_temp": (
            float(np.mean([np.mean(t) for t in tracker.zone_temps]))
            if tracker.zone_temps
            else 0.0
        ),
        "productivity_weight": float(tracker.productivity_weight),
        "energy_weight": float(tracker.energy_weight),
        "carbon_weight": float(tracker.carbon_weight),
    }

    if tracker.reward_components:
        summary["total_normalized_productivity_regret"] = float(
            sum(
                r.get("normalized_productivity_regret", 0)
                for r in tracker.reward_components
            )
        )
        summary["total_normalized_energy_cost"] = float(
            sum(r.get("normalized_energy_cost", 0) for r in tracker.reward_components)
        )
        summary["total_normalized_carbon_emission"] = float(
            sum(
                r.get("normalized_carbon_emission", 0)
                for r in tracker.reward_components
            )
        )

    with open(policy_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def save_charts(
    tracker: SimulationTracker, scenario_results_dir: Path, policy_name: str
) -> None:
    """Save tracker charts as PNG files."""
    import matplotlib

    matplotlib.use("Agg")
    policy_dir = scenario_results_dir / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)

    original_show = plt.show

    def save_show():
        pass

    plt.show = save_show
    try:
        tracker.plot_overview(color_by_ahu=True)
        plt.savefig(policy_dir / "overview.png", dpi=150, bbox_inches="tight")
        plt.close()

        tracker.plot_rewards()
        plt.savefig(policy_dir / "rewards.png", dpi=150, bbox_inches="tight")
        plt.close()

        tracker.plot_ahu()
        plt.savefig(policy_dir / "ahu.png", dpi=150, bbox_inches="tight")
        plt.close()
    finally:
        plt.show = original_show


def evaluate_policy_for_config(
    scenario_results_dir: Path,
    input_dir: Path,
    policy_key: str,
    policy_class: type[Any],
    num_steps: int,
    save_video: bool = False,
) -> dict[str, Any]:
    """Evaluate one policy on one generated config directory."""
    gin_config_path = input_dir / "config.gin"
    if not gin_config_path.exists():
        raise FileNotFoundError(f"Missing config.gin: {gin_config_path}")

    env = load_hybrid_environment(gin_config_path)
    # Only retain per-step temperature grids when a video will actually be
    # written. The buffer is a full float grid per step and reaches tens of GB
    # over a 30-day episode, which OOMs long headless runs for no benefit.
    tracker = SimulationTracker(
        env, vmin=280, vmax=310, capture_snapshots=save_video
    )

    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0

    building_meta_data = create_building_meta_data(env, input_dir)
    action_spec = env.action_spec()

    policy = initialize_policy(policy_class, building_meta_data, action_spec)
    policy_name = get_policy_name(policy, policy_key)

    compact_summary = run_policy(env, tracker, policy, num_steps)

    if save_video:
        video_path = scenario_results_dir / policy_name / "simulation.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Creating video: {video_path}")
        tracker.create_video(str(video_path), fps=10)

    print("Saving charts...")
    save_charts(tracker, scenario_results_dir, policy_name)

    print("Saving data...")
    summary = save_data(tracker, scenario_results_dir, policy_name)
    summary.update(
        {
            "seconds_per_step": compact_summary["seconds_per_step"],
            "n_discrete_action": building_meta_data["n_discrete_action"],
            "n_continuous_action": building_meta_data["n_continuous_action"],
            "num_zone_temp_features": len(building_meta_data["zone_temp_indices"]),
            "num_ahus": len(building_meta_data["ahu_zone_assignments"]),
        }
    )

    with open(scenario_results_dir / policy_name / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def evaluate_config_dir(
    results_dir: Path,
    input_dir: Path,
    policy_keys: Iterable[str],
    num_steps: int,
    save_video: bool = False,
) -> dict[str, Any]:
    """Evaluate selected policies on one generated scenario directory."""
    print("\n" + "=" * 80)
    print("Scenario:", input_dir)
    print("Config:", input_dir / "config.gin")
    print("=" * 80)
    scenario_results_dir = results_dir / input_dir.parent.name / input_dir.name
    scenario_results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[str, Any] = {}
    for policy_key in policy_keys:
        if policy_key not in POLICY_MAP:
            raise ValueError(
                f"Unknown policy key: {policy_key}. Options: {POLICY_MAP.keys()}"
            )

        print(f"Running {policy_key}...")
        summary = evaluate_policy_for_config(
            scenario_results_dir=scenario_results_dir,
            input_dir=input_dir,
            policy_key=policy_key,
            policy_class=POLICY_MAP[policy_key],
            num_steps=num_steps,
            save_video=save_video,
        )
        all_summaries[policy_key] = summary
        print(f"Total reward: {summary['total_reward']:.2f}")

    write_comparison_json(scenario_results_dir, all_summaries)

    return all_summaries


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for participant policy evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HVAC control policies on generated smart-building scenarios. "
            "Ships with a trivial 'default' policy; register your own in POLICY_MAP."
        ),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["transfer"],
        choices=sorted(POLICY_MAP.keys()),
        help="Policy keys to evaluate (default: transfer, the editable template)",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=DEFAULT_SCENARIOS_DIR,
        help="Directory containing generated training scenarios",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--buildings",
        nargs="*",
        default=None,
        help="Building YAML paths, globs, or folder stems (default: all)",
    )
    parser.add_argument(
        "--weathers",
        nargs="*",
        default=None,
        help="Weather preset names (default: all discovered scenarios)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8640,
        help=(
            "Simulation steps per policy (default: 8640 = the full 30-day "
            "episode, which is the length the shipped baseline_returns.json "
            "was run at). Pass a shorter horizon only for a smoke test: it "
            "skips eco-setback periods, flatters schedule-aware policies, and "
            "the return is not comparable to the baselines."
        ),
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save simulation video for each policy run",
    )
    parser.add_argument(
        "--generate-scenarios",
        action="store_true",
        help="Generate scenarios before evaluation (default: use existing training_scenarios/)",
    )
    parser.add_argument(
        "--plot-one-config",
        action="store_true",
        help="Plot floor plan and zone map for the first scenario and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    config_dirs = resolve_config_dirs(
        scenarios_dir=args.scenarios_dir,
        buildings=args.buildings,
        weathers=args.weathers,
        generate_scenarios=args.generate_scenarios,
    )
    if not config_dirs:
        raise RuntimeError("No scenario config directories matched the filters.")

    print(f"Found {len(config_dirs)} scenario config directories.")

    if args.plot_one_config:
        plot_config_dir(config_dirs[0])
        return

    all_results: dict[str, Any] = {}
    for input_dir in config_dirs:
        scenario_key = str(Path(input_dir.parent.name) / input_dir.name)
        all_results[scenario_key] = evaluate_config_dir(
            results_dir=args.results_dir,
            input_dir=input_dir,
            policy_keys=args.policies,
            num_steps=args.steps,
            save_video=args.save_video,
        )

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for scenario_name, summaries in all_results.items():
        print(f"\n{scenario_name}:")
        for policy, summary in summaries.items():
            print(f"  {policy:12s}: reward={summary['total_reward']:8.2f}")

    print(f"\nAll results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
