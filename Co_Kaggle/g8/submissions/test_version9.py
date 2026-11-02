"""Synthetic contract and behavior checks for the submitted Version 9 notebook.

These tests execute the *exact* notebook code with a stub gateway. They do not
claim to measure comfort, energy, or leaderboard performance.
"""
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np


NOTEBOOK = Path(__file__).with_name("submission-nb - Version 9.ipynb")
notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert len(notebook["cells"]) == 3

policy_source = "".join(notebook["cells"][1]["source"])
entrypoint_source = "".join(notebook["cells"][2]["source"])
compile(policy_source, "<V9 policy>", "exec")
compile(entrypoint_source, "<V9 entrypoint>", "exec")

package = types.ModuleType("kaggle_evaluation")
gateway = types.ModuleType("smart_buildings_inference_server")
gateway.Policy = type("Policy", (), {"__init__": lambda self, meta, spec: None})
package.smart_buildings_inference_server = gateway
sys.modules["kaggle_evaluation"] = package
sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = gateway
namespace = {}
exec(policy_source, namespace)
Policy = namespace["TransferPolicy"]


def scenario(ahu_count=2, zones_each=3, kelvin=295.0, occupants=-125/126):
    """Construct disordered action indices and metadata-driven sensor fields."""
    names = ["comfort_mode_now", "comfort_mode_soon", "num_occupants"]
    assignments = {}
    zone_actions = {}
    count = ahu_count * zones_each
    action_offset = 3 * ahu_count + 2

    for ahu in range(ahu_count):
        assignments[f"ahu_{ahu}"] = []
        for local in range(zones_each):
            index = ahu * zones_each + local
            vav = f"VAV_{index}"
            assignments[f"ahu_{ahu}"].append(f"room_{index}")
            names.extend([
                vav + "_zone_air_temperature_sensor",
                vav + "_zone_air_heating_temperature_setpoint_sensor",
                vav + "_zone_air_cooling_temperature_setpoint_sensor",
            ])
            zone_actions[vav] = {
                "supply_air_damper_percentage_command": action_offset + 2*index,
                "reheat_valve_percentage_command": action_offset + 2*index + 1,
            }

    metadata = {
        "observation_field_names": names,
        "n_continuous_action": action_offset + 2*count,
        "n_discrete_action": ahu_count + 1,
        "temp_indices": [3*i for i in range(ahu_count)],
        "pressure_indices": [3*i + 1 for i in range(ahu_count)]
        + [3*ahu_count + 1],
        "recirculation_indices": [3*i + 2 for i in range(ahu_count)],
        "water_indices": [3*ahu_count],
        "ahu_zone_assignments": assignments,
        "zone_action_indices": zone_actions,
        "comfort_temp_window": [293.0, 297.0],
        "eco_temp_window": [288.0, 302.0],
        "morning_start_hour": 8.0,
        "evening_start_hour": 18.0,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {
                "sample_mean": 295.0, "sample_std": 5.0
            }
        },
    }
    observation = np.zeros(len(names), dtype=float)
    observation[names.index("num_occupants")] = occupants
    for i, field in enumerate(names):
        if field.endswith("zone_air_temperature_sensor"):
            observation[i] = (kelvin - 295.0) / 5.0
        elif field.endswith("heating_temperature_setpoint_sensor"):
            observation[i] = 293.0
        elif field.endswith("cooling_temperature_setpoint_sensor"):
            observation[i] = 297.0

    def set_zone(index, temperature):
        i = names.index(f"VAV_{index}_zone_air_temperature_sensor")
        observation[i] = (temperature - 295.0)/5.0

    def set_global(field, value):
        observation[names.index(field)] = value

    return metadata, observation, set_zone, set_global


def checked(policy, observation):
    result = policy.get_action(observation)
    assert set(result) == {"continuous_action", "discrete_action"}
    cont, disc = result["continuous_action"], result["discrete_action"]
    assert cont.shape == (policy.n_continuous,)
    assert disc.shape == (policy.n_discrete,)
    assert cont.dtype == np.float32 and disc.dtype == np.int32
    assert np.isfinite(cont).all() and (np.abs(cont) <= 1.00001).all()
    assert np.isin(disc, (0, 1)).all()
    return cont, disc


# Basic contract on different building sizes; all neutral -> off.
for shape in [(1, 1), (2, 3), (4, 9)]:
    meta, obs, set_zone, set_global = scenario(*shape)
    policy = Policy(meta, None)
    assert not checked(policy, obs)[1].any()
    set_zone(0, 305.0)
    continuous, discrete = checked(policy, obs)
    assert discrete[0] == 1 and discrete[-1] == 0
    assert not discrete[1:shape[0]].any()
    assert continuous[meta["temp_indices"][0]] < 1.0
    assert continuous[meta["recirculation_indices"][0]] == 1.0

    set_zone(0, 283.0)
    continuous, discrete = checked(policy, obs)
    assert discrete[0] == 1 and discrete[-1] == 1
    assert continuous[meta["temp_indices"][0]] == 1.0
    assert continuous[meta["water_indices"][0]] > -1.0

# Occupancy signal changes the threshold for *identical* mild violations.
meta, obs, set_zone, set_global = scenario(1, 2)
set_zone(0, 297.3)
assert checked(Policy(meta, None), obs)[1][0] == 0
set_global("num_occupants", -0.50)
assert checked(Policy(meta, None), obs)[1][0] == 1
# Occupancy is a global building signal; do not claim to identify zone occupancy.
set_global("num_occupants", -125/126)
set_global("comfort_mode_now", 1.0)
assert checked(Policy(meta, None), obs)[1][0] == 1

# Comfort-mode-now / soon schedule provides fallback with missing live setpoints.
meta, obs, set_zone, set_global = scenario(1, 2)
meta["observation_field_names"][
    meta["observation_field_names"].index(
        "VAV_0_zone_air_heating_temperature_setpoint_sensor"
    )
] = "missing_heat"
policy = Policy(meta, None)
set_zone(0, 290.0)  # inside eco but below comfort
assert checked(policy, obs)[1][0] == 0
set_global("comfort_mode_now", 1.0)
assert checked(policy, obs)[1][-1] == 1

# Mixed hot/cold on a shared AHU must preserve independent reheat commands.
meta, obs, set_zone, set_global = scenario(1, 2)
set_zone(0, 307.0)
set_zone(1, 283.0)
cont, disc = checked(Policy(meta, None), obs)
valve = meta["zone_action_indices"]["VAV_1"]["reheat_valve_percentage_command"]
assert disc[0] == disc[-1] == 1 and cont[valve] > -1.0

# Severe temperature violation overrides the absence of occupancy.
meta, obs, set_zone, set_global = scenario(1, 1)
set_zone(0, 301.2)
assert checked(Policy(meta, None), obs)[1][0] == 1

# Unreliable sensor in one zone cannot poison all outputs.
meta, obs, set_zone, set_global = scenario(2, 3)
set_zone(0, float("nan"))
set_zone(1, 308.0)
assert checked(Policy(meta, None), obs)[1][0] == 1
set_zone(1, float("nan"))
set_zone(2, float("nan"))
assert checked(Policy(meta, None), obs)[1][0] == 0
set_global("num_occupants", float("nan"))
assert not checked(Policy(meta, None), obs)[1].any()

# Reset is safe; arbitrary-length sequences do not accumulate state.
meta, obs, set_zone, set_global = scenario(4, 9)
policy = Policy(meta, None)
begin = time.perf_counter()
for i in range(128):
    set_zone(0, 300.0 + 0.02*i)
    checked(policy, obs)
elapsed = time.perf_counter() - begin
assert elapsed < 10.0, f"128 actions too slow: {elapsed:.2f}s"
assert policy.reset() is None
assert not hasattr(policy, "history")

class FakeInferenceServer:
    def __init__(self, cls):
        assert cls is Policy
    def run(self):
        pass

gateway.SmartBuildingsInferenceServer = FakeInferenceServer
exec(entrypoint_source, namespace)
print(f"PASS: V9 contract, occupancy, fallback, mixed load, NaN, reset, runtime {elapsed:.3f}s/128")
