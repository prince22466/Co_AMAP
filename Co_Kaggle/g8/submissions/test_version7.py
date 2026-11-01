"""Version 7 synthetic contract, physics and runtime tests.

Executes the exact notebook policy cell with a stub Kaggle server.
These tests do not substitute for the 30-scenario official score.
"""
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np


ROOT = Path(__file__).parent
notebook = json.loads((ROOT / "submission-nb - Version 7.ipynb").read_text())
assert len(notebook["cells"]) == 3
source = "".join(notebook["cells"][1]["source"])
compile(source, "<version7 notebook policy>", "exec")
compile("".join(notebook["cells"][2]["source"]), "<version7 entrypoint>", "exec")

package = types.ModuleType("kaggle_evaluation")
gateway = types.ModuleType("smart_buildings_inference_server")
gateway.Policy = type("Policy", (), {"__init__": lambda self, meta, spec: None})
package.smart_buildings_inference_server = gateway
sys.modules["kaggle_evaluation"] = package
sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = gateway
namespace = {}
exec(source, namespace)
Policy = namespace["TransferPolicy"]


def scenario(ahu_count=2, zones_each=4, temp=295.0, outdoor=298.0):
    """Construct metadata with scattered AHU/VAV indices and unit names."""
    names = ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000"]
    assignments = {}
    zone_actions = {}
    total_zones = ahu_count * zones_each
    action_base = 3 * ahu_count + 2
    for ahu in range(ahu_count):
        key = f"ahu_{ahu}"
        assignments[key] = []
        names.append(f"ahs_{key}_outside_air_temperature_sensor")
        for local in range(zones_each):
            i = ahu * zones_each + local
            assignments[key].append(f"room_{i}")
            vav = f"VAV_{i}"
            names.extend([
                vav + "_zone_air_temperature_sensor",
                vav + "_zone_air_heating_temperature_setpoint_sensor",
                vav + "_zone_air_cooling_temperature_setpoint_sensor",
            ])
            zone_actions[vav] = {
                "supply_air_damper_percentage_command": action_base + i * 2,
                "reheat_valve_percentage_command": action_base + i * 2 + 1,
            }
    meta = {
        "observation_field_names": names,
        "n_continuous_action": 3 * ahu_count + 2 + 2 * total_zones,
        "n_discrete_action": ahu_count + 1,
        "temp_indices": [3 * i for i in range(ahu_count)],
        "pressure_indices": [3 * i + 1 for i in range(ahu_count)] + [3 * ahu_count + 1],
        "recirculation_indices": [3 * i + 2 for i in range(ahu_count)],
        "water_indices": [3 * ahu_count],
        "ahu_zone_assignments": assignments,
        "zone_action_indices": zone_actions,
        "comfort_temp_window": [293., 297.],
        "eco_temp_window": [288., 302.],
        "morning_start_hour": 8.,
        "evening_start_hour": 18.,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 295., "sample_std": 5.}
        },
    }
    obs = np.zeros(len(names), dtype=np.float64)
    obs[names.index("hod_sin_000")] = math.sin(2 * math.pi * 12 / 24)
    obs[names.index("hod_cos_000")] = math.cos(2 * math.pi * 12 / 24)
    obs[names.index("dow_sin_000")] = math.sin(2 * math.pi * 2 / 7)
    obs[names.index("dow_cos_000")] = math.cos(2 * math.pi * 2 / 7)
    for i, field in enumerate(names):
        if field.endswith("_zone_air_temperature_sensor"):
            obs[i] = (temp - 295.) / 5.
        elif field.endswith("_zone_air_heating_temperature_setpoint_sensor"):
            obs[i] = 293.
        elif field.endswith("_zone_air_cooling_temperature_setpoint_sensor"):
            obs[i] = 297.
        elif field.endswith("_outside_air_temperature_sensor"):
            obs[i] = outdoor

    def set_zone(index, kelvin):
        obs[names.index(f"VAV_{index}_zone_air_temperature_sensor")] = (kelvin - 295.) / 5.

    def set_outside(kelvin):
        for field in names:
            if field.endswith("outside_air_temperature_sensor"):
                obs[names.index(field)] = kelvin

    return meta, obs, set_zone, set_outside


def check(policy, obs):
    """Every call must obey the exact hybrid-action interface."""
    result = policy.get_action(obs)
    c, d = result["continuous_action"], result["discrete_action"]
    assert c.shape == (policy.n_continuous,) and c.dtype == np.float32
    assert d.shape == (policy.n_discrete,) and d.dtype == np.int32
    assert np.isfinite(c).all() and (np.abs(c) <= 1.00001).all()
    assert np.isin(d, [0, 1]).all()
    return c, d


# Different building sizes and both heat/cool/mixed demand.
for shape in [(1, 1), (2, 4), (4, 7)]:
    meta, obs, set_zone, set_outside = scenario(*shape)
    policy = Policy(meta, None)
    c, d = check(policy, obs)
    assert not np.any(d), "AHU and pump should be off inside the comfort band"
    assert np.all(c[meta["temp_indices"]] == 1.), "Idle coils should not call for cooling"
    set_zone(0, 305.)
    c, d = check(policy, obs)
    assert d[0] == 1 and d[-1] == 0, "Only hot-zone cooling"
    assert c[meta["temp_indices"][0]] < 1., "Cooling coil setpoint should decrease"
    set_zone(0, 285.)
    c, d = check(policy, obs)
    assert d[0] == 1 and d[-1] == 1, "Cold zone requires AHU and central water"
    assert c[meta["temp_indices"][0]] == 1., "AHU cannot directly heat"
    if shape[1] > 1:
        set_zone(0, 305.)
        set_zone(1, 286.)
        c, d = check(policy, obs)
        assert d[0] == d[-1] == 1, "Mixed load must retain VAV reheat"
        assert c[meta["zone_action_indices"]["VAV_1"]["reheat_valve_percentage_command"]] > -1
    policy.reset()
    c_after, d_after = check(policy, obs)
    assert np.array_equal(c, c_after) and np.array_equal(d, d_after)


# Free cooling must use less recirculation at moderate outdoor temperature;
# with extreme cold avoid drawing more outdoor air than the floor requires.
meta, obs, set_zone, set_outside = scenario(1, 4)
set_zone(0, 305.)
policy = Policy(meta, None)
set_outside(292.)
free_c, _ = check(policy, obs)
set_outside(309.)
hot_c, _ = check(policy, obs)
set_outside(235.)
cold_c, _ = check(policy, obs)
index = meta["recirculation_indices"][0]
assert free_c[index] < hot_c[index], "Use more outside air when useful for cooling"
assert np.isclose(cold_c[index], hot_c[index]), "Protect against subfreezing free-cool"


# One invalid zone must not invalidate a whole AHU. Missing/invalid setpoints
# should fall back to public schedule limits.
meta, obs, set_zone, _ = scenario(2, 4)
meta["observation_field_names"][meta["observation_field_names"].index(
    "VAV_1_zone_air_heating_temperature_setpoint_sensor")] = "missing_heat"
policy = Policy(meta, None)
set_zone(1, float("nan"))
check(policy, obs)
set_zone(1, 284.)
c, d = check(policy, obs)
assert d[0] == 1 and d[-1] == 1


# Complexity smoke: no nested candidate search or linear growth in history.
meta, obs, set_zone, _ = scenario(4, 8)
policy = Policy(meta, None)
start = time.perf_counter()
for step in range(128):
    set_zone(0, 302. - 0.02 * step)
    check(policy, obs)
duration = time.perf_counter() - start
assert duration < 10., f"Too slow: {duration:.3f}s for 128 steps"
print(f"PASS: notebook syntax, dynamic shapes, heat/cool/mixed, free-cooling, "
      f"sensor failure, reset and O(zones) latency: {duration:.3f}s/128")
