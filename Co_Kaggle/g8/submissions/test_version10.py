"""Run exact notebook source with a stub Kaggle gateway (not a reward test)."""
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np

PATH = Path(__file__).with_name("submission-nb - Version 10.ipynb")
nb = json.loads(PATH.read_text(encoding="utf-8"))
assert len(nb["cells"]) == 3
source = "".join(nb["cells"][1]["source"])
entry = "".join(nb["cells"][2]["source"])
compile(source, "<V10 policy>", "exec")
compile(entry, "<V10 gateway>", "exec")

package = types.ModuleType("kaggle_evaluation")
gateway = types.ModuleType("smart_buildings_inference_server")
gateway.Policy = type("Policy", (), {"__init__": lambda self, meta, spec: None})
package.smart_buildings_inference_server = gateway
sys.modules["kaggle_evaluation"] = package
sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = gateway
namespace = {}
exec(source, namespace)
Policy = namespace["TransferPolicy"]


def scenario(ahu_count=2, zones_each=3, use_flow=True):
    """Exercise dynamically scattered action indices and optional AHU sensors."""
    names = ["comfort_mode_now"]
    assignments = {}
    zone_actions = {}
    n = ahu_count * zones_each
    base = 3 * ahu_count + 2
    for a in range(ahu_count):
        assignments[f"ahu_{a}"] = []
        if use_flow:
            names.extend([f"ahs_ahu_{a}_supply_air_flowrate_sensor",
                          f"ahs_ahu_{a}_supply_air_flowrate_setpoint"])
        for j in range(zones_each):
            i = a * zones_each + j
            assignments[f"ahu_{a}"].append(f"room_{i}")
            vav = f"VAV_{i}"
            names.extend([vav + "_zone_air_temperature_sensor",
                          vav + "_zone_air_heating_temperature_setpoint_sensor",
                          vav + "_zone_air_cooling_temperature_setpoint_sensor"])
            zone_actions[vav] = {
                "supply_air_damper_percentage_command": base + 2*i,
                "reheat_valve_percentage_command": base + 2*i + 1,
            }
    meta = {
        "observation_field_names": names,
        "n_continuous_action": base + 2*n,
        "n_discrete_action": ahu_count + 1,
        "temp_indices": [3*a for a in range(ahu_count)],
        "pressure_indices": [3*a+1 for a in range(ahu_count)] + [3*ahu_count+1],
        "recirculation_indices": [3*a+2 for a in range(ahu_count)],
        "water_indices": [3*ahu_count],
        "ahu_zone_assignments": assignments,
        "zone_action_indices": zone_actions,
        "comfort_temp_window": [293., 297.],
        "eco_temp_window": [288., 302.],
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 295., "sample_std": 5.}
        },
    }
    obs = np.zeros(len(names))
    obs[0] = 1.
    for i, name in enumerate(names):
        if name.endswith("_zone_air_temperature_sensor"):
            obs[i] = 0.
        elif name.endswith("_zone_air_heating_temperature_setpoint_sensor"):
            obs[i] = 293.
        elif name.endswith("_zone_air_cooling_temperature_setpoint_sensor"):
            obs[i] = 297.
        elif name.endswith("_supply_air_flowrate_sensor"):
            obs[i] = 10.
        elif name.endswith("_supply_air_flowrate_setpoint"):
            obs[i] = 10.

    def set_temp(zone, kelvin):
        obs[names.index(f"VAV_{zone}_zone_air_temperature_sensor")] = (kelvin - 295.) / 5.

    def set_flow(a, sensor, target):
        obs[names.index(f"ahs_ahu_{a}_supply_air_flowrate_sensor")] = sensor
        obs[names.index(f"ahs_ahu_{a}_supply_air_flowrate_setpoint")] = target

    return meta, obs, set_temp, set_flow


def checked(policy, obs):
    result = policy.get_action(obs)
    assert set(result) == {"continuous_action", "discrete_action"}
    c, d = result["continuous_action"], result["discrete_action"]
    assert c.shape == (policy.n_continuous,) and c.dtype == np.float32
    assert d.shape == (policy.n_discrete,) and d.dtype == np.int32
    assert np.isfinite(c).all() and (np.abs(c) <= 1.00001).all()
    assert np.isin(d, (0, 1)).all()
    return c, d


for ahus, zones in [(1, 1), (2, 4), (4, 8)]:
    meta, obs, temp, flow = scenario(ahus, zones)
    policy = Policy(meta, None)
    c, d = checked(policy, obs)
    assert np.all(d == 0), "In-band is idle"
    temp(0, 306.)
    c, d = checked(policy, obs)
    assert d[0] == 1 and d[-1] == 0, "Hot-zone cooling"
    assert c[meta["temp_indices"][0]] < 1.0
    temp(0, 282.)
    c, d = checked(policy, obs)
    assert d[0] == 1 and d[-1] == 1, "Cold-zone VAV reheat"
    assert c[meta["temp_indices"][0]] == 1.0, "AHU cannot heat"
    if zones > 1:
        temp(0, 306.)
        temp(1, 282.)
        c, d = checked(policy, obs)
        assert d[0] == d[-1] == 1, "Mixed heating and cooling"
        reheat = meta["zone_action_indices"]["VAV_1"]["reheat_valve_percentage_command"]
        assert c[reheat] > -1.0
    policy.reset()
    before = checked(policy, obs)
    policy.reset()
    after = checked(policy, obs)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])

# The main innovation: mild demand gradually TRIMS fan pressure.
meta, obs, temp, flow = scenario(1, 3)
policy = Policy(meta, None)
temp(0, 297.7)  # 0.7 K cooling error, damper command unsaturated
pidx = meta["pressure_indices"][0]
values = []
for _ in range(5):
    c, d = checked(policy, obs)
    values.append((c[pidx] + 1.)*1000.)
assert all(values[k+1] < values[k] for k in range(4)), values

# Strong single-zone deficit RESPONDS faster, even when most zones are neutral.
temp(0, 313.)
c, d = checked(policy, obs)
assert (c[pidx] + 1)*1000 > values[-1] + 100
assert d[0] == 1

# Measured flow shortage also responds, but only if target is valid.
policy.reset()
temp(0, 297.7)
flow(0, 1., 10.)
c, d = checked(policy, obs)
assert (c[pidx] + 1)*1000 > policy.START_PRESSURE_PA
policy.reset()
flow(0, float("nan"), 10.)
c, d = checked(policy, obs)
assert (c[pidx] + 1)*1000 < policy.START_PRESSURE_PA

# Pressure does not silently leak from one building episode to the next.
policy.reset()
assert policy.pressure_pa == [policy.START_PRESSURE_PA]

# Absent optional fields, invalid temperatures, setpoint fallback.
meta, obs, temp, _ = scenario(2, 4, use_flow=False)
policy = Policy(meta, None)
temp(1, float("nan"))
checked(policy, obs)
temp(0, 303.)
c, d = checked(policy, obs)
assert d[0] == 1
heat_idx = meta["observation_field_names"].index(
    "VAV_0_zone_air_heating_temperature_setpoint_sensor")
obs[heat_idx] = float("nan")
checked(policy, obs)

# Time budget: linear per-step and no candidate search/model fitting.
meta, obs, temp, _ = scenario(6, 10)
policy = Policy(meta, None)
temp(0, 307.)
start = time.perf_counter()
for i in range(128):
    temp(0, 307. - (i % 10) * 0.1)
    checked(policy, obs)
duration = time.perf_counter() - start
assert duration < 10., duration

class FakeServer:
    def __init__(self, klass):
        assert klass is Policy
    def run(self):
        self.called = True

gateway.SmartBuildingsInferenceServer = FakeServer
exec(entry, namespace)
print(f"PASS V10: notebook/gateway, shapes, reset, mixed thermal loads, "
      f"pressure trim/respond, optional flow, invalid sensors, "
      f"128-step runtime={duration:.3f}s")
