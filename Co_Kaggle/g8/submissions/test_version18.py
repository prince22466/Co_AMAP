"""Version 18 notebook interface and disturbance-observer regression tests.

This script executes the EXACT notebook policy cell with a stub of the Kaggle
gateway. It checks causality, thermal-residual signs, guards, and control
outputs, but it cannot establish simulator comfort/energy performance.
"""
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np


ROOT = Path(__file__).parent
notebook = json.loads((ROOT / "submission-nb - Version 18.ipynb").read_text())
assert len(notebook["cells"]) == 3
source = "".join(notebook["cells"][1]["source"])
compile(source, "<version18 notebook policy>", "exec")
compile("".join(notebook["cells"][2]["source"]), "<version18 entrypoint>", "exec")

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



# 1. Action contract: temperatures, mixed loads, and changing building size.
for shape in [(1, 1), (2, 4), (4, 7)]:
    meta, obs, set_zone, set_outdoor = scenario(*shape)
    policy = Policy(meta, None)
    c, d = check(policy, obs)
    assert not d.any(), "No conditioning for steady in-band temperatures"
    set_zone(0, 305.)
    c, d = check(policy, obs)
    assert d[0] == 1 and c[meta["temp_indices"][0]] < 1.
    policy.reset()
    set_zone(0, 285.)
    c, d = check(policy, obs)
    assert d[0] == d[-1] == 1 and c[meta["temp_indices"][0]] == 1.
    if shape[1] > 1:
        policy.reset()
        set_zone(0, 305.)
        set_zone(1, 286.)
        c, d = check(policy, obs)
        assert d[0] == d[-1] == 1
        assert c[meta["zone_action_indices"]["VAV_1"]["reheat_valve_percentage_command"]] > -1.

# 2. Positive hidden heat gain, before the cooling deadband is breached:
# a true DOB must respond earlier than the exact same policy without observer.
meta, obs, set_zone, _ = scenario(1, 1, temp=296.8, outdoor=296.8)
dob, ablation = Policy(meta, None), Policy(meta, None)
ablation.DISABLE_OBSERVER = True
for temp in [296.8, 296.95, 297.10]:
    set_zone(0, temp)
    cd, dd = check(dob, obs)
    cb, db = check(ablation, obs)
key = dob.ahus[0]["zones"][0]["temperature"]
assert dob.disturbance_k_per_step[key] > 0.0
assert dd[0] == 1 and db[0] == 0, "Positive residual must advance cooling"
assert cd[meta["temp_indices"][0]] < cb[meta["temp_indices"][0]]

# 3. Negative disturbance should advance heating, without guessing a
# physical sign from a simple temperature derivative.
meta, obs, set_zone, _ = scenario(1, 1, temp=293.2, outdoor=293.2)
dob, ablation = Policy(meta, None), Policy(meta, None)
ablation.DISABLE_OBSERVER = True
for temp in [293.2, 293.05, 292.90]:
    set_zone(0, temp)
    _, dd = check(dob, obs)
    _, db = check(ablation, obs)
key = dob.ahus[0]["zones"][0]["temperature"]
assert dob.disturbance_k_per_step[key] < 0.0
assert dd[0] == dd[-1] == 1 and db[0] == 0 and db[-1] == 0

# 4. Action-conditioned innovation differs from raw trend. With the same
# positive measured delta, turning on reheat makes the residual *smaller*.
meta, obs, set_zone, _ = scenario(1, 1, temp=292., outdoor=292.)
p1, p2 = Policy(meta, None), Policy(meta, None)
set_zone(0, 292.)
check(p1, obs)
check(p2, obs)
key = p1.ahus[0]["zones"][0]["temperature"]
p1.previous_zone[key]["valve"] = 1.0
p2.previous_zone[key]["valve"] = 0.0
set_zone(0, 292.08)
check(p1, obs)
check(p2, obs)
assert p1.disturbance_k_per_step[key] < p2.disturbance_k_per_step[key]

# 5. Disturbance clamps and outlier rejection; an unrealistic jump cannot
# poison the observer and there is never an unbounded integral windup.
meta, obs, set_zone, _ = scenario(1, 1, temp=295., outdoor=295.)
policy = Policy(meta, None)
check(policy, obs)
key = policy.ahus[0]["zones"][0]["temperature"]  # Never reuse another scenario's key.
set_zone(0, 310.)
check(policy, obs)
assert key not in policy.disturbance_k_per_step, "15 K spike must be ignored"
assert key not in policy.previous_zone, "A jumped sensor must not seed the next model"
policy.reset()
set_zone(0, 295.)
check(policy, obs)
for _ in range(100):
    set_zone(0, 295.1 if _ % 2 else 295.0)
    check(policy, obs)
assert all(abs(d) <= policy.DISTURBANCE_LIMIT_K_PER_STEP + 1e-8
           for d in policy.disturbance_k_per_step.values())

# An established positive estimate must be cleared on a bad sensor step.
# Otherwise an invalid sample can keep an old offset alive (false cooling).
meta, obs, set_zone, _ = scenario(1, 1, temp=296.8, outdoor=296.8)
policy = Policy(meta, None)
for measured_k in [296.8, 296.95, 297.10]:
    set_zone(0, measured_k)
    check(policy, obs)
key = policy.ahus[0]["zones"][0]["temperature"]
assert policy.disturbance_k_per_step[key] > 0.0
set_zone(0, 312.0)
check(policy, obs)
assert key not in policy.disturbance_k_per_step
assert key not in policy.previous_zone
set_zone(0, 297.10)
check(policy, obs)
assert key not in policy.disturbance_k_per_step, (
    "First valid sample after a jump must reinitialize, not estimate from a spike"
)

# With a missing/invalid VAV command index, no unknown action may contaminate
# another command slot or produce a false action-conditioned disturbance.
meta, obs, set_zone, _ = scenario(1, 2, temp=295.0, outdoor=295.0)
meta["zone_action_indices"]["VAV_0"]["supply_air_damper_percentage_command"] = -1
meta["zone_action_indices"]["VAV_0"]["reheat_valve_percentage_command"] = 1_000_000
policy = Policy(meta, None)
assert policy.ahus[0]["zones"][0]["damper"] is None
assert policy.ahus[0]["zones"][0]["valve"] is None
check(policy, obs)
for temperature_k in [295.1, 295.2]:
    set_zone(0, temperature_k)
    continuous, discrete = check(policy, obs)
assert policy.disturbance_k_per_step == {}, (
    "Observer must not estimate heat gains from unobservable actuator commands"
)
# The valid second terminal still has the proper metadata-resolved indices.
set_zone(1, 285.0)
continuous, discrete = check(policy, obs)
valid = meta["zone_action_indices"]["VAV_1"]
assert discrete[-1] == 1
assert continuous[valid["reheat_valve_percentage_command"]] > -1.0

# The nominal residual must subtract cooling and reheat effects, not just use
# a raw temperature derivative.
sample = {
    "temperature": 295.0,
    "outdoor": 300.0,
    "supply": 290.0,
    "damper": 0.5,
    "valve": 0.5,
}
expected_k = (0.003 * 5.0 + 0.006 * 0.5 * (-5.0) + 0.06 * 0.5)
assert abs(policy._predict_nominal_change_k(sample) - expected_k) < 1e-12

# 6. Reset matches a fresh policy, including all state and current actions;
# observer state must not leak between independent building episodes.
policy.reset()
assert policy.previous_zone == {} and policy.disturbance_k_per_step == {}
fresh = Policy(meta, None)
a, d = check(policy, obs)
a0, d0 = check(fresh, obs)
assert np.array_equal(a, a0) and np.array_equal(d, d0)

# 7. Invalid sensors remove stale previous action/observer state.
set_zone(0, float("nan"))
check(policy, obs)
assert policy.previous_zone == {} and policy.disturbance_k_per_step == {}
set_zone(0, 295.0)
check(policy, obs)
assert policy.disturbance_k_per_step == {}

# 8. AHU isolation: only the disturbed zone receives an observer correction.
meta, obs, set_zone, _ = scenario(2, 1, temp=296.8, outdoor=296.8)
policy = Policy(meta, None)
check(policy, obs)
set_zone(0, 296.95)
check(policy, obs)
set_zone(0, 297.10)
_, disc = check(policy, obs)
assert disc[0] == 1 and disc[1] == 0
assert len(policy.previous_zone) == 2
assert len(policy.disturbance_k_per_step) <= 2

# 9. Outdoor air is field-normalized in the official data; the sensor reader
# must not treat a z-score as Kelvin or convert with zone temperature stats.
meta, obs, set_zone, set_outside = scenario(1, 2, temp=305., outdoor=295.)
meta["observation_normalization_metadata"]["outside_air_temperature_sensor"] = {
    "sample_mean": 295., "sample_std": 8.}
set_outside((288. - 295.) / 8.)
policy = Policy(meta, None)
assert abs(policy._outside_temperature(obs, policy.ahus[0]["outdoor"]) - 288.) < 1e-8
check(policy, obs)
set_outside(float("nan"))
check(policy, obs)

# 10. Runtime remains bounded and the notebook entrypoint is official.
meta, obs, set_zone, _ = scenario(4, 8)
policy = Policy(meta, None)
start = time.perf_counter()
for step in range(128):
    set_zone(0, 302. - 0.02 * step)
    check(policy, obs)
duration = time.perf_counter() - start
assert duration < 10., f"Too slow: {duration:.3f}s for 128 steps"
class FakeInferenceServer:
    def __init__(self, cls):
        assert cls is Policy
    def run(self):
        self.called = True
gateway.SmartBuildingsInferenceServer = FakeInferenceServer
exec("".join(notebook["cells"][-1]["source"]), namespace)
print(f"PASS Version 18: exact-notebook syntax/gateway, contract, DOB ± signs, "
      f"action-conditioned innovation, clipping, reset, invalid sensors, "
      f"multiple AHUs, normalization and runtime {duration:.3f}s/128")
