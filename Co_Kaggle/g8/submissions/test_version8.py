"""Version 8 synthetic interface, temporal behavior and runtime tests.

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
notebook = json.loads((ROOT / "submission-nb - Version 8.ipynb").read_text())
assert len(notebook["cells"]) == 3
source = "".join(notebook["cells"][1]["source"])
compile(source, "<version8 notebook policy>", "exec")
compile("".join(notebook["cells"][2]["source"]), "<version8 entrypoint>", "exec")

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


# Different building sizes and hot, cold, and simultaneous loads.
for shape in [(1, 1), (2, 4), (4, 7)]:
    meta, obs, set_zone, _ = scenario(*shape)
    policy = Policy(meta, None)
    _, d = check(policy, obs)
    assert not np.any(d), "Comfortable zones must not start equipment"

    set_zone(0, 305.)
    c, d = check(policy, obs)
    assert d[0] == 1 and d[-1] == 0
    if shape[0] > 1:
        assert not np.any(d[1:shape[0]]), "Other AHUs stay idle"
    assert c[meta["temp_indices"][0]] < 1.
    assert c[meta["recirculation_indices"][0]] == 1.0

    policy.reset()
    set_zone(0, 285.)
    c, d = check(policy, obs)
    assert d[0] == d[-1] == 1, "Cold zones need VAV reheat + water"
    assert c[meta["temp_indices"][0]] == 1., "Do not heat using the cooling coil"

    if shape[1] > 1:
        policy.reset()
        set_zone(0, 305.)
        set_zone(1, 285.)
        c, d = check(policy, obs)
        assert d[0] == 1 and d[-1] == 1
        valve = meta["zone_action_indices"]["VAV_1"]["reheat_valve_percentage_command"]
        assert c[valve] > -1., "Heat in one zone must survive shared cooling"


# Start/stop hysteresis and minimum two-sample on time.
meta, obs, set_zone, _ = scenario(1, 2)
policy = Policy(meta, None)
set_zone(0, 297.20)
assert check(policy, obs)[1][0] == 0, "Below start threshold: stay off"
set_zone(0, 297.50)
assert check(policy, obs)[1][0] == 1, "Above start threshold: turn on"
set_zone(0, 297.03)
assert check(policy, obs)[1][0] == 1, "Minimum run protects against chatter"
assert check(policy, obs)[1][0] == 0, "Turn off after dwell expires"
set_zone(0, 297.20)
assert check(policy, obs)[1][0] == 0, "Do not restart within hysteresis band"
set_zone(0, 297.60)
assert check(policy, obs)[1][0] == 1, "Re-start for true new violation"
policy.reset()
set_zone(0, 295.)
assert check(policy, obs)[1][0] == 0, "Reset clears temporal state"


# Slew limits cap sharp commands only while an AHU is continuously running.
meta, obs, set_zone, _ = scenario(1, 4)
policy = Policy(meta, None)
set_zone(0, 298.)
c1, d1 = check(policy, obs)
set_zone(0, 320.)
c2, d2 = check(policy, obs)
assert d1[0] == d2[0] == 1
supply, fan = meta["temp_indices"][0], meta["pressure_indices"][0]
damper = meta["zone_action_indices"]["VAV_0"]["supply_air_damper_percentage_command"]
assert abs(float(c2[supply] - c1[supply])) <= 0.3501
assert abs(float(c2[fan] - c1[fan])) <= 0.3501
assert abs(float(c2[damper] - c1[damper])) <= 0.5001
assert c2[supply] < c1[supply], "Coil cools further after large hot excursion"


# One isolated severe zone cannot vanish in mean damper pressure.
meta, obs, set_zone, _ = scenario(1, 30)
set_zone(0, 320.)
c, d = check(Policy(meta, None), obs)
pressure_pa = (float(c[meta["pressure_indices"][0]]) + 1.) * 1000.
assert d[0] == 1 and pressure_pa > 900., "Protect one extreme zone"


# Invalid zone temperatures do not disable other zones; all-invalid resets mode.
meta, obs, set_zone, _ = scenario(2, 4)
policy = Policy(meta, None)
set_zone(0, 305.)
set_zone(1, float("nan"))
assert check(policy, obs)[1][0] == 1
set_zone(0, float("nan"))
for i in range(2, 4):
    set_zone(i, float("nan"))
assert check(policy, obs)[1][0] == 0, "Sensor outage must not keep AHU latched"
set_zone(0, 285.)
c, d = check(policy, obs)
assert d[0] == d[-1] == 1


# Missing setpoints use schedule defaults. No hardcoded observation offsets.
meta, obs, set_zone, _ = scenario(2, 4)
name = "VAV_0_zone_air_heating_temperature_setpoint_sensor"
meta["observation_field_names"][meta["observation_field_names"].index(name)] = "missing_heat"
policy = Policy(meta, None)
set_zone(0, 284.)
assert check(policy, obs)[1][-1] == 1


# Larger topology: bounded time, no nested optimization or growing histories.
meta, obs, set_zone, _ = scenario(4, 8)
policy = Policy(meta, None)
start = time.perf_counter()
for step in range(128):
    set_zone(0, 302. - 0.02 * step)
    check(policy, obs)
elapsed = time.perf_counter() - start
assert elapsed < 10., f"Too slow: {elapsed:.3f}s / 128 steps"
assert len(policy.running) == 4 and len(policy.hold) == 4

class FakeInferenceServer:
    def __init__(self, cls):
        assert cls is Policy

    def run(self):
        self.called = True

gateway.SmartBuildingsInferenceServer = FakeInferenceServer
exec("".join(notebook["cells"][-1]["source"]), namespace)
print(f"PASS: Version 8 contract, modes, dwell, slew, mixed demand, "
      f"invalid sensors, reset and runtime: {elapsed:.3f}s/128")
