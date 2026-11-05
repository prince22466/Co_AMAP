"""Offline Version 27 Smith-predictor and Kaggle interface regressions.

Uses exact notebook code with a fake gateway; no official simulator is run.
The tests use controlled synthetic observations, not the Kaggle simulator.
"""
import ast
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

NOTEBOOK = Path(__file__).with_name("submission-nb - Version 27.ipynb")
notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert len(notebook["cells"]) == 2
assert [c["cell_type"] for c in notebook["cells"]] == ["markdown", "code"]
source = "".join(notebook["cells"][1]["source"])
assert source.count("SmartBuildingsInferenceServer(TransferPolicy).run()") == 1
ast.parse(source)
compile(source, str(NOTEBOOK), "exec")

package = types.ModuleType("kaggle_evaluation")
gateway = types.ModuleType("smart_buildings_inference_server")
gateway.Policy = type("Policy", (), {"__init__": lambda self, meta, spec: None})
gateway_calls = []


class FakeInferenceServer:
    def __init__(self, policy_type):
        gateway_calls.append(policy_type)

    def run(self):
        pass


gateway.SmartBuildingsInferenceServer = FakeInferenceServer
package.smart_buildings_inference_server = gateway
sys.modules["kaggle_evaluation"] = package
sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = gateway
namespace = {}
exec(source, namespace)
Policy = namespace["TransferPolicy"]
assert gateway_calls == [Policy], "Gateway must execute in the same notebook cell"


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


# 1. Robust hybrid action contract on varying topologies.
for shape in [(1, 1), (2, 4), (4, 7)]:
    meta, obs, set_zone, set_outside = scenario(*shape)
    policy = Policy(meta, None)
    _, discrete = check(policy, obs)
    assert not discrete.any(), "A neutral building should be idle"
    set_zone(0, 301.0)
    _, discrete = check(policy, obs)
    assert discrete[0] == 1, "Cooling demand should activate the AHU"
    policy.reset()
    set_zone(0, 290.0)
    _, discrete = check(policy, obs)
    assert discrete[0] == 1 and discrete[-1] == 1, "Heating must start water"

# 2. Direct thermal model and explicit delay indexing.
meta, obs, set_zone, set_outside = scenario(1, 1, temp=298., outdoor=298.)
policy = Policy(meta, None)
check(policy, obs)
key = policy.ahus[0]["zones"][0]["temperature"]
queue = policy.smith_states[key]["queue"]
assert len(queue) == 1
first_input = queue[0]
assert first_input["supply"] is not None
expected_delta = policy._thermal_change(298., 298., first_input)
assert expected_delta < 0, "The first issued command must cool the room"
check(policy, obs)
assert abs(policy.last_smith_correction_k[key] - expected_delta) < 1e-6, (
    "At t=1 delayed model must still use idle, while instant model uses u[0]"
)
assert len(policy.smith_states[key]["queue"]) == 2
check(policy, obs)
assert len(policy.smith_states[key]["queue"]) == 3
# At t=3 delayed plant must be allowed to apply the oldest stored input.
assert policy.smith_states[key]["queue"][0] is first_input
check(policy, obs)
assert len(policy.smith_states[key]["queue"]) == 3
assert all(np.isfinite(v) for v in policy.last_smith_correction_k.values())

# 3. Controlled, same-dispatcher comparison: on a hot unresponsive room,
# predicted pending cooling should reduce repeated airflow requests.
meta, obs, set_zone, _ = scenario(1, 1, temp=298., outdoor=298.)
smith, baseline = Policy(meta, None), Policy(meta, None)
baseline.ENABLE_SMITH = False
damper_idx = meta["zone_action_indices"]["VAV_0"]["supply_air_damper_percentage_command"]
smith_air = base_air = 0.0
for i in range(6):
    c_s, d_s = check(smith, obs)
    c_b, d_b = check(baseline, obs)
    assert d_s[0] == d_b[0] == 1
    smith_air += (c_s[damper_idx] + 1.0) / 2.0
    base_air += (c_b[damper_idx] + 1.0) / 2.0
assert smith.last_smith_correction_k[key] < 0.0
assert smith_air < base_air, "Smith should not keep issuing full airflow while cooling is pending"

# 4. Under steady cold readings, pending reheat shifts compensated T UP
# and reduces repeated valve demand, with no fictitious cooling request.
meta, obs, set_zone, _ = scenario(1, 1, temp=292., outdoor=292.)
smith, baseline = Policy(meta, None), Policy(meta, None)
baseline.ENABLE_SMITH = False
valve_idx = meta["zone_action_indices"]["VAV_0"]["reheat_valve_percentage_command"]
valve_s = valve_b = 0.0
for _ in range(6):
    c_s, d_s = check(smith, obs)
    c_b, d_b = check(baseline, obs)
    assert d_s[0] == d_b[0] == d_s[-1] == d_b[-1] == 1
    valve_s += (c_s[valve_idx] + 1.) / 2.
    valve_b += (c_b[valve_idx] + 1.) / 2.
assert smith.last_smith_correction_k[smith.ahus[0]["zones"][0]["temperature"]] > 0.
assert valve_s < valve_b, "Pending reheat should reduce repeated valve demand"

# 5. Zero-delay model is exactly the ablated physical dispatcher, not
# simply a fixed-temperature anticipator.
meta, obs, _, _ = scenario(1, 2, temp=298., outdoor=298.)
zero, baseline = Policy(meta, None), Policy(meta, None)
zero.ACTUATOR_DELAY_STEPS = 0
baseline.ENABLE_SMITH = False
for _ in range(10):
    ca, da = check(zero, obs)
    cb, db = check(baseline, obs)
    assert np.array_equal(ca, cb) and np.array_equal(da, db)
assert all(abs(v) < 1e-9 for v in zero.last_smith_correction_k.values())

# 6. Severe *measured* violations must retain their FULL demand.
# Merely requiring the AHU to remain ON misses a dangerous failure mode:
# a 10 K hot error can otherwise be reduced to an almost-idle 0.18 K request.
for severe_k in (307.0, 285.0):
    meta, obs, _, _ = scenario(1, 1, temp=severe_k, outdoor=severe_k)
    policy, baseline = Policy(meta, None), Policy(meta, None)
    baseline.ENABLE_SMITH = False
    for _ in range(16):
        corrected, running = check(policy, obs)
        reference, reference_run = check(baseline, obs)
        assert np.array_equal(running, reference_run), (
            "Model-based compensation must not disable emergency equipment"
        )
        assert np.array_equal(corrected, reference), (
            "Severe measured comfort error must never lose cooling/reheat "
            "intensity, even with significant pending actuator input"
        )
    assert all(abs(v) <= policy.MAX_CORRECTION_K + 1e-9
               for v in policy.last_smith_correction_k.values())
    assert max(len(v["queue"]) for v in policy.smith_states.values()) <= 3

# 7. Outlier, invalid sample, reset, and fresh-instance equivalence.
meta, obs, set_zone, _ = scenario(1, 1, temp=298., outdoor=298.)
policy = Policy(meta, None)
for _ in range(3):
    check(policy, obs)
set_zone(0, 320.)
check(policy, obs)
assert not policy.smith_states, "Unphysical sensor jump must discard pending inputs"
set_zone(0, 298.)
check(policy, obs)
assert not policy.last_smith_correction_k
set_zone(0, float("nan"))
check(policy, obs)
assert not policy.smith_states
set_zone(0, 298.)
check(policy, obs)
policy.reset()
assert not policy.smith_states and not policy.last_smith_correction_k
fresh = Policy(meta, None)
a, d = check(policy, obs)
a0, d0 = check(fresh, obs)
assert np.array_equal(a, a0) and np.array_equal(d, d0)

# 8. Invalid action indices are rejected rather than wrapping, and the
# predictor does not create phantom inputs for an unobservable actuator.
meta, obs, _, _ = scenario(1, 2, temp=298., outdoor=298.)
meta["zone_action_indices"]["VAV_0"]["supply_air_damper_percentage_command"] = -1
meta["zone_action_indices"]["VAV_0"]["reheat_valve_percentage_command"] = 99999
policy = Policy(meta, None)
assert policy.ahus[0]["zones"][0]["damper"] is None
for _ in range(3):
    check(policy, obs)
bad_key = policy.ahus[0]["zones"][0]["temperature"]
assert bad_key not in policy.smith_states

# 9. Field-specific temperature normalization.
meta, obs, set_zone, set_outside = scenario(1, 1, temp=298., outdoor=298.)
meta["observation_normalization_metadata"]["outside_air_temperature_sensor"] = {
    "sample_mean": 295.0, "sample_std": 8.0
}
set_outside((285.0 - 295.0) / 8.0)
policy = Policy(meta, None)
assert abs(policy._outside_temperature(obs, policy.ahus[0]["outdoor"]) - 285.) < 1e-10
check(policy, obs)
set_outside(float("nan"))
check(policy, obs)

# 10. AHU isolation: correcting one hot zone must not activate an
# unrelated neutral AHU or create a Smith offset for that other zone.
meta, obs, set_zone, _ = scenario(2, 1, temp=295.0, outdoor=295.0)
set_zone(0, 298.0)
policy = Policy(meta, None)
for _ in range(5):
    _, running = check(policy, obs)
    assert running[0] == 1 and running[1] == 0
first_key = policy.ahus[0]["zones"][0]["temperature"]
second_key = policy.ahus[1]["zones"][0]["temperature"]
assert policy.last_smith_correction_k[first_key] < 0.0
assert abs(policy.last_smith_correction_k[second_key]) < 1e-9

# 11. No-leak cross-AHU memory, bounded runtime and episode reset.
meta, obs, set_zone, _ = scenario(4, 8)
policy = Policy(meta, None)
start = time.perf_counter()
for step in range(128):
    set_zone(0, 297.8 + 0.01 * (step % 8))
    check(policy, obs)
duration = time.perf_counter() - start
assert duration < 10.0, f"128 decisions too slow: {duration:.3f} s"
assert len(policy.smith_states) <= 32
assert all(len(state["queue"]) <= 3 for state in policy.smith_states.values())
policy.reset()
assert not policy.smith_states
print(f"PASS Version 27: notebook/gateway, thermal-delay indexing, cooling/heating "
      f"overcorrection reduction, zero-delay ablation, safety, reset, mapping, "
      f"normalization, multi-AHU and latency ({duration:.3f}s/128).")
