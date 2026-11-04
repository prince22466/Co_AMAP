"""Regression checks for Version 16 LQR/LQG (synthetic, not Kaggle scoring).

Run: python Co_Kaggle/g8/submissions/test_version16.py
Requires numpy. The notebook's official gateway import is mocked for tests.
"""
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

NOTEBOOK = Path(__file__).with_name("submission-nb - Version 16.ipynb")
notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert len(notebook["cells"]) == 3
assert notebook["cells"][2]["cell_type"] == "code"
assert "SmartBuildingsInferenceServer(TransferPolicy).run()" in "".join(
    notebook["cells"][2]["source"])
source = "".join(notebook["cells"][1]["source"])
compile(source, str(NOTEBOOK), "exec")

gateway = types.ModuleType("smart_buildings_inference_server")
gateway.Policy = type("BasePolicy", (), {"__init__": lambda self, meta, spec: None})
kaggle = types.ModuleType("kaggle_evaluation")
kaggle.smart_buildings_inference_server = gateway
saved = sys.modules.get("kaggle_evaluation")
sys.modules["kaggle_evaluation"] = kaggle
namespace = {}
try:
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
finally:
    if saved is None:
        sys.modules.pop("kaggle_evaluation", None)
    else:
        sys.modules["kaggle_evaluation"] = saved
Policy = namespace["TransferPolicy"]
Observer = namespace["ZoneKalman"]
gain = namespace["scalar_discrete_lqr_gain"]()
a, b = namespace["A_THERMAL"], namespace["B_THERMAL"]
assert 0 < gain < 2.0 and abs(a-b*gain) < 1.0
assert abs(namespace["scalar_discrete_lqr_gain"](a,b,1.0,4.0)-gain) < 1e-12
for bad in [(a,0,1,4), (a,b,0,4), (a,b,1,0), (float("nan"),b,1,4)]:
    try:
        namespace["scalar_discrete_lqr_gain"](*bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Riccati parameters accepted")

# Verify gain solves scalar DARE, not a heuristic error-to-action gain.
q, r = 1.0, 4.0
c = r*(1-a*a)-q*b*b
p = (math.sqrt(c*c+4*b*b*q*r)-c)/(2*b*b)
assert abs(p-(q+a*a*p-(a*b*p)**2/(r+b*b*p))) < 1e-9
assert abs(gain-a*b*p/(r+b*b*p)) < 1e-9


def synthetic(n_ahu=2, per_ahu=3, outside=295.0, normalized_outside=True,
              missing_actuator=False):
    fields = ["hod_sin_000","hod_cos_000","dow_sin_000","dow_cos_000"]
    assignments, actions = {}, {}
    for ai in range(n_ahu):
        ahu = f"ahu_{ai}"
        assignments[ahu] = []
        fields.append(f"ahs_{ahu}_outside_air_temperature_sensor")
        for j in range(per_ahu):
            k = ai*per_ahu+j
            assignments[ahu].append(f"room_{k}")
            vav = f"VAV_{k}"
            fields.extend([vav+"_zone_air_temperature_sensor",
                           vav+"_zone_air_heating_temperature_setpoint_sensor",
                           vav+"_zone_air_cooling_temperature_setpoint_sensor"])
            base = 3*n_ahu+2+2*k
            actions[vav] = {
                "supply_air_damper_percentage_command": base,
                "reheat_valve_percentage_command": base+1}
            if missing_actuator and k == 0:
                actions[vav].pop("reheat_valve_percentage_command")
    stats = {"zone_air_temperature_sensor":
             {"sample_mean":295.0, "sample_std":5.0}}
    if normalized_outside:
        stats["outside_air_temperature_sensor"] = {
            "sample_mean":295.0, "sample_std":10.0}
    meta = {
        "n_continuous_action":3*n_ahu+2+2*n_ahu*per_ahu,
        "n_discrete_action":n_ahu+1,
        "observation_field_names":fields,
        "temp_indices":[3*i for i in range(n_ahu)],
        "pressure_indices":[3*i+1 for i in range(n_ahu)]+[3*n_ahu+1],
        "recirculation_indices":[3*i+2 for i in range(n_ahu)],
        "water_indices":[3*n_ahu],
        "ahu_zone_assignments":assignments,
        "zone_action_indices":actions,
        "comfort_temp_window":[293.0,297.0],
        "eco_temp_window":[288.0,302.0],
        "morning_start_hour":8.0,
        "evening_start_hour":18.0,
        "observation_normalization_metadata":stats,
    }
    obs = np.zeros(len(fields), dtype=np.float64)
    obs[0], obs[1] = math.sin(math.pi), math.cos(math.pi)
    obs[2], obs[3] = math.sin(4*math.pi/7), math.cos(4*math.pi/7)
    positions = {field:i for i,field in enumerate(fields)}
    for field, i in positions.items():
        if field.endswith("zone_air_temperature_sensor"):
            obs[i] = 0.0
        elif field.endswith("heating_temperature_setpoint_sensor"):
            obs[i] = 293.0
        elif field.endswith("cooling_temperature_setpoint_sensor"):
            obs[i] = 297.0
        elif field.endswith("outside_air_temperature_sensor"):
            obs[i] = ((outside-295.0)/10.0
                      if normalized_outside else outside)
    def zone(k, kelvin):
        obs[positions[f"VAV_{k}_zone_air_temperature_sensor"]] = (kelvin-295.0)/5.0
    return meta, obs, zone


def action(policy, obs):
    output = policy.get_action(obs)
    assert set(output) == {"continuous_action", "discrete_action"}
    c, d = output["continuous_action"], output["discrete_action"]
    assert c.shape == (policy.n_continuous,) and c.dtype == np.float32
    assert d.shape == (policy.n_discrete,) and d.dtype == np.int32
    assert np.isfinite(c).all() and np.max(np.abs(c)) <= 1.000001
    assert np.isin(d, [0,1]).all()
    return c, d


# Dynamic topologies and physical operations: idle, strong heat/cool, mixed.
for n, per in [(1,1), (2,4), (4,7), (12,1)]:
    meta, obs, set_zone = synthetic(n, per)
    policy = Policy(meta, None)
    c, d = action(policy, obs)
    assert not d.any(), "wide comfort band must not chase midpoint"
    assert (c[meta["temp_indices"]] == 1).all()
    set_zone(0, 306.0)
    c, d = action(policy, obs)
    assert d[0] == 1 and d[-1] == 0, "cooling must not start boiler"
    assert c[meta["temp_indices"][0]] < 1.0
    set_zone(0, 286.0)  # large sensor change tests observer reinitialization
    c, d = action(policy, obs)
    assert d[0] == 1 and d[-1] == 1, "reheat must start boiler"
    assert c[meta["temp_indices"][0]] == 1.0
    if per > 1:
        set_zone(1, 307.0)
        c, d = action(policy, obs)
        assert d[0] == 1 and d[-1] == 1
        valve = meta["zone_action_indices"]["VAV_0"]["reheat_valve_percentage_command"]
        assert c[valve] > -1.0
    assert len(policy.zone_filters) == n*per
    policy.reset()
    assert not policy.zone_filters
    action(policy, obs)

# Normalized outdoor and raw Kelvin are equivalent in a new episode.
meta_n, obs_n, set_n = synthetic(1,4,outside=288.,normalized_outside=True)
meta_r, obs_r, set_r = synthetic(1,4,outside=288.,normalized_outside=False)
set_n(0, 305.)
set_r(0, 305.)
policy_n, policy_r = Policy(meta_n,None), Policy(meta_r,None)
c_n, _ = action(policy_n,obs_n)
c_r, _ = action(policy_r,obs_r)
assert np.allclose(c_n,c_r,atol=1e-6)
assert policy_n._outside_temperature(obs_n, policy_n.ahus[0]["outdoor"]) == 288.0

# Without outside normalization, an outside z-score must be ignored.
bad_field = next(i for i,s in enumerate(meta_r["observation_field_names"])
                 if s.endswith("outside_air_temperature_sensor"))
obs_r[bad_field] = -0.7
unknown = Policy(meta_r,None)
assert np.isnan(unknown._outside_temperature(obs_r,unknown.ahus[0]["outdoor"]))
action(unknown,obs_r)

# LQR-only ablation must not create Kalman observers.
class LQROnly(Policy):
    USE_KALMAN = False

meta, obs, zone = synthetic(1,2,outside=310)
zone(0,304)
lqr, lqg = LQROnly(meta,None), Policy(meta,None)
action(lqr,obs)
action(lqg,obs)
assert not lqr.zone_filters and len(lqg.zone_filters)==2
assert abs(lqr.LQR_GAIN - lqg.LQR_GAIN) < 1e-12

# Two-state filter should learn a persistent drift without process divergence.
k = Observer(295.0)
temperature, ambient, true_drift = 295.0, 295.0, 0.06
for i in range(160):
    temperature = a*temperature+(1-a)*ambient+true_drift
    k.previous_effort = 0.0
    k.previous_outside = ambient
    k.observe(temperature + 0.03*math.sin(i*0.15))
assert abs(k.drift-true_drift) < 0.035, k.drift
assert math.isfinite(k.temperature) and math.isfinite(k.p_tt)
assert 0 < k.p_tt <= 1 and 0 < k.p_dd <= 0.1
k.observe(315.0)
assert abs(k.temperature-315.0) < 1e-9 and k.drift == 0

# Invalid sensors, missing valve and bounded return under repeated decisions.
meta, obs, zone = synthetic(2,4,missing_actuator=True)
zone(0,286)
zone(1,float("nan"))
policy = Policy(meta,None)
for _ in range(10):
    action(policy,obs)

# Check no hidden per-step optimization and a bounded CPU-only synthetic budget.
assert "import scipy" not in source and "from scipy" not in source and "import cma" not in source
meta, obs, zone = synthetic(4,8)
policy = Policy(meta,None)
start = time.perf_counter()
for i in range(128):
    zone(0,304.0-0.02*i)
    action(policy,obs)
elapsed = time.perf_counter()-start
assert elapsed < 10.0, f"Slow synthetic inference: {elapsed:.2f}s / 128"
print(f"PASS V16: exact notebook, DARE, LQR ablation, two-state Kalman, "
      f"hybrid actions, weather, topology, safety and latency ({elapsed:.3f}s/128)")
