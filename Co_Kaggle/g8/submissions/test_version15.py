"""Version 15 exact-notebook regression tests (no simulator/optimizer required).

Run: python Co_Kaggle/g8/submissions/test_version15.py
These synthetic checks are NOT an official rollout or leaderboard score.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
tuner_spec = importlib.util.spec_from_file_location(
    "tune_version15", ROOT / "tune_version15.py")
tuner = importlib.util.module_from_spec(tuner_spec)
tuner_spec.loader.exec_module(tuner)

notebook, Policy, BOUNDS, DEFAULTS = tuner.load_notebook_policy()
assert len(notebook["cells"]) == 3
assert "SmartBuildingsInferenceServer(TransferPolicy).run()" in "".join(
    notebook["cells"][2]["source"])
source = "".join(notebook["cells"][1]["source"])
compile(source, "Version 15 notebook", "exec")
assert "import cma" not in source and "skopt" not in source
assert len(BOUNDS) == len(DEFAULTS) == 8
assert all(lo <= DEFAULTS[key] <= hi for key, (lo, hi) in BOUNDS.items())


def synthetic_building(ahu_count=2, zones_each=4, outside=295.0,
                       normalized_outside=True):
    fields = ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000"]
    assignments, zone_actions = {}, {}
    for ahu in range(ahu_count):
        key = f"ahu_{ahu}"
        assignments[key] = []
        fields.append(f"ahs_{key}_outside_air_temperature_sensor")
        for j in range(zones_each):
            number = ahu * zones_each + j
            vav = f"VAV_{number}"
            assignments[key].append(f"room_{number}")
            fields += [f"{vav}_zone_air_temperature_sensor",
                       f"{vav}_zone_air_heating_temperature_setpoint_sensor",
                       f"{vav}_zone_air_cooling_temperature_setpoint_sensor"]
            start = 3 * ahu_count + 2 + number * 2
            zone_actions[vav] = {
                "supply_air_damper_percentage_command": start,
                "reheat_valve_percentage_command": start + 1}
    normalizers = {"zone_air_temperature_sensor":
                   {"sample_mean": 295.0, "sample_std": 5.0}}
    if normalized_outside:
        normalizers["outside_air_temperature_sensor"] = {
            "sample_mean": 295.0, "sample_std": 10.0}
    metadata = {
        "n_continuous_action": 3*ahu_count + 2 + 2*ahu_count*zones_each,
        "n_discrete_action": ahu_count+1, "observation_field_names": fields,
        "temp_indices": [3*i for i in range(ahu_count)],
        "pressure_indices": [3*i+1 for i in range(ahu_count)]+[3*ahu_count+1],
        "recirculation_indices": [3*i+2 for i in range(ahu_count)],
        "water_indices": [3*ahu_count],
        "ahu_zone_assignments": assignments, "zone_action_indices": zone_actions,
        "comfort_temp_window": [293.0, 297.0],
        "eco_temp_window": [288.0, 302.0],
        "morning_start_hour": 8.0, "evening_start_hour": 18.0,
        "observation_normalization_metadata": normalizers,
    }
    obs = np.zeros(len(fields), dtype=np.float64)
    obs[0], obs[1] = math.sin(math.pi), math.cos(math.pi)  # Noon.
    obs[2], obs[3] = math.sin(4*math.pi/7), math.cos(4*math.pi/7)
    positions = {name: i for i, name in enumerate(fields)}
    for name, i in positions.items():
        if name.endswith("zone_air_temperature_sensor"):
            obs[i] = 0.0
        elif name.endswith("heating_temperature_setpoint_sensor"):
            obs[i] = 293.0
        elif name.endswith("cooling_temperature_setpoint_sensor"):
            obs[i] = 297.0
        elif name.endswith("outside_air_temperature_sensor"):
            obs[i] = (outside - 295.0) / 10.0 if normalized_outside else outside

    def set_zone(number, kelvin):
        obs[positions[f"VAV_{number}_zone_air_temperature_sensor"]] = (
            kelvin - 295.0) / 5.0

    def set_outside(kelvin):
        for name, index in positions.items():
            if name.endswith("outside_air_temperature_sensor"):
                obs[index] = ((kelvin - 295.0) / 10.0
                              if normalized_outside else kelvin)
    return metadata, obs, set_zone, set_outside


def check(policy, observation):
    result = policy.get_action(observation)
    assert set(result) == {"continuous_action", "discrete_action"}
    c, d = result["continuous_action"], result["discrete_action"]
    assert c.dtype == np.float32 and d.dtype == np.int32
    assert c.shape == (policy.n_continuous,)
    assert d.shape == (policy.n_discrete,)
    assert np.isfinite(c).all() and np.max(np.abs(c)) <= 1.000001
    assert np.isin(d, [0, 1]).all()
    return c, d


def candidate(meta, updates):
    return type("CandidatePolicy", (Policy,),
                {"PARAMETERS": {**DEFAULTS, **updates}})(meta, None)


# Hybrid interface, cooling/heating/mixed loads, no tempering of idle output.
for n_ahu, per in [(1, 1), (2, 4), (4, 7), (12, 1)]:
    meta, obs, set_zone, _ = synthetic_building(n_ahu, per)
    p = Policy(meta, None)
    c, d = check(p, obs)
    assert not d.any(), "AHUs and boiler off when all zones are comfortable"
    assert np.all(c[meta["temp_indices"]] == 1.0)
    set_zone(0, 307.0)
    c, d = check(p, obs)
    assert d[0] == 1 and d[-1] == 0
    assert c[meta["temp_indices"][0]] < 1.
    set_zone(0, 286.0)
    c, d = check(p, obs)
    assert d[0] == 1 and d[-1] == 1
    assert c[meta["temp_indices"][0]] == 1., "AHU has no heater"
    if per > 1:
        set_zone(1, 306.)
        c, d = check(p, obs)
        assert d[0] == 1 and d[-1] == 1
        valve = meta["zone_action_indices"]["VAV_0"]["reheat_valve_percentage_command"]
        assert c[valve] > -1
    p.reset()
    check(p, obs)

# Weather is standardized by official metadata; equivalent Kelvin must yield
# identical outdoor economizer commands for raw/normalized sensor versions.
meta_n, obs_n, set_n, _ = synthetic_building(1, 4, outside=288.)
meta_r, obs_r, set_r, _ = synthetic_building(1, 4, outside=288.,
                                             normalized_outside=False)
set_n(0, 305.)
set_r(0, 305.)
c_norm, _ = check(Policy(meta_n, None), obs_n)
c_raw, _ = check(Policy(meta_r, None), obs_r)
recirc = meta_n["recirculation_indices"][0]
assert c_norm[recirc] < 1.0, "Moderate weather should enable free cooling"
assert np.isclose(c_norm[recirc], c_raw[recirc], atol=1e-5)
# Missing normalization with a z-score should safely fall back to recirculation.
obs_r[[i for i, k in enumerate(meta_r["observation_field_names"])
       if "outside_air_temperature_sensor" in k]] = -0.7
c_unknown, _ = check(Policy(meta_r, None), obs_r)
assert c_unknown[recirc] == 1.0

# Parameter perturbations must change *physical* control response.
meta, obs, set_zone, set_outside = synthetic_building(1, 4)
set_zone(0, 305.)
low = candidate(meta, {"cool_airflow_gain": 1.0})
high = candidate(meta, {"cool_airflow_gain": 3.4})
damper = meta["zone_action_indices"]["VAV_0"]["supply_air_damper_percentage_command"]
assert check(high, obs)[0][damper] > check(low, obs)[0][damper]
low = candidate(meta, {"fan_pressure_gain_pa": 800.})
high = candidate(meta, {"fan_pressure_gain_pa": 1700.})
fan = meta["pressure_indices"][0]
assert check(high, obs)[0][fan] > check(low, obs)[0][fan]
set_zone(0, 291.)
low = candidate(meta, {"reheat_gain": 0.15})
high = candidate(meta, {"reheat_gain": 0.55})
valve = meta["zone_action_indices"]["VAV_0"]["reheat_valve_percentage_command"]
assert check(high, obs)[0][valve] > check(low, obs)[0][valve]
low = candidate(meta, {"water_shortfall_gain": 3.0})
high = candidate(meta, {"water_shortfall_gain": 8.0})
water = meta["water_indices"][0]
assert check(high, obs)[0][water] > check(low, obs)[0][water]

# Malformed sensor values or tuned parameters cannot silently leak NaNs.
set_zone(1, np.nan)
check(Policy(meta, None), obs)
try:
    candidate(meta, {"active_threshold_k": float("nan")})
except ValueError:
    pass
else:
    raise AssertionError("Expected nonfinite-tuned-gain validation failure")

# Validate search-bounds transformations, exact export, cache and split safety.
vector = tuner.normalized_vector(DEFAULTS, BOUNDS)
assert all(0 <= value <= 1 for value in vector)
roundtrip = tuner.physical_parameters(vector, BOUNDS)
assert all(abs(DEFAULTS[name] - roundtrip[name]) < 1e-9 for name in BOUNDS)
try:
    tuner.physical_parameters([1, 2], BOUNDS)
except ValueError:
    pass
else:
    raise AssertionError("Expected dimensionality failure")

with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp)
    tuned = {**DEFAULTS, "active_threshold_k": 0.25}
    exported = output / "submission-nb - Version 15.tuned.ipynb"
    tuner.export_notebook(notebook, tuned, exported)
    _, policy_cls, bounds, exported_defaults = tuner.load_notebook_policy(exported)
    assert policy_cls.PARAMETERS == tuned and exported_defaults == tuned
    assert bounds == BOUNDS

    a = output / "building_train_1" / "atlanta_2"
    b = output / "building_train_7" / "fairbanks_7"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    tuner.check_split([a], [b])
    try:
        tuner.check_split([a], [a.parent / "denver_7"])
    except ValueError:
        pass
    else:
        raise AssertionError("Train/validation building overlap accepted")

    previous_simulator = tuner.simulate_episode
    try:
        calls = []
        def fake_sim(policy_class, params, config, steps):
            calls.append((config, steps))
            return (params["active_threshold_k"] - 0.25)**2 + 0.01
        tuner.simulate_episode = fake_sim
        obj = tuner.ScenarioObjective(
            Policy, BOUNDS, [a, b], 32, output / "trials.jsonl")
        one = obj.evaluate(vector)
        two = obj.evaluate(vector)
        assert one == two and len(calls) == 2 and len(obj.trials) == 1
        assert len((output / "trials.jsonl").read_text().splitlines()) == 1
        assert json.loads((output / "trials.jsonl").read_text())["loss"] == one
    finally:
        tuner.simulate_episode = previous_simulator

# An actual optimizer integration smoke runs only if optional local deps exist.
for method, package in [("cmaes", "cma"), ("bayes", "skopt")]:
    if importlib.util.find_spec(package) is not None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_simulator = tuner.simulate_episode
            tuner.simulate_episode = (
                lambda p, gains, c, steps: sum(
                    ((gains[key] - DEFAULTS[key]) /
                     (BOUNDS[key][1]-BOUNDS[key][0]))**2 for key in BOUNDS))
            try:
                dummy = tuner.ScenarioObjective(
                    Policy, BOUNDS, [Path("/tmp/building_train_1/atlanta_2")],
                    16, Path(tmp) / "search.jsonl")
                baseline, best = tuner.run_search(
                    method, 9, 123, vector, dummy)
                assert np.isfinite(baseline)
                assert best["loss"] <= baseline + 1e-12
            finally:
                tuner.simulate_episode = previous_simulator

# Per-decision latency regression: no optimizer inside get_action.
meta, obs, set_zone, _ = synthetic_building(4, 8)
policy = Policy(meta, None)
start = time.perf_counter()
for step in range(128):
    set_zone(0, 304. - .02*step)
    check(policy, obs)
duration = time.perf_counter()-start
assert duration < 10., f"Excessive per-step latency: {duration:.2f}s/128"
print(f"PASS V15: notebook/params, weather units, dynamic topologies, "
      f"heat/cool, bounds, tunability, offline export, trace and latency "
      f"({duration:.3f}s / 128 actions)")
