"""Synthetic contract/behavior tests for the EXACT submitted V10 notebook.

No Kaggle simulator or reward is used here. The gateway is stubbed so the
actual policy cell can run with only NumPy installed.
"""
import json
from pathlib import Path
import sys
import time
import types

import numpy as np


NOTEBOOK_PATH = Path(__file__).with_name("submission-nb - Version 10.ipynb")
notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4 and len(notebook["cells"]) == 3
assert all(cell.get("id") for cell in notebook["cells"])
policy_code = "".join(notebook["cells"][1]["source"])
gateway_code = "".join(notebook["cells"][2]["source"])
compile(policy_code, "<V10 submitted policy>", "exec")
compile(gateway_code, "<V10 inference gateway>", "exec")

kaggle_package = types.ModuleType("kaggle_evaluation")
gateway = types.ModuleType("smart_buildings_inference_server")
gateway.Policy = type("Policy", (), {"__init__": lambda self, meta, spec: None})
kaggle_package.smart_buildings_inference_server = gateway
sys.modules["kaggle_evaluation"] = kaggle_package
sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = gateway
namespace = {}
exec(policy_code, namespace)
Policy = namespace["TransferPolicy"]


def scenario(ahu_count=2, zones_each=3, with_airflow=True):
    """Build arbitrary-size building metadata with metadata-indexed controls."""
    fields = ["comfort_mode_now"]
    assignments, vav_actions = {}, {}
    first_vav_index = 3 * ahu_count + 2
    for ahu in range(ahu_count):
        ahu_name = f"ahu_{ahu}"
        assignments[ahu_name] = []
        if with_airflow:
            fields.extend([
                f"ahs_{ahu_name}_supply_air_flowrate_sensor",
                f"ahs_{ahu_name}_supply_air_flowrate_setpoint",
            ])
        for local in range(zones_each):
            number = ahu * zones_each + local
            vav_name = f"VAV_{number}"
            assignments[ahu_name].append(f"room_{number}")
            fields.extend([
                f"{vav_name}_zone_air_temperature_sensor",
                f"{vav_name}_zone_air_heating_temperature_setpoint_sensor",
                f"{vav_name}_zone_air_cooling_temperature_setpoint_sensor",
            ])
            vav_actions[vav_name] = {
                "supply_air_damper_percentage_command": first_vav_index + 2*number,
                "reheat_valve_percentage_command": first_vav_index + 2*number + 1,
            }

    meta = {
        "observation_field_names": fields,
        "n_continuous_action": first_vav_index + 2*ahu_count*zones_each,
        "n_discrete_action": ahu_count + 1,
        "temp_indices": [3*a for a in range(ahu_count)],
        "pressure_indices": [3*a+1 for a in range(ahu_count)] + [3*ahu_count+1],
        "recirculation_indices": [3*a+2 for a in range(ahu_count)],
        "water_indices": [3*ahu_count],
        "ahu_zone_assignments": assignments,
        "zone_action_indices": vav_actions,
        "comfort_temp_window": [293.0, 297.0],
        "eco_temp_window": [288.0, 302.0],
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 295.0, "sample_std": 5.0}
        },
    }
    observation = np.zeros(len(fields), dtype=np.float64)
    observation[0] = 1.0
    field_lookup = {name: index for index, name in enumerate(fields)}
    for name, index in field_lookup.items():
        if name.endswith("zone_air_temperature_sensor"):
            observation[index] = 0.0  # 295 K
        elif name.endswith("zone_air_heating_temperature_setpoint_sensor"):
            observation[index] = 293.0
        elif name.endswith("zone_air_cooling_temperature_setpoint_sensor"):
            observation[index] = 297.0
        elif name.endswith("supply_air_flowrate_sensor"):
            observation[index] = 10.0
        elif name.endswith("supply_air_flowrate_setpoint"):
            observation[index] = 10.0

    def zone_temp(number, temperature_k):
        index = field_lookup[f"VAV_{number}_zone_air_temperature_sensor"]
        observation[index] = (temperature_k - 295.0)/5.0

    def ahu_flow(number, measured, requested):
        observation[field_lookup[f"ahs_ahu_{number}_supply_air_flowrate_sensor"]] = measured
        observation[field_lookup[f"ahs_ahu_{number}_supply_air_flowrate_setpoint"]] = requested

    return meta, observation, zone_temp, ahu_flow


def checked(policy, observation):
    """Verify the complete Kaggle hybrid-action contract on every call."""
    output = policy.get_action(observation)
    assert set(output) == {"continuous_action", "discrete_action"}
    continuous = output["continuous_action"]
    discrete = output["discrete_action"]
    assert continuous.shape == (policy.continuous_count,)
    assert discrete.shape == (policy.discrete_count,)
    assert continuous.dtype == np.float32 and discrete.dtype == np.int32
    assert np.isfinite(continuous).all() and np.all(np.abs(continuous) <= 1.0)
    assert np.isin(discrete, (0, 1)).all()
    return continuous, discrete


def fan_pa(continuous, meta, ahu_id=0):
    """Decode AHU fan's normalized action into duct pressure (Pa)."""
    return (float(continuous[meta["pressure_indices"][ahu_id]]) + 1.0)*1000.0


def test_heating_cooling_and_reset():
    for ahus, zones in ((1, 1), (2, 4), (4, 8)):
        meta, obs, zone_temp, _ = scenario(ahus, zones)
        policy = Policy(meta, None)
        c, d = checked(policy, obs)
        assert not np.any(d), "Zones in-band should not run conditioning"
        zone_temp(0, 306.0)
        c, d = checked(policy, obs)
        assert d[0] == 1 and d[-1] == 0, "Hot-zone cooling"
        assert c[meta["temp_indices"][0]] < 1.0
        zone_temp(0, 282.0)
        c, d = checked(policy, obs)
        assert d[0] == 1 and d[-1] == 1, "Cold-zone reheat"
        assert c[meta["temp_indices"][0]] == 1.0, "AHU must not heat"
        if zones > 1:
            zone_temp(0, 306.0)
            zone_temp(1, 282.0)
            c, d = checked(policy, obs)
            valve = meta["zone_action_indices"]["VAV_1"]["reheat_valve_percentage_command"]
            assert d[0] == d[-1] == 1 and c[valve] > -1.0
        policy.reset()
        first = checked(policy, obs)
        policy.reset()
        second = checked(policy, obs)
        assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_trim_respond_and_comfort_guard():
    meta, obs, zone_temp, _ = scenario(1, 3)
    policy = Policy(meta, None)
    zone_temp(0, 297.7)  # Mild cooling, damper not fully open.
    pressures = [fan_pa(checked(policy, obs)[0], meta) for _ in range(6)]
    assert all(a > b for a, b in zip(pressures, pressures[1:]))

    zone_temp(0, 299.1)  # 2.1 K; comfort guard requests before saturation.
    recovered = fan_pa(checked(policy, obs)[0], meta)
    assert recovered >= pressures[-1] + 119

    zone_temp(0, 313.0)
    severe_pressure = fan_pa(checked(policy, obs)[0], meta)
    assert severe_pressure >= recovered + 119
    for _ in range(40):
        checked(policy, obs)
    assert policy.pressure_pa[0] == policy.MAX_PRESSURE_PA

    zone_temp(0, 297.7)
    for _ in range(80):
        checked(policy, obs)
    assert policy.pressure_pa[0] == policy.MIN_PRESSURE_PA


def test_flow_warmup_missing_and_bad_telemetry():
    meta, obs, zone_temp, ahu_flow = scenario(1, 3)
    policy = Policy(meta, None)
    zone_temp(0, 297.7)
    ahu_flow(0, 0.0, 10.0)  # Could be the reading before the AHU starts.
    for _ in range(policy.FLOW_SENSOR_WARMUP_STEPS):
        c, _ = checked(policy, obs)
        assert fan_pa(c, meta) <= policy.INITIAL_PRESSURE_PA
    c, _ = checked(policy, obs)
    assert policy.pressure_pa[0] > policy.INITIAL_PRESSURE_PA

    policy.reset()
    ahu_flow(0, float("nan"), 10.0)
    for _ in range(3):
        checked(policy, obs)
    assert policy.pressure_pa[0] < policy.INITIAL_PRESSURE_PA
    policy.reset()
    ahu_flow(0, 1.0, -10.0)
    for _ in range(3):
        checked(policy, obs)
    assert policy.pressure_pa[0] < policy.INITIAL_PRESSURE_PA

    meta, obs, zone_temp, _ = scenario(1, 3, with_airflow=False)
    zone_temp(0, 297.7)
    policy = Policy(meta, None)
    for _ in range(3):
        checked(policy, obs)
    assert policy.pressure_pa[0] < policy.INITIAL_PRESSURE_PA


def test_independent_ahus_and_stop():
    meta, obs, zone_temp, _ = scenario(2, 2)
    policy = Policy(meta, None)
    zone_temp(0, 313.0)
    zone_temp(2, 297.8)
    c, d = checked(policy, obs)
    assert d[0] == 1 and d[1] == 1
    assert fan_pa(c, meta, 0) > 700 and fan_pa(c, meta, 1) < 700
    zone_temp(0, 295.0)
    c, d = checked(policy, obs)
    assert d[0] == 0 and d[1] == 1
    assert policy.pressure_pa[0] == policy.INITIAL_PRESSURE_PA
    assert policy.running_steps[0] == 0
    assert policy.pressure_pa[1] < 700
    policy.reset()
    assert policy.pressure_pa == [700.0, 700.0]
    assert policy.running_steps == [0, 0]


def test_missing_sensors_setpoints_and_long_ahu_list():
    # AHUs with names 10 and 2 must sort numerically, not lexicographically.
    meta, obs, zone_temp, _ = scenario(12, 1, with_airflow=False)
    zone_temp(10, 309.0)
    policy = Policy(meta, None)
    _, d = checked(policy, obs)
    assert d[10] == 1 and d[2] == 0

    meta, obs, zone_temp, _ = scenario(2, 3, with_airflow=False)
    policy = Policy(meta, None)
    zone_temp(0, 302.5)
    zone_temp(1, float("nan"))
    c, d = checked(policy, obs)
    assert d[0] == 1, "One invalid zone cannot disable its AHU"
    obs[meta["observation_field_names"].index(
        "VAV_0_zone_air_heating_temperature_setpoint_sensor")] = float("nan")
    checked(policy, obs)  # Missing live setpoint uses comfort fallback.
    # If a reheat valve action is unavailable, don't enable idle hot water.
    meta["zone_action_indices"]["VAV_0"]["reheat_valve_percentage_command"] = None
    zone_temp(0, 282.0)
    zone_temp(1, 295.0)
    _, d = checked(Policy(meta, None), obs)
    assert d[-1] == 0


def test_latency_and_entrypoint():
    meta, obs, zone_temp, _ = scenario(6, 10)
    policy = Policy(meta, None)
    start = time.perf_counter()
    for step in range(128):
        zone_temp(0, 307.0 - (step % 10)*0.1)
        checked(policy, obs)
    duration = time.perf_counter() - start
    assert duration < 10.0, f"128 steps exceeded lightweight budget: {duration:.2f}s"

    class FakeServer:
        def __init__(self, klass):
            assert klass is Policy
        def run(self):
            self.called = True

    gateway.SmartBuildingsInferenceServer = FakeServer
    exec(gateway_code, namespace)
    return duration


if __name__ == "__main__":
    test_heating_cooling_and_reset()
    test_trim_respond_and_comfort_guard()
    test_flow_warmup_missing_and_bad_telemetry()
    test_independent_ahus_and_stop()
    test_missing_sensors_setpoints_and_long_ahu_list()
    seconds = test_latency_and_entrypoint()
    print("PASS V10: 6 grouped checks (notebook/gateway, hybrid actions, "
          "mixed comfort, trim/respond/limits, startup gating, missing sensors, "
          f"independent AHUs, reset, 128-step runtime {seconds:.3f}s)")
