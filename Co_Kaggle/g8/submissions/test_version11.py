"""Synthetic contract/behavior tests for the EXACT submitted V11 notebook.

No Kaggle simulator or reward is used here. The gateway is stubbed so the
actual policy cell can run with only NumPy installed.
"""
import json
from pathlib import Path
import sys
import time
import types

import numpy as np


NOTEBOOK_PATH = Path(__file__).with_name("submission-nb - Version 11.ipynb")
notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4 and len(notebook["cells"]) == 3
assert all(cell.get("id") for cell in notebook["cells"])
policy_code = "".join(notebook["cells"][1]["source"])
gateway_code = "".join(notebook["cells"][2]["source"])
compile(policy_code, "<V11 submitted policy>", "exec")
compile(gateway_code, "<V11 inference gateway>", "exec")

kaggle_package = types.ModuleType("kaggle_evaluation")
gateway = types.ModuleType("smart_buildings_inference_server")
gateway.Policy = type("Policy", (), {"__init__": lambda self, meta, spec: None})
kaggle_package.smart_buildings_inference_server = gateway
sys.modules["kaggle_evaluation"] = kaggle_package
sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = gateway
namespace = {}
exec(policy_code, namespace)
Policy = namespace["TransferPolicy"]


def scenario(ahu_count=2, zones_each=3, with_airflow=True,
             with_supply_sensors=False):
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
        if with_supply_sensors:
            fields.append(f"ahs_{ahu_name}_supply_air_temperature_sensor")
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
            "zone_air_temperature_sensor": {
                "sample_mean": 295.0, "sample_std": 5.0
            },
            "supply_air_temperature_sensor": {
                "sample_mean": 290.0, "sample_std": 5.0
            } if with_supply_sensors else {},
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
        elif name.endswith("supply_air_temperature_sensor"):
            observation[index] = 2.0  # Normalized (300 K - 290 K) / 5 K
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



def test_hybrid_contract_and_plant_modes():
    """Heating, cooling and mixed loads produce valid independent actions."""
    for ahus, zones in ((1, 1), (2, 4), (4, 8)):
        meta, obs, set_temp, _ = scenario(ahus, zones)
        policy = Policy(meta, None)
        _, switches = checked(policy, obs)
        assert not np.any(switches), "In-band zones should be idle"

        set_temp(0, 306.0)
        actions, switches = checked(policy, obs)
        assert switches[0] == 1 and switches[-1] == 0
        assert actions[meta["temp_indices"][0]] < 1.0

        set_temp(0, 282.0)
        actions, switches = checked(policy, obs)
        assert switches[0] == 1 and switches[-1] == 1
        assert actions[meta["temp_indices"][0]] == 1.0, "Coil cannot heat"

        if zones > 1:
            set_temp(0, 299.0)
            set_temp(1, 290.0)
            actions, switches = checked(policy, obs)
            valve_idx = meta["zone_action_indices"]["VAV_1"][
                "reheat_valve_percentage_command"
            ]
            assert switches[0] == switches[-1] == 1
            assert actions[valve_idx] > -1.0
        policy.reset()
        previous = checked(policy, obs)
        current = checked(policy, obs)
        assert all(np.array_equal(a, b) for a, b in zip(previous, current))


def test_reheat_aware_supply_reset_and_guard():
    """Heating conflict must raise SAT, but not past hot-zone headroom."""
    meta, obs, set_temp, _ = scenario(1, 4)
    policy = Policy(meta, None)
    set_temp(0, 300.0)  # 3 K cooling error
    baseline, _ = checked(policy, obs)
    pure_cooling_k = 295.0 + 10.0 * float(baseline[meta["temp_indices"][0]])

    set_temp(1, 290.0)
    set_temp(2, 290.0)
    mixed, _ = checked(policy, obs)
    mixed_k = 295.0 + 10.0 * float(mixed[meta["temp_indices"][0]])
    assert mixed_k > pure_cooling_k + 0.5
    headroom_k = (policy.STRONG_COOLING_HEADROOM_K
                  + policy.STRONG_COOLING_HEADROOM_GAIN * 3.0)
    assert mixed_k <= 300.0 - headroom_k + 1e-5
    assert 285.0 <= mixed_k <= 305.0

    # Disabling the new mechanism should restore the cooling-only SAT.
    policy.MAX_REHEAT_AVOIDANCE_RESET_K = 0.0
    ablated, _ = checked(policy, obs)
    ablated_k = 295.0 + 10.0 * float(ablated[meta["temp_indices"][0]])
    assert abs(ablated_k - pure_cooling_k) < 1e-4


def test_mild_conflict_and_terminal_coupling():
    meta, obs, set_temp, _ = scenario(1, 2)
    policy = Policy(meta, None)
    set_temp(0, 298.0)
    cooling, _ = checked(policy, obs)
    supply_cool = 295.0 + 10 * float(cooling[meta["temp_indices"][0]])
    damper_idx = meta["zone_action_indices"]["VAV_0"][
        "supply_air_damper_percentage_command"
    ]
    cooling_damper = float(cooling[damper_idx])

    set_temp(1, 290.0)
    mixed, switches = checked(policy, obs)
    supply_mixed = 295.0 + 10 * float(mixed[meta["temp_indices"][0]])
    assert supply_mixed > supply_cool
    assert supply_mixed <= 298.0 - policy.MIN_COOLING_HEADROOM_K + 1e-5
    assert mixed[damper_idx] > cooling_damper, "Warmer SAT requires more damper"
    assert switches[-1] == 1
    assert abs(fan_pa(mixed, meta) - policy.ACTIVE_FAN_PRESSURE_PA) < 1e-4


def test_severe_cooling_and_multiple_ahus():
    """Large cooling error is not erased by other zones' heating need."""
    meta, obs, set_temp, _ = scenario(2, 4)
    policy = Policy(meta, None)
    set_temp(0, 307.0)  # 10 K cooling error, clamped headroom guard
    for room in (1, 2, 3):
        set_temp(room, 283.0)
    set_temp(4, 290.0)
    actions, switches = checked(policy, obs)
    sat0 = 295.0 + 10 * float(actions[meta["temp_indices"][0]])
    sat1 = 295.0 + 10 * float(actions[meta["temp_indices"][1]])
    assert sat0 < 297.0
    assert sat1 == 305.0, "Heating-only AHU should disable its cooling coil"
    assert switches[0] == switches[1] == switches[-1] == 1
    assert abs(fan_pa(actions, meta, 0) - 850.0) < 1e-4
    assert abs(fan_pa(actions, meta, 1) - 850.0) < 1e-4


def test_bad_sensors_mapping_and_missing_valves():
    meta, obs, set_temp, _ = scenario(12, 1, with_airflow=False)
    set_temp(10, 310.0)
    policy = Policy(meta, None)
    _, switches = checked(policy, obs)
    assert switches[10] == 1 and switches[2] == 0

    meta, obs, set_temp, _ = scenario(2, 3, with_airflow=False)
    set_temp(0, 301.0)
    set_temp(1, float("nan"))
    policy = Policy(meta, None)
    _, switches = checked(policy, obs)
    assert switches[0] == 1

    heat_name = "VAV_0_zone_air_heating_temperature_setpoint_sensor"
    obs[meta["observation_field_names"].index(heat_name)] = float("nan")
    checked(policy, obs)
    meta["zone_action_indices"]["VAV_0"]["reheat_valve_percentage_command"] = None
    set_temp(0, 280.0)
    set_temp(1, 295.0)
    _, switches = checked(Policy(meta, None), obs)
    assert switches[-1] == 0, "No valid reheat valve, no hot-water demand"


def test_safeguard_binds_for_mild_cooling():
    """An actual headroom cap must bind, not merely remain inactive."""
    meta, obs, set_temp, _ = scenario(1, 4)
    set_temp(0, 297.35)  # 0.35 K above cooling setpoint
    for zone in (1, 2, 3):
        set_temp(zone, 289.0)
    policy = Policy(meta, None)
    actions, switches = checked(policy, obs)
    supply_k = 295.0 + 10.0 * float(actions[meta["temp_indices"][0]])
    assert switches[0] == switches[-1] == 1
    assert abs(supply_k - 296.95) < 1e-3, "Hot-zone guard must bind"

    policy.MAX_REHEAT_AVOIDANCE_RESET_K = 0.0
    no_reset, _ = checked(policy, obs)
    no_reset_k = 295.0 + 10.0 * float(no_reset[meta["temp_indices"][0]])
    assert supply_k > no_reset_k + 0.5, "The guard must allow some reset"


def test_measured_supply_telemetry_and_fallbacks():
    """A cold supply sensor must strengthen reheat only for its own AHU."""
    meta, obs, set_temp, _ = scenario(2, 2, with_supply_sensors=True)
    set_temp(0, 291.5)
    set_temp(2, 291.5)
    policy = Policy(meta, None)
    field_idx = {
        name: index for index, name in enumerate(meta["observation_field_names"])
    }
    vav_indices = meta["zone_action_indices"]
    first_valve = vav_indices["VAV_0"]["reheat_valve_percentage_command"]
    other_valve = vav_indices["VAV_2"]["reheat_valve_percentage_command"]
    first_sensor = field_idx["ahs_ahu_0_supply_air_temperature_sensor"]
    other_sensor = field_idx["ahs_ahu_1_supply_air_temperature_sensor"]

    initial, switches = checked(policy, obs)
    assert switches[0] == switches[1] == switches[-1] == 1
    assert abs(float(initial[first_valve]) - float(initial[other_valve])) < 1e-6

    # One AHU delivers 275 K air despite the 305 K cooling-coil command.
    obs[first_sensor] = (275.0 - 290.0) / 5.0
    cold, _ = checked(policy, obs)
    assert cold[first_valve] > initial[first_valve] + 0.5
    assert abs(float(cold[other_valve]) - float(initial[other_valve])) < 1e-6

    # Reject malformed and physically implausible normalized observations.
    obs[first_sensor] = float("nan")
    invalid, _ = checked(policy, obs)
    assert abs(float(invalid[first_valve]) - float(initial[first_valve])) < 1e-6
    obs[first_sensor] = (500.0 - 290.0) / 5.0
    invalid, _ = checked(policy, obs)
    assert abs(float(invalid[first_valve]) - float(initial[first_valve])) < 1e-6

    # Missing supply normalization does not treat normalized data as kelvin.
    meta["observation_normalization_metadata"].pop("supply_air_temperature_sensor")
    obs[first_sensor] = (275.0 - 290.0) / 5.0
    unsupported, _ = checked(Policy(meta, None), obs)
    assert abs(float(unsupported[first_valve]) - float(initial[first_valve])) < 1e-6

    # Cooling damper should not increase because of a warm *command* if the
    # observed supply is already much colder; matched telemetry matters.
    meta["observation_normalization_metadata"]["supply_air_temperature_sensor"] = {
        "sample_mean": 290.0, "sample_std": 5.0
    }
    set_temp(0, 298.0)
    set_temp(2, 295.0)
    obs[first_sensor] = (300.0 - 290.0) / 5.0
    obs[other_sensor] = (300.0 - 290.0) / 5.0
    policy = Policy(meta, None)
    warm_supply, _ = checked(policy, obs)
    obs[first_sensor] = (280.0 - 290.0) / 5.0
    cold_supply, _ = checked(policy, obs)
    damper_idx = vav_indices["VAV_0"]["supply_air_damper_percentage_command"]
    assert cold_supply[damper_idx] < warm_supply[damper_idx]


def test_runtime_and_gateway():
    """128 invocations and the exact organizer gateway cell."""
    meta, obs, set_temp, _ = scenario(6, 10)
    policy = Policy(meta, None)
    start = time.perf_counter()
    for step in range(128):
        set_temp(0, 307.0 - (step % 10) * 0.1)
        checked(policy, obs)
    duration = time.perf_counter() - start
    assert duration < 10.0, f"128 steps exceeded budget: {duration:.2f}s"

    class FakeServer:
        def __init__(self, klass):
            assert klass is Policy

        def run(self):
            self.called = True

    gateway.SmartBuildingsInferenceServer = FakeServer
    exec(gateway_code, namespace)
    return duration


if __name__ == "__main__":
    test_hybrid_contract_and_plant_modes()
    test_reheat_aware_supply_reset_and_guard()
    test_mild_conflict_and_terminal_coupling()
    test_severe_cooling_and_multiple_ahus()
    test_bad_sensors_mapping_and_missing_valves()
    test_safeguard_binds_for_mild_cooling()
    test_measured_supply_telemetry_and_fallbacks()
    seconds = test_runtime_and_gateway()
    print("PASS V11: 8 grouped checks (actual notebook/gateway, dynamic "
          "action contract, reheat-aware reset/ablation, binding cooling "
          "headroom guard, matched normalized AHU telemetry, independent "
          "multi-AHU control, missing sensors, and 128-step "
          f"runtime {seconds:.3f}s)")
