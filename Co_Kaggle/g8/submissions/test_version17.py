"""Synthetic regression tests for the exact Version 17 Kaggle notebook code.

No organizer simulator or hidden scenario data is used. Invoke:
    python Co_Kaggle/g8/submissions/test_version17.py
"""
import json
import pathlib
import sys
import types
import unittest

import numpy as np


NOTEBOOK = pathlib.Path(__file__).with_name("submission-nb - Version 17.ipynb")


def load_policy_code():
    """Execute exactly the notebook's controller cell with a stubbed gateway."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) >= 3
    code = "".join(notebook["cells"][1]["source"])
    gateway = "".join(notebook["cells"][-1]["source"])
    assert "SmartBuildingsInferenceServer(TransferPolicy).run()" in gateway

    evaluation = types.ModuleType("kaggle_evaluation")
    inference = types.ModuleType("smart_buildings_inference_server")

    class StubPolicy:
        def __init__(self, building_meta_data, action_spec):
            self.action_spec = action_spec

    inference.Policy = StubPolicy
    evaluation.smart_buildings_inference_server = inference
    previous = sys.modules.get("kaggle_evaluation")
    sys.modules["kaggle_evaluation"] = evaluation
    try:
        namespace = {"__name__": "version17_under_test"}
        exec(compile(code, str(NOTEBOOK), "exec"), namespace)
    finally:
        if previous is None:
            sys.modules.pop("kaggle_evaluation", None)
        else:
            sys.modules["kaggle_evaluation"] = previous
    return namespace["TransferPolicy"], namespace["PressureExtremumSeeker"]


TransferPolicy, PressureESC = load_policy_code()


def make_case(zones_per_ahu=(2,), zone_temp_k=297.0, outdoor_k=300.0):
    """Provide a realistic indexed action/observation layout for N AHUs."""
    ahus = len(zones_per_ahu)
    num_zones = sum(zones_per_ahu)
    water_supply = 3 * ahus
    water_pressure = water_supply + 1
    fields, readings = [], []
    assignments, action_indices = {}, {}
    zone_start = water_pressure + 1
    offset = 0
    for ahu, count in enumerate(zones_per_ahu):
        assignments[str(ahu)] = []
        fields.append(f"ahs_{ahu}_outside_air_temperature_sensor")
        readings.append((outdoor_k - 290.0) / 5.0)
        for _ in range(count):
            room = f"room_{offset}"
            vav = f"VAV_{offset}"
            assignments[str(ahu)].append(room)
            action_indices[vav] = {
                "supply_air_damper_percentage_command": zone_start + 2 * offset,
                "reheat_valve_percentage_command": zone_start + 2 * offset + 1,
            }
            fields.extend([
                f"{vav}_zone_air_temperature_sensor",
                f"{vav}_zone_air_heating_temperature_setpoint_sensor",
                f"{vav}_zone_air_cooling_temperature_setpoint_sensor",
            ])
            readings.extend([(zone_temp_k - 295.0) / 2.0, 294.0, 296.0])
            offset += 1
    fields += ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000"]
    readings += [0.0, 1.0, 0.0, 1.0]
    metadata = {
        "n_continuous_action": zone_start + 2 * num_zones,
        "n_discrete_action": ahus + 1,
        "observation_field_names": fields,
        "temp_indices": [3 * i for i in range(ahus)],
        "pressure_indices": [3 * i + 1 for i in range(ahus)] + [water_pressure],
        "recirculation_indices": [3 * i + 2 for i in range(ahus)],
        "water_indices": [water_supply],
        "ahu_zone_assignments": assignments,
        "zone_action_indices": action_indices,
        "comfort_temp_window": [294.0, 296.0],
        "eco_temp_window": [292.0, 300.0],
        "morning_start_hour": 8.0,
        "evening_start_hour": 18.0,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 295.0, "sample_std": 2.0},
            "outside_air_temperature_sensor": {"sample_mean": 290.0, "sample_std": 5.0},
        },
    }
    return metadata, np.asarray(readings, dtype=np.float64)


def check_contract(test, policy, observation):
    result = policy.get_action(observation)
    test.assertEqual(set(result), {"continuous_action", "discrete_action"})
    continuous = result["continuous_action"]
    discrete = result["discrete_action"]
    test.assertEqual(continuous.dtype, np.float32)
    test.assertEqual(discrete.dtype, np.int32)
    test.assertEqual(continuous.shape, (policy.n_continuous,))
    test.assertEqual(discrete.shape, (policy.n_discrete,))
    test.assertTrue(np.all(np.isfinite(continuous)))
    test.assertTrue(np.all(continuous >= -1.0) and np.all(continuous <= 1.0))
    test.assertTrue(np.all((discrete == 0) | (discrete == 1)))
    return result


class ESCUnitTests(unittest.TestCase):
    def test_uses_previous_command_cost_and_moves_downhill(self):
        seeker = PressureESC(block_steps=4, settling_steps=1)
        last_pressure = 1000.0
        for _ in range(4 * 4 + 2):
            observed_comfort = ((last_pressure - 850.0) / 200.0) ** 2
            last_pressure = seeker.choose_pressure(
                1000.0, observed_comfort, ((294.0, 296.0),), True)
        self.assertGreaterEqual(seeker.cycles_completed, 1)
        self.assertLess(seeker.center_pa, 0.0)
        self.assertLessEqual(abs(seeker.center_pa), seeker.MAX_UPDATE_PA)
        self.assertTrue(np.isfinite(seeker.gradient_ema))

    def test_symmetric_abba_probe_and_saturation(self):
        seeker = PressureESC(block_steps=4, settling_steps=1)
        commands = []
        for _ in range(4 * 4 + 1):
            commands.append(seeker.choose_pressure(
                1000, 0.0, ("constant",), True))
        self.assertEqual(commands[0], 1080.0)
        self.assertEqual(commands[4], 920.0)
        self.assertEqual(commands[8], 920.0)
        self.assertEqual(commands[12], 1080.0)
        seeker.reset()
        self.assertEqual(seeker.choose_pressure(
            240, 0.0, ("constant",), True), 240.0)
        self.assertIsNone(seeker.pending_pressure_pa)

    def test_aborts_invalid_or_changed_context_without_forgetting_center(self):
        seeker = PressureESC(block_steps=4, settling_steps=1)
        for _ in range(3):
            seeker.choose_pressure(900, 0.7, ((294, 296),), True)
        seeker.center_pa = -55.0
        seeker.choose_pressure(900, 0.7, ((292, 300),), True)
        self.assertEqual(seeker.steps, 0)
        self.assertEqual(seeker.block, 0)
        self.assertEqual(seeker.center_pa, -55.0)
        seeker.choose_pressure(900, float("nan"), ((292, 300),), True)
        self.assertIsNone(seeker.pending_pressure_pa)
        seeker.reset()
        self.assertEqual(seeker.center_pa, 0.0)
        self.assertEqual(seeker.cycles_completed, 0)


class PolicyTests(unittest.TestCase):
    def test_contract_across_multi_ahu_topologies(self):
        for layout in [(1,), (3,), (1, 2), (2, 3, 1)]:
            with self.subTest(layout=layout):
                metadata, obs = make_case(layout)
                policy = TransferPolicy(metadata, {})
                result = check_contract(self, policy, obs)
                self.assertTrue(np.all(result["discrete_action"][:-1] == 1))
                self.assertEqual(len(policy.seekers), len(layout))

    def test_normalization_and_hot_cold_idle(self):
        metadata, obs = make_case()
        policy = TransferPolicy(metadata, {})
        outside_index = policy.ahus[0]["outdoor"]
        self.assertAlmostEqual(policy._outside_temperature(obs, outside_index), 300.0)
        hot = check_contract(self, policy, obs)
        self.assertEqual(hot["discrete_action"][0], 1)

        cold_meta, cold_obs = make_case(zone_temp_k=292.0)
        cold = check_contract(self, TransferPolicy(cold_meta, {}), cold_obs)
        self.assertEqual(cold["discrete_action"][0], 1)
        self.assertEqual(cold["discrete_action"][-1], 1)

        idle_meta, idle_obs = make_case(zone_temp_k=295.0)
        idle = check_contract(self, TransferPolicy(idle_meta, {}), idle_obs)
        self.assertEqual(idle["discrete_action"][0], 0)

    def test_severe_case_reverts_to_original_pressure(self):
        metadata, obs = make_case(zone_temp_k=299.0)
        esc = TransferPolicy(metadata, {})
        fixed = TransferPolicy(metadata, {})
        fixed.ENABLE_ESC = False
        esc_result = check_contract(self, esc, obs)
        fixed_result = check_contract(self, fixed, obs)
        idx = metadata["pressure_indices"][0]
        self.assertEqual(esc_result["continuous_action"][idx],
                         fixed_result["continuous_action"][idx])
        self.assertIsNone(esc.seekers[0].pending_pressure_pa)

    def test_learning_is_disabled_by_ablation_switch(self):
        metadata, obs = make_case()
        esc = TransferPolicy(metadata, {})
        fixed = TransferPolicy(metadata, {})
        fixed.ENABLE_ESC = False
        experimental = check_contract(self, esc, obs)
        reference = check_contract(self, fixed, obs)
        idx = metadata["pressure_indices"][0]
        self.assertGreater(experimental["continuous_action"][idx],
                           reference["continuous_action"][idx])
        self.assertIsNotNone(esc.seekers[0].pending_pressure_pa)
        self.assertIsNone(fixed.seekers[0].pending_pressure_pa)

    def test_reset_and_missing_sensor_recovery(self):
        metadata, obs = make_case((1, 2))
        policy = TransferPolicy(metadata, {})
        for _ in range(28):
            check_contract(self, policy, obs)
        self.assertGreater(policy.seekers[0].steps + policy.seekers[0].block, 0)
        with_missing = obs.copy()
        with_missing[policy.ahus[0]["zones"][0]["temperature"]] = float("nan")
        check_contract(self, policy, with_missing)
        self.assertIsNone(policy.seekers[0].pending_pressure_pa)
        policy.reset()
        self.assertEqual(policy.seekers[0].center_pa, 0.0)
        self.assertEqual(policy.seekers[0].block, 0)
        self.assertEqual(policy.seekers[0].cycles_completed, 0)


if __name__ == "__main__":
    unittest.main()
