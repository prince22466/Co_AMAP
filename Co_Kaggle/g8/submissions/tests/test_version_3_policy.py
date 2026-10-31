"""Synthetic interface tests for the exact Version 3 notebook policy.

Run: python -m unittest discover -s Co_Kaggle/g8/submissions/tests -v
These are NOT simulator, energy/comfort, or hidden Kaggle score tests.
"""
import json
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

NOTEBOOK = Path(__file__).resolve().parents[1] / "submission-nb - Version 3.ipynb"


def load_policy():
    server = types.ModuleType("smart_buildings_inference_server")
    server.Policy = type("Policy", (), {
        "__init__": lambda self, building_meta_data, action_spec: None})
    server.SmartBuildingsInferenceServer = type("Server", (), {
        "__init__": lambda self, policy: None, "run": lambda self: None})
    package = types.ModuleType("kaggle_evaluation")
    package.smart_buildings_inference_server = server
    sys.modules["kaggle_evaluation"] = package
    sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = server
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "".join(notebook["cells"][1]["source"])
    scope = {}
    exec(compile(src, str(NOTEBOOK), "exec"), scope)
    return scope["TransferPolicy"]


TransferPolicy = load_policy()


def fixture():
    names = ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000"]
    for i in range(3):
        names += [
            f"VAV_{i}_zone_air_temperature_sensor",
            f"VAV_{i}_zone_air_heating_temperature_setpoint_sensor",
            f"VAV_{i}_zone_air_cooling_temperature_setpoint_sensor",
        ]
    names += ["ahs_ahu_0_outside_air_temperature_sensor",
              "ahs_ahu_1_outside_air_temperature_sensor"]
    meta = {
        "n_continuous_action": 14, "n_discrete_action": 3,
        "observation_field_names": names,
        "temp_indices": [0, 3], "pressure_indices": [1, 4, 6],
        "recirculation_indices": [2, 5], "water_indices": [7],
        "comfort_temp_window": [293., 297.],
        "eco_temp_window": [289., 302.],
        "morning_start_hour": 8, "evening_start_hour": 18,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 295., "sample_std": 2.}},
        "zone_action_indices": {
            f"VAV_{i}": {
                "supply_air_damper_percentage_command": 8 + 2 * i,
                "reheat_valve_percentage_command": 9 + 2 * i,
            } for i in range(3)},
        "ahu_zone_assignments": {
            "ahu_0": ["room_0", "room_1"], "ahu_1": ["room_2"]},
    }
    obs = np.zeros(len(names), dtype=np.float64)
    obs[0] = math.sin(2 * math.pi * 10 / 24)
    obs[1] = math.cos(2 * math.pi * 10 / 24)
    obs[2] = 0.0
    obs[3] = 1.0
    for i in range(3):
        obs[4 + 3 * i] = 0.0
        obs[5 + 3 * i] = 293.0
        obs[6 + 3 * i] = 297.0
    obs[-2:] = [290., 290.]
    return meta, obs


class Version3InterfaceTests(unittest.TestCase):

    def make_action(self, temps, tail=True, water=True):
        meta, obs = fixture()
        for i, temp in enumerate(temps):
            obs[4 + 3 * i] = (temp - 295.) / 2.
        policy = TransferPolicy(meta, None)
        policy.TAIL_GUARD = tail
        policy.WATER_ON_DEMAND = water
        action = policy.get_action(obs)
        cont, disc = action["continuous_action"], action["discrete_action"]
        self.assertEqual(cont.shape, (14,))
        self.assertEqual(disc.shape, (3,))
        self.assertEqual(cont.dtype, np.float32)
        self.assertEqual(disc.dtype, np.int32)
        self.assertTrue(np.all(np.isfinite(cont)))
        self.assertTrue(np.all((cont >= -1) & (cont <= 1)))
        self.assertTrue(np.all((disc == 0) | (disc == 1)))
        return policy, action

    def test_no_demand_equipment_idle(self):
        _, action = self.make_action([295., 295., 295.])
        np.testing.assert_array_equal(action["discrete_action"], [0, 0, 0])
        self.assertEqual(action["continuous_action"][6], -1.)

    def test_cooling_only_no_hot_water(self):
        _, action = self.make_action([301., 301., 301.])
        self.assertEqual(action["discrete_action"][0], 1)
        self.assertEqual(action["discrete_action"][-1], 0)
        self.assertEqual(action["continuous_action"][6], -1.)
        self.assertLess(action["continuous_action"][0], 0.)

    def test_heating_activates_water_and_reheat(self):
        _, action = self.make_action([288., 288., 288.])
        self.assertEqual(action["discrete_action"][-1], 1)
        self.assertEqual(action["continuous_action"][6], 1.)
        self.assertGreater(action["continuous_action"][0], 0.)
        self.assertEqual(action["continuous_action"][9], 1.)

    def test_tail_guard_catches_cancelled_zone_mean(self):
        _, action = self.make_action([287., 303., 295.])
        self.assertEqual(action["discrete_action"][0], 1)
        self.assertGreater(action["continuous_action"][0], 0.)
        _, control = self.make_action([287., 303., 295.], tail=False)
        self.assertEqual(control["discrete_action"][0], 0)

    def test_mixed_mode_reheat(self):
        _, action = self.make_action([292., 307., 295.])
        self.assertLess(action["continuous_action"][0], 0.)
        self.assertEqual(action["discrete_action"][-1], 1)

    def test_disabled_water_gate_mimics_original_pump(self):
        _, action = self.make_action([301., 301., 301.], water=False)
        self.assertEqual(action["discrete_action"][-1], 1)

    def test_reset_and_nan_sensor(self):
        meta, obs = fixture()
        policy = TransferPolicy(meta, None)
        policy.get_action(obs)
        policy.reset()
        self.assertTrue(np.all(policy.integral == 0))
        obs[4] = np.nan
        obs[5] = np.nan
        action = policy.get_action(obs)
        self.assertTrue(np.all(np.isfinite(action["continuous_action"])))

    def test_other_layout_one_ahu(self):
        meta, obs = fixture()
        meta["ahu_zone_assignments"] = {"ahu_0": ["room_0"]}
        meta["temp_indices"] = [0]
        meta["recirculation_indices"] = [2]
        meta["pressure_indices"] = [1, 6]
        meta["n_discrete_action"] = 2
        action = TransferPolicy(meta, None).get_action(obs)
        self.assertEqual(action["discrete_action"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
