"""Offline contract/regression tests for the Version 20 Kaggle notebook.

Run: python -m unittest discover -s Co_Kaggle/g8/tests -p 'test_version20.py'
A stub inference server is used; these are NOT organizer simulator scores.
"""
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def load_policy():
    fake = types.ModuleType("kaggle_evaluation")
    server = types.ModuleType("smart_buildings_inference_server")

    class Policy:
        def __init__(self, building_meta_data, action_spec):
            pass

    server.Policy = Policy
    fake.smart_buildings_inference_server = server
    sys.modules["kaggle_evaluation"] = fake
    sys.modules["smart_buildings_inference_server"] = server

    notebook_file = (Path(__file__).resolve().parents[1] / "submissions"
                     / "submission-nb - Version 20.ipynb")
    data = json.loads(notebook_file.read_text(encoding="utf-8"))
    assert len(data["cells"]) == 3
    namespace = {"__name__": "version20_under_test"}
    exec("".join(data["cells"][1]["source"]), namespace)
    return namespace["TransferPolicy"]


TransferPolicy = load_policy()


def metadata():
    names = [
        "VAV_1_zone_air_temperature_sensor",
        "VAV_1_zone_air_heating_temperature_setpoint_sensor",
        "VAV_1_zone_air_cooling_temperature_setpoint_sensor",
        "VAV_2_zone_air_temperature_sensor",
        "VAV_2_zone_air_heating_temperature_setpoint_sensor",
        "VAV_2_zone_air_cooling_temperature_setpoint_sensor",
        "ahs_ahu_1_outside_air_temperature_sensor",
    ]
    return {
        "n_continuous_action": 9,
        "n_discrete_action": 2,
        "observation_field_names": names,
        "temp_indices": [0],
        "pressure_indices": [1, 4],
        "recirculation_indices": [2],
        "water_indices": [3],
        "ahu_zone_assignments": {"ahu_1": ["room_1", "room_2"]},
        "zone_action_indices": {
            "VAV_1": {"supply_air_damper_percentage_command": 5,
                      "reheat_valve_percentage_command": 6},
            "VAV_2": {"supply_air_damper_percentage_command": 7,
                      "reheat_valve_percentage_command": 8},
        },
        "comfort_temp_window": [294.0, 297.0],
        "eco_temp_window": [290.0, 300.0],
        "morning_start_hour": 8,
        "evening_start_hour": 18,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {
                "sample_mean": 295.0, "sample_std": 5.0},
            "outside_air_temperature_sensor": {
                "sample_mean": 295.0, "sample_std": 10.0},
        },
    }


def observation(first=295.0, second=295.0, outdoor=291.0):
    return np.array([
        (first - 295.0) / 5.0, 294.0, 297.0,
        (second - 295.0) / 5.0, 294.0, 297.0,
        (outdoor - 295.0) / 10.0,
    ], dtype=np.float64)


class Version20Test(unittest.TestCase):
    def setUp(self):
        self.policy = TransferPolicy(metadata(), None)

    def assert_valid(self, action):
        self.assertEqual(set(action), {"continuous_action", "discrete_action"})
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        self.assertEqual(cont.shape, (9,))
        self.assertEqual(disc.shape, (2,))
        self.assertEqual(cont.dtype, np.float32)
        self.assertEqual(disc.dtype, np.int32)
        self.assertTrue(np.isfinite(cont).all())
        self.assertTrue(((cont >= -1.0) & (cont <= 1.0)).all())
        self.assertTrue(np.isin(disc, [0, 1]).all())

    def test_idle_equipment_is_off(self):
        action = self.policy.get_action(observation())
        self.assert_valid(action)
        self.assertEqual(action["discrete_action"].tolist(), [0, 0])
        self.assertEqual(action["continuous_action"][6], -1.0)

    def test_hot_zone_runs_ahu_but_not_boiler(self):
        action = self.policy.get_action(observation(first=300.0, outdoor=288.0))
        self.assert_valid(action)
        self.assertEqual(action["discrete_action"].tolist(), [1, 0])
        self.assertLess(action["continuous_action"][0], 1.0)
        self.assertGreater(action["continuous_action"][5], -0.8)
        self.assertEqual(action["continuous_action"][6], -1.0)

    def test_cold_zone_uses_reheat_and_hot_water(self):
        action = self.policy.get_action(observation(first=290.0, outdoor=280.0))
        self.assert_valid(action)
        self.assertEqual(action["discrete_action"].tolist(), [1, 1])
        self.assertEqual(action["continuous_action"][0], 1.0)
        self.assertGreater(action["continuous_action"][6], -1.0)

    def test_outdoor_normalization_uses_kelvin(self):
        obs = observation(outdoor=285.0)
        self.assertAlmostEqual(self.policy._outside_air(obs, self.policy.ahus[0]),
                               285.0)
        meta = metadata()
        del meta["observation_normalization_metadata"]["outside_air_temperature_sensor"]
        other = TransferPolicy(meta, None)
        self.assertTrue(np.isnan(other._outside_air(obs, other.ahus[0])))
        self.assert_valid(other.get_action(obs))

    def test_warming_trend_increases_cooling_surface(self):
        first = observation(first=296.8)
        second = observation(first=297.2)
        self.policy.get_action(first)
        self.policy.get_action(second)
        zone_index = self.policy.ahus[0]["zones"][0]["temperature"]
        self.assertGreater(self.policy._filtered_zone_trend[zone_index], 0.0)
        fresh = TransferPolicy(metadata(), None)
        old = fresh._read_zone(second, fresh.ahus[0]["zones"][0], fresh.comfort)
        new = self.policy._read_zone(second, self.policy.ahus[0]["zones"][0],
                                     self.policy.comfort)
        self.assertGreater(new["hot_surface"], old["hot_surface"])

    def test_hysteresis_and_episode_reset(self):
        self.policy.get_action(observation(first=297.25))
        keep = self.policy.get_action(observation(first=297.16))
        self.assertEqual(keep["discrete_action"][0], 1)
        self.policy.reset()
        reset_result = self.policy.get_action(observation(first=297.16))
        # 0.16 K is between the OFF (0.06) and ON (0.17) thresholds:
        # it remains on only when the previous AHU state has been retained.
        self.assertEqual(reset_result["discrete_action"][0], 0)
        self.assertEqual(len(self.policy._previous_zone_temp), 2)
        self.assertTrue(np.isclose(
            self.policy._filtered_zone_trend[self.policy.ahus[0]["zones"][0]["temperature"]],
            0.0))
        fresh = TransferPolicy(metadata(), None)
        np.testing.assert_array_equal(
            reset_result["continuous_action"],
            fresh.get_action(observation(first=297.16))["continuous_action"])

    def test_invalid_or_missing_observations_do_not_leak_history(self):
        self.policy.get_action(observation(first=300.0))
        obs = observation()
        obs[0] = np.nan
        action = self.policy.get_action(obs)
        self.assert_valid(action)
        self.assertNotIn(0, self.policy._previous_zone_temp)
        self.assertEqual(action["discrete_action"][0], 0)

    def test_bounded_for_extreme_temperatures(self):
        for t in (245.0, 259.0, 280.0, 298.0, 315.0, 340.0):
            self.policy.reset()
            self.assert_valid(self.policy.get_action(
                observation(first=t, second=315.0, outdoor=325.0)))


if __name__ == "__main__":
    unittest.main()
