"""Synthetic tests for Version 19 daily iterative learning control.

No organizer simulator is required for this gateway/learning unit test.
"""
import json
import math
from pathlib import Path
import sys
import types
import unittest

import numpy as np


NOTEBOOK = Path(__file__).with_name("submission-nb - Version 19.ipynb")


def policy_classes():
    server = types.ModuleType("smart_buildings_inference_server")
    server.Policy = type("Policy", (), {"__init__": lambda self, m, s: None})
    gateway = types.ModuleType("kaggle_evaluation")
    gateway.smart_buildings_inference_server = server
    original = sys.modules.get("kaggle_evaluation")
    sys.modules["kaggle_evaluation"] = gateway
    try:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert "SmartBuildingsInferenceServer(TransferPolicy).run()" in "".join(
            notebook["cells"][-1]["source"])
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"][1:-1])
        namespace = {}
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    finally:
        if original is None:
            sys.modules.pop("kaggle_evaluation", None)
        else:
            sys.modules["kaggle_evaluation"] = original
    return namespace["PhysicsBaselinePolicy"], namespace["TransferPolicy"]


BASE, ILC = policy_classes()


def metadata():
    names = []
    actions = {}
    for i in range(2):
        vav = f"VAV_{i}"
        names.extend([
            vav + "_zone_air_temperature_sensor",
            vav + "_zone_air_heating_temperature_setpoint_sensor",
            vav + "_zone_air_cooling_temperature_setpoint_sensor",
        ])
        actions[vav] = {
            "supply_air_damper_percentage_command": 5 + 2*i,
            "reheat_valve_percentage_command": 6 + 2*i,
        }
    names.extend([
        "ahs_0_outside_air_temperature_sensor",
        "hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000",
    ])
    return {
        "n_continuous_action": 9,
        "n_discrete_action": 2,
        "observation_field_names": names,
        "temp_indices": [0],
        "pressure_indices": [1, 3],
        "recirculation_indices": [2],
        "water_indices": [4],
        "ahu_zone_assignments": {"0": ["room_0", "room_1"]},
        "zone_action_indices": actions,
        "comfort_temp_window": [294.0, 297.0],
        "eco_temp_window": [292.0, 299.0],
        "morning_start_hour": 8.0,
        "evening_start_hour": 18.0,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 295.0, "sample_std": 10.0},
            "outside_air_temperature_sensor": {"sample_mean": 295.0, "sample_std": 10.0},
        }
    }


def observation(slot, dow=1, zone_temperatures=(300.0, 291.0),
                outdoor_k=294.0, valid_clock=True):
    values = []
    for temp in zone_temperatures:
        values.extend([(temp - 295.0)/10.0, 294.0, 297.0])
    hour = slot * 2.0 * math.pi/288.0
    day = dow * 2.0 * math.pi/7.0
    values.extend([
        (outdoor_k-295.0)/10.0,
        math.sin(hour), math.cos(hour),
        math.sin(day), math.cos(day),
    ])
    if not valid_clock:
        values[-4] = float("nan")
    return np.asarray(values, dtype=np.float64)


def assert_action(test, action):
    continuous, discrete = action["continuous_action"], action["discrete_action"]
    test.assertEqual(continuous.shape, (9,))
    test.assertEqual(discrete.shape, (2,))
    test.assertEqual(continuous.dtype, np.float32)
    test.assertEqual(discrete.dtype, np.int32)
    test.assertTrue(np.isfinite(continuous).all())
    test.assertTrue(np.all((-1.0 <= continuous) & (continuous <= 1.0)))
    test.assertTrue(set(discrete.tolist()).issubset({0, 1}))


class IterativeLearningTests(unittest.TestCase):
    def setUp(self):
        self.policy = ILC(metadata(), None)
        self.baseline = BASE(metadata(), None)

    def first_day(self, dow=1, outdoor=294.0):
        for slot in range(288):
            self.policy.get_action(observation(slot, dow=dow, outdoor_k=outdoor))

    def test_first_day_equals_baseline_and_is_safe(self):
        for slot in range(288):
            obs = observation(slot, dow=1)
            actual = self.policy.get_action(obs)
            baseline = self.baseline.get_action(obs)
            np.testing.assert_array_equal(
                actual["continuous_action"], baseline["continuous_action"])
            np.testing.assert_array_equal(
                actual["discrete_action"], baseline["discrete_action"])
            assert_action(self, actual)

    def test_second_weekday_learns_separately_for_cooling_and_reheat(self):
        self.first_day()
        obs = observation(0, dow=2)
        action = self.policy.get_action(obs)
        original = self.baseline.get_action(obs)
        # Slot zero updates from the temperature measured at slot three.
        self.assertGreater(float(action["continuous_action"][5]),
                           float(original["continuous_action"][5]))
        self.assertGreater(float(action["continuous_action"][8]),
                           float(original["continuous_action"][8]))
        np.testing.assert_array_equal(
            action["discrete_action"], original["discrete_action"])
        assert_action(self, action)

    def test_weekend_profile_is_independent(self):
        self.first_day(dow=1)
        obs = observation(0, dow=6)
        action = self.policy.get_action(obs)
        original = self.baseline.get_action(obs)
        np.testing.assert_array_equal(
            action["continuous_action"], original["continuous_action"])

    def test_changed_weather_disables_old_profile(self):
        self.first_day()
        obs = observation(0, dow=2, outdoor_k=304.0)
        action = self.policy.get_action(obs)
        original = self.baseline.get_action(obs)
        np.testing.assert_array_equal(
            action["continuous_action"], original["continuous_action"])

    def test_reset_clears_all_learning(self):
        self.first_day()
        self.policy.get_action(observation(0, dow=2))
        self.policy.reset()
        obs = observation(1, dow=2)
        np.testing.assert_array_equal(
            self.policy.get_action(obs)["continuous_action"],
            self.baseline.get_action(obs)["continuous_action"])

    def test_partial_day_does_not_train(self):
        for slot in range(100, 288):
            self.policy.get_action(observation(slot))
        obs = observation(0, dow=2)
        np.testing.assert_array_equal(
            self.policy.get_action(obs)["continuous_action"],
            self.baseline.get_action(obs)["continuous_action"])

    def test_invalid_clock_falls_back_without_training(self):
        self.first_day()
        obs = observation(0, dow=2, valid_clock=False)
        np.testing.assert_array_equal(
            self.policy.get_action(obs)["continuous_action"],
            self.baseline.get_action(obs)["continuous_action"])
        self.assertIsNone(self.policy.day_data)

    def test_invalid_zone_and_saturation(self):
        obs = observation(0, zone_temperatures=(float("nan"), 250.0))
        assert_action(self, self.policy.get_action(obs))
        self.first_day()
        for slot in range(288):
            assert_action(self, self.policy.get_action(
                observation(slot, dow=2, zone_temperatures=(320.0, 250.0))))

    def test_true_delayed_error_update_and_bounds(self):
        self.first_day()
        cool = self.policy.profiles["weekday"]["cool"]
        heat = self.policy.profiles["weekday"]["heat"]
        self.assertGreater(float(cool[0, 0]), 0.0)
        self.assertGreater(float(heat[1, 0]), 0.0)
        self.assertEqual(float(cool[0, -1]), 0.0)  # no future response within day
        self.assertLessEqual(float(np.max(np.abs(cool))), ILC.MAX_COOL_CORRECTION)
        self.assertLessEqual(float(np.max(np.abs(heat))), ILC.MAX_HEAT_CORRECTION)


if __name__ == "__main__":
    unittest.main()
