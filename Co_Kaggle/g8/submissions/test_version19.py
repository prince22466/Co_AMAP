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
    """Run the *entire* Kaggle submission cell in a fresh mocked kernel.

    This fails if TransferPolicy is in an unexecuted earlier cell or if the
    official gateway gets called before the class exists.
    """
    gateway_calls = []
    server = types.ModuleType("smart_buildings_inference_server")
    server.Policy = type("Policy", (), {
        "__init__": lambda self, building_meta_data, action_spec: None,
    })

    class FakeInferenceServer:
        def __init__(self, policy_type):
            self.policy_type = policy_type

        def run(self):
            # Simulate the gateway resolving TransferPolicy at call time.
            assert isinstance(self.policy_type, type)
            assert issubclass(self.policy_type, server.Policy)
            gateway_calls.append(self.policy_type)

    server.SmartBuildingsInferenceServer = FakeInferenceServer
    gateway = types.ModuleType("kaggle_evaluation")
    gateway.smart_buildings_inference_server = server
    original = sys.modules.get("kaggle_evaluation")
    sys.modules["kaggle_evaluation"] = gateway
    try:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        # Standalone Kaggle execution: all imports, policies and gateway must
        # be in the same cell. Running the last cell alone must be sufficient.
        assert len(code_cells) == 1, "V19 must have exactly one runnable code cell"
        source = "".join(code_cells[0]["source"])
        assert source.count("SmartBuildingsInferenceServer(TransferPolicy).run()") == 1
        namespace = {}
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
        assert gateway_calls == [namespace["TransferPolicy"]]
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


def observation(slot, dow=1, zone_temperatures=(300.0, 293.0),
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


def assert_action(test, action, continuous_size=9, discrete_size=2):
    continuous, discrete = action["continuous_action"], action["discrete_action"]
    test.assertEqual(continuous.shape, (continuous_size,))
    test.assertEqual(discrete.shape, (discrete_size,))
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

    def test_iteration_uses_executed_not_planned_correction(self):
        self.first_day()
        self.policy.get_action(observation(0, dow=2))  # commit day one
        applied = float(self.policy.day_data["applied"][1, 0])
        self.assertGreater(applied, 0.0)
        for slot in range(1, 288):
            self.policy.get_action(observation(slot, dow=2))
        self.policy.get_action(observation(0, dow=3))  # commit day two
        expected = ILC.RETENTION * applied + ILC.LEARNING_RATE * 1.2
        self.assertAlmostEqual(
            float(self.policy.profiles["weekday"]["heat"][1, 0]),
            expected, places=5)

    def test_weather_gated_day_does_not_replay_unexecuted_profile(self):
        self.first_day()
        self.policy.get_action(observation(0, dow=2, outdoor_k=304.0))
        old_profile = float(self.policy.profiles["weekday"]["cool"][0, 0])
        self.assertGreater(old_profile, 0.0)
        self.assertEqual(float(self.policy.day_data["applied"][0, 0]), 0.0)
        for slot in range(1, 288):
            self.policy.get_action(observation(slot, dow=2, outdoor_k=304.0))
        self.policy.get_action(observation(0, dow=3, outdoor_k=304.0))
        # Same temperature error, but the old correction was never applied.
        # An invalid ILC implementation would accumulate 90% of it anyway.
        self.assertAlmostEqual(
            float(self.policy.profiles["weekday"]["cool"][0, 0]),
            old_profile, places=5)

    def test_backward_clock_jump_cannot_train_partial_day(self):
        # 246 samples would satisfy the minimum size, but no midnight was seen.
        for slot in range(246):
            self.policy.get_action(observation(slot, dow=1))
        obs = observation(0, dow=2)
        np.testing.assert_array_equal(
            self.policy.get_action(obs)["continuous_action"],
            self.baseline.get_action(obs)["continuous_action"],
        )
        self.assertFalse(self.policy.profiles["weekday"]["ready"].any())

    def test_midnight_requires_next_weekday(self):
        # A repeated clock-day number is inconsistent with a new calendar day.
        self.first_day(dow=1)
        obs = observation(0, dow=1)
        self.policy.get_action(obs)
        self.assertFalse(self.policy.profiles["weekday"]["ready"].any())

    def test_weekend_to_weekday_rollover_is_valid(self):
        # Sunday -> Monday should commit the Sunday history but *not* use it
        # on Monday (the weekday and weekend profiles remain independent).
        self.first_day(dow=6)
        obs = observation(0, dow=0)
        new_action = self.policy.get_action(obs)
        self.assertTrue(self.policy.profiles["weekend"]["ready"].any())
        np.testing.assert_array_equal(
            new_action["continuous_action"],
            self.baseline.get_action(obs)["continuous_action"],
        )

    def test_mode_reversal_does_not_replay_opposite_actuator(self):
        self.first_day()
        obs = observation(0, dow=2, zone_temperatures=(293.0, 300.0))
        action = self.policy.get_action(obs)
        reference = self.baseline.get_action(obs)
        np.testing.assert_array_equal(
            action["continuous_action"], reference["continuous_action"])
        np.testing.assert_array_equal(
            action["discrete_action"], reference["discrete_action"])

    def test_time_slot_decode_matches_float32_observations(self):
        for weekday in (0, 4, 5, 6):
            for slot in range(288):
                obs = observation(slot, dow=weekday).astype(np.float32)
                self.assertEqual(
                    self.policy._time_key(obs),
                    (slot, weekday, "weekday" if weekday < 5 else "weekend"),
                )

    def test_multiple_ahus_use_metadata_indices(self):
        # Two independent AHUs share one hot-water system. None of the action
        # or observation indices have been derived from an assumed fixed size.
        meta = metadata()
        meta.update({
            "n_continuous_action": 12,
            "n_discrete_action": 3,
            "temp_indices": [0, 3],
            "pressure_indices": [1, 4, 7],
            "recirculation_indices": [2, 5],
            "water_indices": [6],
            "ahu_zone_assignments": {
                "0": ["room_0"],
                "1": ["room_1"],
            },
            "zone_action_indices": {
                "VAV_0": {
                    "supply_air_damper_percentage_command": 8,
                    "reheat_valve_percentage_command": 9,
                },
                "VAV_1": {
                    "supply_air_damper_percentage_command": 10,
                    "reheat_valve_percentage_command": 11,
                },
            },
        })
        fields = list(meta["observation_field_names"])
        fields.insert(7, "ahs_1_outside_air_temperature_sensor")
        meta["observation_field_names"] = fields

        learner = ILC(meta, None)
        base = BASE(meta, None)

        def two_ahu_obs(slot, dow):
            single = observation(slot, dow=dow)
            return np.insert(single, 7, single[6])

        for slot in range(288):
            obs = two_ahu_obs(slot, 1)
            a = learner.get_action(obs)
            np.testing.assert_array_equal(
                a["continuous_action"], base.get_action(obs)["continuous_action"],
            )
            assert_action(self, a, continuous_size=12, discrete_size=3)

        obs = two_ahu_obs(0, 2)
        a = learner.get_action(obs)
        b = base.get_action(obs)
        self.assertEqual(a["discrete_action"].tolist(), b["discrete_action"].tolist())
        self.assertGreater(float(a["continuous_action"][8]),
                           float(b["continuous_action"][8]))
        self.assertGreater(float(a["continuous_action"][11]),
                           float(b["continuous_action"][11]))
        assert_action(self, a, continuous_size=12, discrete_size=3)

    def test_true_delayed_error_update_and_bounds(self):
        self.first_day()
        self.policy.get_action(observation(0, dow=2))  # commit day 1
        cool = self.policy.profiles["weekday"]["cool"]
        heat = self.policy.profiles["weekday"]["heat"]
        self.assertGreater(float(cool[0, 0]), 0.0)
        self.assertGreater(float(heat[1, 0]), 0.0)
        self.assertEqual(float(cool[0, -1]), 0.0)  # no future response within day
        self.assertLessEqual(float(np.max(np.abs(cool))), ILC.MAX_COOL_CORRECTION)
        self.assertLessEqual(float(np.max(np.abs(heat))), ILC.MAX_HEAT_CORRECTION)


if __name__ == "__main__":
    unittest.main()
