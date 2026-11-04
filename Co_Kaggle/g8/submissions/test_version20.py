"""Offline contract and nonlinear-control regression tests for Version 20.

Loads the actual notebook controller code with a stub Kaggle Policy class.
Does not require the Smart Buildings simulator or Kaggle evaluation package.
"""

import ast
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


NOTEBOOK = Path(__file__).with_name("submission-nb - Version 20.ipynb")
NOTEBOOK_SOURCE = "".join(json.loads(NOTEBOOK.read_text())["cells"][1]["source"])


def load_policy():
    """Compile the submitted code against a minimal official-gateway stub."""
    ast.parse(NOTEBOOK_SOURCE)

    kaggle = types.ModuleType("kaggle_evaluation")
    server = types.ModuleType("smart_buildings_inference_server")
    server.Policy = type(
        "Policy", (), {"__init__": lambda self, *args: None}
    )
    kaggle.smart_buildings_inference_server = server

    with patch.dict(sys.modules, {"kaggle_evaluation": kaggle}):
        namespace = {}
        exec(compile(NOTEBOOK_SOURCE, str(NOTEBOOK), "exec"), namespace)
    return namespace["TransferPolicy"]


TransferPolicy = load_policy()


def building_metadata():
    """One AHU, one zone, and independently indexed physical actuators."""
    return {
        "n_continuous_action": 6,
        "n_discrete_action": 2,
        "observation_field_names": [
            "VAV_0_zone_air_temperature_sensor",
            "VAV_0_zone_air_heating_temperature_setpoint_sensor",
            "VAV_0_zone_air_cooling_temperature_setpoint_sensor",
            "ahs_0_outside_air_temperature_sensor",
        ],
        "temp_indices": [0],
        "pressure_indices": [1],
        "recirculation_indices": [2],
        "water_indices": [3],
        "comfort_temp_window": [293.0, 297.0],
        "eco_temp_window": [289.0, 301.0],
        "morning_start_hour": 8,
        "evening_start_hour": 18,
        "ahu_zone_assignments": {"0": ["room_0"]},
        "zone_action_indices": {
            "VAV_0": {
                "reheat_valve_percentage_command": 4,
                "supply_air_damper_percentage_command": 5,
            },
        },
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {
                "sample_mean": 295.0,
                "sample_std": 2.0,
            },
            "outside_air_temperature_sensor": {
                "sample_mean": 290.0,
                "sample_std": 10.0,
            },
        },
    }


def observation(temperature_k, outside_normalized=0.0):
    """Convert zone Kelvin into normalized input; setpoints remain Kelvin."""
    return np.array(
        [(temperature_k - 295.0) / 2.0, 293.0, 297.0, outside_normalized],
        dtype=np.float64,
    )


class Version20Tests(unittest.TestCase):
    def setUp(self):
        self.policy = TransferPolicy(building_metadata(), None)

    def test_output_contract_across_modes(self):
        for temperature_k in (290.0, 295.0, 296.7, 297.2, 300.0):
            with self.subTest(temperature_k=temperature_k):
                action = self.policy.get_action(observation(temperature_k))
                continuous = action["continuous_action"]
                discrete = action["discrete_action"]
                self.assertEqual(continuous.shape, (6,))
                self.assertEqual(discrete.shape, (2,))
                self.assertEqual(continuous.dtype, np.float32)
                self.assertEqual(discrete.dtype, np.int32)
                self.assertTrue(np.isfinite(continuous).all())
                self.assertTrue(np.all((continuous >= -1) & (continuous <= 1)))
                self.assertTrue(np.isin(discrete, [0, 1]).all())

    def test_idle_zone_stays_off(self):
        action = self.policy.get_action(observation(295.0))
        self.assertEqual(action["discrete_action"].tolist(), [0, 0])
        self.assertEqual(float(action["continuous_action"][5]), -1.0)
        self.assertEqual(float(action["continuous_action"][4]), -1.0)

    def test_cooling_commands_airflow_without_reheat(self):
        action = self.policy.get_action(observation(300.0))
        self.assertEqual(action["discrete_action"].tolist(), [1, 0])
        self.assertGreater(float(action["continuous_action"][5]), -0.8)
        self.assertEqual(float(action["continuous_action"][4]), -1.0)

    def test_heating_commands_reheat_and_water(self):
        action = self.policy.get_action(observation(290.0))
        self.assertEqual(action["discrete_action"].tolist(), [1, 1])
        self.assertGreater(float(action["continuous_action"][4]), -1.0)

    def test_denormalize_outdoor_and_reject_bad_readings(self):
        p = self.policy
        self.assertEqual(p._outside_temperature(observation(295.0), 3), 290.0)
        self.assertTrue(np.isnan(p._outside_temperature(
            observation(295.0, float("nan")), 3
        )))
        self.assertTrue(np.isnan(p._outside_temperature(
            observation(295.0, 1000.0), 3
        )))

    def test_predictive_rate_enables_early_cooling(self):
        p = self.policy
        zone = {
            "actions": {"temperature": 0},
            "temp": 296.8,
            "low": 293.0,
            "high": 297.0,
        }
        self.assertEqual(p._sliding_status(dict(zone))["smc_cooling"], 0.0)
        p._history[0] = (296.0, 0.0)
        status = p._sliding_status(dict(zone))
        self.assertGreater(status["smc_cooling"], 0.0)
        self.assertEqual(status["smc_heating"], 0.0)

    def test_saturation_mode_exclusivity_and_reset(self):
        p = self.policy
        zone = {"actions": {"temperature": 0}, "low": 293, "high": 297}
        hot = p._sliding_status({**zone, "temp": 300.0})
        self.assertEqual((hot["smc_cooling"], hot["smc_heating"]), (1.0, 0.0))
        p.reset()
        self.assertFalse(p._history)
        cold = p._sliding_status({**zone, "temp": 290.0})
        self.assertEqual((cold["smc_cooling"], cold["smc_heating"]), (0.0, 1.0))

    def test_discontinuity_does_not_create_artificial_rate(self):
        p = self.policy
        zone = {"actions": {"temperature": 0}, "low": 293, "high": 297}
        p._history[0] = (295.0, 0.9)
        p._sliding_status({**zone, "temp": 300.0})
        self.assertEqual(p._history[0], (300.0, 0.0))

    def test_bad_zone_observation_invalidates_history(self):
        p = self.policy
        p.get_action(observation(300.0))
        self.assertTrue(p._history)
        bad = observation(295.0)
        bad[0] = float("nan")
        p.get_action(bad)
        self.assertFalse(p._history)
        p.get_action(observation(295.0))
        self.assertEqual(p._history[0][1], 0.0)


if __name__ == "__main__":
    unittest.main()
