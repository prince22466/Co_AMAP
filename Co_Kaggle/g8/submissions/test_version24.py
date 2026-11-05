"""Offline contract and gain-scheduling regression tests for Version 24.

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


NOTEBOOK = Path(__file__).with_name("submission-nb - Version 24.ipynb")
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


class Version24Tests(unittest.TestCase):
    def setUp(self):
        self.policy = TransferPolicy(building_metadata(), None)

    @staticmethod
    def status(temp):
        return {
            "actions": {"temperature": 0},
            "temp": temp,
            "low": 293.0,
            "high": 297.0,
        }

    def test_notebook_structure_and_official_gateway(self):
        notebook = json.loads(NOTEBOOK.read_text())
        code_cells = [cell for cell in notebook["cells"]
                      if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 2)
        self.assertIn("class TransferPolicy", "".join(code_cells[0]["source"]))
        self.assertIn("SmartBuildingsInferenceServer(TransferPolicy).run()",
                      "".join(code_cells[1]["source"]))

    def test_action_contract_and_equipment_modes(self):
        for temp in (290.0, 295.0, 297.3, 300.0):
            with self.subTest(temp=temp):
                action = TransferPolicy(building_metadata(), None).get_action(
                    observation(temp))
                c = action["continuous_action"]
                d = action["discrete_action"]
                self.assertEqual((c.shape, d.shape), ((6,), (2,)))
                self.assertEqual((c.dtype, d.dtype), (np.float32, np.int32))
                self.assertTrue(np.isfinite(c).all())
                self.assertTrue(np.all((c >= -1.0) & (c <= 1.0)))
                self.assertTrue(np.isin(d, (0, 1)).all())

        idle = TransferPolicy(building_metadata(), None).get_action(
            observation(295.0))
        self.assertEqual(idle["discrete_action"].tolist(), [0, 0])
        self.assertEqual(float(idle["continuous_action"][5]), -1.0)
        cool = TransferPolicy(building_metadata(), None).get_action(
            observation(300.0))
        self.assertEqual(cool["discrete_action"].tolist(), [1, 0])
        self.assertEqual(float(cool["continuous_action"][4]), -1.0)
        heat = TransferPolicy(building_metadata(), None).get_action(
            observation(290.0))
        self.assertEqual(heat["discrete_action"].tolist(), [1, 1])
        self.assertGreater(float(heat["continuous_action"][4]), -1.0)

    def test_convex_interpolation_and_continuity(self):
        p = self.policy
        self.assertEqual(p._scheduled_gains(280.0), p.COOLING_GAINS[0])
        self.assertEqual(p._scheduled_gains(294.0), p.COOLING_GAINS[1])
        self.assertEqual(p._scheduled_gains(308.0), p.COOLING_GAINS[2])
        expected = tuple((a + b) / 2 for a, b in zip(
            p.COOLING_GAINS[0], p.COOLING_GAINS[1]))
        self.assertTrue(np.allclose(p._scheduled_gains(287.0), expected))
        self.assertTrue(np.allclose(
            p._scheduled_gains(294.0 - 1e-6),
            p._scheduled_gains(294.0 + 1e-6), atol=1e-5))

    def test_cooling_stronger_in_hot_regime(self):
        cold = TransferPolicy(building_metadata(), None)
        hot = TransferPolicy(building_metadata(), None)
        cold_status = cold._gain_scheduled_status(self.status(297.5), 280.0)
        hot_status = hot._gain_scheduled_status(self.status(297.5), 308.0)
        self.assertGreater(hot_status["lpv_cooling"],
                           cold_status["lpv_cooling"])
        self.assertEqual(hot_status["lpv_heating"], 0.0)

    def test_heating_stronger_in_cold_regime(self):
        cold = TransferPolicy(building_metadata(), None)
        hot = TransferPolicy(building_metadata(), None)
        cold_status = cold._gain_scheduled_status(self.status(292.5), 280.0)
        hot_status = hot._gain_scheduled_status(self.status(292.5), 308.0)
        self.assertGreater(cold_status["lpv_heating"],
                           hot_status["lpv_heating"])
        self.assertEqual(cold_status["lpv_cooling"], 0.0)

    def test_fixed_gain_ablation_and_missing_weather(self):
        p = self.policy
        p.ENABLE_GAIN_SCHEDULING = False
        for outside_k in (275.0, 294.0, 313.0, float("nan")):
            self.assertEqual(p._scheduled_gains(outside_k),
                             p.COOLING_GAINS[1])
            self.assertEqual(p._scheduled_gains(outside_k, heating=True),
                             p.HEATING_GAINS[1])
        p.ENABLE_GAIN_SCHEDULING = True
        self.assertEqual(p._scheduled_gains(float("nan")),
                         p.COOLING_GAINS[1])

    def test_integral_bounded_and_deadband_resets(self):
        p = self.policy
        for _ in range(250):
            status = p._gain_scheduled_status(self.status(297.4), 308.0)
            self.assertTrue(0.0 <= status["lpv_cooling"] <= 1.0)
            self.assertLessEqual(
                p._state[0][2], p.MAX_INTEGRAL_ERROR_K_STEPS)
        p._gain_scheduled_status(self.status(295.0), 308.0)
        self.assertEqual(p._state[0][2:], (0.0, 0.0))
        p.reset()
        self.assertEqual(p._state, {})

    def test_discontinuity_and_invalid_sensor_clear_state(self):
        p = self.policy
        p._gain_scheduled_status(self.status(297.4), 308.0)
        p._gain_scheduled_status(self.status(298.0), 308.0)
        self.assertGreater(p._state[0][1], 0.0)
        p._gain_scheduled_status(self.status(302.0), 308.0)
        self.assertEqual(p._state[0][1], 0.0)
        p.reset()
        p.get_action(observation(300.0))
        self.assertIn(0, p._state)
        bad = observation(295.0)
        bad[0] = float("nan")
        p.get_action(bad)
        self.assertEqual(p._state, {})

    def test_setpoint_change_discards_old_integral(self):
        """Past integral error must not survive a new comfort-band target."""
        p = self.policy
        for _ in range(24):
            p._gain_scheduled_status(self.status(297.12), 294.0)
        self.assertGreater(p._state[0][2], 0.5)

        changed_band = {**self.status(297.12), "high": 297.10}
        result = p._gain_scheduled_status(changed_band, 294.0)
        self.assertAlmostEqual(p._state[0][2], 0.02, places=6)
        self.assertGreater(result["lpv_cooling"], 0.0)
        self.assertEqual(p._state[0][3], 1)

    def test_saturation_does_not_wind_up_integral(self):
        """Large disturbances should not accumulate saturated PI error."""
        p = self.policy
        for _ in range(50):
            result = p._gain_scheduled_status(self.status(301.0), 308.0)
            self.assertEqual(result["lpv_cooling"], 1.0)
        self.assertEqual(p._state[0][2], 0.0)

    def test_missing_actuators_cannot_activate_hvac(self):
        """A heating request without a reheat valve cannot heat the zone."""
        metadata = building_metadata()
        del metadata["zone_action_indices"]["VAV_0"][
            "reheat_valve_percentage_command"
        ]
        policy = TransferPolicy(metadata, None)
        output = policy.get_action(observation(290.0))
        self.assertEqual(output["discrete_action"].tolist(), [0, 0])

        # A cooling request without a damper is also unserviceable.
        metadata = building_metadata()
        del metadata["zone_action_indices"]["VAV_0"][
            "supply_air_damper_percentage_command"
        ]
        policy = TransferPolicy(metadata, None)
        output = policy.get_action(observation(300.0))
        self.assertEqual(output["discrete_action"].tolist(), [0, 0])

    def test_low_demand_zone_remains_idle_when_neighbor_heats(self):
        """Shared AHU load must not amplify an insignificant zone error."""
        metadata = building_metadata()
        metadata["n_continuous_action"] = 8
        metadata["ahu_zone_assignments"]["0"].append("room_1")
        metadata["zone_action_indices"]["VAV_1"] = {
            "reheat_valve_percentage_command": 6,
            "supply_air_damper_percentage_command": 7,
        }
        metadata["observation_field_names"] += [
            "VAV_1_zone_air_temperature_sensor",
            "VAV_1_zone_air_heating_temperature_setpoint_sensor",
            "VAV_1_zone_air_cooling_temperature_setpoint_sensor",
        ]
        obs = np.array(
            [(297.03 - 295.0) / 2.0, 293.0, 297.0, 0.0,
             (290.0 - 295.0) / 2.0, 293.0, 297.0]
        )
        policy = TransferPolicy(metadata, None)
        action = policy.get_action(obs)
        self.assertEqual(action["discrete_action"].tolist(), [1, 1])
        # Minimum VAV opening of 10% maps to -0.8.
        self.assertAlmostEqual(float(action["continuous_action"][5]), -0.8, places=6)
        self.assertGreater(float(action["continuous_action"][6]), -1.0)

    def test_two_ahus_dispatch_independently_and_share_hot_water(self):
        """No hard-coded single-AHU indexes or cross-zone controller states."""
        meta = building_metadata()
        meta["n_continuous_action"] = 12
        meta["n_discrete_action"] = 3
        meta["temp_indices"] = [0, 1]
        meta["pressure_indices"] = [2, 3, 4]
        meta["recirculation_indices"] = [5, 6]
        meta["water_indices"] = [7]
        meta["ahu_zone_assignments"] = {
            "1": ["room_1"],
            "0": ["room_0"],
        }
        meta["zone_action_indices"] = {
            "VAV_0": {
                "reheat_valve_percentage_command": 8,
                "supply_air_damper_percentage_command": 9,
            },
            "VAV_1": {
                "reheat_valve_percentage_command": 10,
                "supply_air_damper_percentage_command": 11,
            },
        }
        meta["observation_field_names"] += [
            "VAV_1_zone_air_temperature_sensor",
            "VAV_1_zone_air_heating_temperature_setpoint_sensor",
            "VAV_1_zone_air_cooling_temperature_setpoint_sensor",
            "ahs_1_outside_air_temperature_sensor",
        ]
        obs = np.array([
            (300.0 - 295.0) / 2.0, 293.0, 297.0, 0.0,
            (290.0 - 295.0) / 2.0, 293.0, 297.0, 0.0,
        ])
        policy = TransferPolicy(meta, None)
        result = policy.get_action(obs)
        self.assertEqual(result["continuous_action"].shape, (12,))
        self.assertEqual(result["discrete_action"].tolist(), [1, 1, 1])
        self.assertEqual(float(result["continuous_action"][8]), -1.0)
        self.assertGreater(float(result["continuous_action"][9]), -1.0)
        self.assertGreater(float(result["continuous_action"][10]), -1.0)
        self.assertGreater(float(result["continuous_action"][11]), -1.0)
        self.assertEqual(len(policy._state), 2)

    def test_missing_outdoor_sensor_falls_back_to_mild_gains(self):
        """Weather failures must not propagate NaNs into actuator actions."""
        p = self.policy
        bad = observation(297.7, float("nan"))
        result = p.get_action(bad)
        self.assertTrue(np.isfinite(result["continuous_action"]).all())
        self.assertEqual(result["discrete_action"].tolist(), [1, 0])

    def test_denormalize_outdoor_reading(self):
        p = self.policy
        self.assertEqual(p._outside_temperature(observation(295.0), 3), 290.0)
        self.assertEqual(p._outside_temperature(observation(295.0, 1.8), 3), 308.0)
        self.assertTrue(np.isnan(p._outside_temperature(
            observation(295.0, float("nan")), 3)))


if __name__ == "__main__":
    unittest.main()
