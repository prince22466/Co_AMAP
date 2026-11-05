"""Contract, QP, and nominal-V7 regression tests for Version 25.

Run offline with NumPy only. The actual notebook code is executed against
a tiny stand-in for the Kaggle Policy base class; the evaluation cell is not
executed, so no simulator package is required.
"""

import ast
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


DIRECTORY = Path(__file__).resolve().parent
V7_NOTEBOOK = DIRECTORY / "submission-nb - Version 7.ipynb"
V25_NOTEBOOK = DIRECTORY / "submission-nb - Version 25.ipynb"


def notebook_code(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 3, "Notebook must retain V7's 3-cell structure"
    assert [c["cell_type"] for c in notebook["cells"]] == [
        "markdown", "code", "code"
    ]
    gateway = "".join(notebook["cells"][-1]["source"])
    assert "SmartBuildingsInferenceServer(TransferPolicy).run()" in gateway
    source = "".join(notebook["cells"][1]["source"])
    ast.parse(source, filename=str(path))
    return source


def load_policies():
    """Compile V7 and V25 classes without importing the Kaggle runtime."""
    stub = types.ModuleType("smart_buildings_inference_server")
    stub.Policy = type("Policy", (), {"__init__": lambda self, *args: None})
    kaggle = types.ModuleType("kaggle_evaluation")
    kaggle.smart_buildings_inference_server = stub
    with patch.dict(sys.modules, {
        "kaggle_evaluation": kaggle,
        "smart_buildings_inference_server": stub,
    }):
        v7_globals, v25_globals = {}, {}
        exec(compile(notebook_code(V7_NOTEBOOK), str(V7_NOTEBOOK), "exec"),
             v7_globals)
        exec(compile(notebook_code(V25_NOTEBOOK), str(V25_NOTEBOOK), "exec"),
             v25_globals)
    return v7_globals["TransferPolicy"], v25_globals["TransferPolicy"]


V7, V25 = load_policies()


def metadata():
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
                "sample_mean": 295.0, "sample_std": 2.0,
            },
            "outside_air_temperature_sensor": {
                "sample_mean": 290.0, "sample_std": 10.0,
            },
        },
    }


def observation(temperature_k, lower=293.0, upper=297.0):
    return np.array([
        (temperature_k - 295.0) / 2.0,
        lower,
        upper,
        0.0,
    ], dtype=np.float64)


def fraction(action, index):
    return (float(action["continuous_action"][index]) + 1.0) / 2.0


class Version25Tests(unittest.TestCase):
    def setUp(self):
        self.nominal = V7(metadata(), None)
        self.filtered = V25(metadata(), None)

    def test_v7_code_preserved_except_class_name(self):
        """V25 uses the V7 implementation, not a partial reimplementation."""
        v7 = notebook_code(V7_NOTEBOOK)
        v25 = notebook_code(V25_NOTEBOOK)
        renamed = v7.replace(
            "class TransferPolicy(smart_buildings_inference_server.Policy):",
            "class PhysicsBaselinePolicy(smart_buildings_inference_server.Policy):",
            1,
        )
        self.assertTrue(v25.startswith(renamed))
        self.assertEqual(v25.count("class TransferPolicy("), 1)

    def test_output_contract(self):
        for temp in (290.0, 291.0, 295.0, 296.5, 297.3, 298.0, 310.0):
            with self.subTest(temp=temp):
                action = self.filtered.get_action(observation(temp))
                continuous, discrete = (action["continuous_action"],
                                        action["discrete_action"])
                self.assertEqual(continuous.shape, (6,))
                self.assertEqual(discrete.shape, (2,))
                self.assertEqual(continuous.dtype, np.float32)
                self.assertEqual(discrete.dtype, np.int32)
                self.assertTrue(np.isfinite(continuous).all())
                self.assertTrue(np.all(np.abs(continuous) <= 1.0))
                self.assertTrue(np.isin(discrete, (0, 1)).all())

    def test_safe_zone_is_bitwise_identical_to_v7(self):
        for temp in (294.0, 295.0, 296.0):
            with self.subTest(temp=temp):
                obs = observation(temp)
                original = self.nominal.get_action(obs)
                filtered = self.filtered.get_action(obs)
                np.testing.assert_array_equal(
                    original["continuous_action"],
                    filtered["continuous_action"],
                )
                np.testing.assert_array_equal(
                    original["discrete_action"],
                    filtered["discrete_action"],
                )

    def test_severe_overheating_adds_cooling_without_reheat(self):
        obs = observation(298.0)
        original = self.nominal.get_action(obs)
        filtered = self.filtered.get_action(obs)
        self.assertEqual(filtered["discrete_action"].tolist(), [1, 0])
        self.assertGreater(fraction(filtered, 5), fraction(original, 5))
        self.assertGreaterEqual(
            float(filtered["continuous_action"][1]),
            float(original["continuous_action"][1]),
        )
        self.assertEqual(float(filtered["continuous_action"][4]), -1.0)
        self.assertLessEqual(
            fraction(filtered, 5) - fraction(original, 5),
            V25.SEVERE_FRACTION_CAP + 1e-6,
        )
        self.assertLessEqual(
            float(filtered["continuous_action"][1])
            - float(original["continuous_action"][1]),
            V25.MAX_FAN_ACTION_CORRECTION + 1e-6,
        )

    def test_severe_undercooling_adds_heat_only_if_pump_runs(self):
        obs = observation(291.0)
        original = self.nominal.get_action(obs)
        filtered = self.filtered.get_action(obs)
        self.assertEqual(filtered["discrete_action"].tolist(), [1, 1])
        self.assertGreater(fraction(filtered, 4), fraction(original, 4))
        # Airflow increases are not guaranteed to heat a cold room.
        self.assertEqual(fraction(filtered, 5), fraction(original, 5))
        self.assertEqual(float(filtered["continuous_action"][1]),
                         float(original["continuous_action"][1]))
        self.assertLessEqual(fraction(filtered, 4) - fraction(original, 4),
                             V25.SEVERE_FRACTION_CAP + 1e-6)

    def test_saturated_cooling_actuator_stays_bounded(self):
        obs = observation(310.0)
        original = self.nominal.get_action(obs)
        filtered = self.filtered.get_action(obs)
        self.assertLessEqual(fraction(filtered, 5), 1.0)
        self.assertGreaterEqual(fraction(filtered, 5), fraction(original, 5))
        self.assertTrue(np.isfinite(filtered["continuous_action"]).all())


    def test_no_cooling_authority_leaves_v7_commands_unchanged(self):
        """Do not assume increased damper/fan always provides cooling."""
        obs = observation(301.0)
        original = self.nominal.get_action(obs)
        with patch.object(self.filtered, "_cooling_headroom",
                          return_value=0.0):
            filtered = self.filtered.get_action(obs)
        np.testing.assert_array_equal(
            filtered["continuous_action"], original["continuous_action"]
        )
        np.testing.assert_array_equal(
            filtered["discrete_action"], original["discrete_action"]
        )

    def test_fan_boost_only_for_saturated_hot_zone(self):
        """A severe unsatisfied cooling zone can trigger a fan boost."""
        # At 298 K, V7's damper is not fully open; do not spend fan power.
        mild_hot = observation(298.0)
        v7_mild = self.nominal.get_action(mild_hot)
        v25_mild = self.filtered.get_action(mild_hot)
        self.assertEqual(
            float(v25_mild["continuous_action"][1]),
            float(v7_mild["continuous_action"][1]),
        )

        # At 310 K the V7 damper is saturated; additional pressure is
        # permitted when the nominal coil setpoint provides cooling.
        self.filtered.reset()
        severe_hot = observation(310.0)
        v7_severe = self.nominal.get_action(severe_hot)
        v25_severe = self.filtered.get_action(severe_hot)
        self.assertGreater(
            float(v25_severe["continuous_action"][1]),
            float(v7_severe["continuous_action"][1]),
        )
        self.assertLessEqual(
            float(v25_severe["continuous_action"][1])
            - float(v7_severe["continuous_action"][1]),
            V25.MAX_FAN_ACTION_CORRECTION + 1e-6,
        )

    def test_heating_does_not_open_damper(self):
        """With a cooling-only AHU, heating should increase only reheat."""
        obs = observation(291.0)
        original = self.nominal.get_action(obs)
        filtered = self.filtered.get_action(obs)
        self.assertEqual(
            float(filtered["continuous_action"][5]),
            float(original["continuous_action"][5]),
        )
        self.assertGreater(fraction(filtered, 4), fraction(original, 4))

    def test_multizone_conflicting_loads(self):
        """The QPs must respect independent valves/dampers on one AHU."""
        meta = metadata()
        meta["n_continuous_action"] = 8
        meta["ahu_zone_assignments"]["0"] = ["room_0", "room_1"]
        meta["zone_action_indices"]["VAV_1"] = {
            "reheat_valve_percentage_command": 6,
            "supply_air_damper_percentage_command": 7,
        }
        meta["observation_field_names"] = [
            "VAV_0_zone_air_temperature_sensor",
            "VAV_0_zone_air_heating_temperature_setpoint_sensor",
            "VAV_0_zone_air_cooling_temperature_setpoint_sensor",
            "VAV_1_zone_air_temperature_sensor",
            "VAV_1_zone_air_heating_temperature_setpoint_sensor",
            "VAV_1_zone_air_cooling_temperature_setpoint_sensor",
            "ahs_0_outside_air_temperature_sensor",
        ]
        obs = np.array([
            (291.0 - 295.0) / 2.0, 293.0, 297.0,
            (299.0 - 295.0) / 2.0, 293.0, 297.0,
            0.0,
        ])
        nominal = V7(meta, None).get_action(obs)
        filtered = V25(meta, None).get_action(obs)
        self.assertEqual(filtered["discrete_action"].tolist(), [1, 1])
        self.assertEqual(float(filtered["continuous_action"][5]),
                         float(nominal["continuous_action"][5]))
        self.assertGreater(float(filtered["continuous_action"][4]),
                           float(nominal["continuous_action"][4]))
        self.assertGreater(float(filtered["continuous_action"][7]),
                           float(nominal["continuous_action"][7]))
        self.assertEqual(float(filtered["continuous_action"][6]),
                         float(nominal["continuous_action"][6]))

    def test_soft_qp_matches_analytic_unconstrained_solution(self):
        """Verify the dual solver's objective optimum, not just bounds."""
        deficit = 0.03
        gains, weights, penalty = [1.0, 2.0], [2.0, 1.0], 10.0
        expected_dual = deficit / (
            1.0 / penalty
            + sum(g * g / w for g, w in zip(gains, weights))
        )
        expected = [expected_dual * g / w
                    for g, w in zip(gains, weights)]
        values, slack = V25._solve_soft_cbf_qp(
            deficit, gains, [0.3, 0.3], weights, penalty
        )
        np.testing.assert_allclose(values, expected, atol=1e-8, rtol=1e-5)
        self.assertAlmostEqual(slack, expected_dual / penalty, places=7)
        self.assertGreaterEqual(
            sum(g * x for g, x in zip(gains, values)) + slack,
            deficit - 1e-7,
        )

    def test_soft_qp_primal_feasibility_and_saturation(self):
        delta, slack = V25._solve_soft_cbf_qp(
            0.03, [0.3], [0.2], [1.0], 20.0,
        )
        self.assertGreater(delta[0], 0.0)
        self.assertLessEqual(delta[0], 0.2)
        self.assertGreaterEqual(slack, 0.0)
        self.assertGreaterEqual(0.3 * delta[0] + slack, 0.03 - 1e-7)

        saturated, residual = V25._solve_soft_cbf_qp(
            0.3, [0.1], [0.05], [1.0], 20.0,
        )
        self.assertAlmostEqual(saturated[0], 0.05, places=6)
        self.assertGreater(residual, 0.0)
        self.assertGreaterEqual(
            0.1 * saturated[0] + residual, 0.3 - 1e-7,
        )

        unchanged, no_slack = V25._solve_soft_cbf_qp(
            -0.1, [0.1], [0.25], [1.0], 20.0,
        )
        self.assertEqual(unchanged, [0.0])
        self.assertEqual(no_slack, 0.0)

    def test_episode_reset_and_invalid_sensor_clear_rate(self):
        self.filtered.get_action(observation(298.0))
        self.filtered.get_action(observation(298.2))
        self.assertGreater(self.filtered._zone_history[0][1], 0.0)
        self.filtered.reset()
        self.assertEqual(self.filtered._zone_history, {})
        self.filtered.get_action(observation(298.2))
        self.assertEqual(self.filtered._zone_history[0][1], 0.0)

        bad = observation(298.0)
        bad[0] = np.nan
        action = self.filtered.get_action(bad)
        self.assertEqual(self.filtered._zone_history, {})
        self.assertTrue(np.isfinite(action["continuous_action"]).all())

    def test_large_sensor_jump_not_extrapolated(self):
        self.filtered.get_action(observation(296.0))
        self.filtered.get_action(observation(305.0))
        self.assertEqual(self.filtered._zone_history[0][1], 0.0)

    def test_invalid_setpoints_use_v7_schedule_fallback(self):
        for value in (np.nan, -1.0, 450.0):
            with self.subTest(value=value):
                obs = observation(299.0, lower=value)
                output = self.filtered.get_action(obs)
                self.assertTrue(np.isfinite(output["continuous_action"]).all())


if __name__ == "__main__":
    unittest.main()
