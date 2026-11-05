"""Offline contract, regression and ablation tests for Version 26.

Executes the actual V26/V12 notebook policy cells with a stub Kaggle gateway.
Includes behavioral coverage for invalid state recovery, missing weather and
one-sided safety risks; does NOT measure official simulator performance.
"""
import ast
import json
import math
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


NOTEBOOK = Path(__file__).with_name("submission-nb - Version 26.ipynb")


def load_policy(notebook_path=NOTEBOOK):
    """Load a real notebook policy while replacing only the Kaggle gateway."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"]
                  if cell["cell_type"] == "code"]
    assert len(code_cells) == 2, "One policy cell + one gateway cell required"
    policy_source = "".join(code_cells[0]["source"])
    gateway_source = "".join(code_cells[1]["source"])
    ast.parse(policy_source)
    ast.parse(gateway_source)
    assert "class TransferPolicy(" in policy_source
    assert "SmartBuildingsInferenceServer(TransferPolicy).run()" in gateway_source

    kaggle = types.ModuleType("kaggle_evaluation")
    server = types.ModuleType("smart_buildings_inference_server")
    server.Policy = type("Policy", (), {"__init__": lambda self, *args: None})
    kaggle.smart_buildings_inference_server = server
    with patch.dict(sys.modules, {"kaggle_evaluation": kaggle}):
        namespace = {}
        exec(compile(policy_source, str(notebook_path), "exec"), namespace)
    return namespace["TransferPolicy"], namespace["ZoneResponseModel"]


TransferPolicy, ZoneResponseModel = load_policy()
# Independent reference: nominal V26 must reproduce V12's issued actions.
Version12Policy, _ = load_policy(
    NOTEBOOK.with_name("submission-nb - Version 12.ipynb")
)


def metadata(two_ahus=False):
    fields = [
        "VAV_0_zone_air_temperature_sensor",
        "VAV_0_zone_air_heating_temperature_setpoint_sensor",
        "VAV_0_zone_air_cooling_temperature_setpoint_sensor",
        "ahs_0_outside_air_temperature_sensor",
    ]
    result = {
        "n_continuous_action": 7,
        "n_discrete_action": 2,
        "observation_field_names": fields,
        "temp_indices": [0],
        "pressure_indices": [1, 4],
        "recirculation_indices": [2],
        "water_indices": [3],
        "comfort_temp_window": [293.0, 297.0],
        "eco_temp_window": [289.0, 301.0],
        "morning_start_hour": 8,
        "evening_start_hour": 18,
        "ahu_zone_assignments": {"0": ["room_0"]},
        "zone_action_indices": {
            "VAV_0": {
                "reheat_valve_percentage_command": 5,
                "supply_air_damper_percentage_command": 6,
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
    if two_ahus:
        result.update({
            "n_continuous_action": 12,
            "n_discrete_action": 3,
            "observation_field_names": fields + [
                "VAV_1_zone_air_temperature_sensor",
                "VAV_1_zone_air_heating_temperature_setpoint_sensor",
                "VAV_1_zone_air_cooling_temperature_setpoint_sensor",
                "ahs_1_outside_air_temperature_sensor",
            ],
            "temp_indices": [0, 1],
            "pressure_indices": [2, 3, 4],
            "recirculation_indices": [5, 6],
            "water_indices": [7],
            "ahu_zone_assignments": {
                "0": ["room_0"], "1": ["room_1"],
            },
            "zone_action_indices": {
                "VAV_0": {
                    "reheat_valve_percentage_command": 8,
                    "supply_air_damper_percentage_command": 9,
                },
                "VAV_1": {
                    "reheat_valve_percentage_command": 10,
                    "supply_air_damper_percentage_command": 11,
                },
            },
        })
    return result


def observation(temp_k=295.0, outdoor_k=290.0, temp2_k=None):
    result = [
        (temp_k - 295.0) / 2.0, 293.0, 297.0,
        (outdoor_k - 290.0) / 10.0,
    ]
    if temp2_k is not None:
        result.extend([
            (temp2_k - 295.0) / 2.0, 293.0, 297.0,
            (outdoor_k - 290.0) / 10.0,
        ])
    return np.asarray(result, dtype=np.float64)


class RobustMPCTests(unittest.TestCase):
    def setUp(self):
        self.policy = TransferPolicy(metadata(), None)

    def state(self, temperature=296.8, outdoor=290.0):
        return {
            "zone": self.policy.ahus[0]["zones"][0],
            "temp": temperature, "low": 293.0, "high": 297.0,
            "outdoor": outdoor,
        }

    def test_output_contract_and_idle(self):
        for temp in (286.0, 291.0, 295.0, 299.0, 306.0):
            with self.subTest(temp=temp):
                action = self.policy.get_action(observation(temp))
                continuous = action["continuous_action"]
                discrete = action["discrete_action"]
                self.assertEqual(continuous.shape, (7,))
                self.assertEqual(discrete.shape, (2,))
                self.assertEqual(continuous.dtype, np.float32)
                self.assertEqual(discrete.dtype, np.int32)
                self.assertTrue(np.isfinite(continuous).all())
                self.assertTrue(((continuous >= -1) & (continuous <= 1)).all())
                self.assertTrue(np.isin(discrete, [0, 1]).all())
        self.policy.reset()
        action = self.policy.get_action(observation(295.0))
        self.assertEqual(action["discrete_action"].tolist(), [0, 0])
        self.assertEqual(float(action["continuous_action"][6]), -1.0)

    def test_severe_temperatures_require_active_ahu(self):
        for temp in (286.0, 306.0):
            self.policy.reset()
            action = self.policy.get_action(observation(temp))
            self.assertEqual(int(action["discrete_action"][0]), 1)

    def test_uncertainty_increases_with_horizon(self):
        model = ZoneResponseModel()
        x = np.array([0.2, 0.7, 0.0, 1.0])
        self.assertGreater(model.forecast_sigma(x, 10),
                           model.forecast_sigma(x, 4))
        self.assertGreater(model.forecast_sigma(x, 4), 0.0)
        self.assertLessEqual(model.forecast_sigma(x, 1000),
                             model.MAX_FORECAST_SIGMA)

    def test_disturbance_learning_uses_previous_action_residual(self):
        stable, noisy = ZoneResponseModel(), ZoneResponseModel()
        x = np.array([0.0, 0.0, 0.0, 1.0])
        stable.remember(295.0, x)
        noisy.remember(295.0, x)
        stable.observe(295.0)
        noisy.observe(295.8)
        self.assertEqual((stable.samples, noisy.samples), (1, 1))
        self.assertGreater(noisy.innovation_variance,
                           stable.innovation_variance)
        self.assertGreater(noisy.forecast_sigma(x, 10),
                           stable.forecast_sigma(x, 10))

    def test_discontinuities_clear_stale_action_features(self):
        model = ZoneResponseModel()
        features = np.array([0.0, 0.0, 0.0, 1.0])
        initial_variance = model.innovation_variance

        model.remember(295.0, features)
        model.observe(310.0)  # impossible five-minute temperature jump
        self.assertEqual(model.samples, 0)
        self.assertEqual(model.innovation_variance, initial_variance)
        self.assertIsNone(model.last_temperature)
        self.assertIsNone(model.last_features)

        # Without a fresh issued action, there must be no retroactive fit.
        model.observe(310.1)
        self.assertEqual(model.samples, 0)
        model.remember(310.0, features)
        model.observe(310.1)
        self.assertEqual(model.samples, 1)
        self.assertTrue(np.isfinite(model.forecast_sigma(features, 10)))

    def test_invalid_rls_state_clears_pending_action(self):
        model = ZoneResponseModel()
        features = np.array([0.0, 0.0, 0.0, 1.0])
        model.remember(295.0, features)
        model.covariance.fill(float("nan"))
        model.observe(295.1)
        self.assertIsNone(model.last_features)
        self.assertEqual(model.samples, 0)
        self.assertTrue(np.isfinite(model.covariance).all())

    def test_unknown_weather_expands_scenario_envelope(self):
        known = self.state(296.0, 290.0)
        unknown = dict(known, outdoor=float("nan"))
        features = (0.0, 0.0, 0.0, 1.0)
        for horizon in self.policy.PLAN_STEPS:
            self.assertGreater(
                self.policy._scenario_radius(unknown, features, horizon),
                self.policy._scenario_radius(known, features, horizon),
            )
        action = self.policy.get_action(observation(
            296.0, outdoor_k=float("nan")))
        self.assertTrue(np.isfinite(action["continuous_action"]).all())

    def test_directional_safety_rejects_counterproductive_modes(self):
        p = self.policy
        model = p.ahus[0]["zones"][0]["model"]
        model.innovation_variance = model.MAX_INNOVATION_VARIANCE

        # Strong hot-only risk: heat-only should not be allowed.
        model.weights[3] = 0.5
        hot_state = self.state(296.8)
        hot, cold = p._off_tail_violations(hot_state)
        self.assertGreaterEqual(hot, p.MAX_OFF_FORECAST_VIOLATION_K)
        self.assertLess(cold, p.MAX_OFF_FORECAST_VIOLATION_K)
        hot_config, _ = p._select_operating_point([hot_state])
        self.assertNotIn(hot_config.mode, ("off", "heat"))

        # Strong cold-only risk: cool-only should not be allowed.
        model.weights[3] = -0.5
        cold_state = self.state(293.2)
        hot, cold = p._off_tail_violations(cold_state)
        self.assertGreaterEqual(cold, p.MAX_OFF_FORECAST_VIOLATION_K)
        self.assertLess(hot, p.MAX_OFF_FORECAST_VIOLATION_K)
        cold_config, _ = p._select_operating_point([cold_state])
        self.assertNotIn(cold_config.mode, ("off", "cool"))

    def test_nominal_v26_matches_independent_v12_actions(self):
        """Compare full issued actions, not just one forecast expression."""
        v12 = Version12Policy(metadata(), None)
        v26 = TransferPolicy(metadata(), None)
        v26.USE_UNCERTAINTY = False

        for i in range(50):
            temperature = (
                295.0 + 4.0 * math.sin(i / 9.0)
            )
            observed = observation(
                temperature, outdoor_k=290.0 + 10.0 * math.cos(i / 13.0),
            )
            baseline = v12.get_action(observed)
            ablation = v26.get_action(observed)
            np.testing.assert_array_equal(
                ablation["continuous_action"], baseline["continuous_action"]
            )
            np.testing.assert_array_equal(
                ablation["discrete_action"], baseline["discrete_action"]
            )

    def test_symmetric_scenario_cost_penalizes_risk(self):
        state = self.state(296.9)
        p = self.policy
        x = (0.0, 0.0, 0.0, 1.0)
        p.USE_UNCERTAINTY = True
        risk = p._forecast_discomfort(state, 0.0, x)
        p.USE_UNCERTAINTY = False
        nominal = p._forecast_discomfort(state, 0.0, x)
        self.assertEqual(nominal, 0.0)
        self.assertGreater(risk, nominal)

    def test_nominal_ablation_matches_v12_forecast(self):
        state = self.state(297.5)
        p = self.policy
        p.USE_UNCERTAINTY = False
        rate = 0.08
        expected = sum(
            p._violation(state["temp"] + rate * steps,
                         state["low"], state["high"])
            for steps in p.PLAN_STEPS
        ) / len(p.PLAN_STEPS)
        self.assertAlmostEqual(
            p._forecast_discomfort(state, rate, (0.0, 0.0, 0.0, 1.0)),
            expected,
        )

    def test_risk_off_guard_and_explicit_ablation(self):
        state = self.state(296.8)
        model = state["zone"]["model"]
        model.innovation_variance = model.MAX_INNOVATION_VARIANCE
        model.weights[3] = 0.5
        self.assertGreaterEqual(
            self.policy._off_forecast_violation(state),
            self.policy.MAX_OFF_FORECAST_VIOLATION_K,
        )
        self.assertTrue(self.policy._candidate_allowed(
            self.policy.OPERATING_POINTS[0], 0.0, 0.0))
        chosen, _ = self.policy._select_operating_point([state])
        self.assertNotEqual(chosen.mode, "off")
        self.policy.USE_UNCERTAINTY = False
        chosen, _ = self.policy._select_operating_point([state])
        self.assertIsNotNone(chosen)  # deterministic controller remains runnable

    def test_separate_outdoor_normalization_and_bad_reading(self):
        p = self.policy
        states = p._read_zones(
            observation(295.0, outdoor_k=305.0),
            p.ahus[0], p.comfort)
        self.assertEqual(states[0]["outdoor"], 305.0)
        p.get_action(observation(300.0))
        self.assertIsNotNone(p.ahus[0]["zones"][0]["model"].last_features)
        bad = observation(295.0)
        bad[0] = np.nan
        action = p.get_action(bad)
        self.assertEqual(action["discrete_action"][0], 0)
        self.assertIsNone(p.ahus[0]["zones"][0]["model"].last_features)

    def test_model_updates_and_reset(self):
        p = self.policy
        p.get_action(observation(299.0))
        p.get_action(observation(298.9))
        self.assertEqual(p.ahus[0]["zones"][0]["model"].samples, 1)
        p.reset()
        model = p.ahus[0]["zones"][0]["model"]
        self.assertEqual(model.samples, 0)
        self.assertIsNone(model.last_features)
        self.assertEqual(model.innovation_variance, 0.0225)

    def test_multi_ahu_shared_water_dispatch(self):
        p = TransferPolicy(metadata(two_ahus=True), None)
        action = p.get_action(observation(289.0, temp2_k=288.0))
        self.assertEqual(action["continuous_action"].shape, (12,))
        self.assertEqual(action["discrete_action"].shape, (3,))
        self.assertTrue(np.isfinite(action["continuous_action"]).all())
        self.assertEqual(action["discrete_action"][0], 1)
        self.assertEqual(action["discrete_action"][1], 1)
        self.assertEqual(action["discrete_action"][2], 1)
        self.assertGreater(float(action["continuous_action"][8]), -1.0)
        self.assertGreater(float(action["continuous_action"][10]), -1.0)

    def test_runtime_bound_on_synthetic_single_zone(self):
        p = self.policy
        start = time.monotonic()
        for index in range(120):
            action = p.get_action(observation(
                295.0 + math.sin(index / 10.0)))
            self.assertTrue(np.isfinite(action["continuous_action"]).all())
        self.assertLess(time.monotonic() - start, 12.0)


if __name__ == "__main__":
    unittest.main()
