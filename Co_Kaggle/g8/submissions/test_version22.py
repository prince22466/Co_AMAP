"""Unit/contract tests for Version 22 economic MPC (no simulator needed).

These tests validate the policy code and *modeled* reward terms. They do
not demonstrate that the proxy power model predicts official energy draw.
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


NOTEBOOK = Path(__file__).with_name("submission-nb - Version 22.ipynb")


def load_controller():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 2, "Keep one policy cell and one Kaggle gateway cell"
    policy_source = "".join(code_cells[0]["source"])
    gateway_source = "".join(code_cells[1]["source"])
    ast.parse(policy_source)
    ast.parse(gateway_source)
    assert "SmartBuildingsInferenceServer(TransferPolicy).run()" in gateway_source

    kaggle = types.ModuleType("kaggle_evaluation")
    server = types.ModuleType("smart_buildings_inference_server")
    server.Policy = type("Policy", (), {"__init__": lambda self, *args: None})
    kaggle.smart_buildings_inference_server = server

    with patch.dict(sys.modules, {"kaggle_evaluation": kaggle}):
        ns = {}
        exec(compile(policy_source, str(NOTEBOOK), "exec"), ns)
    return ns["TransferPolicy"]


TransferPolicy = load_controller()


def metadata():
    return {
        "n_continuous_action": 7,
        "n_discrete_action": 2,
        "observation_field_names": [
            "VAV_0_zone_air_temperature_sensor",
            "VAV_0_zone_air_heating_temperature_setpoint_sensor",
            "VAV_0_zone_air_cooling_temperature_setpoint_sensor",
            "ahs_0_outside_air_temperature_sensor",
            "num_occupants", "hod_sin_000", "hod_cos_000",
            "dow_sin_000", "dow_cos_000",
        ],
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


def observation(kelvin=295.0, outside_k=290.0, occupancy=0.0,
                hour=13.0, weekday=2.0):
    return np.array([
        (kelvin - 295.0) / 2.0, 293.0, 297.0,
        (outside_k - 290.0) / 10.0, occupancy,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * weekday / 7),
        math.cos(2 * math.pi * weekday / 7),
    ], dtype=np.float64)


class Version22Tests(unittest.TestCase):
    def setUp(self):
        self.p = TransferPolicy(metadata(), None)

    def test_output_contract_and_idle_state(self):
        for kelvin in (285.0, 290.0, 295.0, 300.0, 307.0):
            with self.subTest(kelvin=kelvin):
                action = self.p.get_action(observation(kelvin))
                continuous, discrete = action["continuous_action"], action["discrete_action"]
                self.assertEqual(continuous.shape, (7,))
                self.assertEqual(discrete.shape, (2,))
                self.assertEqual(continuous.dtype, np.float32)
                self.assertEqual(discrete.dtype, np.int32)
                self.assertTrue(np.all(np.isfinite(continuous)))
                self.assertTrue(np.all((-1 <= continuous) & (continuous <= 1)))
                self.assertTrue(np.all(np.isin(discrete, [0, 1])))

        self.p.reset()
        idle = self.p.get_action(observation(295.0))
        self.assertEqual(idle["discrete_action"].tolist(), [0, 0])
        self.assertEqual(float(idle["continuous_action"][6]), -1.0)

    def test_severe_heat_and_cold_guardrails(self):
        for kelvin in (286.0, 305.0):
            with self.subTest(kelvin=kelvin):
                self.p.reset()
                action = self.p.get_action(observation(kelvin))
                self.assertEqual(int(action["discrete_action"][0]), 1)

    def test_official_piecewise_comfort_regret(self):
        penalty = self.p._violation
        self.assertEqual(penalty(295.0, 293.0, 297.0), 0.0)
        self.assertAlmostEqual(penalty(298.0, 293.0, 297.0), 0.2)
        self.assertEqual(penalty(302.0, 293.0, 297.0), 1.0)
        self.assertEqual(penalty(307.0, 293.0, 297.0), 2.0)
        self.assertEqual(penalty(283.0, 293.0, 297.0), 2.0)

    def test_economic_terms_and_separate_normalizers(self):
        ctx = self.p._reward_context(observation())
        # At nominal 80 kW electricity + 205 kW gas, both official normalized
        # energy and carbon terms equal one -> 0.2 + 0.2.
        self.assertAlmostEqual(self.p._economic_energy_score(80.0, 205.0, ctx), 0.4)
        self.assertAlmostEqual(self.p._economic_energy_score(0.0, 0.0, ctx), 0.0)
        self.assertGreater(self.p._economic_energy_score(1.0, 1.0, ctx), 0.0)
        self.assertAlmostEqual(
            self.p._economic_energy_score(2.0, 3.0, ctx),
            self.p._economic_energy_score(2.0, 0.0, ctx)
            + self.p._economic_energy_score(0.0, 3.0, ctx),
        )

    def test_tariffs_and_carbon_vary_with_time(self):
        weekday = self.p._reward_context(observation(hour=13, weekday=2))
        weekend = self.p._reward_context(observation(hour=13, weekday=6))
        early = self.p._reward_context(observation(hour=3, weekday=2))
        self.assertAlmostEqual(weekday[0], 0.20)
        self.assertAlmostEqual(weekend[0], 0.16)
        self.assertAlmostEqual(early[0], 0.16)
        self.assertNotEqual(weekday[1], early[1])

    def test_unambiguously_empty_occupancy_suppresses_comfort_only(self):
        # The total count is normalized using (count-k)/(k+1), and k can
        # differ by scenario. An explicitly known k=125 identifies zero.
        known = metadata()
        known["occupancy_normalization_constant"] = 125.0
        self.p = TransferPolicy(known, None)
        occupied = self.p._reward_context(observation(occupancy=0.0))
        empty = self.p._reward_context(observation(occupancy=-125.0 / 126))
        self.assertEqual(occupied[2], 1.0)
        self.assertEqual(empty[2], 0.0)
        state = {
            "zone": self.p.ahus[0]["zones"][0],
            "temp": 298.0, "low": 293.0, "high": 297.0, "outdoor": 290.0,
        }
        off = self.p.OPERATING_POINTS[0]
        self.assertGreater(self.p._best_vav(state, off, occupied)[0], 0.0)
        self.assertAlmostEqual(self.p._best_vav(state, off, empty)[0], 0.0)

        # The public simulator's default k=0 has a different empty signal.
        known["occupancy_normalization_constant"] = 0.0
        self.p = TransferPolicy(known, None)
        self.assertEqual(
            self.p._reward_context(observation(occupancy=0.0))[2], 0.0)
        self.assertEqual(
            self.p._reward_context(observation(occupancy=1.0))[2], 1.0)
        # Do not infer absence from a magic -125/126 without metadata.
        self.p = TransferPolicy(metadata(), None)
        self.assertEqual(
            self.p._reward_context(observation(occupancy=-125.0/126.0))[2], 1.0)

    def test_cost_model_monotonic_and_zero_when_off(self):
        state = {"temp": 300.0, "outdoor": 305.0}
        off, cool, heat = (self.p.OPERATING_POINTS[i] for i in (0, 3, 4))
        self.assertEqual(self.p._zone_power_kw(state, off, 1.0, 1.0), (0.0, 0.0))
        e_min, gas_min = self.p._zone_power_kw(state, cool, 0.1, 0.0)
        e_max, gas_max = self.p._zone_power_kw(state, cool, 1.0, 0.0)
        self.assertGreater(e_max, e_min)
        self.assertEqual(gas_min, 0.0)
        self.assertEqual(gas_max, 0.0)
        _, gas_heat = self.p._zone_power_kw(state, heat, 1.0, 1.0)
        self.assertGreater(gas_heat, 0.0)

    def test_reward_objective_matches_weighted_components(self):
        state = {
            "zone": self.p.ahus[0]["zones"][0],
            "temp": 298.0, "low": 293.0, "high": 297.0, "outdoor": 290.0,
        }
        ctx = self.p._reward_context(observation())
        off = self.p.OPERATING_POINTS[0]
        rate = state["zone"]["model"].predict_delta(
            state["zone"]["model"].features(
                state["temp"], state["outdoor"], 0.0,
                off.supply_k, 0.0, 0.0, 0.0, off.water_k))
        expected = (self.p.REWARD_COMFORT /
                    self.p.total_zones *
                    sum(self.p._violation(
                        state["temp"] + rate * steps, state["low"], state["high"])
                        for steps in self.p.PLAN_STEPS) /
                    len(self.p.PLAN_STEPS))
        self.assertAlmostEqual(self.p._best_vav(state, off, ctx)[0], expected)

    def test_building_reward_not_divided_twice_by_ahu_zone_count(self):
        """Aggregate every zone's normalized contribution exactly once.

        Construct a tradeoff where averaging two zone scores for a second
        time would incorrectly choose "off" instead of a running mode.
        """
        two_zones = [
            {"temp": 295.0, "low": 293.0, "high": 297.0},
            {"temp": 295.0, "low": 293.0, "high": 297.0},
        ]
        self.p._economic_energy_score = (
            lambda electric_kw, gas_kw, ctx: 0.12 if electric_kw > 0 else 0.0)
        self.p._best_vav = (
            lambda state, config, ctx, recirculation: (
                0.02 if config.running else 0.09, 0.1, 0.0))
        chosen, profiles = self.p._select_operating_point(
            two_zones, (0.16, 0.09, 1.0))
        self.assertEqual(chosen.mode, "vent")
        self.assertEqual(len(profiles), 2)

    def test_recirculation_forecast_matches_issued_ahu_action(self):
        """Free cooling must reduce forecast chiller work and be commanded."""
        config = self.p.OPERATING_POINTS[3]  # 289 K cooling mode
        state = {
            "zone": self.p.ahus[0]["zones"][0],
            "temp": 300.0, "low": 293.0, "high": 297.0, "outdoor": 280.0,
        }
        fraction = self.p._planned_recirculation(config, [state])
        self.assertGreater(fraction, 0.0)
        self.assertLess(fraction, 0.7)
        electric_economizer, gas = self.p._zone_power_kw(
            state, config, 1.0, 0.0, fraction)
        electric_fixed, _ = self.p._zone_power_kw(
            state, config, 1.0, 0.0, 0.7)
        self.assertLess(electric_economizer, electric_fixed)
        self.assertEqual(gas, 0.0)

        continuous, discrete = self.p._initial_actions()
        self.p._apply_ahu_action(
            self.p.ahus[0], config, fraction, continuous, discrete)
        self.assertAlmostEqual(continuous[2], 2 * fraction - 1.0, places=5)

        # RLS must remember the *issued* recirculation, not fixed 70%.
        pending = []
        self.p._apply_vav_actions(
            [state], config, [(0.0, 1.0, 0.0)], fraction,
            continuous, [], pending)
        self.p._remember_issued_actions(pending, None)
        expected = state["zone"]["model"].features(
            300.0, 280.0, 1.0, config.supply_k,
            1.0, 0.0, 0.0, config.water_k, fraction)
        np.testing.assert_allclose(
            state["zone"]["model"].last_features, expected)

    def test_severe_excursion_requires_corrective_mode(self):
        configs = {point.mode: point for point in self.p.OPERATING_POINTS}
        for name in ("off", "vent", "mild_cool", "cool"):
            with self.subTest(cold=name):
                self.assertFalse(self.p._candidate_allowed(
                    configs[name], worst_hot=0.0, worst_cold=5.0))
        for name in ("off", "vent", "heat"):
            with self.subTest(hot=name):
                self.assertFalse(self.p._candidate_allowed(
                    configs[name], worst_hot=5.0, worst_cold=0.0))
        self.assertTrue(self.p._candidate_allowed(
            configs["mixed"], worst_hot=5.0, worst_cold=5.0))

    def test_without_recirculation_mapping_uses_fixed_mix(self):
        m = metadata()
        m["recirculation_indices"] = []
        p = TransferPolicy(m, None)
        state = {
            "zone": p.ahus[0]["zones"][0],
            "temp": 300.0, "low": 293.0, "high": 297.0, "outdoor": 280.0,
        }
        self.assertEqual(
            p._planned_recirculation(p.OPERATING_POINTS[3], [state], False),
            0.7)
        a = p.get_action(observation(300.0, outside_k=280.0))
        self.assertTrue(np.isfinite(a["continuous_action"]).all())

    def test_multiple_ahus_share_one_water_system(self):
        m = metadata()
        m.update({
            "n_continuous_action": 12,
            "n_discrete_action": 3,
            "temp_indices": [0, 3],
            "pressure_indices": [1, 4, 7],
            "recirculation_indices": [2, 5],
            "water_indices": [6],
            "ahu_zone_assignments": {
                "0": ["room_0"], "1": ["room_1"]},
            "zone_action_indices": {
                "VAV_0": {
                    "reheat_valve_percentage_command": 8,
                    "supply_air_damper_percentage_command": 9},
                "VAV_1": {
                    "reheat_valve_percentage_command": 10,
                    "supply_air_damper_percentage_command": 11},
            },
        })
        m["observation_field_names"] = (
            m["observation_field_names"] + [
                "VAV_1_zone_air_temperature_sensor",
                "VAV_1_zone_air_heating_temperature_setpoint_sensor",
                "VAV_1_zone_air_cooling_temperature_setpoint_sensor",
                "ahs_1_outside_air_temperature_sensor",
            ]
        )
        p = TransferPolicy(m, None)
        obs = np.concatenate((
            observation(kelvin=285.0),
            np.array([(305.0-295.0)/2.0, 293.0, 297.0, 1.0])))
        a = p.get_action(obs)
        self.assertEqual(a["continuous_action"].shape, (12,))
        self.assertEqual(a["discrete_action"].shape, (3,))
        self.assertEqual(a["discrete_action"][:2].tolist(), [1, 1])
        self.assertEqual(int(a["discrete_action"][-1]), 1)
        self.assertTrue(np.all(np.isfinite(a["continuous_action"])))

    def test_surrogate_ablation_is_runnable(self):
        self.p.USE_ECONOMIC_OBJECTIVE = False
        for t in (295.0, 298.0, 290.0):
            a = self.p.get_action(observation(t))
            self.assertTrue(np.isfinite(a["continuous_action"]).all())

    def test_model_updates_and_reset(self):
        zone = self.p.ahus[0]["zones"][0]
        self.p.get_action(observation(300.0))
        self.p.get_action(observation(299.8))
        self.assertEqual(zone["model"].samples, 1)
        self.p.reset()
        self.assertEqual(zone["model"].samples, 0)
        self.assertIsNone(zone["model"].last_features)

    def test_invalid_sensor_and_long_run(self):
        for _ in range(60):
            action = self.p.get_action(observation(296.0, outside_k=-30.0 + 273.15))
            self.assertTrue(np.isfinite(action["continuous_action"]).all())
        bad = observation()
        bad[0] = float("nan")
        self.p.get_action(bad)
        self.assertIsNone(self.p.ahus[0]["zones"][0]["model"].last_features)

    def test_bounded_compute(self):
        start = time.monotonic()
        for i in range(120):
            self.p.get_action(observation(295.0 + math.sin(i / 10)))
        self.assertLess(time.monotonic() - start, 12.0)


if __name__ == "__main__":
    unittest.main()
