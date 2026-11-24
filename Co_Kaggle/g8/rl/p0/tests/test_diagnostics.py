"""P0.1 diagnostic unit tests without the physical simulator."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest

import numpy as np

from Co_Kaggle.g8.rl.p0.diagnostics_core import (
    DiagnosticAccumulator, action_groups, zone_excursion,
)


def fixture_metadata():
    return {
        "temp_indices": [0], "pressure_indices": [1, 3],
        "recirculation_indices": [2], "water_indices": [4],
        "n_discrete_action": 2,
        "zone_action_indices": {"VAV_1": {
            "supply_air_damper_percentage_command": 5,
            "reheat_valve_percentage_command": 6,
        }},
    }


def zone(temp, low, high, occupancy):
    return SimpleNamespace(zone_air_temperature=temp,
                           heating_setpoint_temperature=low,
                           cooling_setpoint_temperature=high,
                           average_occupancy=occupancy)


def fake_response():
    return SimpleNamespace(
        normalized_productivity_regret=-0.1,
        normalized_energy_cost=0.2,
        normalized_carbon_emission=0.3,
        total_occupancy=2.0,
        electricity_energy_cost=1.0,
        natural_gas_energy_cost=2.0,
        carbon_emitted=0.5,
    )


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.acc = DiagnosticAccumulator(fixture_metadata())
        self.weights = {"productivity_weight": 0.6,
                        "energy_cost_weight": 0.2,
                        "carbon_emission_weight": 0.2}
        self.info = SimpleNamespace(zone_reward_infos={
            "a": zone(299.0, 294.0, 297.0, 2.0),  # 2 K, occupied.
            "b": zone(289.0, 294.0, 297.0, 0.0),  # 5 K, unoccupied.
        })
        self.action = {
            "continuous_action": np.array([0.95, -1., 0., 1., -1., 1., -1.],
                                          dtype=np.float32),
            "discrete_action": np.array([1, 0], dtype=np.int32),
        }

    def record(self, **kwargs):
        defaults = dict(
            timestamp=datetime(2026, 11, 24, 10), comfort_mode=True,
            action=self.action, reward_info=self.info,
            response=fake_response(), reward=-0.16, weights=self.weights,
        )
        defaults.update(kwargs)
        return self.acc.record(**defaults)

    def test_action_group_indices(self):
        groups = action_groups(fixture_metadata())
        self.assertEqual(groups["ahu_static_pressure"], ("continuous", [1]))
        self.assertEqual(groups["water_pressure"], ("continuous", [3]))
        self.assertEqual(groups["vav_damper"], ("continuous", [5]))
        self.assertEqual(groups["water_run"], ("discrete", [1]))

    def test_deviation_zero_inside_band_and_linear_outside(self):
        self.assertEqual(zone_excursion(zone(295, 294, 297, 1)), 0.0)
        self.assertEqual(zone_excursion(zone(299, 294, 297, 1)), 2.0)
        self.assertEqual(zone_excursion(zone(290, 294, 297, 1)), 4.0)
        with self.assertRaises(ValueError):
            zone_excursion(zone(290, 298, 297, 1))

    def test_reward_occupancy_excursions_actions(self):
        row = self.record()
        self.assertAlmostEqual(row["comfort_loss"], 0.06)
        self.assertAlmostEqual(row["energy_loss"], 0.04)
        self.assertAlmostEqual(row["carbon_loss"], 0.06)
        self.assertEqual(row["exceed_1k_zones"], 2)
        self.assertEqual(row["exceed_1k_occupied_zones"], 1)
        self.assertEqual(row["exceed_5k_zones"], 0)  # STRICT > threshold.
        self.assertEqual(row["max_zone_excursion_k"], 5.0)
        self.assertEqual(row["occupancy"], 2.0)

        a = self.acc.summary()
        self.assertAlmostEqual(a["mean_loss"], 0.16)
        self.assertEqual(a["by_hour"]["10"]["steps"], 1)
        self.assertEqual(a["by_occupancy"]["occupied"]["steps"], 1)
        self.assertEqual(a["by_schedule"]["comfort_mode"]["steps"], 1)
        self.assertAlmostEqual(
            a["comfort_excursions"]["thresholds_k"]["1"]["occupied_zone_fraction"],
            1.0,
        )
        self.assertAlmostEqual(
            a["comfort_excursions"]["thresholds_k"]["1"]["occupant_weighted_fraction"],
            1.0,
        )
        self.assertEqual(a["action_utilization"]["ahu_run"]["on_fraction"], 1.0)
        self.assertEqual(a["action_utilization"]["water_run"]["on_fraction"], 0.0)
        self.assertEqual(
            a["action_utilization"]["vav_damper"]["near_upper_bound_fraction"], 1.0
        )

    def test_second_step_discrete_switching(self):
        self.record()
        next_action = {
            "continuous_action": self.action["continuous_action"].copy(),
            "discrete_action": np.array([0, 1], dtype=np.int32),
        }
        self.record(action=next_action, timestamp=datetime(2026, 11, 24, 11),
                    comfort_mode=False)
        summ = self.acc.summary()
        self.assertAlmostEqual(summ["action_utilization"]["ahu_run"]["on_fraction"], 0.5)
        self.assertAlmostEqual(
            summ["action_utilization"]["ahu_run"]["switch_fraction_per_transition"], 1.0
        )
        self.assertEqual(summ["by_schedule"]["eco_mode"]["steps"], 1)

    def test_mismatched_reward_is_rejected(self):
        with self.assertRaises(ValueError):
            self.record(reward=-0.12)

    def test_occupancy_inconsistency_is_rejected(self):
        wrong = fake_response()
        wrong.total_occupancy = 3.0
        with self.assertRaises(ValueError):
            self.record(response=wrong)


if __name__ == "__main__":
    unittest.main()
