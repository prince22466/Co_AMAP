"""Synthetic contract and coupled-equipment regressions for g8 Version 23.

Run: python Co_Kaggle/g8/submissions/test_version23.py
No Kaggle simulator is required; this does NOT measure leaderboard quality.
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


NOTEBOOK = Path(__file__).with_name("submission-nb - Version 23.ipynb")
book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert len(book["cells"]) == 3, "Expected documentation, policy, gateway"
assert [c["cell_type"] for c in book["cells"]] == [
    "markdown", "code", "code"]
assert "SmartBuildingsInferenceServer(TransferPolicy).run()" in "".join(
    book["cells"][2]["source"])
source = "".join(book["cells"][1]["source"])
ast.parse(source)
ast.parse("".join(book["cells"][2]["source"]))

gateway = types.ModuleType("smart_buildings_inference_server")
gateway.Policy = type("GatewayPolicy", (), {"__init__": lambda self, *args: None})
kaggle = types.ModuleType("kaggle_evaluation")
kaggle.smart_buildings_inference_server = gateway
with patch.dict(sys.modules, {"kaggle_evaluation": kaggle}):
    namespace = {}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
Policy = namespace["TransferPolicy"]
project = namespace["project_shared_capacity"]


def fixture(ahu_count=2, per_ahu=3, outdoors=290.0):
    """Independent action offsets and different observation/sensor indices."""
    assignments, actions, fields = {}, {}, [
        "hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000"]
    n_zones = ahu_count * per_ahu
    for ai in range(ahu_count):
        name = f"ahu_{ai}"
        assignments[name] = []
        fields.append(f"ahs_{name}_outside_air_temperature_sensor")
        for j in range(per_ahu):
            k = ai * per_ahu + j
            assignments[name].append(f"room_{k}")
            zone = f"VAV_{k}"
            fields.extend([
                f"{zone}_zone_air_temperature_sensor",
                f"{zone}_zone_air_heating_temperature_setpoint_sensor",
                f"{zone}_zone_air_cooling_temperature_setpoint_sensor"])
            base = 3 * ahu_count + 2 + 2 * k
            actions[zone] = {
                "supply_air_damper_percentage_command": base,
                "reheat_valve_percentage_command": base + 1}
    metadata = {
        "n_continuous_action": 3 * ahu_count + 2 + 2 * n_zones,
        "n_discrete_action": ahu_count + 1,
        "temp_indices": [3 * i for i in range(ahu_count)],
        "pressure_indices": [3 * i + 1 for i in range(ahu_count)]
                            + [3 * ahu_count + 1],
        "recirculation_indices": [3 * i + 2 for i in range(ahu_count)],
        "water_indices": [3 * ahu_count],
        "observation_field_names": fields,
        "ahu_zone_assignments": assignments,
        "zone_action_indices": actions,
        "comfort_temp_window": [293.0, 297.0],
        "eco_temp_window": [289.0, 301.0],
        "morning_start_hour": 8.0,
        "evening_start_hour": 18.0,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {
                "sample_mean": 295.0, "sample_std": 5.0},
            "outside_air_temperature_sensor": {
                "sample_mean": 290.0, "sample_std": 10.0},
        },
    }
    obs = np.zeros(len(fields), dtype=np.float64)
    # Monday 12:00; occupied comfort fallback on bad setpoint readings.
    obs[0:4] = [0.0, -1.0, 0.0, 1.0]
    for k, name in enumerate(fields):
        if name.endswith("_outside_air_temperature_sensor"):
            obs[k] = (outdoors - 290.0) / 10.0
        elif name.endswith("_zone_air_temperature_sensor"):
            obs[k] = 0.0
        elif name.endswith("_zone_air_heating_temperature_setpoint_sensor"):
            obs[k] = 293.0
        elif name.endswith("_zone_air_cooling_temperature_setpoint_sensor"):
            obs[k] = 297.0
    return metadata, obs


def set_zones(meta, obs, kelvin):
    """Supply one physical temperature per zone; normalize each observation."""
    positions = {name: i for i, name in enumerate(meta["observation_field_names"])}
    for i, temperature in enumerate(kelvin):
        obs[positions[f"VAV_{i}_zone_air_temperature_sensor"]] = (
            temperature - 295.0) / 5.0
    return obs


def zone_states(policy, obs, ahu_index=0):
    fallback = policy._schedule_fallback(obs)
    states = [policy._read_zone(obs, z, fallback)
              for z in policy.ahus[ahu_index]["zones"]]
    return [s for s in states if s is not None]


class Version23Tests(unittest.TestCase):
    def test_projection_capacity_and_priority(self):
        desired = np.array([0.9, 0.9, 0.9])
        allocation = project(desired, np.array([15.0, 1.0, 1.0]), 1.1)
        self.assertLessEqual(float(allocation.sum()), 1.1 + 1e-8)
        self.assertTrue(np.all(allocation >= 0.0))
        self.assertTrue(np.all(allocation <= desired))
        self.assertGreater(allocation[0], allocation[1])
        np.testing.assert_allclose(project(desired, np.ones(3), 4.0), desired)
        np.testing.assert_array_equal(project(desired, np.ones(3), 0.0),
                                      np.zeros(3))
        for invalid in (-1.0, float("nan")):
            with self.assertRaises(ValueError):
                project(desired, np.ones(3), invalid)

    def test_idle_and_action_contract(self):
        meta, obs = fixture()
        policy = Policy(meta, None)
        result = policy.get_action(obs)
        a, d = result["continuous_action"], result["discrete_action"]
        self.assertEqual(a.shape, (meta["n_continuous_action"],))
        self.assertEqual(d.shape, (meta["n_discrete_action"],))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(d.dtype, np.int32)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(((a >= -1.0) & (a <= 1.0)).all())
        self.assertTrue(np.isin(d, [0, 1]).all())
        self.assertEqual(d.tolist(), [0, 0, 0])
        np.testing.assert_array_equal(policy.get_action(obs)["continuous_action"], a)

    def test_outdoor_denormalization_and_invalid_fallback(self):
        meta, obs = fixture(1, 2, outdoors=279.0)
        policy = Policy(meta, None)
        idx = policy.ahus[0]["outdoor"]
        self.assertAlmostEqual(policy._outdoor_temperature(obs, idx), 279.0)
        bad = obs.copy()
        bad[idx] = 999.0
        self.assertTrue(math.isnan(policy._outdoor_temperature(bad, idx)))
        bad[idx] = np.nan
        self.assertTrue(math.isnan(policy._outdoor_temperature(bad, idx)))

    def test_shared_supply_changes_both_zone_predictions(self):
        meta, obs = fixture(1, 2)
        policy = Policy(meta, None)
        obs = set_zones(meta, obs, [301.0, 295.0])
        states = zone_states(policy, obs)
        plan = policy._plan_ahu(states, 290.0)
        self.assertIsNotNone(plan)
        self.assertIn(plan["coil_k"], policy.SUPPLY_CANDIDATES_K)
        self.assertEqual(plan["damper"].shape, (2,))
        self.assertEqual(plan["predicted_change"].shape, (2,))
        self.assertLess(plan["requested_change"][0], 0.0)
        self.assertEqual(plan["requested_change"][1], 0.0)
        _, _, cold_air = policy._mixed_supply(285.0, [301.0, 295.0], 290.0)
        _, _, warm_air = policy._mixed_supply(305.0, [301.0, 295.0], 290.0)
        # Same upstream supply setpoint changes the model response of BOTH
        # downstream zones (including one inside the comfort band).
        self.assertLess(cold_air, warm_air)
        self.assertLess(policy.AIR_GAIN * 0.2 * (cold_air - 295),
                        policy.AIR_GAIN * 0.2 * (warm_air - 295))

    def test_ahu_flow_budget_respected_for_multiple_zones(self):
        meta, obs = fixture(1, 8)
        obs = set_zones(meta, obs, [302.0] * 8)
        policy = Policy(meta, None)
        plan = policy._plan_ahu(zone_states(policy, obs), 290.0)
        extras = plan["damper"] - policy.MIN_VAV_FRACTION
        self.assertGreater(float(extras.sum()), 0.0)
        self.assertLessEqual(float(extras.sum()), plan["air_budget"] + 1e-8)
        self.assertTrue(np.all(plan["damper"] <= 0.98 + 1e-10))

    def test_mixed_heating_cooling_and_independent_ahus(self):
        meta, obs = fixture(2, 2, outdoors=285.0)
        obs = set_zones(meta, obs, [302.0, 289.0, 295.0, 295.0])
        policy = Policy(meta, None)
        action = policy.get_action(obs)
        continuous, discrete = action["continuous_action"], action["discrete_action"]
        self.assertEqual(discrete[0], 1)
        self.assertEqual(discrete[1], 0)
        self.assertEqual(continuous[meta["zone_action_indices"]["VAV_0"][
            "reheat_valve_percentage_command"]], -1.0)
        self.assertGreater(continuous[meta["zone_action_indices"]["VAV_1"][
            "reheat_valve_percentage_command"]], -1.0)
        self.assertEqual(discrete[-1], 1)

    def test_water_is_shared_across_ahus(self):
        meta, obs = fixture(2, 2, outdoors=276.0)
        policy = Policy(meta, None)
        obs = set_zones(meta, obs, [291.0, 290.0, 289.0, 288.0])
        requests = []
        for i, ahu in enumerate(policy.ahus):
            for state in zone_states(policy, obs, i):
                requests.append((state, 0.90, 1.0))
        continuous, discrete = policy._make_idle_actions()
        policy._dispatch_water(requests, [276.0], continuous, discrete)
        self.assertEqual(discrete[-1], 1)
        load = sum(
            0.90 * ((continuous[state["actions"]["valve"]] + 1.0) / 2.0)
            for state, _, _ in requests)
        budget = (policy.WATER_CAPACITY_FIRST +
                  (len(requests) - 1) * policy.WATER_CAPACITY_ADDITIONAL)
        self.assertLessEqual(load, budget + 1e-6)
        self.assertLess(load, len(requests) * 0.9)
        # More severely cold rooms retain larger portions of a scarce boiler.
        loads = [
            (continuous[state["actions"]["valve"]] + 1.0) / 2.0
            for state, _, _ in requests]
        self.assertGreater(loads[-1], loads[0])

    def test_missing_sensor_and_missing_valve_are_safe(self):
        meta, obs = fixture(1, 2)
        meta["zone_action_indices"]["VAV_0"].pop(
            "reheat_valve_percentage_command")
        policy = Policy(meta, None)
        obs = set_zones(meta, obs, [289.0, 295.0])
        self.assertEqual(policy.get_action(obs)["discrete_action"].tolist(),
                         [0, 0])
        bad = obs.copy()
        bad[policy.ahus[0]["zones"][0]["temperature"]] = np.nan
        result = policy.get_action(bad)
        self.assertTrue(np.isfinite(result["continuous_action"]).all())
        self.assertEqual(result["discrete_action"].tolist(), [0, 0])

    def test_reheat_capacity_depends_on_installed_equipment(self):
        """Boiler capacity must not grow when an extra zone asks for heat."""
        meta, obs = fixture(2, 3)
        policy = Policy(meta, None)
        expected = (policy.WATER_CAPACITY_FIRST +
                    5 * policy.WATER_CAPACITY_ADDITIONAL)
        self.assertEqual(policy.zone_count, 6)
        self.assertAlmostEqual(policy._water_capacity(), expected)

        # The priority/projection cap stays fixed for different request sets.
        cold = set_zones(meta, obs.copy(), [291.0] * 6)
        all_states = [
            state for i in range(2)
            for state in zone_states(policy, cold, i)]
        requests = [(state, 0.95, 1.0) for state in all_states]
        action, switches = policy._make_idle_actions()
        policy._dispatch_water(requests, [290.0], action, switches)
        full_load = sum(
            0.95 * ((action[state["actions"]["valve"]] + 1) / 2)
            for state in all_states)
        self.assertLessEqual(full_load, expected + 1e-6)
        self.assertAlmostEqual(policy._water_capacity(), expected)

        # Passing a smaller request set cannot alter installed capacity.
        action, switches = policy._make_idle_actions()
        policy._dispatch_water(requests[:2], [290.0], action, switches)
        self.assertAlmostEqual(policy._water_capacity(), expected)

    def test_absent_boiler_does_not_repurpose_last_ahu_switch(self):
        """A missing boiler must not toggle or overwrite an AHU switch."""
        meta, obs = fixture(2, 2)
        meta["water_indices"] = []
        meta["n_discrete_action"] = 2  # Exactly one run switch per AHU.
        meta["pressure_indices"].pop()  # No hot-water pump pressure.
        obs = set_zones(meta, obs, [290.0, 289.0, 291.0, 290.0])
        policy = Policy(meta, None)
        self.assertIsNone(policy.water_run)
        self.assertIsNone(policy._plan_ahu(
            zone_states(policy, obs, 0), 290.0))
        action = policy.get_action(obs)
        self.assertEqual(action["discrete_action"].tolist(), [0, 0])
        for zone in meta["zone_action_indices"].values():
            self.assertEqual(action["continuous_action"][
                zone["reheat_valve_percentage_command"]], -1.0)

    def test_broken_zone_observation_and_missing_damper(self):
        """A missing damper is not a source of fictitious cooling capacity."""
        meta, obs = fixture(1, 2)
        meta["zone_action_indices"]["VAV_0"].pop(
            "supply_air_damper_percentage_command")
        obs = set_zones(meta, obs, [301.0, 295.0])
        policy = Policy(meta, None)
        self.assertIsNone(policy._plan_ahu(
            zone_states(policy, obs), 290.0))
        self.assertEqual(policy.get_action(obs)["discrete_action"].tolist(),
                         [0, 0])
        obs[:] = float("nan")
        self.assertTrue(np.isfinite(
            policy.get_action(obs)["continuous_action"]).all())

    def test_large_building_bounded_runtime(self):
        meta, obs = fixture(4, 12, outdoors=287.0)
        obs = set_zones(meta, obs, [289.0, 302.0, 295.0] * 16)
        policy = Policy(meta, None)
        start = time.perf_counter()
        for _ in range(12):
            result = policy.get_action(obs)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 8.0)  # Extremely generous CI guardrail.
        self.assertTrue(np.isfinite(result["continuous_action"]).all())


if __name__ == "__main__":
    unittest.main()
