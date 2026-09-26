#!/usr/bin/env python3
"""Black-box event tests for v20_1.forecast() across five real v20 replays.

forecast() itself is not modified or instrumented. For every detected event, the suite
runs the real observation and a same-turn counterfactual that changes only the event
field. The forecast delta is then checked against an independent test specification.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC_REPLY = HERE.parent
G5_ROOT = HERE.parents[2]
AGENT_PATH = STATIC_REPLY / "v20_1.py"
HISTORY_DIR = G5_ROOT / "game_history" / "v20"
HISTORY_NAMES = (
    "111548564.json",
    "111549675.json",
    "111551968.json",
    "111553265.json",
    "111587965.json",
)
EPS = 1e-9

# Independent forecast specification. Expected results deliberately do not use the
# production module's SHOPS/ANIMALS/CROPS/dist/nearest_shed helpers.
SPEC_TURNS_PER_DAY = 24
SPEC_SEASON_TURNS = 720
SPEC_SHOP_INTERVAL = 4
SPEC_SHOP_UNLOCK_TURNS = 72
SPEC_MAX_SHOPS = 8
SPEC_SHED = ((4, 4), (5, 4), (4, 5), (5, 5))

SPEC_SHOPS = {
    "BAKERY": {"WHEAT": 1, "EGG": 1},
    "PIZZA_SHOP": {"MILK": 1, "TOMATO": 1, "WHEAT": 1},
    "BRUNCH_SPOT": {"EGG": 1, "WHEAT": 1, "STRAWBERRY": 1},
    "YARN_STORE": {"WOOL": 2},
    "ICE_CREAM_SHOP": {"STRAWBERRY": 1, "MILK": 1, "WHEAT": 1},
    "PET_CAFE": {"CARROT": 2},
    "SMOOTHIE_SHOP": {"STRAWBERRY": 1, "MILK": 1},
    "FARMERS_MARKET": {"WHEAT": 1, "CARROT": 1, "TOMATO": 1, "STRAWBERRY": 1},
}

SPEC_ANIMALS = {
    "COW": {"product": "MILK", "first_day": 8, "interval_days": 2, "units": 3},
    "SHEEP": {"product": "WOOL", "first_day": 6, "interval_days": 3, "units": 4},
    "GOOSE": {"product": "EGG", "first_day": 4, "interval_days": 1, "units": 2},
}

SPEC_CROPS = {
    "WHEAT": {"ongoing": False, "events": ((4, 4),)},
    "CARROT": {"ongoing": False, "events": ((3, 3),)},
    "TOMATO": {"ongoing": True, "events": ((8, 1), (9, 1), (10, 1), (11, 1))},
    "STRAWBERRY": {"ongoing": True, "events": ((10, 1), (12, 1), (14, 1), (16, 1))},
    "MELON": {"ongoing": False, "events": ((10, 6),)},
}


def load_agent():
    spec = importlib.util.spec_from_file_location("_forecast_test_v20_1", AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_history(name):
    path = HISTORY_DIR / name
    history = json.loads(path.read_text(encoding="utf-8"))
    if len(history["steps"]) != SPEC_SEASON_TURNS:
        raise AssertionError(
            f"{name}: expected {SPEC_SEASON_TURNS} steps, got {len(history['steps'])}"
        )
    return history


def losing_seat(history):
    final = history["steps"][-1]
    rewards = [float(final[i]["reward"]) for i in (0, 1)]
    if rewards[0] == rewards[1]:
        raise AssertionError("cannot infer v20 seat from a tied replay")
    return 0 if rewards[0] < rewards[1] else 1


def observation(history, history_step, seat):
    return copy.deepcopy(history["steps"][history_step][seat]["observation"])


def absolute_turn(obs):
    return int(obs["day"]) * SPEC_TURNS_PER_DAY + int(obs["hour"])


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shed_distance(p):
    return min(manhattan(p, shed) for shed in SPEC_SHED)


def assert_close(testcase, actual, expected, msg):
    testcase.assertTrue(
        math.isclose(actual, expected, abs_tol=EPS, rel_tol=0.0),
        f"{msg}: actual={actual}, expected={expected}",
    )


def assert_forecast_shape(projected, net_flow, items):
    expected_products = set(items)
    if set(projected) != expected_products:
        raise AssertionError(
            f"projected products mismatch: {set(projected)} != {expected_products}"
        )
    if set(net_flow) != expected_products:
        raise AssertionError(
            f"net_flow products mismatch: {set(net_flow)} != {expected_products}"
        )
    for product in expected_products:
        if len(net_flow[product]) != SPEC_SEASON_TURNS:
            raise AssertionError(
                f"net_flow[{product}] length={len(net_flow[product])}, "
                f"expected {SPEC_SEASON_TURNS}"
            )
        if len(projected[product]) != 31:
            raise AssertionError(
                f"projected[{product}] length={len(projected[product])}, expected 31"
            )


def run_forecast(agent, obs):
    agent.STEP = absolute_turn(obs)
    projected, net_flow = agent.forecast(copy.deepcopy(obs))
    assert_forecast_shape(projected, net_flow, obs["market"]["inventory"])
    return projected, net_flow


def subtract_series(actual, baseline):
    if set(actual) != set(baseline):
        raise AssertionError(
            f"series products mismatch: {set(actual)} != {set(baseline)}"
        )
    result = {}
    for product in actual:
        if len(actual[product]) != len(baseline[product]):
            raise AssertionError(
                f"series length mismatch for {product}: "
                f"{len(actual[product])} != {len(baseline[product])}"
            )
        result[product] = [
            actual[product][i] - baseline[product][i]
            for i in range(len(actual[product]))
        ]
    return result


def assert_projection_delta_consistent(testcase, delta_flow, delta_projected, step):
    """Daily projected delta must equal cumulative turn-flow delta before that day."""
    testcase.assertEqual(set(delta_flow), set(delta_projected))
    for product in delta_flow:
        testcase.assertEqual(len(delta_flow[product]), SPEC_SEASON_TURNS)
        testcase.assertEqual(len(delta_projected[product]), 31)
        running = 0.0
        for day in range(31):
            sample_turn = min(SPEC_SEASON_TURNS - 1, day * SPEC_TURNS_PER_DAY)
            if sample_turn <= step:
                expected = 0.0
            else:
                running = sum(delta_flow[product][step:sample_turn])
                expected = running
            assert_close(
                testcase,
                delta_projected[product][day],
                expected,
                f"PROJECTED day={day} turn={sample_turn} product={product}",
            )


def assert_flow_exact(testcase, delta_flow, expected, label):
    testcase.assertEqual(set(delta_flow), set(expected))
    for product in delta_flow:
        testcase.assertEqual(len(delta_flow[product]), SPEC_SEASON_TURNS)
        testcase.assertEqual(len(expected[product]), SPEC_SEASON_TURNS)
        for t in range(SPEC_SEASON_TURNS):
            assert_close(
                testcase,
                delta_flow[product][t],
                expected[product][t],
                f"{label} t={t} product={product}",
            )


def empty_expected(flow):
    return {product: [0.0] * SPEC_SEASON_TURNS for product in flow}


def added_shops(prev_obs, obs):
    before = Counter(prev_obs["town"]["unlocked_shops"])
    after = Counter(obs["town"]["unlocked_shops"])
    return list((after - before).elements())


def public_asset(tile_state):
    return isinstance(tile_state, dict) and (
        "animal" in tile_state or "crop" in tile_state
    )


def find_shop_events(history, seat):
    events = []
    for i in range(1, SPEC_SEASON_TURNS):
        prev_obs = observation(history, i - 1, seat)
        obs = observation(history, i, seat)
        added = added_shops(prev_obs, obs)
        if added:
            events.append((i, prev_obs, obs, added))
    return events


def find_clean_opponent_asset_events(history, seat, asset_key):
    opponent = 1 - seat
    events = []
    for i in range(1, SPEC_SEASON_TURNS):
        prev_obs = observation(history, i - 1, seat)
        obs = observation(history, i, seat)
        prev_tiles = prev_obs["farms"][opponent]["tiles"]
        tiles = obs["farms"][opponent]["tiles"]
        for y, row in enumerate(tiles):
            for x, tile_state in enumerate(row):
                if not isinstance(tile_state, dict) or asset_key not in tile_state:
                    continue
                previous = prev_tiles[y][x]
                if public_asset(previous):
                    continue
                events.append((i, obs, x, y, tile_state))
    return events


def find_opponent_yield_events(history, seat, asset_key):
    opponent = 1 - seat
    events = []
    for i in range(1, SPEC_SEASON_TURNS):
        prev_obs = observation(history, i - 1, seat)
        obs = observation(history, i, seat)
        prev_tiles = prev_obs["farms"][opponent]["tiles"]
        tiles = obs["farms"][opponent]["tiles"]
        for y, row in enumerate(tiles):
            for x, tile_state in enumerate(row):
                previous = prev_tiles[y][x]
                if not isinstance(tile_state, dict) or not isinstance(previous, dict):
                    continue
                asset = tile_state.get(asset_key)
                if asset is None or previous.get(asset_key) != asset:
                    continue
                placed_key = "placed_day" if asset_key == "animal" else "planted_day"
                if previous.get(placed_key) != tile_state.get(placed_key):
                    continue
                before = previous.get("yield_units", 0)
                after = tile_state.get("yield_units", 0)
                if before == after:
                    continue
                events.append((i, obs, x, y, before, after, tile_state))
    return events


class ForecastMultiHistoryEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = load_agent()
        cls.cases = []
        for name in HISTORY_NAMES:
            history = load_history(name)
            cls.cases.append((name, history, losing_seat(history)))

    def test_fixture_clock_and_model_spec(self):
        """All five fixtures and the independent model specification remain coherent."""
        agent = self.agent
        self.assertEqual(agent.TURNS_PER_DAY, SPEC_TURNS_PER_DAY)
        self.assertEqual(agent.SEASON_TURNS, SPEC_SEASON_TURNS)
        self.assertEqual(agent.SHOP_INTERVAL, SPEC_SHOP_INTERVAL)
        self.assertEqual(agent.SHOP_UNLOCK_TURNS, SPEC_SHOP_UNLOCK_TURNS)
        self.assertEqual(agent.MAX_SHOPS, SPEC_MAX_SHOPS)
        self.assertEqual(tuple(agent.SHED), SPEC_SHED)
        self.assertEqual(agent.SHOPS, SPEC_SHOPS)

        actual_animals = {
            animal: {
                "product": cfg[1],
                "first_day": cfg[2],
                "interval_days": cfg[3],
                "units": cfg[4],
            }
            for animal, cfg in agent.ANIMALS.items()
        }
        self.assertEqual(actual_animals, SPEC_ANIMALS)

        for name, history, seat in self.cases:
            for i in range(SPEC_SEASON_TURNS):
                obs = observation(history, i, seat)
                self.assertEqual(
                    absolute_turn(obs),
                    i,
                    f"{name}: history step {i} != observation day/hour",
                )

    def test_shop_open_events_change_full_season_demand_exactly(self):
        seen_shops = Counter()
        total_events = 0

        for name, history, seat in self.cases:
            events = find_shop_events(history, seat)
            self.assertGreater(len(events), 0, f"{name}: no shop-open events")
            total_events += len(events)

            for i, _prev_obs, obs, added in events:
                step = absolute_turn(obs)
                counterfactual = copy.deepcopy(obs)
                previous_shops = list(obs["town"]["unlocked_shops"])
                for shop in added:
                    previous_shops.remove(shop)
                    seen_shops[shop] += 1
                counterfactual["town"]["unlocked_shops"] = previous_shops

                projected_actual, flow_actual = run_forecast(self.agent, obs)
                projected_before, flow_before = run_forecast(self.agent, counterfactual)
                delta_flow = subtract_series(flow_actual, flow_before)
                delta_projected = subtract_series(projected_actual, projected_before)

                added_demand = {c: 0.0 for c in obs["market"]["inventory"]}
                for shop in added:
                    for product, units in SPEC_SHOPS[shop].items():
                        if product in added_demand:
                            added_demand[product] += units

                expected_random = {
                    product: sum(
                        products.get(product, 0) for products in SPEC_SHOPS.values()
                    ) / len(SPEC_SHOPS)
                    for product in obs["market"]["inventory"]
                }
                first_unlock = (
                    (step // SPEC_SHOP_UNLOCK_TURNS) + 1
                ) * SPEC_SHOP_UNLOCK_TURNS
                actual_unlocks = list(
                    range(first_unlock, SPEC_SEASON_TURNS, SPEC_SHOP_UNLOCK_TURNS)
                )[: max(0, SPEC_MAX_SHOPS - len(obs["town"]["unlocked_shops"]))]
                baseline_unlocks = list(
                    range(first_unlock, SPEC_SEASON_TURNS, SPEC_SHOP_UNLOCK_TURNS)
                )[: max(0, SPEC_MAX_SHOPS - len(counterfactual["town"]["unlocked_shops"]))]

                expected = empty_expected(flow_actual)
                for t in range(step, SPEC_SEASON_TURNS):
                    if t % SPEC_SHOP_INTERVAL:
                        continue
                    actual_new = sum(unlock <= t for unlock in actual_unlocks)
                    baseline_new = sum(unlock <= t for unlock in baseline_unlocks)
                    future_count_delta = actual_new - baseline_new
                    for product in expected:
                        expected[product][t] = -(
                            added_demand.get(product, 0.0)
                            + future_count_delta * expected_random.get(product, 0.0)
                        )

                assert_flow_exact(
                    self, delta_flow, expected, f"{name} SHOP_OPEN step={step}"
                )
                assert_projection_delta_consistent(
                    self, delta_flow, delta_projected, step
                )

            print(f"[{name}] shop-open events={len(events)}")

        self.assertEqual(
            set(seen_shops),
            set(SPEC_SHOPS),
            f"five histories did not cover every shop type: {seen_shops}",
        )
        print(f"[all histories] shop events={total_events} by_type={dict(seen_shops)}")

    def test_opponent_animal_appear_events_all_species(self):
        seen = Counter()
        total_events = 0

        for name, history, seat in self.cases:
            opponent = 1 - seat
            events = find_clean_opponent_asset_events(history, seat, "animal")
            total_events += len(events)

            for i, obs, x, y, animal_state in events:
                animal = animal_state["animal"]
                seen[animal] += 1
                cfg = SPEC_ANIMALS[animal]
                step = absolute_turn(obs)
                p = (x, y)

                counterfactual = copy.deepcopy(obs)
                counterfactual["farms"][opponent]["tiles"][y][x].pop("animal", None)

                projected_actual, flow_actual = run_forecast(self.agent, obs)
                projected_before, flow_before = run_forecast(self.agent, counterfactual)
                delta_flow = subtract_series(flow_actual, flow_before)
                delta_projected = subtract_series(projected_actual, projected_before)
                expected = empty_expected(flow_actual)

                # One additional animal adds exactly one WHEAT unit to each daily herd tick.
                first_feed = ((step // SPEC_TURNS_PER_DAY) + 1) * SPEC_TURNS_PER_DAY
                for t in range(first_feed, SPEC_SEASON_TURNS, SPEC_TURNS_PER_DAY):
                    expected["WHEAT"][t] -= 1.0

                product = cfg["product"]
                held = animal_state.get("yield_units", 0)
                positions = [tuple(obs["farms"][opponent]["farmer"])] + [
                    tuple(pos) for pos in obs["farms"][opponent]["hands"]
                ]
                if held:
                    approach = min(manhattan(pos, p) for pos in positions)
                    at = step + approach + shed_distance(p) + 1
                    if at < SPEC_SEASON_TURNS:
                        expected[product][at] += held

                first_day = animal_state["placed_day"] + cfg["first_day"]
                for at_day in range(max(int(obs["day"]) + 1, first_day), 30):
                    if (at_day - first_day) % cfg["interval_days"]:
                        continue
                    at = at_day * SPEC_TURNS_PER_DAY + shed_distance(p) + 1
                    if at < SPEC_SEASON_TURNS:
                        expected[product][at] += cfg["units"]

                assert_flow_exact(
                    self,
                    delta_flow,
                    expected,
                    f"{name} OPP_{animal}_APPEAR step={step}",
                )
                assert_projection_delta_consistent(
                    self, delta_flow, delta_projected, step
                )

            print(f"[{name}] clean opponent animal appearances={len(events)}")

        self.assertEqual(
            set(seen),
            set(SPEC_ANIMALS),
            f"five histories did not cover every animal type: {seen}",
        )
        print(f"[all histories] animal appearances={total_events} by_type={dict(seen)}")

    def test_opponent_animal_yield_changes_all_species(self):
        seen = Counter()
        total_events = 0

        for name, history, seat in self.cases:
            opponent = 1 - seat
            events = find_opponent_yield_events(history, seat, "animal")
            total_events += len(events)

            for i, obs, x, y, before, after, animal_state in events:
                animal = animal_state["animal"]
                seen[animal] += 1
                product = SPEC_ANIMALS[animal]["product"]
                step = absolute_turn(obs)
                p = (x, y)

                counterfactual = copy.deepcopy(obs)
                counterfactual["farms"][opponent]["tiles"][y][x]["yield_units"] = before

                projected_actual, flow_actual = run_forecast(self.agent, obs)
                projected_before, flow_before = run_forecast(self.agent, counterfactual)
                delta_flow = subtract_series(flow_actual, flow_before)
                delta_projected = subtract_series(projected_actual, projected_before)
                expected = empty_expected(flow_actual)

                positions = [tuple(obs["farms"][opponent]["farmer"])] + [
                    tuple(pos) for pos in obs["farms"][opponent]["hands"]
                ]
                approach = min(manhattan(pos, p) for pos in positions)
                arrival = step + approach + shed_distance(p) + 1
                if arrival < SPEC_SEASON_TURNS:
                    expected[product][arrival] += after - before

                assert_flow_exact(
                    self,
                    delta_flow,
                    expected,
                    f"{name} OPP_{animal}_YIELD step={step}",
                )
                assert_projection_delta_consistent(
                    self, delta_flow, delta_projected, step
                )

            print(f"[{name}] opponent animal-yield changes={len(events)}")

        self.assertEqual(
            set(seen),
            set(SPEC_ANIMALS),
            f"five histories did not cover yield changes for every animal type: {seen}",
        )
        print(f"[all histories] animal-yield events={total_events} by_type={dict(seen)}")

    def test_opponent_crop_appear_events_all_types(self):
        seen = Counter()
        total_events = 0

        for name, history, seat in self.cases:
            opponent = 1 - seat
            events = find_clean_opponent_asset_events(history, seat, "crop")
            total_events += len(events)

            for i, obs, x, y, crop_state in events:
                crop = crop_state["crop"]
                seen[crop] += 1
                crop_spec = SPEC_CROPS[crop]
                step = absolute_turn(obs)
                p = (x, y)

                counterfactual = copy.deepcopy(obs)
                counterfactual["farms"][opponent]["tiles"][y][x].pop("crop", None)

                projected_actual, flow_actual = run_forecast(self.agent, obs)
                projected_before, flow_before = run_forecast(self.agent, counterfactual)
                delta_flow = subtract_series(flow_actual, flow_before)
                delta_projected = subtract_series(projected_actual, projected_before)
                expected = empty_expected(flow_actual)

                held = crop_state.get("yield_units", 0)
                positions = [tuple(obs["farms"][opponent]["farmer"])] + [
                    tuple(pos) for pos in obs["farms"][opponent]["hands"]
                ]
                if held:
                    approach = min(manhattan(pos, p) for pos in positions)
                    at = step + approach + shed_distance(p) + 1
                    if at < SPEC_SEASON_TURNS:
                        expected[crop][at] += held

                planted = crop_state["planted_day"]
                for age, scheduled_units in crop_spec["events"]:
                    at_day = planted + age
                    ready_turn = at_day * SPEC_TURNS_PER_DAY
                    if ready_turn <= step or at_day >= 30:
                        continue
                    if crop_spec["ongoing"]:
                        units = (
                            2
                            if crop_state.get("fertilized_until_day", -1) >= at_day
                            else 1
                        )
                    else:
                        units = max(0, scheduled_units - held)
                    if not units:
                        continue
                    at = ready_turn + shed_distance(p) + 1
                    if at < SPEC_SEASON_TURNS:
                        expected[crop][at] += units

                assert_flow_exact(
                    self,
                    delta_flow,
                    expected,
                    f"{name} OPP_{crop}_APPEAR step={step}",
                )
                assert_projection_delta_consistent(
                    self, delta_flow, delta_projected, step
                )

            print(f"[{name}] clean opponent crop appearances={len(events)}")

        self.assertEqual(
            set(seen),
            set(SPEC_CROPS),
            f"five histories did not cover every crop type: {seen}",
        )
        print(f"[all histories] crop appearances={total_events} by_type={dict(seen)}")

    def test_opponent_crop_yield_changes_all_types(self):
        seen = Counter()
        total_events = 0

        for name, history, seat in self.cases:
            opponent = 1 - seat
            events = find_opponent_yield_events(history, seat, "crop")
            total_events += len(events)

            for i, obs, x, y, before, after, crop_state in events:
                crop = crop_state["crop"]
                seen[crop] += 1
                crop_spec = SPEC_CROPS[crop]
                step = absolute_turn(obs)
                p = (x, y)

                counterfactual = copy.deepcopy(obs)
                counterfactual["farms"][opponent]["tiles"][y][x]["yield_units"] = before

                projected_actual, flow_actual = run_forecast(self.agent, obs)
                projected_before, flow_before = run_forecast(self.agent, counterfactual)
                delta_flow = subtract_series(flow_actual, flow_before)
                delta_projected = subtract_series(projected_actual, projected_before)
                expected = empty_expected(flow_actual)

                positions = [tuple(obs["farms"][opponent]["farmer"])] + [
                    tuple(pos) for pos in obs["farms"][opponent]["hands"]
                ]
                approach = min(manhattan(pos, p) for pos in positions)
                arrival = step + approach + shed_distance(p) + 1
                if arrival < SPEC_SEASON_TURNS:
                    expected[crop][arrival] += after - before

                # For one-time crops, visible held yield is subtracted from the later
                # scheduled total. Changing yield_units therefore changes both the
                # visible-supply ETA and any still-future remainder.
                if not crop_spec["ongoing"]:
                    planted = crop_state["planted_day"]
                    for age, scheduled_units in crop_spec["events"]:
                        at_day = planted + age
                        ready_turn = at_day * SPEC_TURNS_PER_DAY
                        if ready_turn <= step or at_day >= 30:
                            continue
                        before_remaining = max(0, scheduled_units - before)
                        after_remaining = max(0, scheduled_units - after)
                        remainder_delta = after_remaining - before_remaining
                        if remainder_delta:
                            at = ready_turn + shed_distance(p) + 1
                            if at < SPEC_SEASON_TURNS:
                                expected[crop][at] += remainder_delta

                assert_flow_exact(
                    self,
                    delta_flow,
                    expected,
                    f"{name} OPP_{crop}_YIELD step={step}",
                )
                assert_projection_delta_consistent(
                    self, delta_flow, delta_projected, step
                )

            print(f"[{name}] opponent crop-yield changes={len(events)}")

        self.assertEqual(
            set(seen),
            set(SPEC_CROPS),
            f"five histories did not cover yield changes for every crop type: {seen}",
        )
        print(f"[all histories] crop-yield events={total_events} by_type={dict(seen)}")


if __name__ == "__main__":
    unittest.main()
