#!/usr/bin/env python3
"""Black-box event tests for v20_1.forecast() using one real v20 replay.

The forecast implementation itself is not modified or instrumented. Each test finds a
real state transition in game_history/v20/111548564.json, then builds a same-time
counterfactual by reverting only that event. The delta between the two forecast outputs
therefore exposes the behavior attributable to that event.

Run:
    python local_arena/static_reply/forecast_tests/test_v20_1_history_events.py -v
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
HISTORY_PATH = G5_ROOT / "game_history" / "v20" / "111548564.json"
EPS = 1e-9

# Independent test specification. Expected behavior deliberately does NOT read these
# quantities from v20_1.py, otherwise a bad production constant/helper could be copied
# into the expected result and make a broken forecast pass its own regression test.
SPEC_TURNS_PER_DAY = 24
SPEC_SEASON_TURNS = 720
SPEC_SHOP_INTERVAL = 4
SPEC_SHOP_UNLOCK_TURNS = 72
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
SPEC_COW = {"product": "MILK", "first_day": 8, "interval_days": 2, "units": 3}
SPEC_CROPS = {
    "WHEAT": {"ongoing": False, "events": ((4, 4),)},
    "CARROT": {"ongoing": False, "events": ((3, 3),)},
    "TOMATO": {"ongoing": True, "events": ((8, 1), (9, 1), (10, 1), (11, 1))},
    "STRAWBERRY": {"ongoing": True, "events": ((10, 1), (12, 1), (14, 1), (16, 1))},
    "MELON": {"ongoing": False, "events": ((10, 6),)},
}
EXPECTED_EVENT_COUNTS = {
    "shop_open": 8,
    "clean_opponent_cow": 8,
    "clean_opponent_crop": 236,
    "opponent_ongoing_yield": 260,
}


def load_agent():
    spec = importlib.util.spec_from_file_location("_forecast_test_v20_1", AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_history():
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    steps = history["steps"]
    if len(steps) != 720:
        raise AssertionError(f"expected 720 steps, got {len(steps)}")
    return history


def losing_seat(history):
    final = history["steps"][-1]
    rewards = [float(final[i]["reward"]) for i in (0, 1)]
    if rewards[0] == rewards[1]:
        raise AssertionError("cannot infer v20 seat from a tied replay")
    return 0 if rewards[0] < rewards[1] else 1


def observation(history, history_step, seat):
    return copy.deepcopy(history["steps"][history_step][seat]["observation"])


def absolute_turn(_agent, obs):
    return int(obs["day"]) * SPEC_TURNS_PER_DAY + int(obs["hour"])


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shed_distance(p):
    return min(manhattan(p, shed) for shed in SPEC_SHED)


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
    # forecast() reads global STEP but does not advance it. Set it from the observation
    # so baseline and counterfactual runs are evaluated at exactly the same game time.
    agent.STEP = absolute_turn(agent, obs)
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


def assert_close(testcase, actual, expected, msg):
    testcase.assertTrue(
        math.isclose(actual, expected, abs_tol=EPS, rel_tol=0.0),
        f"{msg}: actual={actual}, expected={expected}",
    )


def assert_projection_delta_consistent(testcase, delta_flow, delta_projected, step):
    """Daily projected delta must equal cumulative turn-flow delta before that day."""
    testcase.assertEqual(set(delta_flow), set(delta_projected))
    for product in delta_flow:
        testcase.assertEqual(len(delta_flow[product]), SPEC_SEASON_TURNS)
        testcase.assertEqual(len(delta_projected[product]), 31)
        for day in range(31):
            sample_turn = min(SPEC_SEASON_TURNS - 1, day * SPEC_TURNS_PER_DAY)
            expected = (
                0.0
                if sample_turn <= step
                else sum(delta_flow[product][step:sample_turn])
            )
            assert_close(
                testcase,
                delta_projected[product][day],
                expected,
                f"PROJECTED day={day} turn={sample_turn} product={product}",
            )


def nonzero_points(series, eps=EPS):
    return [(i, value) for i, value in enumerate(series) if abs(value) > eps]


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
    for i in range(1, len(history["steps"])):
        prev_obs = observation(history, i - 1, seat)
        obs = observation(history, i, seat)
        added = added_shops(prev_obs, obs)
        if added:
            events.append((i, prev_obs, obs, added))
    return events


def find_clean_opponent_animal_events(history, seat, animal=None):
    opponent = 1 - seat
    events = []
    for i in range(1, len(history["steps"])):
        prev_obs = observation(history, i - 1, seat)
        obs = observation(history, i, seat)
        prev_tiles = prev_obs["farms"][opponent]["tiles"]
        tiles = obs["farms"][opponent]["tiles"]
        for y, row in enumerate(tiles):
            for x, tile_state in enumerate(row):
                if not isinstance(tile_state, dict) or "animal" not in tile_state:
                    continue
                if animal is not None and tile_state["animal"] != animal:
                    continue
                previous = prev_tiles[y][x]
                # Keep the first suite strictly attributable to one new animal. If the
                # previous tile held another crop/animal, reverting the transition would
                # mix two forecast effects and is intentionally excluded here.
                if public_asset(previous):
                    continue
                events.append((i, prev_obs, obs, x, y, previous, tile_state))
    return events



def find_clean_opponent_crop_events(history, seat):
    opponent = 1 - seat
    events = []
    for i in range(1, len(history["steps"])):
        prev_obs = observation(history, i - 1, seat)
        obs = observation(history, i, seat)
        prev_tiles = prev_obs["farms"][opponent]["tiles"]
        tiles = obs["farms"][opponent]["tiles"]
        for y, row in enumerate(tiles):
            for x, tile_state in enumerate(row):
                if not isinstance(tile_state, dict) or "crop" not in tile_state:
                    continue
                previous = prev_tiles[y][x]
                if public_asset(previous):
                    continue
                events.append((i, prev_obs, obs, x, y, previous, tile_state))
    return events


def find_opponent_ongoing_yield_events(history, seat):
    opponent = 1 - seat
    events = []
    for i in range(1, len(history["steps"])):
        prev_obs = observation(history, i - 1, seat)
        obs = observation(history, i, seat)
        prev_tiles = prev_obs["farms"][opponent]["tiles"]
        tiles = obs["farms"][opponent]["tiles"]
        for y, row in enumerate(tiles):
            for x, tile_state in enumerate(row):
                previous = prev_tiles[y][x]
                if not isinstance(tile_state, dict) or not isinstance(previous, dict):
                    continue
                crop = tile_state.get("crop")
                if crop not in ("TOMATO", "STRAWBERRY"):
                    continue
                if previous.get("crop") != crop:
                    continue
                if previous.get("planted_day") != tile_state.get("planted_day"):
                    continue
                before = previous.get("yield_units", 0)
                after = tile_state.get("yield_units", 0)
                if before == after:
                    continue
                events.append((i, prev_obs, obs, x, y, before, after, tile_state))
    return events

class ForecastOneHistoryEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = load_agent()
        cls.history = load_history()
        cls.seat = losing_seat(cls.history)

    def test_fixture_clock_and_model_spec(self):
        """The fixed replay and independent test specification must remain coherent."""
        agent = self.agent
        self.assertEqual(agent.TURNS_PER_DAY, SPEC_TURNS_PER_DAY)
        self.assertEqual(agent.SEASON_TURNS, SPEC_SEASON_TURNS)
        self.assertEqual(agent.SHOP_INTERVAL, SPEC_SHOP_INTERVAL)
        self.assertEqual(agent.SHOP_UNLOCK_TURNS, SPEC_SHOP_UNLOCK_TURNS)
        self.assertEqual(tuple(agent.SHED), SPEC_SHED)
        self.assertEqual(agent.SHOPS, SPEC_SHOPS)
        self.assertEqual(
            {
                "product": agent.ANIMALS["COW"][1],
                "first_day": agent.ANIMALS["COW"][2],
                "interval_days": agent.ANIMALS["COW"][3],
                "units": agent.ANIMALS["COW"][4],
            },
            SPEC_COW,
        )
        for i in range(len(self.history["steps"])):
            obs = observation(self.history, i, self.seat)
            self.assertEqual(
                absolute_turn(agent, obs),
                i,
                f"history step {i} does not match observation day/hour",
            )

    def test_shop_open_events_change_demand_exactly_until_next_unlock(self):
        """A newly opened shop must immediately add its known product demand."""
        agent = self.agent
        events = find_shop_events(self.history, self.seat)
        self.assertEqual(
            len(events), EXPECTED_EVENT_COUNTS["shop_open"],
            "shop event extractor changed for the fixed history fixture",
        )

        print(f"\n[history={HISTORY_PATH.name}] seat={self.seat} shop events={len(events)}")
        for i, prev_obs, obs, added in events:
            step = absolute_turn(agent, obs)
            counterfactual = copy.deepcopy(obs)
            counterfactual["town"]["unlocked_shops"] = copy.deepcopy(
                prev_obs["town"]["unlocked_shops"]
            )

            projected_actual, flow_actual = run_forecast(agent, obs)
            projected_before, flow_before = run_forecast(agent, counterfactual)
            delta_flow = subtract_series(flow_actual, flow_before)
            delta_projected = subtract_series(projected_actual, projected_before)

            added_demand = {c: 0.0 for c in obs["market"]["inventory"]}
            for shop in added:
                for product, units in SPEC_SHOPS[shop].items():
                    if product in added_demand:
                        added_demand[product] += units

            # Before the NEXT unlock, future-shop uncertainty is identical in both
            # forecasts. The only difference must therefore be exact demand from the
            # shop(s) that just became known.
            next_unlock = min(
                SPEC_SEASON_TURNS,
                ((step // SPEC_SHOP_UNLOCK_TURNS) + 1) * SPEC_SHOP_UNLOCK_TURNS,
            )
            for t in range(step, next_unlock):
                shop_tick = t % SPEC_SHOP_INTERVAL == 0
                for product in delta_flow:
                    expected = -added_demand.get(product, 0.0) if shop_tick else 0.0
                    assert_close(
                        self,
                        delta_flow[product][t],
                        expected,
                        f"SHOP_OPEN step={step} t={t} product={product}",
                    )

            demand_per_tick = sum(added_demand.values())
            assert_projection_delta_consistent(
                self, delta_flow, delta_projected, step
            )
            ticks = sum(
                1 for t in range(step, next_unlock)
                if t % SPEC_SHOP_INTERVAL == 0
            )
            affected = {c: n for c, n in added_demand.items() if n}
            projected_changes = {
                c: nonzero_points(delta_projected[c])
                for c in affected
                if nonzero_points(delta_projected[c])
            }
            print(
                f"  SHOP_OPEN history_step={i} turn={step} "
                f"day={obs['day']} hour={obs['hour']} added={added} "
                f"demand/tick=+{demand_per_tick:g} by_product={affected} "
                f"ticks_before_next_unlock={ticks}"
            )
            for product, points in projected_changes.items():
                print(f"    projected_delta[{product}]={points[:6]}")

    def test_new_opponent_cow_changes_milk_supply_and_wheat_demand(self):
        """A clean opponent COW placement exposes both future MILK supply and +1 herd demand."""
        agent = self.agent
        events = find_clean_opponent_animal_events(
            self.history, self.seat, animal="COW"
        )
        self.assertEqual(
            len(events), EXPECTED_EVENT_COUNTS["clean_opponent_cow"],
            "COW event extractor changed for the fixed history fixture",
        )

        opponent = 1 - self.seat
        print(f"\n[history={HISTORY_PATH.name}] opponent={opponent} clean COW events={len(events)}")

        for i, prev_obs, obs, x, y, previous, cow_state in events:
            step = absolute_turn(agent, obs)
            counterfactual = copy.deepcopy(obs)
            counter_tile = copy.deepcopy(cow_state)
            counter_tile.pop("animal", None)
            counterfactual["farms"][opponent]["tiles"][y][x] = counter_tile

            projected_actual, flow_actual = run_forecast(agent, obs)
            projected_before, flow_before = run_forecast(agent, counterfactual)
            delta_flow = subtract_series(flow_actual, flow_before)
            delta_projected = subtract_series(projected_actual, projected_before)

            expected = {
                product: [0.0] * SPEC_SEASON_TURNS
                for product in flow_actual
            }

            # One additional visible animal adds exactly one unit to the simple daily
            # herd-WHEAT forecast. The +3 reserve is per player, so it does not change
            # when the herd increases from N to N+1.
            if "WHEAT" in expected:
                first_feed = ((step // SPEC_TURNS_PER_DAY) + 1) * SPEC_TURNS_PER_DAY
                for t in range(first_feed, SPEC_SEASON_TURNS, SPEC_TURNS_PER_DAY):
                    expected["WHEAT"][t] -= 1.0

            animal_cfg = SPEC_COW
            product = animal_cfg["product"]
            held = cow_state.get("yield_units", 0)
            p = (x, y)
            opp_farm = obs["farms"][opponent]
            positions = [tuple(opp_farm["farmer"])] + [
                tuple(pos) for pos in opp_farm["hands"]
            ]

            # Visible yield already on the new animal tile uses worker approach + harvest
            # + efficient transport, exactly as forecast() currently models it.
            if held:
                approach = min(manhattan(pos, p) for pos in positions) if positions else 0
                at = step + approach + shed_distance(p) + 1
                if at < SPEC_SEASON_TURNS:
                    expected[product][at] += held

            # Future well-cared cow production: +3 MILK per event, then efficient
            # harvest/transport to the shed/market.
            first_day = cow_state["placed_day"] + animal_cfg["first_day"]
            for at_day in range(max(int(obs["day"]) + 1, first_day), 30):
                if (at_day - first_day) % animal_cfg["interval_days"]:
                    continue
                ready_turn = at_day * SPEC_TURNS_PER_DAY
                at = ready_turn + shed_distance(p) + 1
                if at < SPEC_SEASON_TURNS:
                    expected[product][at] += animal_cfg["units"]

            for product_name in delta_flow:
                for t, (actual_value, expected_value) in enumerate(
                    zip(delta_flow[product_name], expected[product_name])
                ):
                    assert_close(
                        self,
                        actual_value,
                        expected_value,
                        f"OPP_COW step={step} t={t} product={product_name}",
                    )

            assert_projection_delta_consistent(
                self, delta_flow, delta_projected, step
            )
            milk_points = nonzero_points(delta_flow[product])
            wheat_points = nonzero_points(delta_flow.get("WHEAT", []))
            projected_milk = nonzero_points(delta_projected[product])
            print(
                f"  OPP_COW_APPEAR history_step={i} turn={step} "
                f"day={obs['day']} hour={obs['hour']} tile=({x},{y}) "
                f"placed_day={cow_state['placed_day']}"
            )
            print(f"    net_flow_delta[{product}]={milk_points[:10]}")
            print(f"    net_flow_delta[WHEAT]={wheat_points[:10]}")
            print(f"    projected_delta[{product}]={projected_milk[:8]}")


    def test_new_opponent_crop_changes_only_that_crop_future_supply(self):
        """A clean opponent crop placement adds exactly its modeled public future supply."""
        agent = self.agent
        events = find_clean_opponent_crop_events(self.history, self.seat)
        self.assertEqual(
            len(events), EXPECTED_EVENT_COUNTS["clean_opponent_crop"],
            "crop event extractor changed for the fixed history fixture",
        )

        opponent = 1 - self.seat
        print(f"\n[history={HISTORY_PATH.name}] opponent={opponent} clean crop events={len(events)}")

        for i, prev_obs, obs, x, y, previous, crop_state in events:
            step = absolute_turn(agent, obs)
            counterfactual = copy.deepcopy(obs)
            counter_tile = copy.deepcopy(crop_state)
            counter_tile.pop("crop", None)
            counterfactual["farms"][opponent]["tiles"][y][x] = counter_tile

            projected_actual, flow_actual = run_forecast(agent, obs)
            projected_before, flow_before = run_forecast(agent, counterfactual)
            delta_flow = subtract_series(flow_actual, flow_before)
            delta_projected = subtract_series(projected_actual, projected_before)

            expected = {
                product: [0.0] * SPEC_SEASON_TURNS
                for product in flow_actual
            }

            crop = crop_state["crop"]
            planted = crop_state["planted_day"]
            crop_spec = SPEC_CROPS[crop]
            ongoing = crop_spec["ongoing"]
            held = crop_state.get("yield_units", 0)
            p = (x, y)
            opp_farm = obs["farms"][opponent]
            positions = [tuple(opp_farm["farmer"])] + [
                tuple(pos) for pos in opp_farm["hands"]
            ]

            if held:
                approach = min(manhattan(pos, p) for pos in positions) if positions else 0
                at = step + approach + shed_distance(p) + 1
                if at < SPEC_SEASON_TURNS:
                    expected[crop][at] += held

            for age, scheduled_units in crop_spec["events"]:
                at_day = planted + age
                ready_turn = at_day * SPEC_TURNS_PER_DAY
                if ready_turn <= step or at_day >= 30:
                    continue
                if ongoing:
                    units = 2 if crop_state.get("fertilized_until_day", -1) >= at_day else 1
                else:
                    units = max(0, scheduled_units - held)
                if not units:
                    continue
                at = ready_turn + shed_distance(p) + 1
                if at < SPEC_SEASON_TURNS:
                    expected[crop][at] += units

            for product_name in delta_flow:
                for t, (actual_value, expected_value) in enumerate(
                    zip(delta_flow[product_name], expected[product_name])
                ):
                    assert_close(
                        self,
                        actual_value,
                        expected_value,
                        f"OPP_CROP step={step} t={t} product={product_name}",
                    )

            assert_projection_delta_consistent(
                self, delta_flow, delta_projected, step
            )
            flow_points = nonzero_points(delta_flow[crop])
            assert_projection_delta_consistent(
                self, delta_flow, delta_projected, step
            )
            projected_points = nonzero_points(delta_projected[crop])
            print(
                f"  OPP_CROP_APPEAR history_step={i} turn={step} "
                f"day={obs['day']} hour={obs['hour']} tile=({x},{y}) "
                f"crop={crop} planted_day={planted}"
            )
            print(f"    net_flow_delta[{crop}]={flow_points[:12]}")
            print(f"    projected_delta[{crop}]={projected_points[:8]}")

    def test_opponent_ongoing_yield_change_moves_visible_supply_only(self):
        """TOMATO/STRAWBERRY yield changes affect only visible held-supply ETA."""
        agent = self.agent
        events = find_opponent_ongoing_yield_events(self.history, self.seat)
        self.assertEqual(
            len(events), EXPECTED_EVENT_COUNTS["opponent_ongoing_yield"],
            "yield event extractor changed for the fixed history fixture",
        )

        opponent = 1 - self.seat
        print(
            f"\n[history={HISTORY_PATH.name}] opponent={opponent} "
            f"ongoing-yield events={len(events)}"
        )

        for i, prev_obs, obs, x, y, before, after, crop_state in events:
            step = absolute_turn(agent, obs)
            counterfactual = copy.deepcopy(obs)
            counterfactual["farms"][opponent]["tiles"][y][x]["yield_units"] = before

            projected_actual, flow_actual = run_forecast(agent, obs)
            projected_before, flow_before = run_forecast(agent, counterfactual)
            delta_flow = subtract_series(flow_actual, flow_before)
            delta_projected = subtract_series(projected_actual, projected_before)

            crop = crop_state["crop"]
            p = (x, y)
            opp_farm = obs["farms"][opponent]
            positions = [tuple(opp_farm["farmer"])] + [
                tuple(pos) for pos in opp_farm["hands"]
            ]
            approach = min(manhattan(pos, p) for pos in positions) if positions else 0
            arrival = step + approach + shed_distance(p) + 1
            delta_units = after - before

            expected = {
                product: [0.0] * SPEC_SEASON_TURNS
                for product in flow_actual
            }
            if arrival < SPEC_SEASON_TURNS:
                expected[crop][arrival] = float(delta_units)

            for product_name in delta_flow:
                for t, (actual_value, expected_value) in enumerate(
                    zip(delta_flow[product_name], expected[product_name])
                ):
                    assert_close(
                        self,
                        actual_value,
                        expected_value,
                        f"OPP_YIELD step={step} t={t} product={product_name}",
                    )

            projected_points = nonzero_points(delta_projected[crop])
            print(
                f"  OPP_YIELD_CHANGE history_step={i} turn={step} "
                f"day={obs['day']} hour={obs['hour']} tile=({x},{y}) "
                f"crop={crop} yield={before}->{after} "
                f"arrival={arrival} flow_delta={delta_units:+g}"
            )
            print(f"    projected_delta[{crop}]={projected_points[:8]}")



if __name__ == "__main__":
    unittest.main()
