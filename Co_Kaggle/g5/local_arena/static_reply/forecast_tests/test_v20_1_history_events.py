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


def absolute_turn(agent, obs):
    return int(obs["day"]) * agent.TURNS_PER_DAY + int(obs["hour"])


def run_forecast(agent, obs):
    # forecast() reads global STEP but does not advance it. Set it from the observation
    # so baseline and counterfactual runs are evaluated at exactly the same game time.
    agent.STEP = absolute_turn(agent, obs)
    projected, net_flow = agent.forecast(copy.deepcopy(obs))
    return projected, net_flow


def subtract_series(actual, baseline):
    return {
        product: [a - b for a, b in zip(actual[product], baseline[product])]
        for product in actual
    }


def assert_close(testcase, actual, expected, msg):
    testcase.assertTrue(
        math.isclose(actual, expected, abs_tol=EPS, rel_tol=0.0),
        f"{msg}: actual={actual}, expected={expected}",
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


class ForecastOneHistoryEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = load_agent()
        cls.history = load_history()
        cls.seat = losing_seat(cls.history)

    def test_shop_open_events_change_demand_exactly_until_next_unlock(self):
        """A newly opened shop must immediately add its known product demand."""
        agent = self.agent
        events = find_shop_events(self.history, self.seat)
        self.assertGreater(len(events), 0, "history contains no shop-open events")

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
                for product, units in agent.SHOPS[shop].items():
                    if product in added_demand:
                        added_demand[product] += units

            # Before the NEXT unlock, future-shop uncertainty is identical in both
            # forecasts. The only difference must therefore be exact demand from the
            # shop(s) that just became known.
            next_unlock = min(
                agent.SEASON_TURNS,
                ((step // agent.SHOP_UNLOCK_TURNS) + 1) * agent.SHOP_UNLOCK_TURNS,
            )
            for t in range(step, next_unlock):
                shop_tick = t % agent.SHOP_INTERVAL == 0
                for product in delta_flow:
                    expected = -added_demand.get(product, 0.0) if shop_tick else 0.0
                    assert_close(
                        self,
                        delta_flow[product][t],
                        expected,
                        f"SHOP_OPEN step={step} t={t} product={product}",
                    )

            demand_per_tick = sum(added_demand.values())
            ticks = sum(1 for t in range(step, next_unlock) if t % agent.SHOP_INTERVAL == 0)
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
        self.assertGreater(
            len(events), 0, "history contains no clean opponent COW placement"
        )

        opponent = 1 - self.seat
        print(f"\n[history={HISTORY_PATH.name}] opponent={opponent} clean COW events={len(events)}")

        for i, prev_obs, obs, x, y, previous, cow_state in events:
            step = absolute_turn(agent, obs)
            counterfactual = copy.deepcopy(obs)
            counterfactual["farms"][opponent]["tiles"][y][x] = copy.deepcopy(previous)

            projected_actual, flow_actual = run_forecast(agent, obs)
            projected_before, flow_before = run_forecast(agent, counterfactual)
            delta_flow = subtract_series(flow_actual, flow_before)
            delta_projected = subtract_series(projected_actual, projected_before)

            expected = {
                product: [0.0] * agent.SEASON_TURNS
                for product in flow_actual
            }

            # One additional visible animal adds exactly one unit to the simple daily
            # herd-WHEAT forecast. The +3 reserve is per player, so it does not change
            # when the herd increases from N to N+1.
            if "WHEAT" in expected:
                first_feed = ((step // agent.TURNS_PER_DAY) + 1) * agent.TURNS_PER_DAY
                for t in range(first_feed, agent.SEASON_TURNS, agent.TURNS_PER_DAY):
                    expected["WHEAT"][t] -= 1.0

            animal_cfg = agent.ANIMALS["COW"]
            product = animal_cfg[1]
            held = cow_state.get("yield_units", 0)
            p = (x, y)
            opp_farm = obs["farms"][opponent]
            positions = [tuple(opp_farm["farmer"])] + [
                tuple(pos) for pos in opp_farm["hands"]
            ]

            # Visible yield already on the new animal tile uses worker approach + harvest
            # + efficient transport, exactly as forecast() currently models it.
            if held:
                approach = min(agent.dist(pos, p) for pos in positions) if positions else 0
                at = step + approach + agent.dist(p, agent.nearest_shed(p)) + 1
                if at < agent.SEASON_TURNS:
                    expected[product][at] += held

            # Future well-cared cow production: +3 MILK per event, then efficient
            # harvest/transport to the shed/market.
            first_day = cow_state["placed_day"] + animal_cfg[2]
            for at_day in range(max(int(obs["day"]) + 1, first_day), 30):
                if (at_day - first_day) % animal_cfg[3]:
                    continue
                ready_turn = at_day * agent.TURNS_PER_DAY
                at = ready_turn + agent.dist(p, agent.nearest_shed(p)) + 1
                if at < agent.SEASON_TURNS:
                    expected[product][at] += animal_cfg[4]

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


if __name__ == "__main__":
    unittest.main()
