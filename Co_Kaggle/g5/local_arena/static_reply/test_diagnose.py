"""Regression checks for diagnostic semantics, independent of the large replay run."""
import copy
import importlib.util
from pathlib import Path
import unittest

import diagnose
from reply_template import load_agent


def observation():
    farm = dict(money=3000, tiles=[[None]*10 for _ in range(10)], farmer=[4, 4],
                hands=[], hires_today=0, unlocked_quadrants=["NW", "NE", "SW", "SE"])
    return dict(player=1, day=3, hour=0, step=72, farms=[copy.deepcopy(farm), copy.deepcopy(farm)],
        private=dict(shed={}, seeds={}, inventories=[{}]), town=dict(unlocked_shops=["YARN_STORE"]),
        market=dict(prices={p: 100 for p in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "FERTILIZER")}))


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.context = load_agent(Path(__file__).with_name("v20_3.py"))
        agent, _ = self.context.__enter__()
        self.g = agent.__globals__

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_duplicate_shop_is_an_additional_demand_instance(self):
        previous = observation()
        obs = copy.deepcopy(previous)
        obs["town"]["unlocked_shops"].append("YARN_STORE")
        signals = self.g["production_signals"](obs)
        action = dict(farmer=["PASS"], hands=[], market=self.g["market_orders"](obs, signals, [["PASS"]]))
        events = diagnose.event_probes(self.g, obs, previous, action, signals)
        self.assertEqual(len(events), 1)
        change = events[0]["changes"][0]
        self.assertEqual(change["product"], "WOOL")
        self.assertEqual(change["demand_delta"], 324)
        self.assertGreater(change["gap_after"], change["gap_before"])
        self.assertEqual(obs["town"]["unlocked_shops"], ["YARN_STORE", "YARN_STORE"])

    def test_opponent_new_animal_reduces_product_gap_and_adds_feed_demand(self):
        previous = observation()
        obs = copy.deepcopy(previous)
        obs["farms"][0]["tiles"][4][4] = dict(kind="PASTURE", animal="SHEEP", placed_day=3, yield_units=0, fed_today=False)
        signals = self.g["production_signals"](obs)
        action = dict(farmer=["PASS"], hands=[], market=self.g["market_orders"](obs, signals, [["PASS"]]))
        event = diagnose.event_probes(self.g, obs, previous, action, signals)[0]
        changes = {r["product"]: r for r in event["changes"]}
        self.assertEqual(changes["WOOL"]["capacity_delta"], 28)
        self.assertEqual(changes["WOOL"]["gap_before"]-changes["WOOL"]["gap_after"], 1)
        self.assertEqual(changes["WHEAT"]["demand_delta"], 27)
        self.assertIn("animal", obs["farms"][0]["tiles"][4][4])

    def test_gate_explains_fractional_shortage_without_calling_it_no_demand(self):
        obs = observation()
        signals = [dict(product="WOOL", producer="SHEEP", score=.3, actionable=True,
                        hard_demand=10, capacity=2, producer_equivalent_gap=.5)]
        row = diagnose.gate_rows(self.g, obs, signals, {"market": []})[0]
        self.assertIn("gap_below_one", row["reasons"])
        self.assertNotIn("cash_reserve", row["reasons"])
        self.assertEqual(row["needed"], 0)

    def test_market_orders_buy_wheat_when_herd_reserve_is_short(self):
        obs = observation()
        obs["day"] = 3
        obs["hour"] = 0
        obs["private"]["shed"]["WHEAT"] = 3
        for x in range(5):
            obs["farms"][obs["player"]]["tiles"][0][x] = dict(
                kind="PASTURE", animal="COW", placed_day=0, yield_units=0,
                fed_today=False, cared_today=False,
            )
        orders = self.g["market_orders"](obs, [], [["PASS"]])
        wheat_buys = [o for o in orders if o[:2] == ["BUY_PRODUCT", "WHEAT"]]
        self.assertEqual(wheat_buys, [["BUY_PRODUCT", "WHEAT", 5]])
        first_non_sell = next(o for o in orders if o[0] != "SELL")
        self.assertEqual(first_non_sell, ["BUY_PRODUCT", "WHEAT", 5])
        self.assertFalse(any(o[:2] == ["SELL", "WHEAT"] for o in orders))

    def test_market_orders_sell_only_wheat_above_herd_reserve(self):
        obs = observation()
        obs["day"] = 3
        obs["hour"] = 5
        obs["private"]["shed"]["WHEAT"] = 10
        for x in range(5):
            obs["farms"][obs["player"]]["tiles"][0][x] = dict(
                kind="PASTURE", animal="COW", placed_day=0, yield_units=0,
                fed_today=False, cared_today=False,
            )
        orders = self.g["market_orders"](obs, [], [["PASS"]])
        self.assertIn(["SELL", "WHEAT", 2], orders)
        self.assertFalse(any(o[:2] == ["BUY_PRODUCT", "WHEAT"] for o in orders))

    def test_market_orders_reserve_counts_same_turn_animal_placement(self):
        obs = observation()
        obs["day"] = 3
        obs["hour"] = 5
        obs["private"]["shed"]["WHEAT"] = 10
        obs["private"]["inventories"][0]["COW"] = 1
        obs["farms"][obs["player"]]["farmer"] = [4, 3]
        obs["farms"][obs["player"]]["tiles"][3][4] = dict(kind="PASTURE")
        for x in range(3):
            obs["farms"][obs["player"]]["tiles"][0][x] = dict(
                kind="PASTURE", animal="COW", placed_day=0, yield_units=0,
                fed_today=False, cared_today=False,
            )
        orders = self.g["market_orders"](obs, [], [["PLACE", "COW"]])
        # Pre-action herd=3 would reserve only 4; post-action herd=4 must reserve 8.
        self.assertIn(["SELL", "WHEAT", 2], orders)
        self.assertNotIn(["SELL", "WHEAT", 6], orders)

    def test_json_nonfinite_gap_is_explicit_null(self):
        self.assertEqual(diagnose.clean({"gap": float("inf"), "score": 1}), {"gap": None, "score": 1})

    def test_same_species_replant_is_capacity_event(self):
        self.assertNotEqual(diagnose.identity(dict(crop="WHEAT", planted_day=0)),
                            diagnose.identity(dict(crop="WHEAT", planted_day=4)))


if __name__ == "__main__":
    unittest.main()
