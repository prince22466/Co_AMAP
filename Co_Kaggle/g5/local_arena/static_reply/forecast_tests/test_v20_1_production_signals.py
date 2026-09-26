#!/usr/bin/env python3
"""Behavior tests for v20_1.production_signals()."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import unittest
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
STATIC_REPLY=HERE.parent
G5_ROOT=HERE.parents[2]
AGENT_PATH=STATIC_REPLY/"v20_1.py"
HISTORY_DIR=G5_ROOT/"game_history"/"v20"
HISTORY_NAMES=(
    "111548564.json",
    "111549675.json",
    "111551968.json",
    "111553265.json",
    "111587965.json",
)
EPS=1e-9
TURNS_PER_DAY=24
SEASON_TURNS=720
SHOP_INTERVAL=4
CENTER_INTERVAL=24

SHOP_SPEC={
    "BAKERY":{"WHEAT":1,"EGG":1},
    "PIZZA_SHOP":{"MILK":1,"TOMATO":1,"WHEAT":1},
    "BRUNCH_SPOT":{"EGG":1,"WHEAT":1,"STRAWBERRY":1},
    "YARN_STORE":{"WOOL":2},
    "ICE_CREAM_SHOP":{"STRAWBERRY":1,"MILK":1,"WHEAT":1},
    "PET_CAFE":{"CARROT":2},
    "SMOOTHIE_SHOP":{"STRAWBERRY":1,"MILK":1},
    "FARMERS_MARKET":{"WHEAT":1,"CARROT":1,"TOMATO":1,"STRAWBERRY":1},
}
ANIMAL_PRODUCT={"COW":"MILK","SHEEP":"WOOL","GOOSE":"EGG"}
CROPS=("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON")
PRODUCTS=CROPS+("MILK","WOOL","EGG")


def load_agent():
    spec=importlib.util.spec_from_file_location("_signal_test_v20_1",AGENT_PATH)
    if spec is None or spec.loader is None:raise RuntimeError(f"cannot import {AGENT_PATH}")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_history(name):
    return json.loads((HISTORY_DIR/name).read_text(encoding="utf-8"))


def losing_seat(history):
    final=history["steps"][-1]
    rewards=[float(final[i]["reward"]) for i in (0,1)]
    if rewards[0]==rewards[1]:raise AssertionError("tied replay")
    return 0 if rewards[0]<rewards[1] else 1


def observation(history,step,seat):
    return copy.deepcopy(history["steps"][step][seat]["observation"])


def by_product(signals):
    return {entry["product"]:entry for entry in signals}


def assert_close(testcase,a,b,msg=""):
    testcase.assertTrue(math.isclose(a,b,abs_tol=EPS,rel_tol=0.0),f"{msg}: {a} != {b}")


def empty_public_tile(obs,player):
    for y,row in enumerate(obs["farms"][player]["tiles"]):
        for x,tile in enumerate(row):
            if isinstance(tile,dict) and "crop" not in tile and "animal" not in tile:
                return x,y,tile
    raise AssertionError("no public non-asset tile available")


class ProductionSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent=load_agent()
        cls.cases=[]
        for name in HISTORY_NAMES:
            history=load_history(name)
            cls.cases.append((name,history,losing_seat(history)))

    def test_signal_contract_formula_and_sorting(self):
        """Scores are [0,1], sorted, and exactly implement 1-exp(-producer_gap)."""
        for name,history,seat in self.cases:
            for step in (0,72,240,480,719):
                signals=self.agent.production_signals(observation(history,step,seat))
                self.assertEqual({x["product"] for x in signals},set(PRODUCTS),name)
                self.assertEqual(
                    [x["score"] for x in signals],
                    sorted((x["score"] for x in signals),reverse=True),
                    name,
                )
                for x in signals:
                    self.assertGreaterEqual(x["score"],0.0)
                    self.assertLessEqual(x["score"],1.0)
                    assert_close(
                        self,x["capacity"],sum(x["capacity_breakdown"].values()),
                        f"{name} {step} {x['product']} capacity",
                    )
                    assert_close(
                        self,x["gap"],max(0.0,x["hard_demand"]-x["capacity"]),
                        f"{name} {step} {x['product']} gap",
                    )
                    if x["gap"]==0:
                        assert_close(self,x["score"],0.0)
                        assert_close(self,x["producer_equivalent_gap"],0.0)
                    elif x["new_producer_capacity"]>0:
                        expected_n=x["gap"]/x["new_producer_capacity"]
                        assert_close(self,x["producer_equivalent_gap"],expected_n)
                        assert_close(self,x["score"],1-math.exp(-expected_n))
                        self.assertTrue(x["actionable"])
                    else:
                        self.assertTrue(math.isinf(x["producer_equivalent_gap"]))
                        assert_close(self,x["score"],1.0)
                        self.assertFalse(x["actionable"])

    def test_hard_demand_is_only_town_plus_currently_open_shops(self):
        for name,history,seat in self.cases:
            for step in (0,73,301,619):
                obs=observation(history,step,seat)
                signals=by_product(self.agent.production_signals(obs))
                town_ticks=sum(1 for t in range(step,SEASON_TURNS) if t%CENTER_INTERVAL==0)
                shop_ticks=sum(1 for t in range(step,SEASON_TURNS) if t%SHOP_INTERVAL==0)
                per_tick={p:0 for p in PRODUCTS}
                for shop in obs["town"]["unlocked_shops"]:
                    for p,n in SHOP_SPEC[shop].items():per_tick[p]+=n
                for p in PRODUCTS:
                    expected=town_ticks+shop_ticks*per_tick[p]
                    assert_close(self,signals[p]["hard_demand"],expected,f"{name} {step} {p}")

    def test_shop_open_increases_only_hard_demand_side_of_signal(self):
        """A new shop raises affected need; it never changes production capacity."""
        strict_score_increases=0
        for name,history,seat in self.cases:
            for step in range(1,SEASON_TURNS):
                prev=observation(history,step-1,seat)
                obs=observation(history,step,seat)
                before=Counter(prev["town"]["unlocked_shops"])
                after=Counter(obs["town"]["unlocked_shops"])
                added=list((after-before).elements())
                if not added:continue

                counter=copy.deepcopy(obs)
                counter["town"]["unlocked_shops"]=list(prev["town"]["unlocked_shops"])
                actual=by_product(self.agent.production_signals(obs))
                baseline=by_product(self.agent.production_signals(counter))
                remaining_ticks=sum(1 for t in range(step,SEASON_TURNS) if t%SHOP_INTERVAL==0)
                added_per_tick={p:0 for p in PRODUCTS}
                for shop in added:
                    for p,n in SHOP_SPEC[shop].items():added_per_tick[p]+=n

                for p in PRODUCTS:
                    assert_close(self,actual[p]["capacity"],baseline[p]["capacity"],f"{name} {step} {p}")
                    assert_close(
                        self,
                        actual[p]["hard_demand"]-baseline[p]["hard_demand"],
                        remaining_ticks*added_per_tick[p],
                        f"{name} {step} {p} demand delta",
                    )
                    self.assertGreaterEqual(actual[p]["gap"]+EPS,baseline[p]["gap"])
                    self.assertGreaterEqual(actual[p]["score"]+EPS,baseline[p]["score"])
                    if actual[p]["score"]>baseline[p]["score"]+EPS:strict_score_increases+=1
        self.assertGreater(strict_score_increases,0)

    def test_clean_opponent_animal_appearance_increases_product_capacity(self):
        seen=Counter()
        strict_score_drops=0
        for name,history,seat in self.cases:
            opponent=1-seat
            for step in range(1,SEASON_TURNS):
                prev=observation(history,step-1,seat)
                obs=observation(history,step,seat)
                prev_tiles=prev["farms"][opponent]["tiles"]
                for y,row in enumerate(obs["farms"][opponent]["tiles"]):
                    for x,tile in enumerate(row):
                        if not isinstance(tile,dict) or "animal" not in tile:continue
                        old=prev_tiles[y][x]
                        if isinstance(old,dict) and ("animal" in old or "crop" in old):continue
                        animal=tile["animal"];product=ANIMAL_PRODUCT[animal];seen[animal]+=1
                        counter=copy.deepcopy(obs)
                        counter["farms"][opponent]["tiles"][y][x].pop("animal",None)
                        actual=by_product(self.agent.production_signals(obs))
                        baseline=by_product(self.agent.production_signals(counter))
                        self.assertGreater(actual[product]["capacity"],baseline[product]["capacity"])
                        assert_close(self,actual[product]["hard_demand"],baseline[product]["hard_demand"])
                        self.assertLessEqual(actual[product]["score"],baseline[product]["score"]+EPS)
                        if actual[product]["score"]+EPS<baseline[product]["score"]:strict_score_drops+=1
        self.assertEqual(set(seen),set(ANIMAL_PRODUCT))
        self.assertGreater(strict_score_drops,0)

    def test_clean_opponent_crop_appearance_increases_crop_capacity(self):
        seen=Counter()
        strict_score_drops=0
        for name,history,seat in self.cases:
            opponent=1-seat
            for step in range(1,SEASON_TURNS):
                prev=observation(history,step-1,seat)
                obs=observation(history,step,seat)
                prev_tiles=prev["farms"][opponent]["tiles"]
                for y,row in enumerate(obs["farms"][opponent]["tiles"]):
                    for x,tile in enumerate(row):
                        if not isinstance(tile,dict) or "crop" not in tile:continue
                        old=prev_tiles[y][x]
                        if isinstance(old,dict) and ("animal" in old or "crop" in old):continue
                        crop=tile["crop"];seen[crop]+=1
                        counter=copy.deepcopy(obs)
                        counter["farms"][opponent]["tiles"][y][x].pop("crop",None)
                        actual=by_product(self.agent.production_signals(obs))
                        baseline=by_product(self.agent.production_signals(counter))
                        self.assertGreater(actual[crop]["capacity"],baseline[crop]["capacity"])
                        assert_close(self,actual[crop]["hard_demand"],baseline[crop]["hard_demand"])
                        self.assertLessEqual(actual[crop]["score"],baseline[crop]["score"]+EPS)
                        if actual[crop]["score"]+EPS<baseline[crop]["score"]:strict_score_drops+=1
        self.assertEqual(set(seen),set(CROPS))
        self.assertGreater(strict_score_drops,0)

    def test_seed_to_planted_is_capacity_neutral_at_same_turn(self):
        """A seed assumed planted immediately should not create a score jump when planted."""
        name,history,seat=self.cases[0]
        obs=observation(history,240,seat)  # day 10
        player=obs["player"]
        for crop in CROPS:
            with self.subTest(crop=crop):
                seed_state=copy.deepcopy(obs)
                seed_state["private"]["seeds"][crop]=seed_state["private"]["seeds"].get(crop,0)+1
                planted_state=copy.deepcopy(obs)
                x,y,tile=empty_public_tile(planted_state,player)
                new_tile=copy.deepcopy(tile)
                new_tile.update({"crop":crop,"planted_day":planted_state["day"],"yield_units":0})
                planted_state["farms"][player]["tiles"][y][x]=new_tile

                seed_signal=by_product(self.agent.production_signals(seed_state))[crop]
                planted_signal=by_product(self.agent.production_signals(planted_state))[crop]
                assert_close(self,seed_signal["capacity"],planted_signal["capacity"],crop)
                assert_close(self,seed_signal["score"],planted_signal["score"],crop)

    def test_ready_harvest_to_owned_and_replanted_is_capacity_neutral(self):
        """Ready crop -> owned stock + same crop replanted preserves modeled capacity."""
        name,history,seat=self.cases[0]
        obs=observation(history,240,seat)  # day 10
        player=obs["player"]
        crop="WHEAT"
        x,y,tile=empty_public_tile(obs,player)

        ready=copy.deepcopy(obs)
        ready_tile=copy.deepcopy(tile)
        ready_tile.update({
            "crop":crop,
            "planted_day":ready["day"]-4,
            "yield_units":4,
        })
        ready["farms"][player]["tiles"][y][x]=ready_tile

        moved=copy.deepcopy(obs)
        moved_tile=copy.deepcopy(tile)
        moved_tile.update({
            "crop":crop,
            "planted_day":moved["day"],
            "yield_units":0,
        })
        moved["farms"][player]["tiles"][y][x]=moved_tile
        moved["private"]["shed"][crop]=moved["private"]["shed"].get(crop,0)+4

        a=by_product(self.agent.production_signals(ready))[crop]
        b=by_product(self.agent.production_signals(moved))[crop]
        assert_close(self,a["capacity"],b["capacity"],"ready->owned/replanted capacity")
        assert_close(self,a["score"],b["score"],"ready->owned/replanted score")


if __name__=="__main__":
    unittest.main()
