#!/usr/bin/env python3
"""Focused behavior tests for v20_3.crop_plan()."""
from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
AGENT_PATH=HERE.parent/"v20_3.py"


def load_agent():
    spec=importlib.util.spec_from_file_location("_crop_plan_test_v20_3",AGENT_PATH)
    if spec is None or spec.loader is None:raise RuntimeError(f"cannot import {AGENT_PATH}")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def obs_with_tiles_and_seeds(tiles,seeds):
    return {
        "player":0,
        "farms":[{"tiles":tiles}],
        "private":{"seeds":dict(seeds)},
    }


def signals_in_order(*products):
    return [{"product":product,"score":1.0-i/100.0} for i,product in enumerate(products)]


class CropPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent=load_agent()

    def test_plans_every_owned_seed_when_all_fit(self):
        tiles=[[None for _ in range(10)] for _ in range(10)]
        obs=obs_with_tiles_and_seeds(
            tiles,
            {"WHEAT":2,"CARROT":1,"STRAWBERRY":3,"MELON":1},
        )
        plan=self.agent.crop_plan(
            obs,
            signals_in_order("MELON","CARROT","WHEAT","STRAWBERRY","TOMATO"),
        )

        self.assertEqual(len(plan),7)
        self.assertEqual(
            Counter(plan.values()),
            Counter({"WHEAT":2,"CARROT":1,"STRAWBERRY":3,"MELON":1}),
        )
        self.assertTrue(all(pos not in self.agent.ANIMAL_POINTS for pos in plan))

    def test_signal_ranking_decides_when_seeds_exceed_empty_tiles(self):
        tiles=[["LOCKED" for _ in range(10)] for _ in range(10)]
        tiles[0][0]=None
        tiles[0][1]=None
        obs=obs_with_tiles_and_seeds(
            tiles,
            {"WHEAT":5,"CARROT":5},
        )
        plan=self.agent.crop_plan(
            obs,
            signals_in_order("CARROT","WHEAT","TOMATO","STRAWBERRY","MELON"),
        )

        self.assertEqual(len(plan),2)
        self.assertEqual(Counter(plan.values()),Counter({"CARROT":2}))

    def test_ranking_never_plants_unowned_seed(self):
        tiles=[["LOCKED" for _ in range(10)] for _ in range(10)]
        tiles[0][0]=None
        obs=obs_with_tiles_and_seeds(
            tiles,
            {"WHEAT":1},
        )
        plan=self.agent.crop_plan(
            obs,
            signals_in_order("STRAWBERRY","CARROT","MELON","WHEAT","TOMATO"),
        )

        self.assertEqual(list(plan.values()),["WHEAT"])


if __name__=="__main__":
    unittest.main()
