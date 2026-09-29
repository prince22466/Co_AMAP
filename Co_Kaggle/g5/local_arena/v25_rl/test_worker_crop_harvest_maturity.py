"""Regression tests for crop harvest maturity gating."""
from __future__ import annotations

import unittest

from worker_policy import WorkerPolicy


class _Executor:
    CROPS = {
        "WHEAT": (10, 25, ((4, 4),), 4),
        "CARROT": (20, 35, ((3, 3),), 3),
        "TOMATO": (50, 60, ((8, 2), (9, 2), (10, 2), (11, 2)), 11),
        "STRAWBERRY": (100, 120, ((10, 2), (12, 2), (14, 2), (16, 2)), 16),
        "MELON": (80, 250, ((10, 6),), 10),
    }

    @staticmethod
    def tile(farm, pos):
        x, y = pos
        return farm["tiles"][y][x]


def _obs(day):
    return {
        "player": 0,
        "day": day,
        "hour": 8,
        "farms": [{
            "farmer": [0, 0],
            "hands": [],
            "tiles": [[{
                "kind": "PLANT",
                "crop": "WHEAT",
                "planted_day": 0,
                "yield_units": 4,
                "watered_today": True,
                "fertilized_until_day": day,
            }]],
        }],
        "private": {
            "inventories": [{}],
            "shed": {},
            "seeds": {},
        },
    }


class CropHarvestMaturityTest(unittest.TestCase):
    def setUp(self):
        self.policy = WorkerPolicy(
            _Executor(),
            model=None,
            device=None,
            deterministic=True,
            collect=False,
        )

    def test_immature_positive_yield_does_not_create_harvest_task(self):
        tasks = self.policy.tasks(_obs(day=2), {}, {(0, 0): "WHEAT"})
        self.assertFalse(any(t.op == "HARVEST" for t in tasks))

    def test_first_yield_day_allows_harvest_task(self):
        tasks = self.policy.tasks(_obs(day=4), {}, {(0, 0): "WHEAT"})
        self.assertTrue(any(t.op == "HARVEST" for t in tasks))


if __name__ == "__main__":
    unittest.main()
