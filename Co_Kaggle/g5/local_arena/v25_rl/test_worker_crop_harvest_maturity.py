"""Regression tests for engine crop HARVEST legality."""
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
    CROP_FIRST_YIELD_DAY = {
        "WHEAT": 2,
        "CARROT": 2,
        "TOMATO": 8,
        "STRAWBERRY": 10,
        "MELON": 10,
    }

    @staticmethod
    def tile(farm, pos):
        x, y = pos
        return farm["tiles"][y][x]


def _obs(day, crop):
    return {
        "player": 0,
        "day": day,
        "hour": 8,
        "farms": [{
            "farmer": [0, 0],
            "hands": [],
            "tiles": [[{
                "kind": "PLANT",
                "crop": crop,
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

    def _has_harvest(self, day, crop):
        tasks = self.policy.tasks(_obs(day, crop), {}, {(0, 0): crop})
        return any(t.op == "HARVEST" for t in tasks)

    def test_wheat_is_blocked_before_age_two(self):
        self.assertFalse(self._has_harvest(1, "WHEAT"))

    def test_wheat_is_legal_at_age_two(self):
        self.assertTrue(self._has_harvest(2, "WHEAT"))

    def test_carrot_is_legal_at_age_two(self):
        self.assertFalse(self._has_harvest(1, "CARROT"))
        self.assertTrue(self._has_harvest(2, "CARROT"))

    def test_tomato_uses_engine_age_eight_threshold(self):
        self.assertFalse(self._has_harvest(7, "TOMATO"))
        self.assertTrue(self._has_harvest(8, "TOMATO"))


if __name__ == "__main__":
    unittest.main()
