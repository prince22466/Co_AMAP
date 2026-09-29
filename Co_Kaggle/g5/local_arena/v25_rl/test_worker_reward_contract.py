"""Reward-contract regression tests for v25 worker RL."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from worker_reward import (
    ANIMAL_PRODUCT_DELIVERED_REWARD,
    ANIMAL_PRODUCT_GENERATED_REWARD,
    ANIMAL_PRODUCT_HARVESTED_REWARD,
    ANIMAL_PRODUCT_VALUE,
    ANIMAL_ESCAPE_PENALTY,
    CRITICAL_FEED_REWARD,
    EFFECTIVE_CARE_REWARD,
    HEALTHY_ANIMAL_DAY_REWARD,
    NORMAL_FEED_REWARD,
    PRODUCT_DELIVERED_REWARD,
    PRODUCT_GENERATED_REWARD,
    PRODUCT_HARVESTED_REWARD,
    PRODUCT_VALUE,
    compute_worker_reward,
)


def _obs(tile, inventory=None, shed=None, day=0, hour=0):
    farm = {
        "farmer": [0, 0],
        "hands": [],
        "tiles": [[tile]],
    }
    return {
        "player": 0,
        "day": day,
        "hour": hour,
        "farms": [farm, {"farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {
            "inventories": [dict(inventory or {})],
            "shed": dict(shed or {}),
            "seeds": {},
        },
    }


EXECUTOR = SimpleNamespace(
    SHED=[(0, 0)],
    SHED_CAPACITY=100,
    CROPS={},
)


class WorkerRewardContractTest(unittest.TestCase):
    def test_crop_lifecycle_remains_eight(self):
        self.assertEqual(
            PRODUCT_GENERATED_REWARD
            + PRODUCT_HARVESTED_REWARD
            + PRODUCT_DELIVERED_REWARD,
            PRODUCT_VALUE,
        )
        self.assertEqual(PRODUCT_VALUE, 8.0)

    def test_animal_lifecycle_is_sixteen_times_crop_value(self):
        self.assertEqual(
            ANIMAL_PRODUCT_GENERATED_REWARD
            + ANIMAL_PRODUCT_HARVESTED_REWARD
            + ANIMAL_PRODUCT_DELIVERED_REWARD,
            ANIMAL_PRODUCT_VALUE,
        )
        self.assertEqual(ANIMAL_PRODUCT_VALUE, 128.0)
        self.assertEqual(ANIMAL_PRODUCT_VALUE, 16.0 * PRODUCT_VALUE)


    def test_actual_generation_uses_animal_premium(self):
        crop_before = _obs({"kind": "PLANT", "crop": "WHEAT", "yield_units": 0})
        crop_after = _obs({"kind": "PLANT", "crop": "WHEAT", "yield_units": 2})
        crop = compute_worker_reward(
            EXECUTOR,
            crop_before,
            {"farmer": ["PASS"], "hands": []},
            crop_after,
        )
        self.assertEqual(crop.reward, 2 * PRODUCT_GENERATED_REWARD)

        animal_before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
        })
        animal_after = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 2,
        })
        animal = compute_worker_reward(
            EXECUTOR,
            animal_before,
            {"farmer": ["PASS"], "hands": []},
            animal_after,
        )
        self.assertEqual(
            animal.reward,
            2 * ANIMAL_PRODUCT_GENERATED_REWARD,
        )
        self.assertEqual(animal.animal_product_units_generated_total, 2)

    def test_actual_harvest_uses_animal_premium(self):
        before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 3,
        })
        after = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
        })
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["HARVEST"], "hands": []},
            after,
        )
        self.assertEqual(
            result.reward,
            3 * ANIMAL_PRODUCT_HARVESTED_REWARD,
        )
        self.assertEqual(result.animal_product_units_harvested_total, 3)

    def test_actual_delivery_uses_animal_premium(self):
        before = _obs(None, inventory={"MILK": 3})
        after = _obs(None, inventory={})
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["PLACE", "MILK", 3], "hands": []},
            after,
        )
        self.assertEqual(
            result.reward,
            3 * ANIMAL_PRODUCT_DELIVERED_REWARD,
        )
        self.assertEqual(result.product_units_moved_to_shed_total, 3)

    def test_normal_feed_is_more_valuable_than_emergency_rescue(self):
        self.assertGreater(NORMAL_FEED_REWARD, CRITICAL_FEED_REWARD)

        normal_before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": False,
            "consecutive_unfed": 0,
        })
        fed_after = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": True,
            "consecutive_unfed": 0,
        })
        normal = compute_worker_reward(
            EXECUTOR,
            normal_before,
            {"farmer": ["FEED"], "hands": []},
            fed_after,
        )
        self.assertEqual(normal.normal_feed, 1)
        self.assertEqual(normal.reward, NORMAL_FEED_REWARD)

        critical_before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": False,
            "consecutive_unfed": 1,
        })
        critical = compute_worker_reward(
            EXECUTOR,
            critical_before,
            {"farmer": ["FEED"], "hands": []},
            fed_after,
        )
        self.assertEqual(critical.critical_feed, 1)
        self.assertEqual(critical.reward, CRITICAL_FEED_REWARD)
        self.assertGreater(normal.reward, critical.reward)

    def test_fed_animal_surviving_day_rollover_gets_dense_credit(self):
        before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": True,
            "consecutive_unfed": 0,
        }, day=0, hour=23)
        after = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": False,
            "consecutive_unfed": 0,
        }, day=1, hour=0)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["PASS"], "hands": []},
            after,
        )
        self.assertEqual(result.healthy_animal_days, 1)
        self.assertEqual(result.reward, HEALTHY_ANIMAL_DAY_REWARD)

    def test_escape_penalty_is_stronger(self):
        before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
        })
        after = _obs(None)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["PASS"], "hands": []},
            after,
        )
        self.assertEqual(result.animals_escaped, 1)
        self.assertEqual(result.reward, ANIMAL_ESCAPE_PENALTY)
        self.assertEqual(ANIMAL_ESCAPE_PENALTY, -100.0)

    def test_care_reward_is_strengthened(self):
        self.assertEqual(EFFECTIVE_CARE_REWARD, 3.0)

    def test_each_animal_product_stage_exceeds_crop_stage(self):
        self.assertGreater(
            ANIMAL_PRODUCT_GENERATED_REWARD,
            PRODUCT_GENERATED_REWARD,
        )
        self.assertGreater(
            ANIMAL_PRODUCT_HARVESTED_REWARD,
            PRODUCT_HARVESTED_REWARD,
        )
        self.assertGreater(
            ANIMAL_PRODUCT_DELIVERED_REWARD,
            PRODUCT_DELIVERED_REWARD,
        )


if __name__ == "__main__":
    unittest.main()
