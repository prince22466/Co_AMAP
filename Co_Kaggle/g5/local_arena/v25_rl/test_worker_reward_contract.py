"""Reward-contract regression tests for v25 worker RL."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from worker_reward import (
    ANIMAL_PRODUCT_DELIVERED_REWARD,
    ANIMAL_PRODUCT_GENERATED_REWARD,
    ANIMAL_PRODUCT_HARVESTED_REWARD,
    ANIMAL_PRODUCT_VALUE,
    PRODUCT_DELIVERED_REWARD,
    PRODUCT_GENERATED_REWARD,
    PRODUCT_HARVESTED_REWARD,
    PRODUCT_VALUE,
    compute_worker_reward,
)


def _obs(tile, inventory=None, shed=None):
    farm = {
        "farmer": [0, 0],
        "hands": [],
        "tiles": [[tile]],
    }
    return {
        "player": 0,
        "day": 0,
        "hour": 0,
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

    def test_animal_lifecycle_is_double_crop_value(self):
        self.assertEqual(
            ANIMAL_PRODUCT_GENERATED_REWARD
            + ANIMAL_PRODUCT_HARVESTED_REWARD
            + ANIMAL_PRODUCT_DELIVERED_REWARD,
            ANIMAL_PRODUCT_VALUE,
        )
        self.assertEqual(ANIMAL_PRODUCT_VALUE, 16.0)
        self.assertEqual(ANIMAL_PRODUCT_VALUE, 2.0 * PRODUCT_VALUE)


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
