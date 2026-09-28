"""Reward-contract regression tests for v25 worker RL."""
from __future__ import annotations

import unittest

from worker_reward import (
    ANIMAL_PRODUCT_DELIVERED_REWARD,
    ANIMAL_PRODUCT_GENERATED_REWARD,
    ANIMAL_PRODUCT_HARVESTED_REWARD,
    ANIMAL_PRODUCT_VALUE,
    PRODUCT_DELIVERED_REWARD,
    PRODUCT_GENERATED_REWARD,
    PRODUCT_HARVESTED_REWARD,
    PRODUCT_VALUE,
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
