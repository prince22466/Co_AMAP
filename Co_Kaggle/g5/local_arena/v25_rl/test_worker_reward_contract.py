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
    AVOIDABLE_PASS_PENALTY,
    CRITICAL_FEED_REWARD,
    CRITICAL_WATER_REWARD,
    CROP_DEATH_PENALTY,
    CROP_TO_WEED_PENALTY,
    EFFECTIVE_CARE_REWARD,
    HEALTHY_ANIMAL_DAY_REWARD,
    LOST_HARVESTABLE_UNIT_PENALTY,
    NORMAL_FEED_REWARD,
    PLANNED_PLACE_ANIMAL_REWARD,
    PLANNED_PLANT_REWARD,
    PRODUCT_DELIVERED_REWARD,
    PRODUCT_GENERATED_REWARD,
    PRODUCT_HARVESTED_REWARD,
    PRODUCT_VALUE,
    ROUTE_PROGRESS_REWARD,
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
    CROPS={
        "WHEAT": (10, 25, ((4, 4),), 4),
    },
    CROP_FIRST_YIELD_DAY={
        "WHEAT": 2,
    },
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




    def test_successful_planned_plant_records_completion_without_turn_plan_bonus(self):
        before = _obs(None, inventory={}, day=0)
        before["private"]["seeds"] = {"WHEAT": 1}
        after = _obs({
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 0,
        }, day=0)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {
                "farmer": ["PLANT", "WHEAT"],
                "hands": [],
                "_plan_credit": [{"op": "PLANT", "item": "WHEAT"}],
            },
            after,
        )
        self.assertEqual(result.planned_plants_completed, 1)
        self.assertEqual(PLANNED_PLANT_REWARD, 0.0)
        self.assertEqual(result.reward, 1.0)

    def test_failed_planned_plant_gets_no_plan_bonus(self):
        before = _obs(None, day=0)
        after = _obs(None, day=0)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {
                "farmer": ["PLANT", "WHEAT"],
                "hands": [],
                "_plan_credit": [{"op": "PLANT", "item": "WHEAT"}],
            },
            after,
        )
        self.assertEqual(result.planned_plants_completed, 0)
        self.assertEqual(result.reward, 0.0)

    def test_successful_planned_animal_placement_records_completion_without_turn_plan_bonus(self):
        before = _obs({"kind": "PASTURE", "animal": None}, inventory={"COW": 1})
        after = _obs({"kind": "PASTURE", "animal": "COW", "yield_units": 0})
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {
                "farmer": ["PLACE", "COW"],
                "hands": [],
                "_plan_credit": [{"op": "PLACE_ANIMAL", "item": "COW"}],
            },
            after,
        )
        self.assertEqual(result.planned_animals_placed, 1)
        self.assertEqual(PLANNED_PLACE_ANIMAL_REWARD, 0.0)
        self.assertEqual(result.reward, 1.0)

    def test_route_progress_and_avoidable_pass_shaping(self):
        before = _obs(None)
        after = _obs(None)

        progress = compute_worker_reward(
            EXECUTOR,
            before,
            {
                "farmer": ["EAST"],
                "hands": [],
                "_route_progress": [True],
            },
            after,
        )
        self.assertEqual(progress.route_progress_steps, 1)
        self.assertEqual(progress.reward, ROUTE_PROGRESS_REWARD)

        avoidable = compute_worker_reward(
            EXECUTOR,
            before,
            {
                "farmer": ["PASS"],
                "hands": [],
                "_avoidable_pass": [True],
            },
            after,
        )
        self.assertEqual(avoidable.avoidable_passes, 1)
        self.assertEqual(AVOIDABLE_PASS_PENALTY, 0.0)
        self.assertEqual(avoidable.reward, 0.0)

        necessary = compute_worker_reward(
            EXECUTOR,
            before,
            {
                "farmer": ["PASS"],
                "hands": [],
                "_avoidable_pass": [False],
            },
            after,
        )
        self.assertEqual(necessary.avoidable_passes, 0)
        self.assertEqual(necessary.reward, 0.0)

    def test_zero_weed_contract_is_deliberately_severe(self):
        self.assertEqual(CROP_TO_WEED_PENALTY, -256.0)
        self.assertEqual(CROP_DEATH_PENALTY, -128.0)
        self.assertEqual(LOST_HARVESTABLE_UNIT_PENALTY, -32.0)
        self.assertEqual(CRITICAL_WATER_REWARD, 16.0)

    def test_crop_to_weed_from_missed_water_is_classified(self):
        before = _obs({
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 0,
            "watered_today": False,
            "consecutive_unwatered": 1,
            "max_lifespan_step": 120,
        }, day=0, hour=23)
        after = _obs({"kind": "WEED"}, day=1, hour=0)

        result = compute_worker_reward(
            EXECUTOR, before, {"farmer": ["PASS"], "hands": []}, after
        )
        self.assertEqual(result.crops_to_weed, 1)
        self.assertEqual(result.crops_to_weed_unwatered, 1)
        self.assertEqual(result.crops_to_weed_decay, 0)
        self.assertEqual(result.crops_to_weed_other, 0)

    def test_crop_to_weed_from_expiry_is_classified_as_decay(self):
        before = _obs({
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 1,
            "watered_today": True,
            "consecutive_unwatered": 0,
            "max_lifespan_step": 120,
        }, day=4, hour=23)
        after = _obs({"kind": "WEED"}, day=5, hour=0)

        result = compute_worker_reward(
            EXECUTOR, before, {"farmer": ["PASS"], "hands": []}, after
        )
        self.assertEqual(result.crops_to_weed, 1)
        self.assertEqual(result.crops_to_weed_unwatered, 0)
        self.assertEqual(result.crops_to_weed_decay, 1)
        self.assertEqual(result.crops_to_weed_other, 0)
        self.assertEqual(result.lost_harvestable_units, 1.0)
        self.assertEqual(
            result.reward,
            CROP_TO_WEED_PENALTY + LOST_HARVESTABLE_UNIT_PENALTY,
        )

    def test_unexplained_crop_to_weed_remains_visible_as_other(self):
        before = _obs({
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 0,
            "watered_today": True,
            "consecutive_unwatered": 0,
            "max_lifespan_step": 120,
        }, day=0, hour=5)
        after = _obs({"kind": "WEED"}, day=0, hour=6)

        result = compute_worker_reward(
            EXECUTOR, before, {"farmer": ["PASS"], "hands": []}, after
        )
        self.assertEqual(result.crops_to_weed, 1)
        self.assertEqual(result.crops_to_weed_unwatered, 0)
        self.assertEqual(result.crops_to_weed_decay, 0)
        self.assertEqual(result.crops_to_weed_other, 1)

    def test_immature_crop_harvest_noop_gets_no_reward(self):
        before = _obs({
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 4,
        }, day=1)
        after = _obs({
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 4,
        }, day=1)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["HARVEST"], "hands": []},
            after,
        )
        self.assertEqual(result.crop_harvest_events_total, 0)
        self.assertEqual(result.crop_units_harvested_total, 0.0)
        self.assertEqual(result.products_harvested, 0.0)
        self.assertEqual(result.products_generated, 0.0)
        self.assertEqual(result.reward, 0.0)


    def test_wheat_age_two_harvest_gets_reward(self):
        before = _obs({
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 4,
        }, day=2)
        after = _obs({
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 0,
        }, day=2)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["HARVEST"], "hands": []},
            after,
        )
        self.assertEqual(result.crop_harvest_events_total, 1)
        self.assertEqual(result.crop_units_harvested_total, 4.0)
        self.assertEqual(result.products_harvested, 4.0)
        self.assertEqual(result.reward, 4 * PRODUCT_HARVESTED_REWARD)

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


    def test_shed_sourced_wheat_redelivery_gets_zero_reward(self):
        before = _obs(None, inventory={"WHEAT": 4})
        after = _obs(None, inventory={})
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {
                "farmer": ["PLACE", "WHEAT", 4],
                "hands": [],
                "_delivery_credit": [{"WHEAT": 0}],
            },
            after,
        )
        self.assertEqual(result.products_delivered, 0.0)
        self.assertEqual(result.product_units_moved_to_shed_total, 0.0)
        self.assertEqual(result.reward, 0.0)

    def test_wheat_delivery_reward_is_capped_by_fresh_credit(self):
        before = _obs(None, inventory={"WHEAT": 6})
        after = _obs(None, inventory={})
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {
                "farmer": ["PLACE", "WHEAT", 6],
                "hands": [],
                "_delivery_credit": [{"WHEAT": 2}],
            },
            after,
        )
        self.assertEqual(result.products_delivered, 2.0)
        self.assertEqual(result.product_units_moved_to_shed_total, 2.0)
        self.assertEqual(result.reward, 2 * PRODUCT_DELIVERED_REWARD)

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
        self.assertEqual(result.animal_product_units_moved_to_shed_total, 3)

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


    def test_care_earlier_then_feed_later_gets_credit_at_rollover(self):
        before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": True,
            "cared_today": True,
            "consecutive_unfed": 0,
            "pending_care_bonus": 0,
        }, day=0, hour=23)
        after = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": False,
            "cared_today": False,
            "consecutive_unfed": 0,
            "pending_care_bonus": 1,
        }, day=1, hour=0)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["PASS"], "hands": []},
            after,
        )
        self.assertEqual(result.effective_care, 1)
        self.assertEqual(
            result.reward,
            HEALTHY_ANIMAL_DAY_REWARD + EFFECTIVE_CARE_REWARD,
        )

    def test_care_without_feed_gets_no_effective_care_credit(self):
        before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": False,
            "cared_today": True,
            "consecutive_unfed": 0,
            "pending_care_bonus": 0,
        }, day=0, hour=23)
        after = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": False,
            "cared_today": False,
            "consecutive_unfed": 1,
            "pending_care_bonus": 0,
        }, day=1, hour=0)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["PASS"], "hands": []},
            after,
        )
        self.assertEqual(result.effective_care, 0)
        self.assertEqual(result.reward, 0.0)


    def test_final_turn_feed_completes_earlier_care_credit(self):
        before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": False,
            "cared_today": True,
            "consecutive_unfed": 0,
            "pending_care_bonus": 0,
        }, day=0, hour=23)
        after = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": False,
            "cared_today": False,
            "consecutive_unfed": 0,
            "pending_care_bonus": 1,
        }, day=1, hour=0)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["FEED"], "hands": []},
            after,
        )
        self.assertEqual(result.effective_care, 1)
        self.assertEqual(result.normal_feed, 1)
        self.assertEqual(
            result.reward,
            NORMAL_FEED_REWARD
            + HEALTHY_ANIMAL_DAY_REWARD
            + EFFECTIVE_CARE_REWARD,
        )

    def test_final_turn_care_is_credited_after_flags_reset(self):
        before = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": True,
            "cared_today": False,
            "consecutive_unfed": 0,
            "pending_care_bonus": 0,
        }, day=0, hour=23)
        after = _obs({
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": False,
            "cared_today": False,
            "consecutive_unfed": 0,
            "pending_care_bonus": 1,
        }, day=1, hour=0)
        result = compute_worker_reward(
            EXECUTOR,
            before,
            {"farmer": ["CARE"], "hands": []},
            after,
        )
        self.assertEqual(result.effective_care, 1)
        self.assertEqual(
            result.reward,
            HEALTHY_ANIMAL_DAY_REWARD + EFFECTIVE_CARE_REWARD,
        )

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
