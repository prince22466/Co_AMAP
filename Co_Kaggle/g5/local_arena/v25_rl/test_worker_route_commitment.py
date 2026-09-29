"""Regression tests for persistent worker task commitment and WHEAT provenance."""
from __future__ import annotations

import unittest

import torch

from worker_policy import (
    AVOIDABLE_PASS_ACTOR_PENALTY,
    DEFER_PLANNED_PLANT_ACTOR_PENALTY,
    PLANNED_PLANT_REWARD_EQUIV,
    CANDIDATE_FEATURE_NAMES,
    Task,
    WorkerPolicy,
)
from worker_reward import RewardBreakdown


class _Executor:
    ANIMALS = {
        "COW": (400, "MILK", 8, 2, 3),
        "SHEEP": (500, "WOOL", 6, 3, 4),
        "GOOSE": (300, "EGG", 4, 1, 2),
    }
    CROPS = {
        "WHEAT": (10, 25, ((4, 4),), 4),
    }
    CROP_FIRST_YIELD_DAY = {
        "WHEAT": 2,
    }
    SHED = ((0, 0),)

    @staticmethod
    def tile(farm, pos):
        x, y = pos
        return farm["tiles"][y][x]

    @staticmethod
    def dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def nearest_shed(pos):
        return (0, 0)

    @staticmethod
    def move(pos, target):
        if target[0] > pos[0]:
            return ["EAST"]
        if target[0] < pos[0]:
            return ["WEST"]
        if target[1] > pos[1]:
            return ["SOUTH"]
        if target[1] < pos[1]:
            return ["NORTH"]
        return ["PASS"]


class _PlantLovingModel:
    def value(self, state):
        return torch.tensor(0.0)

    def logits(self, candidates):
        plant_col = CANDIDATE_FEATURE_NAMES.index("op_PLANT")
        return candidates[:, plant_col] * 100.0


class _PassLovingModel:
    def value(self, state):
        return torch.tensor(0.0)

    def logits(self, candidates):
        pass_col = CANDIDATE_FEATURE_NAMES.index("op_PASS")
        return candidates[:, pass_col] * 100.0


def _obs(worker_x=0, fed=False, wheat=1):
    cow = {
        "kind": "PASTURE",
        "animal": "COW",
        "yield_units": 0,
        "fed_today": fed,
        "cared_today": True,
        "consecutive_unfed": 0,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
        "placed_day": 0,
    }
    farm = {
        "farmer": [worker_x, 0],
        "hands": [],
        "tiles": [[None, None, None, cow]],
    }
    return {
        "player": 0,
        "day": 0,
        "hour": 5,
        "farms": [farm],
        "private": {
            "inventories": [{"WHEAT": wheat}],
            "shed": {"WHEAT": 20},
            "seeds": {},
        },
    }


def _critical_animal_obs(worker_x=0, hour=5, wheat=1):
    obs = _obs(worker_x=worker_x, fed=False, wheat=wheat)
    obs["hour"] = hour
    cow = obs["farms"][0]["tiles"][0][3]
    cow["consecutive_unfed"] = 1
    return obs


def _critical_crop_obs(worker_x=0, hour=5):
    obs = _obs(worker_x=worker_x, fed=False, wheat=1)
    obs["day"] = 1
    obs["hour"] = hour
    obs["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "yield_units": 0,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "fertilized_until_day": -1,
    }
    return obs


def _two_worker_critical_obs():
    obs = _critical_crop_obs(worker_x=0, hour=5)
    obs["farms"][0]["hands"] = [[1, 0]]
    obs["farms"][0]["tiles"][0][3]["cared_today"] = False
    obs["private"]["inventories"] = [{"WHEAT": 1}, {"WHEAT": 1}]
    return obs


def _decay_crop_obs(worker_x=1, hour=22, max_lifespan_step=120):
    obs = _obs(worker_x=worker_x, fed=False, wheat=1)
    obs["day"] = 4
    obs["hour"] = hour
    obs["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "yield_units": 1,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "max_lifespan_step": max_lifespan_step,
        "fertilized_until_day": -1,
    }
    return obs


class WorkerRouteCommitmentTest(unittest.TestCase):
    def test_committed_worker_keeps_heading_to_same_feed_task(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        policy.active_day = 0
        policy.active_tasks[0] = Task((3, 0), "FEED", "WHEAT")

        first = policy.unit_actions(_obs(worker_x=0), {}, {})
        self.assertEqual(first[0], ["EAST"])
        self.assertEqual(policy.turn_route_progress, [True])
        self.assertEqual(policy.active_tasks[0].key, ((3, 0), "FEED", "WHEAT", 0))

        second = policy.unit_actions(_obs(worker_x=1), {}, {})
        self.assertEqual(second[0], ["EAST"])
        self.assertEqual(policy.active_tasks[0].key, ((3, 0), "FEED", "WHEAT", 0))

    def test_critical_feed_preempts_pass(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _critical_animal_obs(worker_x=3, wheat=1)

        actions = policy.unit_actions(obs, {}, {})
        self.assertEqual(actions[0], ["FEED"])

    def test_critical_feed_preempts_noncritical_route(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _critical_animal_obs(worker_x=0, wheat=1)
        obs["farms"][0]["tiles"][0][3]["cared_today"] = False
        policy.active_day = 0
        policy.active_tasks[0] = Task((3, 0), "CARE", "COW")

        actions = policy.unit_actions(obs, {}, {})
        self.assertEqual(actions[0], ["EAST"])
        self.assertEqual(policy.active_tasks[0].op, "FEED")

    def test_critical_feed_without_carried_wheat_forces_pickup(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _critical_animal_obs(worker_x=0, wheat=0)
        obs["private"]["shed"] = {"WHEAT": 20}

        actions = policy.unit_actions(obs, {}, {})
        self.assertEqual(actions[0], ["PICKUP", "WHEAT", 1])
        self.assertEqual(policy.turn_avoidable_pass, [False])

    def test_stranded_carried_wheat_does_not_suppress_emergency_pickup(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _critical_animal_obs(worker_x=0, hour=20, wheat=1)
        obs["farms"][0]["hands"] = [[0, 0]]
        obs["private"]["inventories"] = [
            {"WHEAT": 1},
            {},
        ]
        # Move the critical animal farther than worker 0 can reach before EOD,
        # while worker 1 can still use the shed route after pickup.
        obs["farms"][0]["farmer"] = [0, 0]
        obs["farms"][0]["hands"] = [[4, 0]]
        obs["farms"][0]["tiles"][0][3]["animal"] = None
        obs["farms"][0]["tiles"].append([
            None, None, None, {"kind": "PASTURE", "animal": "COW",
            "yield_units": 0, "fed_today": False, "cared_today": True,
            "consecutive_unfed": 1, "fertilizer_available": False,
            "pending_care_bonus": 0, "placed_day": 0}
        ])
        # Put the critical animal at (3, 1): worker 0's carried wheat is not
        # reachable in time under the synthetic deadline, so a critical pickup
        # must still be generated.
        critical = [
            t for t in policy.tasks(obs, {}, {})
            if t.op == "PICKUP" and t.item == "WHEAT" and t.critical >= 1
        ]
        self.assertTrue(critical)

    def test_emergency_wheat_pickup_leaves_a_later_feed_turn(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _critical_animal_obs(worker_x=0, hour=19, wheat=0)
        pickup = next(
            t for t in policy.tasks(obs, {}, {})
            if t.op == "PICKUP" and t.item == "WHEAT" and t.critical >= 1
        )
        self.assertTrue(
            policy.feasible(
                obs, 0, pickup, {}, dict(obs["private"]["shed"])
            )
        )

        obs["hour"] = 20
        pickup = next(
            t for t in policy.tasks(obs, {}, {})
            if t.op == "PICKUP" and t.item == "WHEAT" and t.critical >= 1
        )
        self.assertFalse(
            policy.feasible(
                obs, 0, pickup, {}, dict(obs["private"]["shed"])
            )
        )

    def test_critical_feed_outranks_imminent_decay_harvest(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _decay_crop_obs(
            worker_x=3, hour=22, max_lifespan_step=122
        )
        cow = obs["farms"][0]["tiles"][0][3]
        cow["consecutive_unfed"] = 1
        cow["fed_today"] = False
        obs["private"]["inventories"] = [{"WHEAT": 1}]

        actions = policy.unit_actions(
            obs,
            {},
            {(0, 0): "WHEAT"},
        )
        self.assertEqual(actions[0], ["FEED"])

    def test_unreachable_critical_feed_does_not_destroy_useful_route(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _critical_animal_obs(worker_x=0, hour=20, wheat=0)
        obs["private"]["shed"] = {"WHEAT": 0}
        obs["farms"][0]["tiles"][0][3]["cared_today"] = False
        policy.active_day = 0
        care = Task((3, 0), "CARE", "COW")
        policy.active_tasks[0] = care

        actions = policy.unit_actions(obs, {}, {})
        self.assertEqual(actions[0], ["EAST"])
        self.assertEqual(policy.active_tasks.get(0), care)

    def test_critical_water_preempts_pass_and_noncritical_route(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        policy.active_day = 1
        policy.active_tasks[0] = Task((3, 0), "FEED", "WHEAT")

        actions = policy.unit_actions(
            _critical_crop_obs(worker_x=0),
            {},
            {(0, 0): "WHEAT"},
        )
        self.assertEqual(actions[0], ["WATER"])
        self.assertNotEqual(policy.active_tasks.get(0), Task((3, 0), "FEED", "WHEAT"))

    def test_imminent_decay_harvest_preempts_pass(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _decay_crop_obs(worker_x=1, hour=22, max_lifespan_step=120)

        actions = policy.unit_actions(
            obs,
            {},
            {(0, 0): "WHEAT"},
        )
        # At step 118, a one-unit crop with decay starting at 120 has reached
        # its latest safe route-start window. PASS is hard-masked.
        self.assertEqual(actions[0], ["WEST"])
        self.assertEqual(policy.active_tasks[0].op, "HARVEST")

        at_target = _decay_crop_obs(
            worker_x=0, hour=23, max_lifespan_step=120
        )
        actions = policy.unit_actions(
            at_target,
            {},
            {(0, 0): "WHEAT"},
        )
        self.assertEqual(actions[0], ["HARVEST"])

    def test_imminent_decay_harvest_preempts_noncritical_route(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _decay_crop_obs(worker_x=1, hour=22, max_lifespan_step=120)
        policy.active_day = 4
        policy.active_tasks[0] = Task((3, 0), "FEED", "WHEAT")

        actions = policy.unit_actions(
            obs,
            {},
            {(0, 0): "WHEAT"},
        )
        self.assertEqual(actions[0], ["WEST"])
        self.assertEqual(policy.active_tasks[0].op, "HARVEST")

    def test_nonurgent_harvest_remains_ppo_choice(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _decay_crop_obs(worker_x=0, hour=5, max_lifespan_step=160)

        actions = policy.unit_actions(
            obs,
            {},
            {(0, 0): "WHEAT"},
        )
        self.assertEqual(actions[0], ["PASS"])

    def test_plant_requires_a_later_turn_for_water(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        task = Task((0, 0), "PLANT", "WHEAT", planned=True)
        obs = _obs(worker_x=0)
        obs["private"]["seeds"] = {"WHEAT": 1}

        obs["hour"] = 23
        self.assertFalse(
            policy.feasible(obs, 0, task, {"WHEAT": 1}, obs["private"]["shed"])
        )
        obs["hour"] = 22
        self.assertTrue(
            policy.feasible(obs, 0, task, {"WHEAT": 1}, obs["private"]["shed"])
        )

    def test_planned_plant_and_pass_get_different_actor_credit(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        plant = Task((0, 0), "PLANT", "WHEAT", planned=True)
        passed = Task((0, 0), "PASS")
        choices = [
            (0, plant, plant.key),
            (0, passed, passed.key),
        ]
        self.assertEqual(
            policy.actor_bonus_for_choice(0, plant, choices, completed=False),
            0.0,
        )
        self.assertEqual(
            policy.actor_bonus_for_choice(0, plant, choices, completed=True),
            0.0,
        )
        self.assertEqual(
            policy.planned_completion_reward_equiv(plant),
            PLANNED_PLANT_REWARD_EQUIV,
        )
        self.assertEqual(
            policy.actor_bonus_for_choice(0, passed, choices),
            AVOIDABLE_PASS_ACTOR_PENALTY
            + DEFER_PLANNED_PLANT_ACTOR_PENALTY,
        )

    def test_one_critical_crop_preempts_only_one_of_two_routes(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        policy.active_day = 1
        feed = Task((3, 0), "FEED", "WHEAT")
        care = Task((3, 0), "CARE", "COW")
        policy.active_tasks[0] = feed
        policy.active_tasks[1] = care

        actions = policy.unit_actions(
            _two_worker_critical_obs(),
            {},
            {(0, 0): "WHEAT"},
        )

        self.assertEqual(actions[0], ["EAST"])
        self.assertEqual(actions[1], ["WEST"])
        self.assertEqual(policy.active_tasks.get(0), feed)
        self.assertNotEqual(policy.active_tasks.get(1), care)

    def test_remote_planned_bonus_arrives_only_on_route_completion(self):
        policy = WorkerPolicy(
            _Executor(),
            _PlantLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=True,
        )
        obs = _obs(worker_x=0)
        obs["private"]["seeds"] = {"WHEAT": 1}
        plan = {(2, 0): "WHEAT"}

        actions = policy.unit_actions(obs, {}, plan)
        self.assertEqual(actions[0], ["EAST"])
        origin = policy.pending.subdecisions[0]
        self.assertEqual(origin.actor_bonus, 0.0)
        policy.finish_turn(RewardBreakdown())

        at_target = _obs(worker_x=2)
        at_target["private"]["seeds"] = {"WHEAT": 1}
        actions = policy.unit_actions(at_target, {}, plan)
        self.assertEqual(actions[0], ["PLANT", "WHEAT"])
        self.assertEqual(origin.actor_bonus, 0.0)
        self.assertEqual(origin.reward_equiv_bonus, 0.0)

        policy.finish_turn(
            RewardBreakdown(planned_plants_completed=1)
        )
        self.assertEqual(
            origin.reward_equiv_bonus,
            PLANNED_PLANT_REWARD_EQUIV,
        )

    def test_emitted_planned_plant_gets_no_credit_if_transition_fails(self):
        policy = WorkerPolicy(
            _Executor(),
            _PlantLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=True,
        )
        obs = _obs(worker_x=0)
        obs["private"]["seeds"] = {"WHEAT": 1}
        plan = {(0, 0): "WHEAT"}

        actions = policy.unit_actions(obs, {}, plan)
        self.assertEqual(actions[0], ["PLANT", "WHEAT"])
        origin = policy.pending.subdecisions[0]
        self.assertEqual(origin.reward_equiv_bonus, 0.0)

        policy.finish_turn(RewardBreakdown(planned_plants_completed=0))
        self.assertEqual(origin.reward_equiv_bonus, 0.0)

    def test_interrupted_planned_route_gets_no_positive_bonus(self):
        policy = WorkerPolicy(
            _Executor(),
            _PlantLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=True,
        )
        obs = _obs(worker_x=0)
        obs["private"]["seeds"] = {"WHEAT": 1}
        plan = {(2, 0): "WHEAT"}
        policy.unit_actions(obs, {}, plan)
        origin = policy.pending.subdecisions[0]
        policy.finish_turn(RewardBreakdown())

        urgent = _critical_crop_obs(worker_x=1)
        urgent["day"] = 0
        urgent["private"]["seeds"] = {"WHEAT": 1}
        actions = policy.unit_actions(
            urgent,
            {},
            {(0, 0): "WHEAT", (2, 0): "WHEAT"},
        )
        self.assertEqual(actions[0], ["WEST"])
        self.assertEqual(origin.actor_bonus, 0.0)
        self.assertEqual(origin.reward_equiv_bonus, 0.0)
        self.assertNotIn(0, policy.active_origins)

    def test_unreachable_critical_water_does_not_destroy_useful_route(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        obs = _critical_crop_obs(worker_x=3, hour=23)
        obs["farms"][0]["tiles"][0][3]["cared_today"] = False
        policy.active_day = 1
        care = Task((3, 0), "CARE", "COW")
        policy.active_tasks[0] = care

        actions = policy.unit_actions(obs, {}, {(0, 0): "WHEAT"})
        self.assertEqual(actions[0], ["CARE"])

    def test_commitment_is_released_when_task_becomes_invalid(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        policy.active_day = 0
        policy.active_tasks[0] = Task((3, 0), "FEED", "WHEAT")

        actions = policy.unit_actions(_obs(worker_x=1, fed=True), {}, {})
        self.assertEqual(actions[0], ["PASS"])
        self.assertEqual(policy.turn_avoidable_pass, [False])
        self.assertNotIn(0, policy.active_tasks)


    def test_pass_is_marked_avoidable_when_feed_work_is_feasible(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        actions = policy.unit_actions(_obs(worker_x=3, fed=False, wheat=1), {}, {})
        self.assertEqual(actions[0], ["PASS"])
        self.assertEqual(policy.turn_avoidable_pass, [True])

    def test_shed_sourced_wheat_is_not_a_delivery_candidate(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        policy.active_day = 0
        policy.shed_sourced_wheat[0] = 4

        obs = _obs(worker_x=0, wheat=4)
        wheat_delivery = [
            t for t in policy.extras(obs, 0)
            if t.op == "DELIVER" and t.item == "WHEAT"
        ]
        self.assertEqual(wheat_delivery, [])

    def test_only_non_shed_wheat_is_deliverable(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        policy.active_day = 0
        policy.shed_sourced_wheat[0] = 4

        obs = _obs(worker_x=0, wheat=6)
        wheat_delivery = [
            t for t in policy.extras(obs, 0)
            if t.op == "DELIVER" and t.item == "WHEAT"
        ]
        self.assertEqual(len(wheat_delivery), 1)
        self.assertEqual(wheat_delivery[0].amount, 2)


if __name__ == "__main__":
    unittest.main()
