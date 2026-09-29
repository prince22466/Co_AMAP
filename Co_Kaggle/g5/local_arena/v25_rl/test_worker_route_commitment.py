"""Regression tests for persistent worker task commitment and WHEAT provenance."""
from __future__ import annotations

import unittest

import torch

from worker_policy import (
    AVOIDABLE_PASS_ACTOR_PENALTY,
    DEFER_PLANNED_PLANT_ACTOR_PENALTY,
    PLANNED_PLANT_ACTOR_BONUS,
    CANDIDATE_FEATURE_NAMES,
    Task,
    WorkerPolicy,
)


class _Executor:
    ANIMALS = {
        "COW": (400, "MILK", 8, 2, 3),
        "SHEEP": (500, "WOOL", 6, 3, 4),
        "GOOSE": (300, "EGG", 4, 1, 2),
    }
    CROPS = {}
    CROP_FIRST_YIELD_DAY = {}
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
            policy.actor_bonus_for_choice(0, plant, choices),
            PLANNED_PLANT_ACTOR_BONUS,
        )
        self.assertEqual(
            policy.actor_bonus_for_choice(0, passed, choices),
            AVOIDABLE_PASS_ACTOR_PENALTY
            + DEFER_PLANNED_PLANT_ACTOR_PENALTY,
        )

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
