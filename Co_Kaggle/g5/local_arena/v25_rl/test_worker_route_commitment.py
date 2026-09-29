"""Regression tests for persistent worker task commitment and WHEAT provenance."""
from __future__ import annotations

import unittest

import torch

from worker_policy import CANDIDATE_FEATURE_NAMES, Task, WorkerPolicy


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
        self.assertEqual(policy.active_tasks[0].key, ((3, 0), "FEED", "WHEAT", 0))

        second = policy.unit_actions(_obs(worker_x=1), {}, {})
        self.assertEqual(second[0], ["EAST"])
        self.assertEqual(policy.active_tasks[0].key, ((3, 0), "FEED", "WHEAT", 0))

    def test_committed_movement_exposes_route_target_for_reward(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        policy.active_day = 0
        policy.active_tasks[0] = Task((3, 0), "FEED", "WHEAT")

        actions = policy.unit_actions(_obs(worker_x=0), {}, {})
        self.assertEqual(actions[0], ["EAST"])
        self.assertEqual(policy.turn_route_targets[0], (3, 0))

    def test_pass_is_marked_avoidable_only_when_same_worker_has_useful_work(self):
        policy = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        actions = policy.unit_actions(_obs(worker_x=3, fed=False, wheat=1), {}, {})
        self.assertEqual(actions[0], ["PASS"])
        self.assertTrue(policy.turn_avoidable_pass[0])

        no_work = WorkerPolicy(
            _Executor(),
            _PassLovingModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        actions = no_work.unit_actions(_obs(worker_x=3, fed=True, wheat=0), {}, {})
        self.assertEqual(actions[0], ["PASS"])
        self.assertFalse(no_work.turn_avoidable_pass[0])

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
        self.assertNotIn(0, policy.active_tasks)

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
