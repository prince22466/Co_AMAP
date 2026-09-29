"""Regression test for worker-specific delivery reservations."""
from __future__ import annotations

import unittest

import torch

from worker_policy import CANDIDATE_FEATURE_NAMES, WorkerPolicy


class _Executor:
    SHED = [(0, 0)]
    ANIMALS = ("COW", "SHEEP", "GOOSE")
    CROPS = {}

    @staticmethod
    def nearest_shed(_pos):
        return (0, 0)

    @staticmethod
    def dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def tile(farm, pos):
        if pos is None:
            return None
        x, y = pos
        return farm["tiles"][y][x]


class _DeliverFirstModel:
    def value(self, _state):
        return torch.tensor(0.0)

    def logits(self, candidates):
        idx = CANDIDATE_FEATURE_NAMES.index("op_DELIVER")
        return candidates[:, idx] * 100.0


def _obs():
    farm = {
        "farmer": [0, 0],
        "hands": [[0, 0]],
        "tiles": [[None]],
    }
    return {
        "player": 0,
        "day": 0,
        "hour": 0,
        "farms": [
            farm,
            {"farmer": [0, 0], "hands": [], "tiles": [[None]]},
        ],
        "private": {
            "inventories": [{"MILK": 1}, {"MILK": 1}],
            "shed": {},
            "seeds": {},
        },
    }


class WorkerDeliveryReservationTest(unittest.TestCase):
    def test_two_workers_can_deliver_same_product_same_shed(self):
        policy = WorkerPolicy(
            _Executor(),
            _DeliverFirstModel(),
            torch.device("cpu"),
            deterministic=True,
            collect=False,
        )
        actions = policy.unit_actions(_obs(), {}, {})
        self.assertEqual(actions[0], ["PLACE", "MILK", 1])
        self.assertEqual(actions[1], ["PLACE", "MILK", 1])


if __name__ == "__main__":
    unittest.main()
