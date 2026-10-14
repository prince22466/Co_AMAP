"""Replay probe using the engine's first legal harvest day."""

import replay_checkpoint_agent as base
from worker_policy import Task


FIRST_YIELD_DAY = {
    "WHEAT": 2,
    "CARROT": 2,
    "TOMATO": 8,
    "STRAWBERRY": 10,
    "MELON": 10,
}
original_tasks = base.policy.tasks


def engine_maturity_tasks(obs, animal_plan, crop_plan):
    tasks = original_tasks(obs, animal_plan, crop_plan)
    farm = obs["farms"][obs["player"]]
    for point in crop_plan:
        point = tuple(point)
        tile = base.v25_rl.tile(farm, point)
        if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
            continue
        crop = tile["crop"]
        age = obs["day"] - tile["planted_day"]
        if tile.get("yield_units", 0) > 0 and age >= FIRST_YIELD_DAY[crop]:
            candidate = Task(point, "HARVEST", crop)
            if candidate not in tasks:
                tasks.append(candidate)
    return tasks


base.policy.tasks = engine_maturity_tasks
agent = base.agent
