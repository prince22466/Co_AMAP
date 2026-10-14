"""Temporary replay probe: reject immature crop HARVEST intents."""

import replay_checkpoint_agent as base


FIRST_YIELD_DAY = {
    "WHEAT": 2,
    "CARROT": 2,
    "TOMATO": 8,
    "STRAWBERRY": 10,
    "MELON": 10,
}
original_tasks = base.policy.tasks


def mature_tasks(obs, animal_plan, crop_plan):
    farm = obs["farms"][obs["player"]]
    result = []
    for task in original_tasks(obs, animal_plan, crop_plan):
        if task.op == "HARVEST" and task.target is not None:
            tile = base.v25_rl.tile(farm, task.target)
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                age = obs["day"] - tile["planted_day"]
                if age < FIRST_YIELD_DAY[tile["crop"]]:
                    continue
        result.append(task)
    return result


base.policy.tasks = mature_tasks
agent = base.agent
