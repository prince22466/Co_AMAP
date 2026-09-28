# v25 worker RL

v25 keeps the existing higher-level production, crop, animal, procurement, and market planners in v25_rl.py. Training replaces the entire unit_actions(obs, animal, crops) worker executor at runtime with WorkerPolicy.unit_actions. None of the legacy worker routing, forced delivery, staging, task weighting, tree prior, or residual-Q worker scheduler is called.

## Static replay

Training uses Co_Kaggle/g5/game_history/v20/*.json. Every history is treated as a recorded v20 loss. The lower-reward seat is replaced by the candidate; the other seat replays its original recorded action stream turn by turn. This is static counterfactual replay, not a live rematch.

A recorded-action parity gate is run before training.

## Worker-only objective

There is no terminal win/loss, final money, money margin, market-price, or opponent-money reward. Episode worker score is simply the sum of per-turn worker rewards. PPO uses the 720-turn reward stream with GAE (gamma=0.99, lambda=0.95 by default).

Reward contract:

| Observable event | Reward |
| --- | ---: |
| animal escapes | -40 |
| PLANT -> WEED | -32 |
| plant disappears without harvest | -32 |
| each harvestable unit lost with destroyed asset | -8 |
| product unit generated | +2 |
| product unit harvested | +2 |
| product unit delivered to shed | +4 |
| successful plant | +1 |
| build coop/pasture | +0.5 |
| place animal | +1 |
| effective care | +1 |
| effective fertilize | +1 |
| collect fertilizer | +1 |
| normal feed / water | +1 |
| critical feed / water (consecutive_* >= 1) | +4 |

All products use one fixed lifecycle value of 8: 2 generated + 2 harvested + 4 delivered. Market price movement is ignored.

PLANT -> WEED is always a heavy worker-efficiency failure, including expiration caused by inefficient harvesting. A plant disappearing without HARVEST is also a heavy failure. Random None -> WEED spawning is not penalized.

## Policy

animal_plan and crop_plan remain workload inputs, but all worker execution is learned. The policy generates feasible intents for planting, watering, harvesting, fertilizing, feeding, care, fertilizer collection, coop/pasture construction, animal placement, resource pickup, product delivery, digging, and pass.

Workers are assigned autoregressively each turn. A remote selected task emits one deterministic Manhattan move toward its target. The PPO record is one environment turn: all worker selections made in that turn form a joint action whose log probability is the sum of its subdecision log probabilities. The one observed turn reward is attached to that joint action.

Files:
- worker_reward.py: fixed worker reward constants and before/after transition reward extraction.
- worker_policy.py: actor/critic, state/candidate features, complete RL unit_actions replacement.
- train_v25_worker_ppo.py: static v20-loss replay, GAE, PPO, validation, logging, checkpoints.
- v25_rl.py: unchanged high-level v25 planner/executor source.

## Run

From Co_Kaggle/g5/local_arena/v25_rl:

    python train_v25_worker_ppo.py --preflight-only

Train:

    python train_v25_worker_ppo.py --updates 100 --episodes-per-update 8 --max-training-hours 2

Resume:

    python train_v25_worker_ppo.py --resume runs/worker_ppo_static_v20/checkpoints/latest.pt --updates 200 --episodes-per-update 8 --max-training-hours 2

Outputs are written under runs/worker_ppo_static_v20:
- metrics.jsonl: PPO loss and mean worker reward plus every reward component.
- episodes.jsonl: per-training-history worker metrics.
- validation.jsonl and validation_episodes.jsonl: deterministic held-out worker metrics.
- checkpoints/latest.pt: latest checkpoint.
- checkpoints/best.pt: highest held-out mean worker reward.

Primary health metrics are mean_animals_escaped, mean_crops_to_weed, mean_products_harvested, mean_products_delivered, and mean_worker_reward.
