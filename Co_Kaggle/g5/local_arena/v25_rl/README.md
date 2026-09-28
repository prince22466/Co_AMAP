# v25 worker RL

v25 keeps the existing higher-level production, crop, and animal planners in `v25_rl.py`. During training, the legacy `unit_actions(obs, animal, crops)` implementation is replaced by `WorkerPolicy.unit_actions`. The candidate does **not** call `market_orders()`.

The old worker executor is therefore not used: its forced delivery logic, livestock routing/staging, heuristic task weights, tree prior, and residual-Q scheduler are bypassed. The candidate's procurement/selling policy is also not optimized in this experiment.

This is a full replacement of the **worker scheduling function**, but it is not primitive-action end-to-end RL. Candidate worker intents and legality checks are generated with rules, the policy learns which worker should take which intent and in what order, and movement toward a selected remote target is one deterministic Manhattan step.

## Static replay

Training uses:

`Co_Kaggle/g5/game_history/v20/*.json`

The corpus contract is that these are v20 losses. For each replay, the lower-final-reward seat is treated as the v20 seat. Only that seat's `farmer` and `hands` actions are replaced by the RL policy. Its recorded v20 `market` action list is replayed unchanged, and the opponent replays its complete original recorded action stream.

This is **static counterfactual replay**, not a live rematch: after the RL workers change the trajectory, neither the recorded opponent nor the recorded v20 market-order stream adapts. A recorded buy/sell/hire/land command may therefore become ineffective if the counterfactual state no longer satisfies its original preconditions.

Before training, `recorded_action_parity()` is run on the first sorted history as a replay-engine sanity check.

## Worker-only objective

Training reward measures worker execution efficiency only. `market_orders()` is not called by the candidate. Recorded v20 market commands are supplied only as frozen exogenous replay inputs so the worker experiment retains the original procurement, hiring, selling, and land-purchase schedule as closely as the counterfactual state permits.

The following are **not** included in PPO reward:

- final WIN / TIE / LOSS
- final money
- money margin
- opponent money
- market prices or price movement

Episode worker score is:

`sum(turn_worker_reward)`

The trainer creates one reward for every replay transition processed by:

`range(1, len(history["steps"]))`

so the number of RL transitions is `len(history["steps"]) - 1`; it is not hard-coded to 720. PPO uses those turn rewards with GAE. Defaults are `gamma=0.99` and `gae_lambda=0.95`.

### Reward contract

| Worker outcome | Reward |
| --- | ---: |
| animal escapes | -40 |
| PLANT -> WEED | -32 |
| still-productive plant is destroyed/disappears without HARVEST | -32 |
| each harvestable unit lost with a destroyed asset | -8 |
| crop/animal product unit generated | +2 |
| crop/animal product unit harvested | +2 |
| crop/animal product unit explicitly delivered to shed | +4 |
| successful PLANT | +1 |
| successful BUILD_COOP / BUILD_PASTURE | +0.5 |
| successful animal placement | +1 |
| effective CARE | +1 |
| effective FERTILIZE | +1 |
| COLLECT_FERTILIZER from an available animal | +1 |
| normal FEED / WATER | +1 |
| critical FEED / WATER where `consecutive_* >= 1` | +4 |

For crop and animal output products in `PRODUCTS`, the maximum lifecycle reward for one unit that completes all three milestones is:

`2 generated + 2 harvested + 4 delivered = 8`

This fixed value ignores market-price movement. Fertilizer is separate from this product lifecycle and receives the collection reward above.

`PLANT -> WEED` is always treated as a heavy worker-efficiency failure, including expiration caused by failing to harvest in time. Random `None -> WEED` spawning is not penalized.

A deliberate `DIG` of a fully exhausted crop with no remaining yield and age beyond its useful production window is treated as valid cleanup and is **not** assigned the crop-death penalty. Destroying a still-productive crop remains a heavy failure.

## Policy

`animal_plan` and `crop_plan` remain workload inputs. The RL worker scheduler can choose among intents for:

- planting
- watering
- harvesting
- fertilizing
- feeding
- care
- fertilizer collection
- coop/pasture construction
- animal placement
- resource pickup
- product delivery
- digging
- pass

Workers are assigned autoregressively within a turn. After each worker choice, feasibility/reservation state is updated and the next worker is selected from the remaining candidates.

If a selected task is remote, the emitted environment action is one deterministic Manhattan move toward its target. The network therefore learns **task/worker assignment and task ordering**, not free-form pathfinding.

One PPO record corresponds to one environment turn. All worker selections made inside that turn are treated as one autoregressive joint action:

`joint_log_prob = sum(subdecision_log_probs)`

The single observed worker reward for the resulting environment transition is attached to that joint turn action.

## Files

- `worker_reward.py` — fixed reward constants and before/after turn reward extraction.
- `worker_policy.py` — actor/critic, state/candidate features, rule-generated feasible worker intents, and the complete runtime replacement for `unit_actions`.
- `train_v25_worker_ppo.py` — static v20-loss replay, GAE, PPO, validation, logging, and checkpoints.
- `v25_rl.py` — unchanged source for `production_signals`, `animal_plan`, `crop_plan`, and shared game helpers. Its `market_orders()` function is not called by the worker-training replay.

## Run

From `Co_Kaggle/g5/local_arena/v25_rl`:

Preflight:

```bash
python train_v25_worker_ppo.py --preflight-only
```

Train:

```bash
python train_v25_worker_ppo.py \
  --updates 100 \
  --episodes-per-update 8 \
  --max-training-hours 2
```

Resume:

```bash
python train_v25_worker_ppo.py \
  --resume runs/worker_ppo_static_v20/checkpoints/latest.pt \
  --updates 200 \
  --episodes-per-update 8 \
  --max-training-hours 2
```

## Outputs

Outputs are written under `runs/worker_ppo_static_v20`.

- `metrics.jsonl` — PPO statistics, mean worker reward, and mean reward-component counts.
- `episodes.jsonl` — per-training-replay worker metrics.
- `validation.jsonl` — deterministic held-out aggregate worker metrics.
- `validation_episodes.jsonl` — deterministic held-out per-replay worker metrics.
- `checkpoints/latest.pt` — latest checkpoint.
- `checkpoints/best.pt` — checkpoint with the highest held-out mean worker reward seen so far, including the initial untrained baseline.

Primary health metrics are:

- `mean_animals_escaped`
- `mean_crops_to_weed`
- `mean_crops_died`
- `mean_products_generated`
- `mean_products_harvested`
- `mean_products_delivered`
- `mean_worker_reward`

The worker policy should improve these operational metrics independently of whether the overall game is ultimately won or lost.
