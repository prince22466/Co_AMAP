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

Training reward measures worker execution efficiency only. `market_orders()` is not called by the candidate. Recorded v20 market commands are supplied only as frozen exogenous inputs so the game can continue hiring workers, buying seeds/animals/resources, selling products, and buying land. Those market actions are required for the environment trajectory, but they are not part of the trainable policy.

The following are **not** included in PPO reward:

- final WIN / TIE / LOSS
- final money
- money margin
- opponent money
- market prices or price movement

The implementation keeps the separation explicit:

```text
worker_action = RL policy output
    farmer
    hands

market_action = recorded v20 market list
    BUY_SEED / BUY_ANIMAL / BUY_PRODUCT
    SELL / HIRE / BUY_LAND

candidate_for_env = worker_action + market_action

env.step(candidate_for_env)

worker_reward = compute_worker_reward(
    state_before,
    worker_action,       # market_action is NOT passed here
    state_after,
)
```

Therefore PPO log-probabilities, GAE advantages, and policy gradients exist only for the worker action. The market list has no PPO log-probability and no direct reward term.

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
| crop product unit generated | +2 |
| crop product unit harvested | +2 |
| crop product unit explicitly delivered to shed | +4 |
| animal product unit generated (MILK/EGG/WOOL) | +4 |
| animal product unit harvested | +4 |
| animal product unit explicitly delivered to shed | +8 |
| successful PLANT | +1 |
| successful BUILD_COOP / BUILD_PASTURE | +0.5 |
| successful animal placement | +1 |
| effective CARE | +1 |
| effective FERTILIZE | +1 |
| COLLECT_FERTILIZER from an available animal | +1 |
| normal FEED / WATER | +1 |
| critical FEED / WATER where `consecutive_* >= 1` | +4 |

Crop products keep the original lifecycle value:

`2 generated + 2 harvested + 4 delivered = 8`

Animal products receive a 2x lifecycle premium because their worker pipeline is longer and more fragile:

`4 generated + 4 harvested + 8 delivered = 16`

The premium applies only to `MILK`, `EGG`, and `WOOL`. Planting, building, animal placement, feeding, care, and fertilizer rewards are unchanged, so the experiment isolates the value of completed animal output rather than broadly inflating all livestock-related actions. These fixed values ignore market-price movement.

`PLANT -> WEED` is always treated as a heavy worker-efficiency failure, including expiration caused by failing to harvest in time. Random `None -> WEED` spawning is not penalized.

A deliberate `DIG` of a fully exhausted crop with no remaining yield and age beyond its useful production window is treated as valid cleanup and is **not** assigned the crop-death penalty. Destroying a still-productive crop remains a heavy failure.

## Production-pipeline measurements

In addition to reward, every turn tracks explicit worker-production counters. These are accumulated into each episode's `reward_breakdown` and averaged in `metrics.jsonl` / `validation.jsonl`.

The tracked pipeline is:

```text
seed planted
    ↓
crop units harvested

animal product generated
    ↓
animal product harvested

harvested product
    ↓
explicit worker delivery to shed
```

The counters are:

- `seeds_planted_total`
- `seeds_planted_by_crop[WHEAT|CARROT|TOMATO|STRAWBERRY|MELON]`
- `crop_harvest_events_total`
- `crop_harvest_events_by_crop[...]`
- `crop_units_harvested_total`
- `crop_units_harvested_by_crop[...]`
- `animal_product_units_generated_total`
- `animal_product_units_generated_by_product[MILK|EGG|WOOL]`
- `animal_product_units_harvested_total`
- `animal_product_units_harvested_by_product[MILK|EGG|WOOL]`
- `product_units_moved_to_shed_total`
- `product_units_moved_to_shed_by_product[WHEAT|CARROT|TOMATO|STRAWBERRY|MELON|MILK|EGG|WOOL]`

A harvest event and harvested units are intentionally separate. For example, harvesting one carrot tile containing 3 units produces:

```text
crop_harvest_events_by_crop["CARROT"] += 1
crop_units_harvested_by_crop["CARROT"] += 3
```

Animal production is measured from tile yield changes. If a worker harvests on the same turn that end-of-day production occurs, generated units are reconstructed as:

```text
generated = max(0, yield_after + harvested_this_turn - yield_before)
```

This avoids missing production when harvest and production occur in the same environment turn.

Product movement to shed counts explicit worker delivery actions, not market selling and not automatic end-of-day inventory dumping. A later recorded `SELL` may remove the delivered units from the shed, but the delivery counter is already credited from the worker action and pre-action inventory.

Planting is counted from the successful tile transition. If a seed is planted on the last hour and immediately becomes `WEED` during day refresh, it is still counted in `seeds_planted_*` and also receives the `PLANT -> WEED` failure penalty.

Aggregate summaries expose both nested dictionaries and flattened metrics, for example:

```text
mean_seeds_planted_by_crop_wheat
mean_crop_units_harvested_by_crop_melon
mean_animal_product_units_generated_by_product_milk
mean_animal_product_units_harvested_by_product_wool
mean_product_units_moved_to_shed_by_product_egg
```

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

One PPO record corresponds to one environment turn. The turn still contains an autoregressive sequence of worker assignments, but PPO **does not form one probability ratio from the sum of all worker log probabilities**.

The single observed worker reward is converted by GAE into one turn-level advantage `A_t`. That same `A_t` is shared by every worker subdecision made in the turn. Each subdecision keeps its own behavior-policy log probability and gets its own clipped PPO ratio:

```text
turn t:
    worker subdecision 1 -> old_log_prob_1
    worker subdecision 2 -> old_log_prob_2
    ...
    worker subdecision N -> old_log_prob_N

one turn-level GAE advantage: A_t

ratio_i = exp(new_log_prob_i - old_log_prob_i)

policy_loss_i =
    -min(
        ratio_i * A_t,
        clip(ratio_i, 0.8, 1.2) * A_t
    )
```

This avoids the unstable previous formulation:

```text
exp(sum(new_log_prob_i - old_log_prob_i))
```

where modest probability changes compounded exponentially with the number of workers. The critic remains turn-level: there is still one `V(s_t)` and one return target per environment turn.

The recorded market list is not part of any PPO action. Market commands may affect the next environment state because they are executed by the game, but they are never passed into `compute_worker_reward()`.

Training uses a conservative stochastic behavior policy rather than full-temperature sampling:

```text
--rollout-temperature 0.20
```

For training rollouts, actor logits are divided by this temperature before sampling. This concentrates probability near the current argmax policy while retaining stochastic exploration. PPO recomputes the old/new log probabilities with the same temperature, so the importance ratio remains mathematically consistent. Deterministic validation still uses the raw actor-logit argmax.

The stability-oriented defaults are:

```text
learning_rate        1e-4
clip_ratio           0.10
target_kl            0.01
entropy_coef         0.001
rollout_temperature  0.20
```

KL is checked before every actor minibatch update. If the current per-subdecision approximate KL is already above `--target-kl`, that minibatch and the remaining actor updates are skipped. Training metrics include `approx_kl`, `clip_fraction`, `ratio_mean`, `ratio_std`, `ratio_min`, `ratio_max`, `actor_samples`, `mean_subdecisions_per_turn`, `actor_minibatches_completed`, `ppo_epochs_completed`, and `kl_early_stop`.

Validation has an automatic catastrophic-collapse guard. The reference reward is the best of the initial deterministic baseline and the best held-out reward achieved so far. If validation falls below:

```text
--collapse-restore-ratio 0.25
```

of that reference, the trainer records `collapse_warning=true`, reloads `best.pt` including optimizer state, and writes the restored policy to `latest.pt`. The raw collapsed post-update checkpoint remains available as `update_NNNN.pt` for diagnosis.

## Files

- `worker_reward.py` — fixed reward constants and before/after turn reward extraction.
- `worker_policy.py` — actor/critic, state/candidate features, rule-generated feasible worker intents, and the complete runtime replacement for `unit_actions`.
- `train_v25_worker_ppo.py` — static v20-loss replay, explicit separation of RL worker actions from frozen recorded market actions, GAE, PPO, validation, logging, and checkpoints.
- `test_ppo_subdecision_ratio.py` — regression coverage for the per-worker ratio formulation and PPO sample accounting.
- `v25_rl.py` — unchanged source for `production_signals`, `animal_plan`, `crop_plan`, and shared game helpers. Its `market_orders()` function is not called by the worker-training replay.

## Run

From `Co_Kaggle/g5/local_arena/v25_rl`:

Preflight:

```bash
python train_v25_worker_ppo.py --preflight-only
```


Regression tests:

```bash
python -m unittest test_ppo_subdecision_ratio.py test_worker_reward_contract.py
```

Train:

```bash
python train_v25_worker_ppo.py \
  --updates 100 \
  --episodes-per-update 8 \
  --max-training-hours 2
```


The animal-reward experiment uses checkpoint algorithm `v25_static_worker_ppo_gae_v4_animal_reward` and writes by default to `runs/worker_ppo_static_v20_v4_animal_reward`. Start a fresh run after this reward-contract change. Older checkpoints are intentionally rejected so critic targets and JSONL metrics from the 8/unit animal contract cannot be mixed with the new 16/unit contract.

`--minibatch-size` now batches worker subdecisions for the actor and turn records for the critic. The default remains 128.

Resume:

```bash
python train_v25_worker_ppo.py \
  --resume runs/worker_ppo_static_v20_v4_animal_reward/checkpoints/latest.pt \
  --updates 200 \
  --episodes-per-update 8 \
  --max-training-hours 2
```

## Outputs

Outputs are written under `runs/worker_ppo_static_v20_v4_animal_reward`.

- `metrics.jsonl` — PPO statistics, mean worker reward, and mean reward-component counts.
- `episodes.jsonl` — per-training-replay worker metrics.
- `validation.jsonl` — deterministic held-out aggregate worker metrics.
- `validation_episodes.jsonl` — deterministic held-out per-replay worker metrics.
- `checkpoints/latest.pt` — latest checkpoint.
- `checkpoints/best.pt` — checkpoint with the highest held-out mean worker reward seen so far, including the initial untrained baseline.

Primary health metrics are:

- `mean_seeds_planted_total`
- `mean_crop_units_harvested_total`
- `mean_animal_product_units_generated_total`
- `mean_animal_product_units_harvested_total`
- `mean_product_units_moved_to_shed_total`
- `mean_animals_escaped`
- `mean_crops_to_weed`
- `mean_crops_died`
- `mean_worker_reward`

The worker policy should improve these operational metrics independently of whether the overall game is ultimately won or lost.
