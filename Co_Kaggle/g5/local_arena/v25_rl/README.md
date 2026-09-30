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
| animal escapes | -100 |
| PLANT -> WEED | -256 |
| still-productive plant is destroyed/disappears without HARVEST | -128 |
| each harvestable unit lost with a destroyed asset | -32 |
| crop product unit generated | +2 |
| crop product unit harvested | +2 |
| crop product unit explicitly delivered to shed | +4 |
| animal product unit generated (MILK/EGG/WOOL) | +16 |
| animal product unit harvested | +16 |
| animal product unit explicitly delivered to shed | +96 |
| successful PLANT | +1 |
| successful PLANT matching `crop_plan` | tracked; actor-only bonus |
| successful BUILD_COOP / BUILD_PASTURE | +0.5 turn reward; planned BUILD gets actor-only bonus |
| successful animal placement | +1 |
| successful animal placement matching `animal_plan` | tracked; actor-only bonus |
| purposeful movement toward a selected/committed task | +0.05 |
| PASS while that worker has feasible non-PASS work | tracked; actor-only penalty |
| PASS with no feasible non-PASS work | 0 |
| effective CARE day (animal finishes day fed + cared) | +3 |
| effective FERTILIZE | +1 |
| COLLECT_FERTILIZER from an available animal | +1 |
| normal FEED | +6 |
| critical FEED where `consecutive_unfed >= 1` | +2 |
| fed animal survives a day rollover | +4 |
| normal WATER | +1 |
| critical WATER where `consecutive_unwatered >= 1` | +16 |

Crop products keep the original lifecycle value:

`2 generated + 2 harvested + 4 delivered = 8`

Animal products now receive a much stronger lifecycle value because the v4 training records showed animal throughput was roughly two orders of magnitude below crop throughput and only ~6.5% of harvested animal output reached the shed at the best validation checkpoint:

`16 generated + 16 harvested + 96 delivered = 128`

The reward deliberately puts most value on **delivery**, so harvesting animal output without moving it to the shed is no longer close to completing the lifecycle. Planner/PASS preference is no longer added to the shared turn reward. It is attached directly to the PPO worker subdecision so one worker's PLANT bonus or PASS penalty cannot incorrectly reinforce or punish every other worker choice made in the same turn. The animal-maintenance shaping is also changed from the previous contract: normal FEED is worth more than critical rescue FEED, effective CARE is credited at day rollover only when the animal actually finished that day both fed and cared, a fed animal surviving a day rollover receives dense credit, and animal escape is more expensive. These fixed values still ignore market-price movement.

`PLANT -> WEED` is treated as a near-catastrophic worker-efficiency failure (`-256` plus `-32` per lost harvestable unit). Random `None -> WEED` spawning is not penalized.

The target is near-zero preventable `PLANT -> WEED`, spoilage, and animal escapes while retaining productive planting, harvesting, and delivery. Crop maintenance scans every live plant, including tiles dropped by a changed crop plan.

See [LOSSLESS_VALIDATION.md](LOSSLESS_VALIDATION.md) for paired replays with the unchanged v9 update-147 actor, including the observed throughput tradeoff and reproduction command.

Harvest deadlines protect the **first lost unit**, using the engine's `max_lifespan_step` and engine lifespan ages as fallback. Held yield never extends that deadline. Worker actions execute before decay at the same engine step; a one-time crop can therefore be harvested at that step. Tomato/strawberry require their final harvest and subsequent DIG cleanup **before the preceding midnight worker reset**: the engine leaves ongoing plants on the map after harvesting and will turn an exhausted zero-yield plant into a weed at expiry. Hands disappear and the farmer returns to the shed at day rollover, so relying on cleanup at expiry itself is unsafe.

On an ongoing crop's final production day, HARVEST becomes urgent immediately and commits that worker to a following DIG after the observed crop is empty. This preserves the final yield and avoids deferring cleanup behind another actor assignment.

The scheduler estimates cumulative travel plus action time for multiple queued jobs per worker. It includes survival work and wheat pickup before feeding. When waiting would lose deadline coverage, PASS and noncritical choices are masked. Urgent PPO assignments are restricted to choices that preserve the estimated remaining coverage when such choices exist, preventing a flexible worker from taking the sole reachable job of another worker.

PLANT requires room for movement, planting, follow-up WATER, existing daily WATER/FEED and imminent harvest/cleanup work. It also reserves the enlarged farm's next-day WATER/FEED/CARE time using the current workforce. Simultaneous plant choices share reservations. The next-day estimate reserves wheat pickup time without assuming today's stock is the future supply ceiling. WHEAT renewal planting can also reserve today's feed time despite depleted wheat, avoiding a feed-production deadlock; actual FEED always requires observed carried wheat. This is a bounded route estimate rather than an optimal-routing guarantee; unknown future market hires are not assumed. Impossible already-existing deadlines can still fail, and new planting is restricted instead of expanding that overload. A missed first-loss deadline still permits salvage while positive yield remains.

`lost_harvestable_units` now includes every observed gradual crop-decay decrement as well as remaining yield destroyed in a terminal asset loss. `crop_units_lost_to_decay` and its per-crop dictionary expose spoilage separately. Successful harvested units are subtracted before computing loss, so a final HARVEST followed by an ongoing plant's expiry cannot double-count collected produce. Historical v9 lost-unit values omit gradual spoilage and are not numerically comparable to the new total.

A random weed spawned on a tile cleared by a successful one-time HARVEST or exhausted DIG is excluded from crop loss. Destroying a productive crop with DIG still counts as crop death and lost yield, even if a random weed then occupies that tile.

Animal survival is also a scheduler invariant. FEED with `consecutive_unfed >= 1` is hard-critical because one more missed end-of-day refresh makes the animal escape. Critical WATER and critical FEED form the first scheduler tier. If there is not enough reachable carried WHEAT, WHEAT PICKUP batches become hard survival prerequisites. When shed stock is insufficient, mature field WHEAT HARVEST can also become a survival prerequisite. Both reserve a reachable follow-up FEED action before rollover, outrank ordinary harvest/PPO work, and do not change frozen market orders.

Hard scheduling is therefore lexicographic:

1. critical WATER / critical FEED
2. WHEAT PICKUP / mature WHEAT HARVEST required for critical FEED
3. pressured first-decay HARVEST / final ongoing HARVEST and DIG
4. ordinary productive PPO work
5. PASS

The same reachability/matching preemption logic is used so unrelated committed routes are preserved whenever the remaining workers can still cover the higher-priority survival work.

A deliberate `DIG` of a fully exhausted crop with no remaining yield is valid cleanup, including an ongoing crop's final production day, and receives no crop-death penalty. Destroying a still-productive crop remains a heavy failure.

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

The counters include cause-specific weed diagnostics:

- `crops_to_weed` — total, retained for checkpoint compatibility
- `crops_to_weed_unwatered`
- `crops_to_weed_decay`
- `crops_to_weed_other`

and the production pipeline counters:

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

If a selected task is remote, the emitted environment action is one deterministic Manhattan move toward its target. The network therefore learns **task/worker assignment and task ordering**, not free-form pathfinding. A small `+0.05` shaping reward is attached only to these target-reducing task-route steps. Avoidable PASS is tracked as an actor-only penalty; necessary idle time remains neutral.

One PPO record corresponds to one environment turn. The turn still contains an autoregressive sequence of worker assignments, but PPO **does not form one probability ratio from the sum of all worker log probabilities**.

The environment reward is converted by GAE into one normalized turn-level advantage `A_t`. Each worker subdecision then gets its own actor-only shaping term `B_i`, while the critic remains trained only on the environment return. Each subdecision keeps its own behavior-policy log probability and clipped PPO ratio:

```text
turn t:
    worker subdecision 1 -> old_log_prob_1
    worker subdecision 2 -> old_log_prob_2
    ...
    worker subdecision N -> old_log_prob_N

one turn-level GAE advantage: A_t

actor_advantage_i = clip(A_t + B_i, -5, +5)

B_i examples:
    completed planned PLANT          +8 raw reward-equivalent / GAE std
    completed planned BUILD          +1.00 normalized
    completed planned animal PICKUP  +1.00 normalized
    completed planned PLACE_ANIMAL   +1.50 normalized
    defer feasible PLANT             -0.75 normalized
    avoidable PASS                   -1.00 normalized

For planned PLANT specifically, completion credit is calibrated on the same raw reward scale as crop HARVEST: +8 reward-equivalent units, approximately one normal crop harvest event. PPO divides that +8 by the rollout's raw GAE standard deviation before adding it only to the originating PLANT subdecision. The credit is withheld until the environment confirms the planned tile transition succeeded; merely reaching the tile or emitting PLANT earns nothing.

For a remote planned task, the originating PPO sample is retained while the worker follows its committed route. If the route becomes invalid, is interrupted for critical WATER, or crosses a day boundary, the pending positive credit is discarded. This prevents repeated incomplete route selections from farming planner shaping.

ratio_i = exp(new_log_prob_i - old_log_prob_i)

policy_loss_i =
    -min(
        ratio_i * actor_advantage_i,
        clip(ratio_i, 0.8, 1.2) * actor_advantage_i
    )
```

This avoids the unstable previous formulation:

```text
exp(sum(new_log_prob_i - old_log_prob_i))
```

where modest probability changes compounded exponentially with the number of workers. The critic remains turn-level: there is still one `V(s_t)` and one return target per environment turn. Actor-only planner/PASS shaping never enters that value target.

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

KL is checked before every actor minibatch update. If the current per-subdecision approximate KL is already above `--target-kl`, that minibatch and the remaining actor updates are skipped. Training metrics include `approx_kl`, `clip_fraction`, `ratio_mean`, `ratio_std`, `ratio_min`, `ratio_max`, `actor_samples`, `mean_subdecisions_per_turn`, `actor_bonus_mean`, `actor_bonus_abs_mean`, positive/negative actor-bonus fractions, `actor_minibatches_completed`, `ppo_epochs_completed`, and `kl_early_stop`.

Validation has an automatic catastrophic-collapse guard. `best.pt` selects strictly higher finite deterministic validation worker reward. The reference reward is the higher of the initial deterministic baseline and the selected checkpoint's reward. Weed, spoilage and escape metrics remain explicit diagnostics alongside throughput. If validation falls below:

```text
--collapse-restore-ratio 0.70
```

of that reference, the trainer records `collapse_warning=true`, reloads `best.pt` including optimizer state, and writes the restored policy to `latest.pt`. The raw collapsed post-update checkpoint remains available as `update_NNNN.pt` for diagnosis. Inspect planted/harvested/delivered units alongside losses to ensure progress is not simply reduced activity.

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
python -m unittest discover -p 'test_*.py'
```

Train:

```bash
python train_v25_worker_ppo.py \
  --updates 100 \
  --episodes-per-update 8 \
  --max-training-hours 2
```


This update keeps checkpoint algorithm `v25_static_worker_ppo_gae_v4_animal_reward` and compatible model architecture. When an older checkpoint is loaded under the new reward contract, the **actor weights are kept**, while the critic is reinitialized and stale optimizer/validation-best state is not reused. A fresh baseline accounts for the corrected scheduler and gradual spoilage penalty.

To keep reward contracts separate, the corrected first-decay/capacity runs write by default to `runs/worker_ppo_static_v20_v10_lossless_harvest`. Resuming a v9 checkpoint retains the actor, resets the critic and optimizer, and evaluates a fresh baseline under the new scheduler and complete spoilage accounting. Candidate/global feature dimensions remain unchanged.

`--minibatch-size` now batches worker subdecisions for the actor and turn records for the critic. The default remains 128.

Resume:

```bash
python train_v25_worker_ppo.py \
  --resume runs/worker_ppo_static_v20_v9_animal_survival/checkpoints/latest.pt \
  --output-dir runs/worker_ppo_static_v20_v10_lossless_harvest \
  --updates 800 \
  --episodes-per-update 4 \
  --max-training-hours 6
```

## Outputs

Outputs are written under `runs/worker_ppo_static_v20_v10_lossless_harvest`.

- `metrics.jsonl` — PPO statistics, mean worker reward, and mean reward-component counts.
- `episodes.jsonl` — per-training-replay worker metrics.
- `validation.jsonl` — deterministic held-out aggregate worker metrics.
- `validation_episodes.jsonl` — deterministic held-out per-replay worker metrics.
- `checkpoints/latest.pt` — latest checkpoint.
- `checkpoints/best.pt` — highest finite deterministic held-out worker reward under the current contract. Earlier runs used a survival selection rule; `selected_best_stats()` rejects stale-contract or differently selected best checkpoints.

Primary health metrics are:

- `mean_seeds_planted_total`
- `mean_crop_units_harvested_total`
- `mean_animal_product_units_generated_total`
- `mean_animal_product_units_harvested_total`
- `mean_animal_product_units_moved_to_shed_total`
- `animal_delivery_ratio`
- `critical_feed_share`
- `animal_escape_per_placed`
- `mean_healthy_animal_days`
- `mean_product_units_moved_to_shed_total`
- `mean_animals_escaped`
- `mean_crops_to_weed`
- `mean_crops_to_weed_unwatered`
- `mean_crops_to_weed_decay`
- `mean_crops_to_weed_other`
- `mean_crops_died`
- `mean_lost_harvestable_units`
- `mean_crop_units_lost_to_decay` and `mean_crop_units_lost_to_decay_by_crop`
- `mean_worker_reward`
- `mean_planned_plants_completed`
- `mean_planned_animals_placed`
- `mean_route_progress_steps`
- `mean_avoidable_passes`

The worker policy should improve these operational metrics independently of whether the overall game is ultimately won or lost.


### Reward-semantics correctness

The current contract is `engine-first-decay-capacity-and-spoilage-v9` and retains the earlier first-yield/CARE corrections.

These corrections mean:

- crop HARVEST legality follows the engine's explicit first-yield day rather than the planner's nominal yield schedule;
- CARE reward is assigned when a surviving animal completes a day both fed and cared, so CARE and FEED may occur on different turns of the same day;
- CARE that is never paired with feeding receives no effective-care reward.

A checkpoint with older reward semantics keeps compatible actor weights, while critic/optimizer/best-score state is reset by the existing reward-contract mismatch handling.


## Persistent worker routes

Worker assignment is now treated as a short macro-action rather than a fresh routing decision every turn.

When PPO assigns a worker to a remote task, the worker keeps that task while moving toward its target. The route is released when:

- the worker reaches the target and executes the task;
- the task disappears or becomes infeasible;
- the day changes.

This prevents stochastic rollouts from repeatedly sending the same worker back and forth between unrelated targets. Route continuation does not create a new actor sample; the original task decision receives delayed credit through turn-level GAE.

## Provenance-qualified delivery reward

Explicit shed delivery reward now measures newly produced logistics rather than raw shed traffic.

WHEAT picked up from the shed is tracked per worker as shed-sourced inventory. That WHEAT remains usable for FEED, but it is excluded from delivery candidates and from delivery reward if returned to the shed. If a worker carries a mixture of shed-sourced and newly harvested WHEAT, only the newly harvested quantity is eligible for delivery credit.

Example:

```text
PICKUP 4 WHEAT from shed
PLACE 4 WHEAT back into shed
→ delivery reward = 0
```

while:

```text
carry 4 shed-sourced WHEAT
HARVEST 2 new WHEAT
PLACE 2 WHEAT into shed
→ 2 units receive delivery credit
```

Reward metadata semantics are versioned as `engine-first-decay-capacity-and-spoilage-v9`. Loading an older checkpoint keeps compatible actor weights but establishes fresh critic, optimizer and validation-best state.
