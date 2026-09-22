# v22 RL: FP16 PPO for selling only

v22 trains **only the `SELL` part of the v20 `market_orders()` function**. The
checked-in `submission_nb/kaggriculture-sub_v20.ipynb` remains the source of
truth for every other game decision.

Frozen v20 behavior includes worker/task allocation, production, crop and animal
planning, movement, hiring, seed/product/animal purchases, land purchases and all
non-SELL market logic. v22 does not import or use the v21 Q-network.

## Training protocol

Training uses the same static counterfactual replay idea as v21:

```text
game_history/v20 replay
    -> v22 runs the v20 agent
    -> only the SELL loop is replaced by v22 PPO
    -> historical opponent action is replayed verbatim
    -> Kaggriculture environment advances
```

The recorded opponent is non-adaptive after v22 changes the trajectory. This is
failure-scenario training/regression, not a live rematch.

## PPO action

The policy produces one factorized categorical decision for each product:

```text
0 = hold
1 = sell 25%
2 = sell 50%
3 = sell 75%
4 = sell 100%
```

Products:

```text
MILK, WOOL, STRAWBERRY, MELON, EGG,
TOMATO, CARROT, FERTILIZER, WHEAT
```

Duplicate quantities are masked for small inventories, so a product with one
sellable unit has only the distinct `hold` and `sell 100%` actions available.

## Hard selling constraints

PPO acts only on legally sellable inventory.

- WHEAT reserve: `max(4, live_animals + 2)`.
- FERTILIZER reserve: v20 behavior before day 10; reserve 4 from day 10 onward.
- Both WHEAT and FERTILIZER reserves are released **only in the final 5 turns of
  the complete episode**.
- Shed capacity is treated as 100 units. Effective stock and overflow pressure
  are explicit observations; expected overflow after the selected sale is also
  logged and receives a small shaping penalty because excess inventory is
  discarded without sale revenue.

The reserve constraints are outside PPO, so the policy cannot sell protected
WHEAT/FERTILIZER.

## Observation state

The PPO state contains global game/economic state plus per-product inventory and
price history.

Global features include:

```text
day, hour, turn, remaining turns,
own money, opponent money, money margin,
effective shed stock, capacity remaining, overflow,
current sellable inventory value, final-5-turn flag
```

For every product the state includes:

```text
sellable quantity
held quantity
total quantity
reserve quantity
current price
1-day moving average (24 turns)
3-day moving average (72 turns)
5-day moving average (120 turns)
5-day minimum / maximum
current price vs 5-day mean
current price vs 5-day maximum
1-turn price change
1-day price change
1-day EMA
5-day EMA
short/long EMA spread
5-day price percentile
```

Early in an episode each statistic uses the price history available so far.

## Exploration

Training uses an explicit PPO-compatible exploration mixture:

```text
70% learned policy
30% forced legal non-greedy alternatives
```

The 30% exploration mass is distributed only across legal actions that differ
from the policy's current greedy action. Deterministic validation does not use
this exploration mixture.

This is implemented inside the behavior distribution itself, so stored PPO
log-probabilities and later PPO probability ratios remain mathematically
consistent. It is configurable with `--exploration-rate` and defaults to
`0.30`.

## Reward

The reward is dominated by the final game result:

```text
terminal = win/loss/tie + bounded final-margin bonus
```

Defaults:

```text
win  = +1
loss = -1
tie  =  0

margin bonus
  = 0.25 * tanh(final_margin / 10000)

improvement bonus
  = 0.50 * tanh((final_margin - original_v20_margin) / 5000)
```

The per-history improvement term gives PPO a useful signal when a v22 selling
policy is still losing but has repaired a substantial part of the original v20
loss. Win/loss remains the dominant objective.

Intermediate shaping is intentionally tiny relative to the terminal objective:

- sale-price quality based on recent price percentile;
- expected 100-unit-capacity overflow penalty.

Worker production/delivery values are **not** PPO rewards.

## FP16

The learnable v22 path is FP16 end to end:

- PPO model parameters;
- forward activations;
- actor logits and critic values;
- stored/optimized returns and advantages;
- PPO losses and gradients;
- SGD updates and gradients.

v22 now uses **plain SGD with no momentum**. This avoids Adam's FP16 first/second
moment buffers and keeps the optimizer state minimal. The default learning rate
is `1e-2`, intentionally larger than the previous Adam `3e-4` because direct
FP16 SGD updates otherwise risk becoming too small to change parameters.

There is no FP32 master-weight copy.

FP16 is expected to be useful primarily on hardware with fast half-precision
execution; CPU FP16 is supported as a correctness path but is not guaranteed to
be faster than FP32.

## v21-style worker telemetry, without the v21 model

v22 reproduces the data-collection logic used in v21 so training/evaluation logs
continue to expose:

```text
produced_value
produced_units
transport_progress_value
delivered_value
delivered_units
produced_value_by_product
delivered_value_by_product
```

This is diagnostics only. The v21 Q-network, worker selector, worker replay,
Double-DQN updates and v21 weights are not used by v22.



## Neutral actor initialization

The PPO actor no longer starts with a +2 logit bias toward `SELL 100%`.

Instead:

```text
actor weights: orthogonal initialization, gain = 0.01
actor biases:  0
```

This makes the initial learned policy approximately neutral across legal sell
fractions while retaining small state-dependent differences. The existing
forced-v20-baseline preflight remains responsible for verifying that the
surgically replaced SELL loop can reproduce v20 exactly.

Because the checkpoint algorithm ID changed, checkpoints from the older
SELL-100-biased initialization are intentionally incompatible. Start a fresh
training run after merging this change.

## Learning diagnostics

Each PPO update now records behavior-change diagnostics in `metrics.jsonl`:

```text
sell_0_pct
sell_25_pct
sell_50_pct
sell_75_pct
sell_100_pct

greedy_sell_0_pct
greedy_sell_25_pct
greedy_sell_50_pct
greedy_sell_75_pct
greedy_sell_100_pct

policy_entropy
approx_kl
clip_fraction
actor_parameter_delta_l2
actor_parameter_delta_relative

mean_margin_improvement_vs_v20
margin_improved_cases
margin_worsened_cases
```

The `sell_*_pct` fields describe the actual sampled training behavior, including
the configured 30% exploration mixture. The `greedy_sell_*_pct` fields rerun
the collected states through the **post-update deterministic policy**, so they
show whether the learned model itself is changing independently of exploration.

Validation logs also include deterministic sell-action percentages. If
`greedy_sell_100_pct` and validation `sell_100_pct` remain near 1.0 for many
updates, the actor has not meaningfully escaped the original v20 sell-all policy.

`approx_kl` measures policy movement during PPO optimization,
`clip_fraction` shows how often PPO ratios hit the clipping region, and
`actor_parameter_delta_relative` measures the relative L2 movement of the
actor parameters during one update.

## Safety/parity gate

Before training, one v20 history must pass three checks:

1. feeding both recorded action streams back into Kaggriculture reproduces the
   saved replay exactly;
2. the checked-in v20 notebook reproduces the recorded v20 action stream and
   final rewards exactly;
3. the surgically modified v22 executor in `forced_v20_baseline` mode reproduces
   the same action stream and rewards exactly.

The third gate verifies that replacing the SELL loop does not alter any other
v20 behavior.

## Train

From `Co_Kaggle/g5`:

```bash
python -m pip install -r local_arena/v22_rl/requirements.txt
python local_arena/v22_rl/train_v22_selling_history.py
```

Preflight only:

```bash
python local_arena/v22_rl/train_v22_selling_history.py --preflight-only
```

Resume:

```bash
python local_arena/v22_rl/train_v22_selling_history.py \
  --resume local_arena/v22_rl/runs/static_v20_history/checkpoints/latest.pt
```

Defaults:

- histories: `game_history/v20/*.json`;
- parent/source of truth: `submission_nb/kaggriculture-sub_v20.ipynb`;
- deterministic 80/20 replay-file train/validation split;
- FP16 PPO, hidden size 128;
- plain SGD, learning rate `1e-2`;
- 8 static replay episodes/update;
- validation every update;
- target validation win rate 0.60;
- maximum training time 2 hours.

## Evaluate

```bash
python local_arena/v22_rl/evaluate_v22_v20_losses.py \
  --checkpoint local_arena/v22_rl/runs/static_v20_history/checkpoints/latest.pt
```

The evaluator reports the v20 and v22 terminal margins, margin improvement,
selling activity, expected overflow, and the same production/delivery telemetry
for every replay.


## PPO target-KL guard

The trainer supports:

```text
--target-kl 0.01
```

During PPO optimization, the current minibatch approximate KL is measured before
the optimizer step. If it exceeds the target, the current step is skipped and
the remaining minibatches/epochs for that PPO update are stopped.

Set `--target-kl <= 0` to disable the guard.

Diagnostics include:

```text
kl_guard_triggered
kl_guard_value
```

This is intended to prevent a single rollout batch from driving an excessively
large policy shift when using FP16 + SGD.

