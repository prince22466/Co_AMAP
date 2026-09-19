# v20 naive constrained-action PPO

This directory contains the first **per-turn constrained RL** experiment built on the v19 Kaggriculture policy.

Unlike `v19_rl`, which only learns `HERD_THRESHOLD` once per day, v20 keeps v19's forecasting, crop planning, livestock planning, market policy, legality checks, task generation, resource constraints, and movement/action execution frozen. PPO only replaces the greedy worker-task selector inside `unit_actions()`.

Each 720-turn game is therefore:

```text
observation
  -> frozen v19 strategic logic
  -> frozen v19 legal task generation
  -> valid (worker, task) candidates
  -> PPO categorical choice
  -> frozen v19 executor
  -> next hourly observation
```

The selector is applied autoregressively inside each hourly turn. Once a worker-task pair is chosen, v19 removes that worker and the chosen tile, then v20 chooses again from the remaining valid pairs. The environment itself still advances only once per hour.

## Action constraints

The policy never receives illegal pairs. v19 still filters candidates using:

- resource availability such as WHEAT/FERTILIZER,
- reachability before the end of the day,
- crop/animal state,
- already reserved workers/tiles,
- hard livestock staging and urgent logistics rules.

So this is RL over the **feasible action set**, not RL over arbitrary Kaggriculture commands.

## Policy

The actor receives the same 21 task features already used by v19's embedded tree ranker. The v19 tree score is kept as a fixed prior logit and the neural actor learns a residual correction:

```text
policy_logit = baseline_scale * v19_tree_score + neural_residual(task_features)
```

At initialization the neural residual is zero, so deterministic inference reproduces the original v19 ranking.

The critic uses a compact global observation vector containing time, money margin, staffing, land, herd counts, resource stocks, and selected market prices.

## Reward

The first experiment deliberately uses a sparse terminal reward:

```text
sign(our_money - opponent_money)
+ 0.05 * tanh((our_money - opponent_money) / 10000)
```

Each internal decision receives that terminal return discounted by the actual game turn.

## Install

From `Co_Kaggle/g5`:

```bash
python -m pip install -r local_arena/v20_rl/requirements.txt
```

## Baseline smoke test

```bash
python local_arena/v20_rl/train_v20_ppo.py --smoke-only
```

This runs the patched selector in forced-baseline mode. The selected pair uses exactly v19's original tuple ordering `(tree_score, -distance, -worker, -task)`.

## Train

```bash
python local_arena/v20_rl/train_v20_ppo.py \
  --updates 100 \
  --episodes-per-update 8
```

Outputs go to `local_arena/v20_rl/runs/task_ppo/`:

- `config.json`
- `episodes.jsonl`
- `metrics.jsonl`
- `checkpoints/update_XXXX.pt`
- `checkpoints/latest.pt`

## Evaluate against frozen v19

```bash
python local_arena/v20_rl/evaluate_v20.py \
  --checkpoint local_arena/v20_rl/runs/task_ppo/checkpoints/latest.pt \
  --games 20
```

Twenty seeds are evaluated from both seats, for 40 games total.

## Scope

v20.0 only asks whether sparse end-of-game RL can improve v19's constrained worker-task decisions. Crop selection, animal expansion, hiring, land purchases, and market orders remain v19 logic and can become later policy heads if this baseline learns.
