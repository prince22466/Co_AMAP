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


## Train on real game-history seeds until >70% vs v19

Use the dynamic-rematch protocol described in `game_history/readme.md`: extract the actual game seed from each replay JSON and rerun v20 against frozen v19.

```bash
python local_arena/v20_rl/train_v20_history.py
```

Defaults:

- scans every JSON under `game_history/`
- reads `seed`, `randomSeed`, or `random_seed` from replay `info` / `configuration`
- deduplicates the real seeds
- deterministic 80/20 train/validation split using `--split-seed 20260919`
- trains only on the training seeds
- opponent is frozen v19 only
- randomly trains from either player seat
- evaluates every update on **all held-out seeds from both seats**
- stops when held-out deterministic win rate is **strictly greater than 70%**, or when wall-clock training time reaches **2 hours**
- the two break conditions are deliberately defined at the top of `main()` in `train_v20_history.py`:

```python
TARGET_WIN_RATE = 0.70
MAX_TRAINING_HOURS = 2.0
```

- saves the passing model as `runs/history_vs_v19/checkpoints/target.pt`
- saves a timeout checkpoint as `runs/history_vs_v19/checkpoints/timeout.pt`

The exact split is written to:

```text
local_arena/v20_rl/runs/history_vs_v19/history_seed_split.json
```

The passing evaluation is written to:

```text
local_arena/v20_rl/runs/history_vs_v19/TARGET_REACHED.json
```

To continue a run that hits the 2-hour timeout before 70%:

```bash
python local_arena/v20_rl/train_v20_history.py \
  --resume local_arena/v20_rl/runs/history_vs_v19/checkpoints/timeout.pt
```

For later experiments, edit only `TARGET_WIN_RATE` and `MAX_TRAINING_HOURS` in `main()`.

This intentionally keeps validation seeds out of PPO updates, so the 70% threshold is not measured on training games.


## Residual Q-learning experiment

`train_v20_q_history.py` is the value-based alternative to the PPO worker-task selector.

It keeps the same frozen v19 strategy and legal candidate generation, but treats
`learned_task_score()` as a fixed action-ranking prior and learns only a neural
Q-value correction:

```text
Q(s,a) = normalized_v19_task_score(s,a) + neural_residual(s,a)
```

The final residual layer is initialized to zero, so deterministic inference starts
with the exact v19 candidate ordering.

Training uses:

- residual **Double-DQN**
- experience replay
- a separate target network
- conservative **top-k epsilon-greedy** exploration
- default epsilon from 2.0% down to 0.5%
- exploration restricted to the current top 3 Q-ranked candidates
- dense reward from the change in money margin between environment hours
- terminal +1 / 0 / -1 for win / tie / loss
- Huber TD loss
- gradient clipping
- held-out real-history seeds, evaluated from both seats against frozen v19
- automatic `best.pt` checkpoint preservation

Run from `Co_Kaggle/g5`:

```bash
python local_arena/v20_rl/train_v20_q_history.py
```

Outputs are isolated from PPO under:

```text
local_arena/v20_rl/runs/history_q_vs_v19/
```

Important outputs:

```text
config.json
history_seed_split.json
episodes.jsonl
metrics.jsonl
validation.jsonl
validation_games.jsonl
BEST.json
checkpoints/latest.pt
checkpoints/best.pt
checkpoints/timeout.pt
checkpoints/target.pt
```

The default stopping conditions are intentionally visible at the top of
`main()`:

```python
TARGET_WIN_RATE = 0.70
MAX_TRAINING_HOURS = 2.0
VALIDATE_EVERY_UPDATES = 1
```

To resume after timeout:

```bash
python local_arena/v20_rl/train_v20_q_history.py \
  --resume local_arena/v20_rl/runs/history_q_vs_v19/checkpoints/timeout.pt
```

The checkpoint restores model, target network, optimizer, and RNG states. The
replay buffer itself is not checkpointed, so resumed training rebuilds replay
experience from new episodes before further replay updates.
