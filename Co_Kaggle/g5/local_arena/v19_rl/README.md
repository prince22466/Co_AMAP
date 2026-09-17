# v19 PPO herd controller

This directory contains the first reinforcement-learning experiment for v19.
It does **not** replace the full v18 agent. Instead, it treats the frozen
`local_arena/v18_c258_compiled.py` policy as the low-level executor and lets a
small PPO actor-critic choose one strategic parameter once per day:

```text
HERD_THRESHOLD in {200, 350, 500, 650, 800}
```

The RL controller is active only on days 3–17, matching the period in which
v18 considers herd expansion. Everything else—forecasting, crop planning,
worker task generation/ranking, logistics, market orders, and action
formatting—remains v18 behavior.

## Why this first

The action space is only five discrete choices and the decision frequency is
15 macro steps per 720-turn game. This gives PPO a tractable delayed-credit
problem while preserving v18's hard-won legal/action logic. The initial actor
is biased toward threshold `500`, so early training stays close to baseline
v18 rather than exploring from a random strategy.

## Install

From `Co_Kaggle/g5`:

```bash
python -m pip install -r local_arena/v19_rl/requirements.txt
```

## Baseline wrapper smoke test

Before training, force the controller to choose the original threshold on all
macro decisions:

```bash
python local_arena/v19_rl/train_v19_ppo.py --smoke-only
```

This checks that a complete Kaggriculture game runs and that the controller
makes exactly one decision on each day 3 through 17, always choosing `500`.
It is an integration check; it is not yet a paired proof that the wrapper is
bit-for-bit behavior-equivalent to the frozen v18 agent.

## Train

```bash
python local_arena/v19_rl/train_v19_ppo.py \
  --updates 100 \
  --episodes-per-update 16
```

The default training opponent mixture is frozen v16, v17, and the v18
executor. Player seat and environment seed are randomized each episode.
Outputs are written under `local_arena/v19_rl/runs/herd_ppo/`:

- `config.json` — exact run configuration and feature/action schemas
- `episodes.jsonl` — episode-level opponent, seat, margin, reward, and chosen thresholds
- `metrics.jsonl` — PPO losses, KL, entropy, action counts, W/L/T, and margins by update
- `checkpoints/update_XXXX.pt` — numbered checkpoints
- `checkpoints/latest.pt` — most recent checkpoint

Resume with:

```bash
python local_arena/v19_rl/train_v19_ppo.py \
  --resume local_arena/v19_rl/runs/herd_ppo/checkpoints/latest.pt \
  --updates 200
```

## State, action, reward

The policy receives 45 compact features derived only from the current live
observation: time, public money/staffing/land, visible herd counts, our private
wheat/fertilizer/seed stocks, crop counts, market prices/inventories, v18
forecast-derived future prices, and visible shop demand.

The action selects one herd threshold. The environment reward is terminal:

```text
sign(our_money - opponent_money)
+ 0.05 * tanh((our_money - opponent_money) / 10000)
```

The first term aligns with win/draw/loss. The small bounded margin term gives
additional learning signal without allowing a rare huge-margin game to
dominate training.

## Promotion criteria

Do not promote a checkpoint because its training win rate is high. Freeze a
candidate, then compare it against unchanged v18 on fresh seeds, both seats,
and multiple frozen opponents using the existing local arena. Keep the old
v16/v17 loss replays as regression/development data, not as the final holdout.

The next v19 step after a genuine held-out gain is to export the trained actor
into a submission-safe inference policy and test it with the official loader.
