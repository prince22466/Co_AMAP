# v21 RL: static training on v20 loss histories

v21 keeps the **same residual Double-DQN model structure and constrained
worker-task action space as v20**. It is warm-started from the selected v20 Q
checkpoint and trains only the residual action-value network.

The change in v21 is the training protocol:

```text
game_history/v20 replay
    -> v21 chooses the candidate-side actions
    -> historical opponent action for that replay step is copied verbatim
    -> Kaggriculture environment advances
    -> transitions go into the same Double-DQN replay buffer used by v20
```

This is **static counterfactual training**. The opponent does not recompute its
policy after v21 changes the trajectory. This follows the project definition in
`game_history/readme.md` and the replacement-replay pattern in:

- `local_arena/v19_rl/evaluate_v19_v18_losses.py`
- `local_arena/v20_rl/evaluate_v20_v19_losses.py`

## Model

The model is imported directly from `v20_rl/train_v20_q_history.py`:

```text
ResidualQ:
    concat(global_state, task_features)
      -> Linear(hidden=64)
      -> Tanh
      -> Linear(64)
      -> Tanh
      -> Linear(1)

Q(s,a) = normalized_v19_task_score(s,a) + neural_residual(s,a)
```

The default initialization checkpoint is:

```text
local_arena/v20_rl/runs/history_q_vs_v19/checkpoints/update_0009.pt
```

## Safety/parity gates

Before any optimizer step, `train_v21_static_history.py` checks every selected
v20 history:

1. replaying both recorded action streams must reproduce the saved replay;
2. the selected v20 checkpoint + frozen v19 executor must reproduce the
   recorded v20 action stream and terminal rewards exactly.

## Train

From `Co_Kaggle/g5`:

```bash
python -m pip install -r local_arena/v21_rl/requirements.txt
python local_arena/v21_rl/train_v21_static_history.py
```

Defaults:

- histories: `game_history/v20/*.json`
- parent: v20 `update_0009.pt`
- deterministic 80/20 replay-file train/validation split
- same v20 residual Double-DQN hyperparameters
- 8 replay episodes per update
- held-out static validation every update
- target validation win rate: >70%
- maximum training time: 8 hours

Useful commands:

```bash
python local_arena/v21_rl/train_v21_static_history.py --preflight-only

python local_arena/v21_rl/train_v21_static_history.py \
  --resume local_arena/v21_rl/runs/static_v20_history/checkpoints/latest.pt
```

## Evaluate

```bash
python local_arena/v21_rl/evaluate_v21_v20_losses.py \
  --v21-checkpoint local_arena/v21_rl/runs/static_v20_history/checkpoints/latest.pt
```

The evaluator requires exact saved-replay parity, exact parent-v20 policy parity,
then runs deterministic v21 while replaying the historical opponent actions.

Static replay is intentionally non-adaptive. Use it as failure-scenario
training/regression; dynamic rematches should remain a separate final
validation step.
