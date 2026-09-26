# v21 RL: static training on v20 loss histories

v21 keeps the same constrained worker-task candidate space and Double-DQN
training loop, but **removes the legacy task-tree scorer completely**. The
neural network is now the full action-value function used for candidate ranking,
bootstrap selection, Bellman targets, validation, and final inference.

It still initializes the 38 -> 64 -> 64 -> 1 network from the weights embedded
in `submission_nb/kaggriculture-sub_v20.ipynb`, then continues training on v20
loss histories.

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
QNetwork:
    concat(global_state, task_features)
      -> Linear(hidden=64)
      -> Tanh
      -> Linear(64)
      -> Tanh
      -> Linear(1)

Q(s,a) = neural_network(global_state, task_features)

There is no `learned_task_score()`, no normalized tree prior, and no
tree-based tie-breaking or bootstrap ranking in v21.
```

The parent weight source is:

```text
submission_nb/kaggriculture-sub_v20.ipynb
```

`train_v21_static_history.py` extracts the notebook's `main.py`, decodes the
embedded `_Q_WEIGHTS_B64` tensor payload, and loads those exact 6,721 float32
parameters into the v20 `ResidualQ` structure (`38 -> 64 -> 64 -> 1`). It does
not require the old v20 `.pt` checkpoint for initialization.



## Tree removal

Training uses a tree-free executor. The v19 notebook is still used as the base
strategy/executor for non-ranking game logic and legal task generation, but
before execution v21 replaces the old task-selection block and removes:

- all `_tree_*` functions;
- `_TREE_FUNCTIONS`;
- `learned_task_score()`.

The final exporter performs the same cleanup and additionally removes the old
normalized-prior helper and residual-pruning bound. The exported v21 `main.py`
therefore ranks legal candidates directly with the neural Q-network.

## Quantization

Quantization is part of training and deployment, not a post-training-only step.
By default, v21 uses **true FP16 end-to-end network training**: model parameters,
forward activations, Q values, TD targets, gradients, and Adam moment tensors are
all FP16. There is no FP32 master-weight copy in default FP16 mode.

Default:

```text
--quantization fp16
```

Experimental FP8:

```text
--quantization fp8_e4m3fn
```

FP8 remains experimental and uses weight/bias QAT. In contrast, the default FP16
path converts the actual trainable network and Double-DQN tensors to FP16 and uses
a pure-FP16 Adam implementation. The only non-FP16 pieces are non-floating-point
bookkeeping such as Python step counters and external environment/replay metadata.

The checkpoint stores the actual FP16 model state in default mode plus a matching
`deployment_state_dict`. The final exporter writes the deployment weights in
the selected representation:

- FP16: 2 bytes/parameter using IEEE-754 binary16; exported residual-Q inference also rounds inputs, accumulations, and tanh outputs to FP16.
- FP8: 1 byte/parameter using E4M3FN-style finite FP8.

`export_v21_submission.py` replaces the residual-Q tensor inside the v20
submission template and creates `main.py` plus `submission.tar.gz`. The final
submission therefore stores the same quantized parameter values used during
QAT.

## Safety/parity gate

Before any optimizer step, `train_v21_static_history.py` runs **one**
deterministic preflight history (the first sorted `game_history/v20/*.json`).

That single preflight verifies:

1. the embedded model payload decodes to exactly 6,721 parameters with the
   expected `38 -> 64 -> 64 -> 1` structure;
2. replaying both recorded action streams reproduces that saved replay exactly;
3. the checked-in `kaggriculture-sub_v20.ipynb` reproduces the recorded v20
   action stream and terminal rewards on that replay.

The separate PyTorch reconstruction is **not** required to reproduce every v20
action. v21 deliberately changes the action scorer from tree-prior-plus-residual
to neural-Q-only, so exact v20 action parity is neither expected nor required.
The decoded-weight check is a model-format check.

The quantized v21 initialization is measured separately as the initial
validation baseline.

## Train

From `Co_Kaggle/g5`:

```bash
python -m pip install -r local_arena/v21_rl/requirements.txt
python local_arena/v21_rl/train_v21_static_history.py
```

Defaults:

- histories: `game_history/v20/*.json`
- parent/source of truth: `submission_nb/kaggriculture-sub_v20.ipynb`
- deterministic 80/20 replay-file train/validation split
- same Double-DQN training machinery and candidate-generation rules
- neural Q-network is the only action scorer; tree ensemble is removed
- FP16 weight/bias QAT by default; FP8 E4M3FN available with `--quantization fp8_e4m3fn`
- 8 replay episodes per update
- held-out static validation every update
- target validation win rate: configured by `--target-win-rate`
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


## Delivered-value reward (superseded by dense worker reward)

v21 training now optimizes worker logistics/production using cumulative delivered product value rather than money margin or terminal win/loss.

At each environment step, newly produced goods that are physically deposited into the shed are valued using the live market price from that turn. The resulting increment is used as the TD reward after scaling/clipping. Same-hour internal worker assignments keep zero immediate reward and discount 1, preserving the existing transition structure.

Only cargo originating from production actions such as `HARVEST` and `COLLECT_FERTILIZER` is reward-eligible. Goods picked up from the shed are never marked eligible, so a `PICKUP -> DROP` loop cannot manufacture reward. The terminal game win/loss bonus is removed from worker-Q training.

Checkpoint algorithm tag:

`v21_static_pure_q_worker_credit_balanced_double_dqn`


## Dense worker reward

The worker-allocation Q-network now receives a dense reward aligned with the actions it controls:

```text
raw_reward_value =
    1.00 * produced_value
  + 0.05 * transport_progress_value
  + 0.25 * delivered_value

reward = clip(raw_reward_value / reward_scale)
```

Default `--reward-scale` is now `1000`.

- **Production:** `HARVEST` and `COLLECT_FERTILIZER` receive immediate live-price product value.
- **Transport:** reward-eligible produced cargo receives signed shaping for moving closer to/farther from the shed.
- **Delivery:** reward-eligible cargo deposited into the shed receives an additional completion bonus.
- **No SELL reward:** market selling remains outside the worker-Q action space.
- **No terminal win/loss bonus:** terminal game outcome is still reported for evaluation but does not train worker allocation.
- **Anti-loop guard:** only production-origin cargo is reward-eligible, so shed `PICKUP -> DROP` cycles do not create product reward.
- **Day rollover:** automatic end-of-day shed deposit of eligible carried cargo is counted as delivery.

Training logs now expose production, delivery, positive-reward fraction, mean reward, and mean absolute TD error so reward flow is immediately visible.

Checkpoint algorithm tag:

`v21_static_pure_q_worker_credit_balanced_double_dqn`


## Per-worker credit assignment and balanced replay

The dense worker reward is now credited to the exact worker Q-decision that caused it.

For one environment hour, v21 records the worker assignments selected by the Q-network, runs the environment step, computes production/transport/delivery value per worker, and writes each worker's realized reward back to that worker's DecisionRecord before building replay transitions.

This removes the previous same-hour credit bug where the entire hour's reward landed on whichever assignment happened to be last in the record sequence.

Training replay is also stratified. By default:

```text
--rewarded-replay-fraction 0.50
```

Each SGD batch targets up to 50% non-zero-reward transitions and fills the remainder from neutral transitions. If there are not enough rewarded samples, the neutral pool fills the shortage.

Training output now includes:

```text
replay_rewarded=...
batch_rewarded=...
```

alongside the existing reward and TD diagnostics.

Checkpoint algorithm tag:

`v21_static_pure_q_worker_credit_balanced_double_dqn`
