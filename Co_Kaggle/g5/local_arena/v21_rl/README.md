# v21 RL: static training on v20 loss histories

v21 keeps the **same residual Double-DQN model structure and constrained
worker-task action space as v20**. It initializes from the exact residual-Q
weights embedded in `submission_nb/kaggriculture-sub_v20.ipynb` and then trains
the residual action-value network with quantization-aware forward passes.

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

The parent/source-of-truth submission is:

```text
submission_nb/kaggriculture-sub_v20.ipynb
```

`train_v21_static_history.py` extracts the notebook's `main.py`, decodes the
embedded `_Q_WEIGHTS_B64` tensor payload, and loads those exact 6,721 float32
parameters into the v20 `ResidualQ` structure (`38 -> 64 -> 64 -> 1`). It does
not require the old v20 `.pt` checkpoint for initialization.


## Quantization

Quantization is part of training and deployment, not a post-training-only step.
The trainer keeps FP32 master parameters for stable Adam updates, but **every Q
forward pass fake-quantizes all three Linear layers' weights and biases** with a
straight-through estimator (STE).

Default:

```text
--quantization fp16
```

Experimental FP8:

```text
--quantization fp8_e4m3fn
```

The current implementation is **weight/bias QAT**. Global-state features, task
features, tanh activations, Bellman targets, gradients, and Adam optimizer state
remain FP32. This avoids unstable low-precision optimizer updates while making
the learned policy adapt to the exact low-precision parameter grid used at
deployment.

The checkpoint stores both FP32 master weights and a quantized
`deployment_state_dict`. The final exporter writes the deployment weights in
the selected representation:

- FP16: 2 bytes/parameter using IEEE-754 binary16.
- FP8: 1 byte/parameter using E4M3FN-style finite FP8.

`export_v21_submission.py` replaces the residual-Q tensor inside the v20
submission template and creates `main.py` plus `submission.tar.gz`. The final
submission therefore stores the same quantized parameter values used during
QAT.

## Safety/parity gates

Before any optimizer step, `train_v21_static_history.py` checks every selected
v20 history:

1. replaying both recorded action streams must reproduce the saved replay;
2. the checked-in `kaggriculture-sub_v20.ipynb` must reproduce the recorded v20
   action stream and terminal rewards exactly;
3. an unquantized `ResidualQ` reconstructed from that notebook's embedded
   weights must also reproduce the recorded v20 trajectory exactly.

The quantized v21 initialization is deliberately **not** required to reproduce
v20 exactly, because FP16/FP8 rounding may change candidate ordering. Its
performance is measured as the initial v21 validation baseline.

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
- same v20 residual Double-DQN hyperparameters
- FP16 weight/bias QAT by default; FP8 E4M3FN available with `--quantization fp8_e4m3fn`
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
