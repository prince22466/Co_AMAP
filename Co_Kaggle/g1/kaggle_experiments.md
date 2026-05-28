# Kaggle Interactive Experiment Feedbacks

**Base model:** `Nemotron-3-Nano-30B`

Use this file to record each Kaggle interactive notebook run and the feedback it gives about fine-tuning time, stopping behavior, and configuration choices.

## Summary Table

| Exp ID | Device | Method | LoRA Parameters | Training Config | Data Config | Runtime | Notes |
|---|---|---|---|---|---|---:|---|
| LoRA-001 | RTX Pro 6000 | LoRA SFT | `r=4`; `alpha=8`; `targets=in_proj,out_proj,up_proj,down_proj`; `dropout=0.05`; `bias=none`; `task=CAUSAL_LM` | `epochs=1`; `max_steps=200`; `batch=1`; `grad_accum=8`; `lr=1e-4`; `max_length=512`; `max_new_tokens=32`; `eval_every=10`; `early_val_rows=50`; `stop_when_val_worse=True`; `save_adapter=True` | 80/20 split on full `train.csv` | 45m | early stop,; due to hitting max_steps=200, ;which is 2.6%(200 / 7600) of 1 epoch; batch=1 means 1 row per step |
