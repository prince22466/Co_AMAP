# Kaggle Interactive Experiment Feedbacks

**Base model:** `Nemotron-3-Nano-30B`

Use this file to record each Kaggle interactive notebook run and the feedback it gives about fine-tuning time, stopping behavior, and configuration choices.

## Summary Table

| Exp ID | Device | Method | LoRA Parameters | Training Config | Data Config | Runtime | Notes |
|---|---|---|---|---|---|---:|---|
| LoRA-001 | RTX Pro 6000 | LoRA SFT | `r=4`; `alpha=8`; `targets=in_proj,out_proj,up_proj,down_proj`; `dropout=0.05`; `bias=none`; `task=CAUSAL_LM` | `epochs=1`; `max_steps=200`; `batch=1`; `grad_accum=8`; `lr=1e-4`; `max_length=512`; `max_new_tokens=32`; `eval_every=10`; `early_val_rows=50`; `stop_when_val_worse=True`; `save_adapter=True` | 80/20 split on full `train.csv` | 45m | Initial fast LoRA run |

## Experiment Details

### LoRA-001

**Device:** RTX Pro 6000

**Method:** LoRA supervised fine-tuning

**Data config:** 80/20 split on full `train.csv`

**Training runtime on Kaggle:** 45m

**LoRA parameters:**

| Parameter | Value |
|---|---|
| `LORA_RANK` | `4` |
| `LORA_ALPHA` | `8` |
| `TARGET_MODULES` | `r".*\.(in_proj\|out_proj\|up_proj\|down_proj)$"` |
| `LORA_DROPOUT` | `0.05` |
| `LORA_BIAS` | `"none"` |
| `TASK_TYPE` | `TaskType.CAUSAL_LM` |

**Training config:**

| Parameter | Value |
|---|---|
| `TRAIN_FRACTION` | `0.80` |
| `EPOCHS` | `1` |
| `MAX_STEPS` | `200` |
| `BATCH_SIZE` | `1` |
| `GRADIENT_ACCUMULATION_STEPS` | `8` |
| `LEARNING_RATE` | `1e-4` |
| `MAX_LENGTH` | `512` |
| `MAX_NEW_TOKENS` | `32` |
| `USE_EARLY_STOPPING` | `True` |
| `EVAL_EVERY_STEPS` | `10` |
| `EARLY_STOPPING_VAL_ROWS` | `50` |
| `STOP_WHEN_VAL_WORSE` | `True` |
| `SAVE_ADAPTER` | `True` |
| `ADAPTER_OUTPUT_DIR` | `OUTPUT_DIR` |

**Notes:**

- Hard stop is `MAX_STEPS = 200`.
- Validation checks use a small subset of 50 rows for faster feedback.
- Stop condition is either max steps reached or validation accuracy worse than the best observed validation accuracy.
