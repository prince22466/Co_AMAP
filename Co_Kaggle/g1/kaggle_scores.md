# Kaggle Experiment Scores


**Base model:** `Nemotron-3-Nano-30B`


Use this file to record each Kaggle submission or serious notebook run. Keep one row per experiment, then add short details below when needed.

## Summary Table

| Exp ID | Kaggle Score | Method | Adapter / Output | Training Runtime on Kaggle | Notes |
|---|---:|---|---|---:|---|
| Base model | 0.5 | NA | NA | 0h | base model as benchmark |
| LoRA-001 | TBD | LoRA SFT | TBD | TBD |  |

## Experiment Details

### LoRA-001

**Kaggle score:** TBD

**Fine-tuning method:** LoRA supervised fine-tuning


**Notebook:** `nvidia-nemotron-submission-lora.ipynb`

**Adapter/component:** `train_LoraAdaptors/LoRA_001.py`

**LoRA parameters:**

| Parameter | Value |
|---|---:|
| `LORA_RANK` | `4` |
| `LORA_ALPHA` | `8` |
| `TARGET_MODULES` | `.*\.(in_proj\|out_proj\|up_proj\|down_proj)$` |
| `LORA_DROPOUT` | `0.05` |
| `LORA_BIAS` | `none` |
| `TASK_TYPE` | `CAUSAL_LM` |

**Training parameters:**

| Parameter | Value |
|---|---:|
| `TRAIN_FRACTION` | `0.80` |
| `EPOCHS` | `1` |
| `MAX_STEPS` | `200` |
| `BATCH_SIZE` | `1` |
| `GRADIENT_ACCUMULATION_STEPS` | `8` |
| `LEARNING_RATE` | `1e-4` |
| `MAX_LENGTH` | `512` |
| `MAX_NEW_TOKENS` | `32` |
| `EVAL_EVERY_STEPS` | `10` |
| `EARLY_STOPPING_VAL_ROWS` | `50` |
| `STOP_WHEN_VAL_WORSE` | `True` |

**Run record:**

| Field | Value |
|---|---|
| Start time | TBD |
| End time | TBD |
| Runtime | TBD |
| Stopped by | TBD |
| Final train summary | TBD |
| Local validation score before training | TBD |
| Local validation score after training | TBD |
| Kaggle public/private score | TBD |

**Observations:**

- TBD

**Next changes to try:**

- TBD
