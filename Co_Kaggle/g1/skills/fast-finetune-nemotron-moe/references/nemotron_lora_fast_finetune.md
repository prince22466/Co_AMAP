# Nemotron LoRA Fast Fine-Tuning Reference

## Competition constraints

- Final submission is a LoRA adapter for `Nemotron-3-Nano-30B`.
- Submission must include `adapter_config.json`.
- Official inference uses vLLM with LoRA enabled.
- Official parameters from evaluation material:
  - `max_lora_rank = 32`
  - `temperature = 0.0`
  - `top_p = 1.0`
  - `max_tokens = 7680`
  - `max_num_seqs = 64`
  - `gpu_memory_utilization = 0.85`
  - `max_model_len = 8192`

## Metric alignment

Validation should mimic the official metric:

1. Extract answer from model output:
   - prioritize final non-empty content inside `\boxed{...}`;
   - fallback to phrases like `Final answer: ...`;
   - fallback to the last numeric value;
   - fallback to the last non-empty line.
2. Verify:
   - if ground truth is binary (`[01]+`), compare exact strings;
   - else try numeric comparison with `math.isclose(..., rel_tol=1e-2, abs_tol=1e-5)`;
   - else compare lowercased strings.
3. Score is `correct / total`.

Training prompt should match the evaluator intent:

```python
def format_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n"
        "Please put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`\n"
    )

def format_training_text(prompt: str, answer: str) -> str:
    return f"{format_prompt(prompt)}\\boxed{{{answer}}}"
```

## Fast Kaggle stopping strategy

Use two exit conditions:

- hard stop by training batches:

```python
if config["MAX_STEPS"] is not None and steps >= config["MAX_STEPS"]:
    stopped_by = "max_steps"
```

- stop when validation gets worse:

```python
if val_score["accuracy"] < best_val_accuracy:
    stopped_by = "validation_worse"
```

Good starting config for fast feedback:

```python
KAGGLE_TRAINING_CONFIG = {
    "TRAIN_FRACTION": 0.80,
    "EPOCHS": 1,
    "MAX_STEPS": 200,
    "BATCH_SIZE": 1,
    "GRADIENT_ACCUMULATION_STEPS": 8,
    "LEARNING_RATE": 1e-4,
    "MAX_LENGTH": 512,
    "MAX_NEW_TOKENS": 32,
    "USE_EARLY_STOPPING": True,
    "EVAL_EVERY_STEPS": 10,
    "EARLY_STOPPING_VAL_ROWS": 50,
    "STOP_WHEN_VAL_WORSE": True,
}
```

Remember: with gradient accumulation, `steps` count batches, while `optimizer_updates` count optimizer updates.

## LoRA targets

Before setting `TARGET_MODULES`, inspect actual linear module names:

```python
for name, module in model.named_modules():
    lower = name.lower()
    if isinstance(module, nn.Linear) and any(
        key in lower for key in ["proj", "attn", "mlp", "moe", "expert", "gate"]
    ):
        print(name, module)
```

For this project a known starting point is:

```python
TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"
```

Keep `LORA_RANK <= 32` for official compatibility.

## Kaggle Triton/Mamba permission fix

If generation fails with a permission error for `ptxas-blackwell`, run before model loading/generation:

```python
import os
import shutil
from pathlib import Path

src = Path(
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/"
    "triton/backends/nvidia/bin/ptxas-blackwell"
)
dst = Path("/kaggle/working/ptxas-blackwell")

shutil.copy(src, dst)
os.chmod(dst, 0o755)
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(dst)
```

## Adapter saving

Only save after confirming the training loop works:

```python
if TRAINING_CONFIG["SAVE_ADAPTER"]:
    model.save_pretrained(TRAINING_CONFIG["ADAPTER_OUTPUT_DIR"])
    tokenizer.save_pretrained(TRAINING_CONFIG["ADAPTER_OUTPUT_DIR"])
```

Submission packaging must preserve `adapter_config.json`.
