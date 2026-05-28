# Kaggle Interactive Experiment feedbacks


**Base model:** `Nemotron-3-Nano-30B`


Use this file to record each Kaggle interactive session notebook run. Which should useful feedback regarding the time on fine tuning.

## Summary Table

| device | Method | LoRA pams | Train Config | Data Config | Training Runtime on Kaggle | Notes |
|---|---:|---|---|---|---|---:|---:|---|
| GPU RTX pro 6000 | LoRA SFT | LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
TASK_TYPE = TaskType.CAUSAL_LM |     "TRAIN_FRACTION": 0.80,
    "EPOCHS": 1,
    "MAX_STEPS": 200,  # hard stop: max number of training batches
    "BATCH_SIZE": 1,
    "GRADIENT_ACCUMULATION_STEPS": 8,
    "LEARNING_RATE": 1e-4,
    "MAX_LENGTH": 512,  # max tokens for each training prompt+answer example
    "MAX_NEW_TOKENS": 32,  # max answer tokens generated during scoring
    "USE_EARLY_STOPPING": True,
    "EVAL_EVERY_STEPS": 10,  # evaluate every N optimizer updates
    "EARLY_STOPPING_VAL_ROWS": 50,  # use a small validation subset for fast stopping checks
    "STOP_WHEN_VAL_WORSE": True,
    "SAVE_ADAPTER": True,
    "ADAPTER_OUTPUT_DIR": OUTPUT_DIR, | 80/20 split on full data | 45m |  |
