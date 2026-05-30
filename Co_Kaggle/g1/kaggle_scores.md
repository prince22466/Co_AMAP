# Kaggle Scores


**Base model:** `Nemotron-3-Nano-30B`


Use this file to record each Kaggle submission or effective notebook run. Keep one row per submission.

## Summary Table

| Exp ID | Kaggle Score | Method | LoRA pams | Train Config | Data Config | Adapter / Output | Training Runtime on Kaggle | Notes |
|---|---:|---|---|---|---|---:|---:|---|
| Base model | 0.5 | NA | NA | NA | NA | NA | 0h | base model as benchmark |
| LoRA-001 | TBD | LoRA SFT | LORA_RANK = 4;
LORA_ALPHA = 8;
TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$";
LORA_DROPOUT = 0.05;
LORA_BIAS = "none";
TASK_TYPE = TaskType.CAUSAL_LM | TBD | TBD | TBD | TBD | TBD |  |
| LoRA-002 | TBD | LoRA SFT | TBD | TBD | TBD | TBD | TBD | TBD |  |
| LoRA-003 | TBD | LoRA SFT | TBD | TBD | TBD | TBD | TBD | TBD |  |
