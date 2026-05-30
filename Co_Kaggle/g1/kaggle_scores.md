# Kaggle Scores


**Base model:** `Nemotron-3-Nano-30B`


Use this file to record each Kaggle submission or effective notebook run. Keep one row per submission.

## Summary Table

| Exp ID | Kaggle Score | Method | LoRA pams | Train Config | Data Config | Adapter / Output | Training Runtime on Kaggle | Notes |
|---|---:|---|---|---|---|---:|---:|---|
| Base model | 0.5 | NA | NA | NA | NA | NA | 0h | base model as benchmark |
| LoRA-001 | 0.56 | LoRA SFT | L_R = 4;L_A = 8; MODULES = in_proj, out_proj, up_proj, down_proj; L_dropout = 0.05; L_BIAS = "none"; TYPE = TaskType.CAUSAL_LM | TRAIN_FRACTION: 0.80; EP: 4; MAX_STEPS: None; MAX_STEPS_PER_EPOCH: 30; BATCH_SIZE: 2; GRAD_ACC_STEPS: 10; L_RATE: 1e-4; MAX_LENGTH: 512; MAX_NEW_TOKENS: 32; EARLY_STOPPING: True; EVAL_EVERY_STEPS: 6; EARLY_STOPPING_VAL_ROWS: 50; STOP_WHEN_VAL_WORSE: True | full train.csv | submission.zip | 45m | early stop at 2.6% ep(reaching MAX_STEP), optimizer_updates: 250 |  |
| LoRA-002 | TBD | LoRA SFT | TBD | TBD | TBD | TBD | TBD | TBD |  |
| LoRA-003 | TBD | LoRA SFT | TBD | TBD | TBD | TBD | TBD | TBD |  |
