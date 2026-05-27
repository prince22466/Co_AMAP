from __future__ import annotations

from peft import TaskType


LORA_RANK = 4
LORA_ALPHA = 8
TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
TASK_TYPE = TaskType.CAUSAL_LM
