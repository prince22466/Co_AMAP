# simply to see if the model works or not

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = Path(
    r"E:\Projects\Co_AMAP\Co_Kaggle\models\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
)

SNAPSHOT_DIR = next((MODEL_DIR / "snapshots").iterdir())

print("Using model snapshot:", SNAPSHOT_DIR)

tokenizer = AutoTokenizer.from_pretrained(
    SNAPSHOT_DIR,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    SNAPSHOT_DIR,
    dtype=torch.float16,   
    low_cpu_mem_usage=True,
    local_files_only=True,
)

prompt = "Explain LoRA fine-tuning in simple words."

messages = [
    {"role": "system", "content": "You are a helpful technical assistant."},
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))