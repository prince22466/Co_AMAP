"""Compare saved rank-one and zero-padded rank-two writes and logits on Colab."""
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from safetensors.torch import load, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path("/content/wsp")
assets = json.loads((root / "assets.json").read_text())
with zipfile.ZipFile(root / "champion.zip") as z:
    weights = z.read("adapters/wsp_r1/adapter_model.safetensors")
assert hashlib.sha256(weights).hexdigest() == "8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815"
champion = load(weights)
zero = root / "zero_champion"
if not zero.exists():
    subprocess.run([sys.executable, str(root / "recipe.py"), "--mode", "explicit_factors", "--champion-zip",
                    str(root / "champion.zip"), "--output", str(zero), "--control", "zero"], check=True)
padded = load_file(str(zero / "adapter_model.safetensors"))
tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(assets["model"], local_files_only=True,
                                            dtype=torch.bfloat16, device_map="cuda").eval()
lm = getattr(model.model, "language_model", model.model)
tokenizer.truncation_side = "left"
batch = tokenizer("After editing a Python package, check its public interface.", return_tensors="pt",
                  truncation=True, max_length=4096).to(model.device)


def evaluate(tensors, *, slotwise=False):
    handles = []
    for key in tensors:
        if not key.endswith(".lora_A.weight"):
            continue
        layer = int(key.split(".layers.")[1].split(".")[0])
        a = tensors[key].to(model.device)
        b = tensors[key.replace("lora_A", "lora_B")].to(model.device)
        def add_write(module, inputs, outputs, a=a, b=b):
            if slotwise:
                # Use the same contraction shape for a slot at both ranks.
                delta = (inputs[0] @ a[0])[:, :, None] * b[:, 0]
                for j in range(1, a.shape[0]):
                    delta = delta + (inputs[0] @ a[j])[:, :, None] * b[:, j]
                return outputs + delta
            return outputs + (inputs[0] @ a.T) @ b.T
        handles.append(lm.layers[layer].mlp.down_proj.register_forward_hook(add_write))
    try:
        with torch.inference_mode():
            return model(**batch, use_cache=False).logits.float().cpu()
    finally:
        for handle in handles:
            handle.remove()


base = evaluate({})
rank1 = evaluate(champion)
rank2 = evaluate(padded)
reference1 = evaluate(champion, slotwise=True)
reference2 = evaluate(padded, slotwise=True)
report = {"input_tokens": batch["input_ids"].numel(), "identical_logits": torch.equal(rank1, rank2),
          "max_abs_logit_delta": float((rank1-rank2).abs().max()),
          "all_position_argmax_equal": torch.equal(rank1.argmax(-1), rank2.argmax(-1)),
          "rank_one_repeat_max_abs_delta": float((rank1-evaluate(champion)).abs().max()),
          "slotwise_identical_logits": torch.equal(reference1, reference2),
          "max_abs_champion_vs_base": float((rank1-base).abs().max()),
          "zero_adapter_sha": hashlib.sha256((zero / "adapter_model.safetensors").read_bytes()).hexdigest()}
(root / "control_logits.json").write_text(json.dumps(report, indent=2))
print(report, flush=True)
assert report["slotwise_identical_logits"] and report["max_abs_champion_vs_base"] > 0
