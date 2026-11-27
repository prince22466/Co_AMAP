"""Check the public base model and the FFN hook contract on the actual Colab L4."""
import gc
import json
import os
import shutil
import zipfile
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path("/content/wsp")
assets = json.loads((root / "assets.json").read_text())
wheel = next(Path(assets["wheels"]).glob("vllm-*.whl"))
with zipfile.ZipFile(wheel) as archive:
    meta = next(p for p in archive.namelist() if p.endswith(".dist-info/METADATA"))
    print([line for line in archive.read(meta).decode().splitlines() if line.startswith("Requires-Dist: torch")])
print("Disk free GiB", shutil.disk_usage("/content").free // 2**30, flush=True)
tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
tokenizer.truncation_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    assets["model"], local_files_only=True, dtype=torch.bfloat16, device_map="cuda"
).eval()
lm = getattr(model.model, "language_model", model.model)
observed = {}
def capture(i):
    def record(module, inputs, outputs):
        h = inputs[0][0, -1].detach().float()
        y = outputs[0, -1].detach().float()
        observed[i] = (h, y)
    return record
hooks = [layer.mlp.down_proj.register_forward_hook(capture(i)) for i, layer in enumerate(lm.layers)]
with torch.inference_mode():
    batch = tokenizer("Check a Python function using its public interface.", return_tensors="pt",
                      truncation=True, max_length=4096).to(model.device)
    model(**batch, use_cache=False, logits_to_keep=1)
for hook in hooks:
    hook.remove()
assert len(lm.layers) == len(observed) == 42
errors = {}
for i, (h, y) in observed.items():
    weight = lm.layers[i].mlp.down_proj.weight
    predicted = torch.mv(weight.float(), h)
    errors[i] = float((predicted - y).norm() / y.norm().clamp_min(1e-8))
report = dict(layers=len(observed), input_dim=h.numel(), output_dim=y.numel(),
              max_relative_error=max(errors.values()), device=torch.cuda.get_device_name(),
              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
(root / "ffn_preflight.json").write_text(json.dumps(report, indent=2))
print(report, flush=True)
assert report["max_relative_error"] < 0.01
del model, lm, weight, hooks, observed
gc.collect()
torch.cuda.empty_cache()
