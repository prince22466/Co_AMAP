import json
import sys
from pathlib import Path
import kagglehub
import torch

assets = json.loads(Path("/content/wsp/assets.json").read_text())
print(sys.version, torch.cuda.get_device_name(), flush=True)
print([p.name for p in Path(assets["wheels"]).glob("*.whl") if p.name.startswith(("torch-", "vllm-", "swegemma-", "adk_", "transformers-"))])
print([p.name for p in Path(assets["model"]).iterdir()])
try:
    public_snapshot = kagglehub.competition_download("gemma-4-developer-agent", path="snapshots/rich_3506.tgz")
    print("Public snapshot", public_snapshot, flush=True)
except Exception as exc:
    print("Published snapshot access:", type(exc).__name__, str(exc).split("?")[0][:220])
