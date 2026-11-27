"""Stage public model and wheelhouse assets on the development Colab VM."""
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import kagglehub

root = Path("/content/wsp")
root.mkdir(exist_ok=True)
assets = {
    "model": kagglehub.model_download("google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2"),
    "wheels": kagglehub.dataset_download("metric/gemma-4-developer-agent-wheelhouse"),
}
(root / "assets.json").write_text(json.dumps(assets))
packages = ("transformers-", "tokenizers-", "safetensors-", "compressed_tensors-", "loguru-")
wheels = sorted(str(w) for w in Path(assets["wheels"]).glob("*.whl") if w.name.startswith(packages))
assert {Path(w).name.split("-")[0] + "-" for w in wheels} == set(packages)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels], check=True)
print("Development environment ready", {k: importlib.metadata.version(k) for k in ("torch", "transformers", "compressed_tensors")})
