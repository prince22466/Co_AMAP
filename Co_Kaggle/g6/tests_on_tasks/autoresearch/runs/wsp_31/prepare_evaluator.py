"""Install the competition's public vLLM and evaluator wheels in an isolated Colab environment."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path("/content/wsp")
assets = json.loads((root / "assets.json").read_text())
uv = shutil.which("uv")
if uv is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    uv = shutil.which("uv")
venv = root / "eval_env"
subprocess.run([uv, "venv", "--python", "3.12", "--seed", "--allow-existing", str(venv)], check=True)
wheels = root / "eval_wheels"
wheels.mkdir(exist_ok=True)
for wheel in Path(assets["wheels"]).glob("*.whl"):
    name = wheel.name
    if "cu128" in name and "+cu128" not in name:
        name = name.replace("cu128", "+cu128")
    if not (wheels / name).exists():
        (wheels / name).symlink_to(wheel)
print(sorted(p.name for p in wheels.iterdir()), flush=True)
packages = ("adk_eval_core-", "adk_submission-", "swegemma-", "vllm-", "transformers-", "compressed_tensors-")
required = [str(p) for p in sorted(wheels.iterdir()) if p.name.startswith(packages)]
assert len(required) == len(packages)
env = dict(os.environ, UV_INDEX_STRATEGY="unsafe-best-match", UV_HTTP_TIMEOUT="120")
cmd = [uv, "pip", "install", "--python", str(venv / "bin/python"), "--find-links", str(wheels),
       "--extra-index-url", "https://download.pytorch.org/whl/cu128",
       "torch==2.10.0+cu128", "torchaudio==2.10.0+cu128", "torchvision==0.25.0+cu128",
       "ipython", "pandas", "pytest", *required]
with (root / "env_setup.log").open("w") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
print("Evaluator environment setup PID", proc.pid, flush=True)
