"""Verify the isolated environment's public competition libraries before inference."""
import os
import shutil
import subprocess
from pathlib import Path

root = Path("/content/wsp")
python = root / "eval_env/bin/python"
wheel_paths = [str(p) for p in sorted((root / "eval_wheels").iterdir()) if "cutlass" not in p.name]
subprocess.run([shutil.which("uv"), "pip", "install", "--no-deps", "--python", str(python), *wheel_paths], check=True)
env = dict(os.environ, PATH=f"{python.parent}:" + os.environ["PATH"],
           VIRTUAL_ENV=str(python.parent.parent), LITELLM_LOCAL_MODEL_COST_MAP="True")
code = """
import importlib.metadata as md
import torch
from vllm import _C
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.evaluate import Evaluator
print({k: md.version(k) for k in ('torch','vllm','google-adk','adk-submission','swegemma')})
print('Local inference runtime imports passed', torch.cuda.get_device_name())
"""
subprocess.run([str(python), "-c", code], env=env, check=True)
