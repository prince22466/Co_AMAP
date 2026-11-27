"""Start a fresh champion discovery run after matching the official wheelhouse."""
import os
import shutil
import subprocess
from pathlib import Path

root = Path("/content/wsp")
python = root / "eval_env/bin/python"
env = dict(os.environ, PATH=f"{python.parent}:" + os.environ["PATH"],
           VIRTUAL_ENV=str(python.parent.parent), PYTHONNOUSERSITE="1",
           MAX_JOBS="2", TORCHINDUCTOR_COMPILE_THREADS="1")
wheels = sorted(str(p) for p in (root / "eval_wheels").glob("*.whl") if "cutlass" not in p.name)
subprocess.run([shutil.which("uv"), "pip", "install", "--no-deps", "--python", str(python), *wheels], check=True)
assert (root / "frozen.ipynb").is_file()
try:
    log = (root / "discovery.log").open("x")
except FileExistsError:
    print("Discovery launch already recorded")
else:
    with log:
        proc = subprocess.Popen([str(python), "-u", str(root / "run_evaluator.py"),
                                 "--split", "discovery", "--name", "discovery01"],
                                env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print("Discovery evaluator PID", proc.pid)
