"""Sequence original serving replay, activations and control adapters on one L4."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

r=Path("/content/wsp")
out=r/"harvest03/native04"
out.mkdir(exist_ok=True)
python=r/"eval_env/bin/python"
env=dict(os.environ,PATH=f"{python.parent}:"+os.environ["PATH"],TRANSFORMERS_OFFLINE="1",
         VLLM_NO_USAGE_STATS="1",MAX_JOBS="2",TORCHINDUCTOR_COMPILE_THREADS="1",
         VLLM_WORKER_MULTIPROC_METHOD="spawn",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def run(name,interpreter,script,*args):
    with (out/(name+".log")).open("w") as log:
        p=subprocess.run([str(interpreter),"-u",str(r/script),*map(str,args)],
                         cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/(name+".status.json")).write_text(json.dumps({"returncode":p.returncode}))
    if p.returncode: raise SystemExit(p.returncode)


if __name__=="__main__":
    # Wait only while the earlier, known process is still performing a harvest.
    pid=Path("/proc/123025/cmdline")
    while pid.exists() and b"/content/wsp/harvest_decisions.py" in pid.read_bytes().split(b"\0"):
        time.sleep(4)
    run("native",python,"native_replays.py","--cases",r/"harvest03/cases.json","--output",out/"original")
    run("columns",sys.executable,"harvest_decisions.py","--run",r/"experiments/discovery02",
        "--output",out/"columns","--native-replays",out/"original/champion.json")
