"""Screen each fold-supported edit against zero and equal-norm random writes on development data."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from recipe import compile_explicit

r=Path("/content/wsp")
base=r/"harvest03/native04"
out=base/"screen_exact"
out.mkdir(exist_ok=True,parents=True)
py=r/"eval_env/bin/python"
env=dict(os.environ,PATH=f"{py.parent}:"+os.environ["PATH"],TRANSFORMERS_OFFLINE="1",
         VLLM_NO_USAGE_STATS="1",MAX_JOBS="2",TORCHINDUCTOR_COMPILE_THREADS="1",VLLM_WORKER_MULTIPROC_METHOD="spawn")


def main():
    ready=base/"columns.status.json"
    while not ready.is_file():
        pid=Path("/proc/124513/cmdline")
        if not pid.exists() or b"/content/wsp/run_native_stage.py" not in pid.read_bytes().split(b"\0"):
            raise RuntimeError("Native replay and column ranking exited before producing results")
        time.sleep(6)
    if json.loads(ready.read_text())["returncode"]: raise RuntimeError("Column stage failed")
    ranking=base/"columns/candidates"
    candidates=json.loads((ranking/"ranking.json").read_text())["candidates"]
    if not candidates:
        print("No supported candidates",flush=True)
        return
    verified=r/"preservation/exact/cases.json"
    if not verified.exists(): raise RuntimeError("Missing exactly formatted preservation histories")
    zero_layers=set()
    for c in candidates:
        factors=ranking/(c["name"]+".safetensors")
        for kind in ("zero","candidate","random"):
            if kind=="zero":
                if c["layer"] in zero_layers: continue
                zero_layers.add(c["layer"])
                name=f'L{c["layer"]}_zero'
            else: name=c["name"]+"_"+kind
            dest=out/"adapters"/name
            if not dest.exists(): compile_explicit(r/"champion.zip",factors,dest,kind)
    cmd=[str(py),"-u",str(r/"native_replays.py"),"--output",str(out/"replays"),"--cases",str(base/"columns/cases.json"),
         "--adapters",str(out/"adapters"),"--verified",str(verified),
         "--roles","unchecked","verified"]
    with (out/"replays.log").open("w") as log:
        result=subprocess.run(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/"done.json").write_text(json.dumps({"returncode":result.returncode}))
    if result.returncode: raise SystemExit(result.returncode)


if __name__=="__main__": main()
