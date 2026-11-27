"""Start the Colab harvest only after the frozen run has saved its exact replay inputs."""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

root=Path("/content/wsp")
run=root/"experiments/discovery02"
out=root/"harvest02"
out.mkdir(exist_ok=True)
state_file=root/"discovery02-state.json"


def checkpoint(stage):
    tmp=root/"harvest02.partial"
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file() and f.name!="summaries.pt":
                z.write(f,str(f.relative_to(out)))
    tmp.replace(root/"harvest02.zip")
    report={"stage":stage,"saved_summaries":(out/"summaries.pt").is_file()}
    (root/"harvest02-state.json").write_text(json.dumps(report))


def execute(script, args):
    with (out/(Path(script).stem+".log")).open("w") as log:
        p=subprocess.run([sys.executable,"-u",str(root/script),*map(str,args)],stdout=log,
                         stderr=subprocess.STDOUT,cwd=root,
                         env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    return p.returncode==0


def main():
    while True:
        if state_file.is_file() and json.loads(state_file.read_text()).get("terminal"):
            break
        time.sleep(12)
    if not (run/"neon_metrics.json").is_file() or not (run/"replay_tokens.jsonl").is_file():
        checkpoint("missing completed replay inputs")
        return
    if not execute("harvest_decisions.py",["--run",run,"--output",out]):
        checkpoint("harvest failed")
        return
    checkpoint("harvest complete")
    candidates=json.loads((out/"candidates/ranking.json").read_text())["candidates"]
    if not candidates:
        checkpoint("no fold-supported candidates")
        return
    if not execute("screen_edits.py",[out]):
        checkpoint("screen failed")
        return
    checkpoint("screen complete")


if __name__=="__main__":
    main()
