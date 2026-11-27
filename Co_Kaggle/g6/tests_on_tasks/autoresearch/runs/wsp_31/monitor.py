"""Download atomic run checkpoints and retry transient Colab reads."""
import argparse
import json
import re
import subprocess
import time
from recover_colab import CLI, HERE, SESSION, execute


def monitor(name, log_name):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or not re.fullmatch(r"[a-zA-Z0-9_-]+\.log", log_name):
        raise ValueError("Invalid run name or log")
    dest = HERE / name
    dest.mkdir(exist_ok=True)
    subprocess.run([CLI,"upload","-s",SESSION,str(HERE/"watch_run.py"),"/content/wsp/watch_run.py"],check=True)
    start = f"""
import subprocess, sys
from pathlib import Path
r = Path("/content/wsp")
try:
    fd = (r/{(name+"-watch.log")!r}).open("x")
except FileExistsError:
    pass
else:
    with fd:
        subprocess.Popen([sys.executable,str(r/"watch_run.py"),{name!r},{log_name!r}],stdout=fd,stderr=subprocess.STDOUT,start_new_session=True)
print("WSP_OK")
"""
    def download(remote, local):
        args = [CLI,"download","-s",SESSION,"/content/wsp/"+remote,str(local)]
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90, check=True)
    saved = -1
    launched = False
    while True:
        try:
            if not launched:
                execute(start)
                launched = True
            download(name+"-state.json",dest/"state.json")
            state = json.loads((dest/"state.json").read_text())
            if state["count"] != saved or state["terminal"]:
                download(name+"-checkpoint.zip",dest/"partial.zip")
                (dest/"partial.zip").replace(dest/"checkpoint.zip")
                saved = state["count"]
                print(name,"saved",saved,ascii(state["log_tail"][-200:]),flush=True)
            if state["complete"]:
                return
            if state["terminal"]:
                raise SystemExit("Evaluator exited before writing metrics")
        except (RuntimeError, subprocess.SubprocessError) as exc:
            print(name,"Colab observation retry",type(exc).__name__,flush=True)
        time.sleep(30)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("log")
    args=p.parse_args()
    monitor(args.name,args.log)
