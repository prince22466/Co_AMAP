#!/usr/bin/env python3
"""SSH bridge for the authorized BI-V150, writing only to owned scratch."""
import argparse
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess

LOCAL = Path(__file__).resolve().parents[1]
HOST = os.getenv("V9_SSH_HOST", "")
ROOT = os.getenv("V9_REMOTE_ROOT", "/root/Optimization/v9_autoresearch")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10"]
CASES = "[[4096,1024,64,256],[16384,1024,64,128]]"

PREPARE = r"""
from pathlib import Path
import sys
p = Path(sys.argv[1])
m = p / ".v9-autoresearch-owned"
if p.exists() and not m.exists() and any(p.iterdir()):
    raise SystemExit("Nonempty unowned remote directory: refusing sync")
p.mkdir(parents=True, exist_ok=True)
m.write_text("Dedicated v9 autoresearch scratch\n")
for name in ("FlagGems", "vllm-plugin-FL"):
    (p / name).mkdir(exist_ok=True)
print("Scratch prepared:", p)
"""

START = r"""
import os, socket, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
assert (p / ".v9-autoresearch-owned").exists()
r = p / ".autoresearch-runs"
r.mkdir(exist_ok=True)
pidfile = r / "server.pid"
if pidfile.exists():
    try: os.kill(int(pidfile.read_text()), 0)
    except ProcessLookupError: pidfile.unlink()
    else: raise SystemExit("Recorded server is still alive; stop first")
try: sock = socket.create_connection(("127.0.0.1", 9031), timeout=1)
except OSError: pass
else:
    sock.close()
    raise SystemExit("Port 9031 occupied; refusing to disturb another server")
env = os.environ.copy()
env["PYTHONPATH"] = str(p / "FlagGems/src") + ":" + str(p / "vllm-plugin-FL") + ":" + env.get("PYTHONPATH", "")
env["VLLM_PLUGINS"] = "fl"
for key in ("VLLM_FL_USE_FLAGGEMS_ATTN", "VLLM_ATTENTION_BACKEND", "VLLM_BATCH_INVARIANT"):
    env.pop(key, None)
cmd = ["vllm","serve","/workspace/MiniCPM5-2B","--port","9031",
       "--compilation-config",'{"cudagraph_mode":"FULL_DECODE_ONLY"}',
       "--served-model-name","minicpm","--gpu-memory-utilization","0.85",
       "--max-model-len","131072"]
with (r / "serve.log").open("ab", buffering=0) as log:
    child = subprocess.Popen(cmd,cwd=p,env=env,stdin=subprocess.DEVNULL,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(child.pid))
print("Started owned vllm PID:",child.pid,"(check readiness before benchmark)")
"""

STOP = r"""
import os, signal, sys
from pathlib import Path
p = Path(sys.argv[1]) / ".autoresearch-runs/server.pid"
if not p.is_file():
    print("No recorded server")
    raise SystemExit(0)
pid = int(p.read_text())
try:
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    pgid = os.getpgid(pid)
except OSError:
    p.unlink(missing_ok=True)
    raise SystemExit(0)
if pgid != pid or b"vllm" not in cmd or b"/workspace/MiniCPM5-2B" not in cmd:
    raise SystemExit("PID identity mismatch; refusing to signal")
os.killpg(pid,signal.SIGTERM)
p.unlink()
print("SIGTERM sent to owned process group",pid)
"""

READY = r"""
import os, sys, urllib.request
from pathlib import Path
p = Path(sys.argv[1]) / ".autoresearch-runs/server.pid"
if not p.exists():
    raise SystemExit("No agent-owned server PID file")
os.kill(int(p.read_text()),0)
with urllib.request.urlopen("http://127.0.0.1:9031/v1/models",timeout=10) as response:
    if response.status != 200: raise SystemExit("Server not ready")
print("Server ready")
"""


def ssh(command):
    subprocess.run([*SSH, HOST, command], check=True)


def remote_python(script):
    ssh(shlex.join(["python3", "-c", script, ROOT]))


def validate():
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", HOST):
        raise SystemExit("Set V9_SSH_HOST to the preapproved SSH alias")
    if PurePosixPath(ROOT) != PurePosixPath("/root/Optimization/v9_autoresearch"):
        raise SystemExit("Remote root is locked to /root/Optimization/v9_autoresearch")
    for program in ("ssh", "rsync"):
        if not shutil.which(program):
            raise SystemExit("Required program not found: " + program)


def sync():
    remote_python(PREPARE)
    for name in ("FlagGems", "vllm-plugin-FL"):
        subprocess.run([
            "rsync","-az","--delete","--exclude=.git","--exclude=__pycache__",
            "--exclude=*.pyc","--exclude=.pytest_cache",
            "-e","ssh -o BatchMode=yes -o StrictHostKeyChecking=yes",
            str(LOCAL / name) + "/", HOST + ":" + ROOT + "/" + name + "/",
        ], check=True)
    print("Synced ONLY the two v9 code trees to owned scratch")


def tests():
    script = ("cd " + shlex.quote(ROOT) +
        " && export PYTHONPATH=" + shlex.quote(ROOT + "/FlagGems/src:" +
        ROOT + "/vllm-plugin-FL") + ":$PYTHONPATH" +
        " && export VLLM_PLUGINS=fl && python3 -m pytest -q -s --noconftest "
        "vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py "
        "vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py")
    ssh("bash -lc " + shlex.quote(script))


def benchmark():
    remote_python(READY)
    script = ("cd /workspace && python3 " +
        shlex.quote(ROOT + "/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py") +
        " --model /workspace/MiniCPM5-2B --served-model-name minicpm "
        "--port 9031 --test-cases " + shlex.quote(CASES))
    ssh("bash -lc " + shlex.quote(script))


def collect():
    dest = LOCAL / ".autoresearch-work/collected"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync","-az","-e","ssh -o BatchMode=yes -o StrictHostKeyChecking=yes",
        "--include=raw_runs_*.csv","--include=summary_*.csv","--exclude=*",
        HOST + ":/workspace/benchmark_results/",str(dest) + "/",
    ], check=True)
    print("Collected official CSVs:",dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check","sync","tests","start","stop","benchmark","collect"))
    action = parser.parse_args().action
    validate()
    if action == "check":
        ssh("test -d /workspace/MiniCPM5-2B && command -v vllm && command -v python3")
        remote_python(PREPARE)
    elif action == "sync": sync()
    elif action == "tests": tests()
    elif action == "start": remote_python(START)
    elif action == "stop": remote_python(STOP)
    elif action == "benchmark": benchmark()
    elif action == "collect": collect()


if __name__ == "__main__":
    main()
