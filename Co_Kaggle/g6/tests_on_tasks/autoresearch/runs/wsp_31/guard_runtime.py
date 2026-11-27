"""Back up each completed research batch and release this particular Colab VM after idle time."""
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

from recover_colab import CLI, HERE, SESSION, execute

ROOT = "/content/wsp"
TOKEN = "wsp31-discovery02-20261125"
REMOTE = '''
from pathlib import Path
import os, time, json, zipfile, hashlib
r=Path("/content/wsp")
key=r/"guard_instance"
if not key.is_file() or key.read_text()!="wsp31-discovery02-20261125":
    raise RuntimeError("Runtime identity changed")
busy=False
for item in Path("/proc").glob("[0-9]*/cmdline"):
    try: argv=item.read_bytes().split(b"\\0")
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if any(a.startswith(b"/content/wsp/") and a.endswith(b".py") and
           not a.endswith(b"watch_run.py") for a in argv):
        busy=True
files=[]
for folder in ("harvest03","native04","preservation","logit_preflight","experiments"):
    for f in (r/folder).rglob("*"):
        if f.is_file() and f.suffix in (".log",".json",".jsonl",".pt",".zip",".safetensors",".npz") and "/input/" not in str(f):
            files.append(f)
stamp=hashlib.sha256(repr(sorted((str(f),f.stat().st_size,f.stat().st_mtime_ns) for f in files)).encode()).hexdigest()
print("GUARD",json.dumps(dict(busy=busy,stamp=stamp)))
'''


def call(code):
    return execute(code+"\nprint('WSP_OK')")


def guard():
    dest=HERE/"colab_backup.zip"
    last=None
    idle_since=time.monotonic()
    while True:
        try:
            output=call(REMOTE)
            state=json.loads(next(line[6:] for line in output.splitlines() if line.startswith("GUARD ")))
            if state["busy"]:
                idle_since=time.monotonic()
            elif state["stamp"]!=last:
                # The source list and identity are re-evaluated in the same command as packing.
                out=call(REMOTE+'''
if not busy:
    with zipfile.ZipFile(r/"guard_backup.zip","w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in files: z.write(f,str(f.relative_to(r)))
    print("DIGEST",hashlib.sha256((r/"guard_backup.zip").read_bytes()).hexdigest())
''')
                digest=next((line[7:] for line in out.splitlines() if line.startswith("DIGEST ")),None)
                if digest:
                    tmp=dest.with_suffix(".partial")
                    subprocess.run([CLI,"download","-s",SESSION,ROOT+"/guard_backup.zip",str(tmp)],check=True,timeout=240)
                    if hashlib.sha256(tmp.read_bytes()).hexdigest()!=digest:
                        raise RuntimeError("Checkpoint digest differs")
                    with zipfile.ZipFile(tmp) as z:
                        if z.testzip(): raise RuntimeError("Invalid checkpoint")
                    tmp.replace(dest)
                    last=state["stamp"]
                    idle_since=time.monotonic()
                    print("Verified and saved",dest,flush=True)
            elif time.monotonic()-idle_since>=600:
                # The workload and the checksum were just rechecked against the same VM.
                subprocess.run([CLI,"stop","-s",SESSION],check=True,timeout=90)
                print("Released backed-up idle runtime",flush=True)
                return
        except (RuntimeError,subprocess.SubprocessError,StopIteration) as err:
            if "Runtime identity changed" in str(err) or "appears to be lost" in str(err):
                return
            print("Retry observation",type(err).__name__,flush=True)
        time.sleep(40)


if __name__=="__main__":
    call(f"from pathlib import Path; Path('{ROOT}/guard_instance').write_text('{TOKEN}')")
    guard()
