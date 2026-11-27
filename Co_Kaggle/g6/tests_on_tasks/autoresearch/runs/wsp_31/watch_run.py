"""Checkpoint an active Colab evaluator without touching the evaluator or its time budget."""
import json
import sys
import time
import zipfile
from pathlib import Path

root = Path("/content/wsp")
name, log_name = sys.argv[1:3]
exp = root / "experiments" / name
saved = -1
seen_alive = False
while True:
    active = False
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = p.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"/content/wsp/run_evaluator.py" in argv and name.encode() in argv:
            active = True
    seen_alive |= active
    diag = None
    for marker in sorted((root/"working/diagnostic_runs").glob("*/task_events.jsonl"),key=lambda p:p.stat().st_mtime):
        if marker.stat().st_mtime >= (exp/"input").stat().st_mtime-10:
            diag = marker.parent
    records = diag/"notebook_task_results.jsonl" if diag else None
    count = len(records.read_text().splitlines()) if records and records.is_file() else 0
    done = (exp/"neon_metrics.json").is_file() and not active
    terminal = not active and (seen_alive or exp.exists())
    log = root / log_name
    if count != saved or terminal:
        tmp = root/(name+"-checkpoint.partial")
        with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as bundle:
            if log.is_file(): bundle.write(log,"evaluator.log")
            for f in exp.iterdir():
                if f.is_file(): bundle.write(f,f.name)
            if diag and not done:
                for f in diag.rglob("*"):
                    if f.is_file() and "configuration_snapshot" not in f.parts:
                        bundle.write(f,"diagnostics/"+str(f.relative_to(diag)))
        tmp.replace(root/(name+"-checkpoint.zip"))
        saved = count
    state = {"active":active,"complete":done,"terminal":terminal,"count":count,
             "log_tail":log.read_text(errors="replace")[-1400:] if log.exists() else ""}
    tmp = root/(name+"-state.partial")
    tmp.write_text(json.dumps(state))
    tmp.replace(root/(name+"-state.json"))
    if terminal: break
    time.sleep(10)
