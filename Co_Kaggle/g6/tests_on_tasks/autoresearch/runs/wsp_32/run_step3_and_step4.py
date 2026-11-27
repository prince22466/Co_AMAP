"""Orchestrate Step 3 (expanded discovery03 + native replay) & Step 4 (selective FFN + GeGLU harvest & native screen)."""
import json
import subprocess
import time
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified


def run_remote_stage(tag: str, bash_cmd: str, timeout_sec: int = 2400) -> str:
    log_path = f"/content/wsp/{tag}.log"
    done_path = f"/content/wsp/{tag}.rc"
    pid_path = f"/content/wsp/{tag}.pid"
    remote_exec(f"""
import os, subprocess
from pathlib import Path
r = Path('/content/wsp')
pidf = Path({pid_path!r})
already_alive = False
if pidf.is_file():
    try:
        old_pid = int(pidf.read_text().strip())
        os.kill(old_pid, 0)
        already_alive = True
        print('Already running', {tag!r}, 'PID', old_pid)
    except Exception:
        pass
if not already_alive:
    done = Path({done_path!r})
    if done.exists():
        done.unlink()
    env = dict(os.environ, PATH=f"{{r/'eval_env/bin'}}:" + os.environ['PATH'],
               TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1',
               MAX_JOBS='2', TORCHINDUCTOR_COMPILE_THREADS='1',
               VLLM_WORKER_MULTIPROC_METHOD='spawn',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    script = ({bash_cmd!r}) + '; rc=$?; rm -f ' + ({pid_path!r}) + '; echo $rc > ' + ({done_path!r})
    with Path({log_path!r}).open('w') as log:
        p = subprocess.Popen(['bash', '-c', script], cwd=r, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pidf.write_text(str(p.pid))
    print('Started', {tag!r}, 'PID', p.pid)
""", timeout=60)

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        st = remote_exec(f"""
from pathlib import Path
done = Path({done_path!r})
log = Path({log_path!r}).read_text(errors='replace')[-1400:] if Path({log_path!r}).exists() else ''
print('TAIL:', log)
if done.is_file() and done.read_text().strip() != '':
    print('STAGE_RC=' + done.read_text().strip())
""", timeout=90)
        if "STAGE_RC=" in st:
            rc = int(next(l for l in st.splitlines() if l.startswith("STAGE_RC=")).split("=")[1])
            if rc != 0:
                raise RuntimeError(f"Stage {tag} failed (rc={rc}):\n{st}")
            print(f"[{tag}] COMPLETED cleanly!", flush=True)
            return st
        # Print one-line heartbeat
        lines = [l for l in st.splitlines() if l.strip() and l != "WSP_OK"]
        if lines:
            print(f"[{tag}] {lines[-1][:140]}", flush=True)
        time.sleep(20)
    raise TimeoutError(f"Stage {tag} timed out after {timeout_sec}s")


def main():
    # Upload latest Step 3 & Step 4 scripts
    for fname in ("run_evaluator_v2.py", "decision_cases_v2.py", "native_replays_v2.py", "harvest_and_rank_v2.py", "geglu_native_v2.py", "screen_native_v2.py"):
        upload_verified(HERE / fname, f"/content/wsp/{fname}")

    py_eval = "/content/wsp/eval_env/bin/python"
    py_base = "python3"

    # 1. Run discovery03 (8 fresh disjoint snapshots across rich, fastapi, requests) if not already completed
    chk = remote_exec("from pathlib import Path; print('DISC3_EXISTS=' + str(Path('/content/wsp/experiments/discovery03/replay_tokens.jsonl').is_file()))")
    if "DISC3_EXISTS=False" in chk:
        print("=== Running discovery03 (8 disjoint public snapshots across rich/fastapi/requests) ===", flush=True)
        run_remote_stage(
            "discovery03",
            f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split discovery03 --name discovery03",
            timeout_sec=2700,
        )
    else:
        print("discovery03 already completed on remote!", flush=True)

    # 2. Extract all cases across discovery02 + discovery03 and run native vLLM replay on champion!
    print("=== Extracting cases & running native vLLM champion replay across discovery02 + discovery03 ===", flush=True)
    run_remote_stage(
        "native_replay_all",
        f"{py_base} -c \""
        f"import json; from pathlib import Path; from decision_cases_v2 import read_cases_from_runs; "
        f"cases = read_cases_from_runs([Path('/content/wsp/experiments/discovery02'), Path('/content/wsp/experiments/discovery03')]); "
        f"Path('/content/wsp/harvest32').mkdir(exist_ok=True); "
        f"Path('/content/wsp/harvest32/raw_cases.json').write_text(json.dumps(cases)); "
        f"from collections import Counter; print('Extracted:', Counter((c['issue'], c['role']) for c in cases))"
        f"\" && "
        f"{py_eval} -u /content/wsp/native_replays_v2.py --cases /content/wsp/harvest32/raw_cases.json "
        f"--output /content/wsp/harvest32/native --roles unchecked check edit localize",
        timeout_sec=900,
    )

    # 3. Step 2 proved r=2 zero control drifts by up to 0.59375 in vLLM (max_lora_rank=8),
    # whereas r=1 disjoint-layer zero controls (L23_r1_zero, L35_r1_zero) have max_abs == 0.0!
    # Strictly enforce r=1 disjoint-layer candidates (allow_r2 = False).
    print("=== Running harvest_and_rank_v2.py (r=1 disjoint layers only) + geglu_native_v2.py ===", flush=True)
    run_remote_stage(
        "harvest_and_rank32",
        f"{py_base} -u /content/wsp/harvest_and_rank_v2.py "
        f"--runs /content/wsp/experiments/discovery02 /content/wsp/experiments/discovery03 "
        f"--native-replays /content/wsp/harvest32/native/champion.json "
        f"--verified-cases /content/wsp/preservation/exact/cases.json "
        f"--output /content/wsp/harvest32 && "
        f"{py_base} -u /content/wsp/geglu_native_v2.py "
        f"--harvest-dir /content/wsp/harvest32 "
        f"--native-replays /content/wsp/harvest32/native/champion.json",
        timeout_sec=900,
    )

    # 4. Screen shortlisted r=1 candidates (both sparse down_proj and GeGLU switch) vs champion, zero, and random controls!
    print("=== Screening r=1 candidates vs champion, zero, and random controls in native vLLM ===", flush=True)
    run_remote_stage(
        "screen_native32",
        f"{py_eval} -u /content/wsp/screen_native_v2.py "
        f"--harvest-dir /content/wsp/harvest32 "
        f"--runs /content/wsp/experiments/discovery02 /content/wsp/experiments/discovery03 "
        f"--bucket-rank 1 "
        f"--extra-adapters-dir /content/wsp/harvest32/geglu_candidates/adapters",
        timeout_sec=1500,
    )

    # 5. Download all discovery03, ranking, and native screen summaries locally!
    for rem, loc in (
        ("/content/wsp/experiments/discovery03/neon_metrics.json", HERE / "discovery03_metrics.json"),
        ("/content/wsp/experiments/discovery03/neon_failures.jsonl", HERE / "discovery03_failures.jsonl"),
        ("/content/wsp/harvest32/candidates/ranking.json", HERE / "ranking_v2.json"),
        ("/content/wsp/harvest32/screen_r1/summary.json", HERE / "screen_r1_summary.json"),
        ("/content/wsp/harvest32/screen_r2/summary.json", HERE / "screen_r2_summary.json"),
    ):
        subprocess.run([CLI, "download", "-s", SESSION, rem, str(loc)], check=False)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/geglu_candidates/geglu_report.json", str(HERE / "geglu_report_v2.json")], check=False)
    print("Downloaded all Step 3 & Step 4 outputs to", HERE, flush=True)


if __name__ == "__main__":
    main()
