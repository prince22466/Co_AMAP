"""Provision Colab L4 session wsp32 (idempotent resume), stage public data, verify compliance, and run Step 2 serving_logits."""
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WSP31 = HERE.parent / "wsp_31"
RESEARCH = HERE.parents[1]
CLI = str(Path.home() / ".local/bin/colab.exe")
SESSION = "wsp32"

DISCOVERY_ALL = [
    # Original discovery02 (5)
    "rich_3506", "rich_3296", "rich_3067", "rich_3469", "rich_3130",
    # Expanded disjoint discovery03 (8 across rich, fastapi, requests)
    "rich_4079", "rich_3480", "rich_3472", "rich_3105", "rich_3006",
    "fastapi_14786", "fastapi_14479", "requests_7328",
    # Reserved fresh holdout (3)
    "rich_3043", "rich_3064", "rich_2725",
]


def remote_exec(code: str, timeout: int = 180) -> str:
    for attempt in range(12):
        try:
            run = subprocess.run(
                [CLI, "exec", "-s", SESSION, "--timeout", str(timeout)],
                input=(code + "\nprint('WSP_OK')\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout + 45,
            )
            out = run.stdout.decode("utf-8", errors="replace")
            if run.returncode == 0 and "WSP_OK" in out.splitlines():
                return out
            if attempt == 11:
                raise RuntimeError(f"Remote command failed (rc={run.returncode}):\n{out[-3000:]}")
        except subprocess.TimeoutExpired:
            if attempt == 11:
                raise
            print(f"  [retry {attempt+1}/12] colab exec timed out, retrying...", flush=True)
        time.sleep(10)
    raise RuntimeError("Unreachable")


def upload_verified(local_path: Path, remote_path: str, chunk_bytes: int = 1_500_000) -> None:
    data = local_path.read_bytes()
    expected_sha = hashlib.sha256(data).hexdigest()
    # Skip if already uploaded with matching sha
    chk = remote_exec(f"""
import hashlib
from pathlib import Path
p = Path({remote_path!r})
print('SHA=' + (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'))
""")
    if f"SHA={expected_sha}" in chk:
        print(f"  Already present {remote_path} (sha={expected_sha[:12]})", flush=True)
        return
    if len(data) <= chunk_bytes:
        for attempt in range(3):
            res = subprocess.run([CLI, "upload", "-s", SESSION, str(local_path), remote_path])
            if res.returncode == 0:
                break
            if attempt == 2:
                res.check_returncode()
            time.sleep(2)
    else:
        n_parts = (len(data) + chunk_bytes - 1) // chunk_bytes
        print(f"  Chunked upload {local_path.name} ({len(data)} bytes in {n_parts} parts)...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            for idx in range(n_parts):
                part = Path(tmp) / f"part_{idx:03d}.bin"
                part.write_bytes(data[idx * chunk_bytes : (idx + 1) * chunk_bytes])
                for attempt in range(3):
                    res = subprocess.run([CLI, "upload", "-s", SESSION, str(part), f"{remote_path}.part{idx:03d}"])
                    if res.returncode == 0:
                        break
                    if attempt == 2:
                        res.check_returncode()
                    time.sleep(2)
        remote_exec(f"""
from pathlib import Path
out = Path({remote_path!r})
parts = sorted(out.parent.glob(out.name + '.part*'))
assert len(parts) == {n_parts}, f'Expected {n_parts} parts, got {{len(parts)}}'
with out.open('wb') as fh:
    for p in parts:
        fh.write(p.read_bytes())
        p.unlink()
""")
    remote_exec(f"""
import hashlib
from pathlib import Path
actual = hashlib.sha256(Path({remote_path!r}).read_bytes()).hexdigest()
assert actual == '{expected_sha}', f'SHA mismatch for {remote_path}: {{actual}} != {expected_sha}'
""")
    print(f"  Verified {remote_path} ({len(data)} bytes, sha={expected_sha[:12]})", flush=True)


def ensure_session() -> None:
    for attempt in range(4):
        try:
            chk = subprocess.run(
                [CLI, "exec", "-s", SESSION, "--timeout", "40"],
                input=b"print('ALIVE_OK')\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=65,
            )
            out = chk.stdout.decode("utf-8", errors="replace")
            if chk.returncode == 0 and "ALIVE_OK" in out:
                print("Existing session wsp32 is alive!", flush=True)
                return
            if "not found" in out.lower():
                break
        except subprocess.TimeoutExpired:
            print(f"  [ensure_session retry {attempt+1}/4] colab exec timed out, retrying...", flush=True)
        time.sleep(6)
    print("Session wsp32 disconnected after idle; recreating fresh L4 session wsp32...", flush=True)
    subprocess.run([CLI, "stop", "-s", SESSION], check=False)
    subprocess.run([CLI, "new", "-s", SESSION, "--gpu", "L4"], check=True)


def main():
    ensure_session()
    print("1. Uploading setup bundle (including Step 3-4 scripts) and checking existing archives...", flush=True)
    files = {
        "champion.zip": RESEARCH.parents[1] / "submission_nb/artifacts/submission_40.zip",
        "frozen.ipynb": RESEARCH / "kaggle_eval_kernel/gemma-4-developer-agent-testing_v12_wsp.ipynb",
        "recipe.py": RESEARCH / "recipe.py",
        "serving_logits_v2.py": HERE / "serving_logits_v2.py",
        "run_evaluator_v2.py": HERE / "run_evaluator_v2.py",
        "decision_cases_v2.py": HERE / "decision_cases_v2.py",
        "native_replays_v2.py": HERE / "native_replays_v2.py",
        "harvest_and_rank_v2.py": HERE / "harvest_and_rank_v2.py",
        "geglu_native_v2.py": HERE / "geglu_native_v2.py",
        "screen_native_v2.py": HERE / "screen_native_v2.py",
    }
    for name in (
        "prepare_colab.py", "prepare_evaluator.py", "fetch_public_data.py",
        "run_evaluator.py", "decision_cases.py", "native_replays.py",
        "verified_states.py", "harvest_decisions.py", "rank_columns.py",
    ):
        files[name] = WSP31 / name

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "wsp_setup.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for arc, src in files.items():
                zf.write(src, arc)
        upload_verified(bundle, "/content/wsp_setup.zip")

        slim_handoff = Path(tmp) / "wsp31_handoff_slim.zip"
        with zipfile.ZipFile(WSP31 / "wsp31_handoff.zip", "r") as zin, zipfile.ZipFile(slim_handoff, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if not item.filename.endswith("summaries.pt"):
                    zout.writestr(item, zin.read(item.filename))
        upload_verified(slim_handoff, "/content/wsp31_handoff_slim.zip")

    upload_verified(WSP31 / "discovery02/latest.zip", "/content/discovery02_latest.zip")

    print("2. Extracting updated setup bundle and checking prepare_colab.py...", flush=True)
    print(remote_exec("""
import os, subprocess, sys, zipfile
from pathlib import Path

root = Path('/content/wsp')
root.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/wsp_setup.zip') as z:
    z.extractall(root)
if not (root / 'harvest03/native04/columns/cases.json').is_file():
    with zipfile.ZipFile('/content/wsp31_handoff_slim.zip') as z:
        z.extractall(root)

disc2 = root / 'experiments/discovery02'
disc2.mkdir(parents=True, exist_ok=True)
if not (disc2 / 'neon_metrics.json').is_file():
    with zipfile.ZipFile('/content/discovery02_latest.zip') as z:
        z.extractall(disc2)
if not (disc2 / 'input/adapters/wsp_r1/adapter_model.safetensors').is_file():
    with zipfile.ZipFile(root / 'champion.zip') as z:
        z.extractall(disc2 / 'input')

env = dict(os.environ, DEBIAN_FRONTEND='noninteractive')
if subprocess.run(['dpkg', '-s', 'python3.12-venv'], capture_output=True).returncode != 0:
    subprocess.run(['apt-get', 'update', '-qq'], check=True, env=env)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'python3.12-venv'], check=True, env=env)

if not (root / 'assets.json').is_file() and not (root / 'prepare.log').is_file():
    with (root / 'prepare.log').open('w') as log:
        subprocess.Popen([sys.executable, '-u', str(root / 'prepare_colab.py')], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
""", timeout=240), flush=True)

    print("3. Waiting for prepare_colab.py...", flush=True)
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        prog = remote_exec("""
from pathlib import Path
plog = Path('/content/wsp/prepare.log').read_text(errors='replace') if Path('/content/wsp/prepare.log').exists() else ''
print(plog[-600:])
""", timeout=90)
        if "Development environment ready" in prog:
            print(prog, flush=True)
            break
        if "Traceback" in prog or "Error:" in prog:
            raise RuntimeError(prog)
        time.sleep(10)
    else:
        raise TimeoutError("prepare_colab.py timed out")

    print("4. Launching prepare_evaluator.py if needed...", flush=True)
    remote_exec("""
from pathlib import Path
import subprocess
py = Path('/content/wsp/eval_env/bin/python')
r = subprocess.run([str(py), '-c', 'import vllm, swegemma, adk_submission'], capture_output=True) if py.exists() else None
if r is None or r.returncode != 0:
    exec(Path('/content/wsp/prepare_evaluator.py').read_text())
else:
    print('eval_env already ready')
""", timeout=90)

    print("5. Staging published competition snapshots & wheels via stage_public_data_v2.py...", flush=True)
    subprocess.run(
        [sys.executable, str(HERE / "stage_public_data_v2.py"), SESSION, *DISCOVERY_ALL, "--with-wheels"],
        check=True,
    )

    print("6. Waiting for eval_env wheel installation to complete...", flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        st = remote_exec("""
from pathlib import Path
import subprocess
py = Path('/content/wsp/eval_env/bin/python')
log = Path('/content/wsp/env_setup.log').read_text(errors='replace')[-400:] if Path('/content/wsp/env_setup.log').exists() else ''
r = subprocess.run([str(py), '-c', 'import vllm, swegemma, adk_submission; print("EVAL_ENV_READY")'], capture_output=True, text=True) if py.exists() else None
print(log)
if r:
    print(r.stdout, r.stderr[-200:])
""", timeout=90)
        if "EVAL_ENV_READY" in st:
            print("eval_env is READY!", flush=True)
            break
        time.sleep(12)
    else:
        raise TimeoutError("eval_env setup timed out")

    print("7. Running official competition compliance preflight + Step 2 serving_logits (rank1 & rank2)...", flush=True)
    remote_exec("""
import os, subprocess
from pathlib import Path
r = Path('/content/wsp')
py = str(r / 'eval_env/bin/python')
env = dict(os.environ, PATH=f"{r/'eval_env/bin'}:" + os.environ['PATH'],
           TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1',
           MAX_JOBS='2', TORCHINDUCTOR_COMPILE_THREADS='1',
           VLLM_WORKER_MULTIPROC_METHOD='spawn',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
with (r / 'step2_logits.log').open('w') as log:
    p = subprocess.Popen(
        ['bash', '-c',
         f'{py} -u /content/wsp/serving_logits_v2.py compliance && '
         f'{py} -u /content/wsp/serving_logits_v2.py rank1 && '
         f'{py} -u /content/wsp/serving_logits_v2.py rank2 && '
         f'echo STEP2_ALL_DONE'],
        cwd=r, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
print('Launched step2 PID', p.pid)
""", timeout=60)

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        tail = remote_exec("from pathlib import Path; print(Path('/content/wsp/step2_logits.log').read_text(errors='replace')[-1500:])", timeout=90)
        if "STEP2_ALL_DONE" in tail:
            print("Step 2 completed!", flush=True)
            break
        if "Traceback (most recent call last):" in tail and any(e in tail for e in ("RuntimeError:", "AssertionError", "ValueError:", "OutOfMemoryError", "ModuleNotFoundError:")):
            raise RuntimeError(tail)
        time.sleep(15)
    else:
        raise TimeoutError("Step 2 serving_logits timed out")

    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/compliance_preflight.json", str(HERE / "compliance_preflight.json")], check=True)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/logit_preflight/comparison.json", str(HERE / "logit_comparison.json")], check=True)
    print("Downloaded compliance_preflight.json and logit_comparison.json!", flush=True)


if __name__ == "__main__":
    main()
