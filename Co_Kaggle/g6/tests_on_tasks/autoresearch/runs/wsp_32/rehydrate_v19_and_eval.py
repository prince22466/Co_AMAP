"""Hydrate wsp32, synthesize & locally back up V19 decoupled adapters, and run QUICK_15 wsp32_iter05 (and wsp32_iter06 if <7/15)."""
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from provision_and_step2 import CLI, HERE, RESEARCH, SESSION, ensure_session, remote_exec, upload_verified
from rehydrate_and_run_v15 import DISCOVERY_19
from run_step3_and_step4 import run_remote_stage
from run_step5_and_step6 import CHAMPION_SOLVES, QUICK_15_IDX, get_task_ids


def hydrate_all() -> None:
    ensure_session()
    wsp31 = HERE.parent / "wsp_31"
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
        "extract_v14_clean.py": HERE / "remote_extract_v14_clean.py",
        "synth_v16_upperband.py": HERE / "remote_synth_v16_upperband.py",
        "synth_v17_surgical.py": HERE / "remote_synth_v17_surgical.py",
        "synth_v18_sweetspot.py": HERE / "remote_synth_v18_sweetspot.py",
    }
    for name in (
        "prepare_colab.py", "prepare_evaluator.py", "fetch_public_data.py",
        "run_evaluator.py", "decision_cases.py", "native_replays.py",
        "verified_states.py", "harvest_decisions.py", "rank_columns.py",
    ):
        files[name] = wsp31 / name

    print("=== 1. Uploading setup bundle & cached 19-snapshot + clean_v14 archives ===", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "wsp_setup_v19.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for arc, src in files.items():
                zf.write(src, arc)
        upload_verified(bundle, "/content/wsp_setup_v19.zip")

        slim_handoff = Path(tmp) / "wsp31_handoff_slim.zip"
        with zipfile.ZipFile(wsp31 / "wsp31_handoff.zip", "r") as zin, zipfile.ZipFile(slim_handoff, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if not item.filename.endswith("summaries.pt"):
                    zout.writestr(item, zin.read(item.filename))
        upload_verified(slim_handoff, "/content/wsp31_handoff_slim.zip")

        clean_zip = Path(tmp) / "clean_v14_cache.zip"
        with zipfile.ZipFile(clean_zip, "w", compression=zipfile.ZIP_DEFLATED) as zc:
            zc.write(HERE / "checkpoints/clean_v14/clean_cases.json", "harvest32/clean_v14/clean_cases.json")
            zc.write(HERE / "checkpoints/clean_v14/champion_clean.json", "harvest32/clean_v14/champion_clean.json")
        upload_verified(clean_zip, "/content/clean_v14_cache.zip")

    upload_verified(wsp31 / "discovery02/latest.zip", "/content/discovery02_latest.zip")
    upload_verified(HERE / "checkpoints/harvest32_19snaps.zip", "/content/harvest32_19snaps.zip")

    print("=== 2. Extracting archives & preparing environment ===", flush=True)
    remote_exec("""
import os, subprocess, sys, zipfile
from pathlib import Path

root = Path('/content/wsp')
root.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/wsp_setup_v19.zip') as z:
    z.extractall(root)
if not (root / 'preservation/exact/cases.json').is_file():
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
if not (root / 'experiments/discovery04/neon_metrics.json').is_file():
    with zipfile.ZipFile('/content/harvest32_19snaps.zip') as z:
        z.extractall(root)
with zipfile.ZipFile('/content/clean_v14_cache.zip') as z:
    z.extractall(root)

env = dict(os.environ, DEBIAN_FRONTEND='noninteractive')
if subprocess.run(['dpkg', '-s', 'python3.12-venv'], capture_output=True).returncode != 0:
    subprocess.run(['apt-get', 'update', '-qq'], check=True, env=env)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'python3.12-venv'], check=True, env=env)

if not (root / 'assets.json').is_file() and not (root / 'prepare.log').is_file():
    with (root / 'prepare.log').open('w') as log:
        subprocess.Popen([sys.executable, '-u', str(root / 'prepare_colab.py')], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
""", timeout=240)

    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        prog = remote_exec("from pathlib import Path; print(Path('/content/wsp/prepare.log').read_text(errors='replace')[-600:] if Path('/content/wsp/prepare.log').exists() else '')", timeout=90)
        if "Development environment ready" in prog:
            print("Base Colab environment ready!", flush=True)
            break
        if "Traceback" in prog or "Error:" in prog:
            raise RuntimeError(prog)
        time.sleep(10)
    else:
        raise TimeoutError("prepare_colab.py timed out")

    remote_exec("""
from pathlib import Path
import subprocess
py = Path('/content/wsp/eval_env/bin/python')
r = subprocess.run([str(py), '-c', 'import vllm, swegemma, adk_submission'], capture_output=True) if py.exists() else None
if r is None or r.returncode != 0:
    exec(Path('/content/wsp/prepare_evaluator.py').read_text())
""", timeout=90)

    print("=== 3. Staging DISCOVERY_19 + QUICK_15 public snapshots & wheels ===", flush=True)
    all_ids = sorted(set(DISCOVERY_19) | set(get_task_ids(QUICK_15_IDX)))
    subprocess.run([sys.executable, str(HERE / "stage_public_data_v2.py"), SESSION, *all_ids, "--with-wheels"], check=True)

    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        st = remote_exec("""
from pathlib import Path
import subprocess
py = Path('/content/wsp/eval_env/bin/python')
r = subprocess.run([str(py), '-c', 'import vllm, swegemma, adk_submission; print("EVAL_ENV_READY")'], capture_output=True, text=True) if py.exists() else None
if r: print(r.stdout, r.stderr[-200:])
""", timeout=90)
        if "EVAL_ENV_READY" in st:
            print("eval_env is READY!", flush=True)
            break
        time.sleep(12)
    else:
        raise TimeoutError("eval_env setup timed out")


def build_and_backup_v19() -> dict:
    py_base = "python3"
    print("=== 4. Synthesizing V17 + V18 factors and compiling V19 decoupled adapters ===", flush=True)
    run_remote_stage("synth_v17_surgical", f"{py_base} -u /content/wsp/synth_v17_surgical.py", timeout_sec=900)
    run_remote_stage("synth_v18_sweetspot", f"{py_base} -u /content/wsp/synth_v18_sweetspot.py", timeout_sec=900)

    remote_exec(r"""
import json, shutil, sys, zipfile
from pathlib import Path
sys.path.insert(0, '/content/wsp')
from safetensors.torch import load_file, save_file
from recipe import compile_explicit

root = Path('/content/wsp')
out_dir = root / 'harvest32/v19_decoupled'
shutil.rmtree(out_dir, ignore_errors=True)
out_dir.mkdir(parents=True, exist_ok=True)
ad_root = out_dir / 'adapters'

f17 = load_file(str(root / 'harvest32/v17_surgical/fac_V17_donly_39_40_41_mid.safetensors'))
f18 = load_file(str(root / 'harvest32/v18_sweetspot/fac_V18_d_39_40_41_firm.safetensors'))

variants = [
    ('V19_39v17_41v18_clean', {
        '39.A': f17['39.A'], '39.B': (0.96 * f17['39.B'].float()).bfloat16(),
        '41.A': f18['41.A'], '41.B': (1.08 * f18['41.B'].float()).bfloat16(),
    }),
    ('V19_39v17_40v17_41v18_bal', {
        '39.A': f17['39.A'], '39.B': (0.96 * f17['39.B'].float()).bfloat16(),
        '40.A': f17['40.A'], '40.B': (0.80 * f17['40.B'].float()).bfloat16(),
        '41.A': f18['41.A'], '41.B': (1.08 * f18['41.B'].float()).bfloat16(),
    }),
]

compiled = []
for name, ft in variants:
    fp = out_dir / f'fac_{name}.safetensors'
    save_file(ft, str(fp))
    inf = compile_explicit(root / 'champion.zip', fp, ad_root / f'{name}_candidate', 'candidate', rank=1)
    compiled.append({'adapter': f'{name}_candidate', 'sha256': inf['adapter_sha256'], 'layers': inf['active_layers']})

(out_dir / 'manifest.json').write_text(json.dumps({'compiled': compiled}, indent=2))
with zipfile.ZipFile('/content/v19_adapters_backup.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zb:
    for p in out_dir.rglob('*'):
        if p.is_file():
            zb.write(p, p.relative_to(out_dir))
print('V19_BACKUP_READY', json.dumps(compiled))
""", timeout=120)

    bkp_p = HERE / "checkpoints/v19_adapters_backup.zip"
    subprocess.run([CLI, "download", "-s", SESSION, "/content/v19_adapters_backup.zip", str(bkp_p)], check=True)
    with zipfile.ZipFile(bkp_p, "r") as z:
        mf = json.loads(z.read("manifest.json"))
    print("Backed up V19 adapters locally:", json.dumps(mf, indent=2), flush=True)
    return {x["adapter"]: x["sha256"] for x in mf["compiled"]}


def run_q15(run_name: str, ad_name: str, sha: str) -> dict:
    ad_path = f"/content/wsp/harvest32/v19_decoupled/adapters/{ad_name}"
    print(f"\n=== Launching QUICK_15 {run_name} on {ad_name} (sha={sha}) ===", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", str(HERE / "run_step5_and_step6.py"),
            "--mode", "quick15",
            "--run-name", run_name,
            "--remote-adapter", ad_path,
            "--sha", sha,
        ],
        check=True,
    )
    return json.loads((HERE / "evals" / run_name / "neon_metrics.json").read_text(encoding="utf-8"))


def main():
    hydrate_all()
    shas = build_and_backup_v19()

    m5 = run_q15("wsp32_iter05", "V19_39v17_41v18_clean_candidate", shas["V19_39v17_41v18_clean_candidate"])
    lost5 = sorted(CHAMPION_SOLVES - set(m5["resolved_ids"]))
    if m5["solved"] >= 7 and not lost5 and m5["syntax_leaks"] <= 1:
        print("=== TARGET >= 7/15 REACHED ON wsp32_iter05! ===", flush=True)
        return

    print("\n=== Evaluating second V19 finalist (with L40 FileEditError->read_file) as wsp32_iter06 ===", flush=True)
    m6 = run_q15("wsp32_iter06", "V19_39v17_40v17_41v18_bal_candidate", shas["V19_39v17_40v17_41v18_bal_candidate"])
    print("Completed wsp32_iter05 and wsp32_iter06!", flush=True)


if __name__ == "__main__":
    main()
