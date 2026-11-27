"""Provision fresh Colab L4 session wsp32, run/checkpoint 19-snapshot disjoint discovery (02+03+04), screen V12, and launch gated QUICK_15 iter03."""
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from provision_and_step2 import CLI, DISCOVERY_ALL, HERE, RESEARCH, SESSION, WSP31, ensure_session, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage
from run_step5_and_step6 import QUICK_15_IDX, get_task_ids

DISCOVERY04 = ["fastapi_15661", "fastapi_15588", "requests_7505", "requests_7502", "rich_4070", "rich_4006"]


def provision_fast() -> None:
    """Idempotently provision wsp32 L4, model, wheelhouse, and all staged public snapshots (skips already-completed step2_logits!)."""
    ensure_session()
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
        "synthesize_multilayer_v2.py": HERE / "synthesize_multilayer_v2.py",
        "synthesize_targeted_dispatch_v3.py": HERE / "synthesize_targeted_dispatch_v3.py",
        "synth_v7_geglu.py": HERE / "remote_synth_v7_bitexact_geglu.py",
        "synth_v12_whitened.py": HERE / "remote_synth_v12_whitened.py",
    }
    for name in (
        "prepare_colab.py", "prepare_evaluator.py", "fetch_public_data.py",
        "run_evaluator.py", "decision_cases.py", "native_replays.py",
        "verified_states.py", "harvest_decisions.py", "rank_columns.py",
    ):
        files[name] = WSP31 / name

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "wsp_setup_v12.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for arc, src in files.items():
                zf.write(src, arc)
        upload_verified(bundle, "/content/wsp_setup_v12.zip")

        slim_handoff = Path(tmp) / "wsp31_handoff_slim.zip"
        with zipfile.ZipFile(WSP31 / "wsp31_handoff.zip", "r") as zin, zipfile.ZipFile(slim_handoff, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if not item.filename.endswith("summaries.pt"):
                    zout.writestr(item, zin.read(item.filename))
        upload_verified(slim_handoff, "/content/wsp31_handoff_slim.zip")

    upload_verified(WSP31 / "discovery02/latest.zip", "/content/discovery02_latest.zip")

    ckpt_zip = HERE / "checkpoints/harvest32_19snaps.zip"
    if ckpt_zip.is_file():
        upload_verified(ckpt_zip, "/content/harvest32_19snaps.zip")

    print("Extracting setup bundle and launching prepare_colab.py...", flush=True)
    remote_exec("""
import os, subprocess, sys, zipfile
from pathlib import Path

root = Path('/content/wsp')
root.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/wsp_setup_v12.zip') as z:
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

if Path('/content/harvest32_19snaps.zip').is_file() and not (root / 'harvest32/cases.json').is_file():
    with zipfile.ZipFile('/content/harvest32_19snaps.zip') as z:
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
        if "Development environment ready" in prog or "True" in remote_exec("from pathlib import Path; print(Path('/content/wsp/assets.json').is_file())"):
            print("Base environment + Gemma 4 model ready!", flush=True)
            break
        if "Traceback" in prog:
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

    all_stage_ids = sorted(set(DISCOVERY_ALL + DISCOVERY04 + get_task_ids(QUICK_15_IDX)))
    print(f"Staging {len(all_stage_ids)} public snapshots + wheels...", flush=True)
    subprocess.run([sys.executable, str(HERE / "stage_public_data_v2.py"), SESSION, *all_stage_ids, "--with-wheels"], check=True)

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
        time.sleep(10)
    else:
        raise TimeoutError("eval_env timed out")


def ensure_19snap_harvest() -> None:
    upload_verified(HERE / "decision_cases_v2.py", "/content/wsp/decision_cases_v2.py")
    py_eval = "/content/wsp/eval_env/bin/python"
    py_base = "python3"

    chk3 = remote_exec("from pathlib import Path; print('D3=' + str(Path('/content/wsp/experiments/discovery03/replay_tokens.jsonl').is_file()))")
    if "D3=False" in chk3:
        print("=== Running discovery03 (8 disjoint snapshots) ===", flush=True)
        run_remote_stage("discovery03", f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split discovery03 --name discovery03", timeout_sec=2700)

    chk4 = remote_exec("from pathlib import Path; print('D4=' + str(Path('/content/wsp/experiments/discovery04/replay_tokens.jsonl').is_file()))")
    if "D4=False" in chk4:
        print(f"=== Running discovery04 (6 disjoint snapshots: {DISCOVERY04}) ===", flush=True)
        run_remote_stage(
            "discovery04",
            f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split custom_dev --tasks {' '.join(DISCOVERY04)} --name discovery04",
            timeout_sec=2400,
        )
        subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/experiments/discovery04/neon_metrics.json", str(HERE / "discovery04_metrics.json")], check=False)
        subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/experiments/discovery04/neon_failures.jsonl", str(HERE / "discovery04_failures.jsonl")], check=False)

    chk_nat = remote_exec("from pathlib import Path; print('NAT=' + str(Path('/content/wsp/harvest32/native/champion.json').is_file()))")
    if "NAT=False" in chk_nat:
        print("=== Extracting all cases across 19 disjoint snapshots (02+03+04) & running native champion replay ===", flush=True)
        run_remote_stage(
            "native_replay_19",
            f"{py_base} -c \""
            f"import json; from pathlib import Path; from decision_cases_v2 import read_cases_from_runs; "
            f"cases = read_cases_from_runs([Path('/content/wsp/experiments/discovery02'), Path('/content/wsp/experiments/discovery03'), Path('/content/wsp/experiments/discovery04')]); "
            f"ver = Path('/content/wsp/preservation/exact/cases.json'); "
            f"cases.extend(json.loads(ver.read_text()) if ver.is_file() else []); "
            f"Path('/content/wsp/harvest32').mkdir(exist_ok=True); "
            f"Path('/content/wsp/harvest32/cases.json').write_text(json.dumps(cases)); "
            f"from collections import Counter; print('19-Snap Cases by role:', Counter(c['role'] for c in cases))"
            f"\" && "
            f"{py_eval} -u /content/wsp/native_replays_v2.py --cases /content/wsp/harvest32/cases.json "
            f"--output /content/wsp/harvest32/native --roles unchecked check verified edit localize",
            timeout_sec=1200,
        )
        # Save durable local checkpoint zip so we NEVER need to re-run discovery03/04/native_replay again!
        ckpt_dir = HERE / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        remote_exec("""
import zipfile
from pathlib import Path
root = Path('/content/wsp')
with zipfile.ZipFile('/content/harvest32_19snaps.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for rel in (
        'experiments/discovery03/neon_metrics.json',
        'experiments/discovery03/neon_failures.jsonl',
        'experiments/discovery03/diagnostics.zip',
        'experiments/discovery03/requests.jsonl',
        'experiments/discovery03/replay_tokens.jsonl',
        'experiments/discovery04/neon_metrics.json',
        'experiments/discovery04/neon_failures.jsonl',
        'experiments/discovery04/diagnostics.zip',
        'experiments/discovery04/requests.jsonl',
        'experiments/discovery04/replay_tokens.jsonl',
        'harvest32/cases.json',
        'harvest32/native/champion.json',
    ):
        p = root / rel
        if p.is_file():
            z.write(p, rel)
""", timeout=180)
        subprocess.run([CLI, "download", "-s", SESSION, "/content/harvest32_19snaps.zip", str(ckpt_dir / "harvest32_19snaps.zip")], check=True)
        print("Saved local checkpoint checkpoints/harvest32_19snaps.zip!", flush=True)


def write_screen_v12() -> None:
    remote_exec("""
import subprocess
from pathlib import Path
subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)
Path('/content/wsp/screen_v12_whitened.py').write_text('''
import json, os, shutil, tempfile, zipfile
from pathlib import Path
from screen_native_v2 import DATA, ROOT, grade_generation, load_returned_patches

def verify_compliance(ad_dir: Path):
    import adk_submission
    from swegemma.config import build_submission_limits
    val_fn = getattr(adk_submission, "validate_directory")
    limits, _ = build_submission_limits()
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "sub"
        with zipfile.ZipFile(ROOT / "champion.zip") as z:
            z.extractall(sub)
        for fn in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(ad_dir / fn, sub / "adapters/wsp_r1" / fn)
        val_fn(sub, limits=limits)
    print("COMPLIANCE_PASS:", ad_dir.name, flush=True)

def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    hdir = Path("/content/wsp/harvest32")
    vdir = hdir / "v12_whitened"
    out_dir = hdir / "screen_v12_whitened"
    out_dir.mkdir(parents=True, exist_ok=True)

    ad_dirs = sorted((vdir / "adapters").iterdir())
    for d in ad_dirs:
        if d.name.endswith("_candidate"):
            verify_compliance(d)

    adapters = {"champion_r1": ROOT / "experiments/discovery02/input/adapters/wsp_r1"}
    for d in ad_dirs:
        if (d / "adapter_config.json").is_file():
            adapters[d.name] = d

    all_cases = json.loads((hdir / "cases.json").read_text())
    selected, seen = [], set()
    for c in all_cases:
        if c["role"] in ("unchecked", "verified", "edit", "localize"):
            key = (c["issue"], c["role"])
            if key not in seen:
                seen.add(key)
                selected.append(c)

    runs = [p for p in (
        Path("/content/wsp/experiments/discovery02"),
        Path("/content/wsp/experiments/discovery03"),
        Path("/content/wsp/experiments/discovery04"),
    ) if p.is_dir()]
    returned_patches = load_returned_patches(runs)
    tasks_map = {r["instance_id"]: r for r in map(json.loads, (DATA / "tasks.jsonl").read_text().splitlines())}
    assets = json.loads((ROOT / "assets.json").read_text())

    engine = LLM(
        model=assets["model"], dtype="bfloat16", max_model_len=32768,
        max_num_seqs=8, max_num_batched_tokens=2048, enable_lora=True,
        max_loras=1, max_lora_rank=_round_up_vllm_lora_rank(1),
        gpu_memory_utilization=0.80, scheduler_reserve_full_isl=False, tensor_parallel_size=1,
    )
    tokenizer = engine.get_tokenizer()
    params = SamplingParams(
        temperature=0.0, top_p=0.95, max_tokens=2048,
        stop_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call|>", "<turn|>")],
        skip_special_tokens=False,
    )
    payload = [{"prompt_token_ids": c["tokens"]} for c in selected]

    summary_rows = []
    champ_edit_tails = {}
    champ_edit_ok = set()
    champ_ver_ok = set()

    for idx, (label, ad_path) in enumerate(adapters.items(), 1):
        save_p = out_dir / f"{label}.json"
        if save_p.is_file():
            records = json.loads(save_p.read_text())
        else:
            req = LoRARequest(label, idx, str(ad_path))
            batch = engine.generate(payload, params, lora_request=req, use_tqdm=False)
            records = []
            for case, r in zip(selected, batch, strict=True):
                txt = tokenizer.decode(r.outputs[0].token_ids, skip_special_tokens=False)
                graded = grade_generation(txt, case, returned_patches, tasks_map)
                records.append({
                    "issue": case["issue"], "role": case["role"], "observed": case["action"],
                    "generated_tokens": len(r.outputs[0].token_ids), "text_tail": txt[-400:], **graded,
                })
            save_p.write_text(json.dumps(records, indent=2))

        if label == "champion_r1":
            champ_edit_ok = {x["issue"] for x in records if x["role"] == "edit" and x.get("preserved")}
            champ_edit_tails = {x["issue"]: x["text_tail"] for x in records if x["role"] == "edit"}
            champ_ver_ok = {x["issue"] for x in records if x["role"] == "verified" and x.get("preserved")}

        unch = [x for x in records if x["role"] == "unchecked"]
        ver = [x for x in records if x["role"] == "verified" and (not champ_ver_ok or x["issue"] in champ_ver_ok)]
        ed = [x for x in records if x["role"] == "edit"]
        ed_vs_c = [x for x in ed if x["issue"] in champ_edit_ok]
        exact_ed_match = sum(1 for x in ed if champ_edit_tails.get(x["issue"]) == x["text_tail"])
        loc = [x for x in records if x["role"] == "localize"]
        agg = {
            "adapter": label,
            "unchecked_total": len(unch),
            "unchecked_submits_left": sum(1 for x in unch if x.get("action") == "submit_patch"),
            "unchecked_targeted_flips": sum(1 for x in unch if x.get("flipped_to_targeted_check")),
            "unchecked_defect_catches": sum(1 for x in unch if x.get("discriminating_defect_catch")),
            "unchecked_bare_pytest": sum(1 for x in unch if x.get("is_bare_pytest")),
            "verified_preserved": f"{sum(1 for x in ver if x.get('preserved'))}/{len(ver)}",
            "edit_preserved_abs": f"{sum(1 for x in ed if x.get('preserved'))}/{len(ed)}",
            "edit_preserved_vs_champ": f"{sum(1 for x in ed_vs_c if x.get('preserved'))}/{len(ed_vs_c)}",
            "edit_exact_match": f"{exact_ed_match}/{len(ed)}",
            "localize_preserved": f"{sum(1 for x in loc if x.get('preserved'))}/{len(loc)}",
        }
        summary_rows.append(agg)
        print("V12_SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
''')
""")


def main():
    try:
        already = "READY=True" in remote_exec("from pathlib import Path; print('READY=' + str(Path('/content/wsp/assets.json').is_file()))", timeout=90)
    except Exception:
        already = False
    if not already:
        provision_fast()
    ensure_19snap_harvest()
    upload_verified(HERE / "remote_synth_v12_whitened.py", "/content/wsp/synth_v12_whitened.py")
    write_screen_v12()

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== Synthesizing V12 Truncated-PCA Whitened Pure down_proj + Group-Locked GeGLU ===", flush=True)
    run_remote_stage("synth_v12_whitened", f"{py_base} -u /content/wsp/synth_v12_whitened.py", timeout_sec=900)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v12_whitened/manifest.json", str(HERE / "v12_whitened_manifest.json")], check=True)
    print((HERE / "v12_whitened_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== Screening V12 Adapters in Native vLLM across 19-Snapshot Suite ===", flush=True)
    run_remote_stage("screen_v12_whitened", f"{py_eval} -u /content/wsp/screen_v12_whitened.py", timeout_sec=3600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v12_whitened/summary.json", str(HERE / "screen_v12_whitened_summary.json")], check=True)

    remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_v12_whitened')
for p in sorted(d.glob('*.json')):
    if p.name == 'summary.json':
        continue
    rows = json.loads(p.read_text())
    unch = [r for r in rows if r.get('role') == 'unchecked']
    if p.stem.endswith('_candidate') or p.stem == 'champion_r1':
        out[p.stem] = [
            {
                'issue': r['issue'], 'action': r.get('action'), 'command': r.get('command'),
                'is_bare_pytest': r.get('is_bare_pytest'), 'defect_catch': r.get('discriminating_defect_catch'),
                'agent_exit': r.get('on_returned_patch_exit'), 'ref_exit': r.get('on_reference_fix_exit'),
            }
            for r in unch
        ]
Path('/content/wsp/harvest32/screen_v12_whitened/unchecked_details.json').write_text(json.dumps(out, indent=2))
""")
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v12_whitened/unchecked_details.json", str(HERE / "screen_v12_whitened_unchecked_details.json")], check=True)
    print("=== V12 WHITENED SUMMARY ===", flush=True)
    print((HERE / "screen_v12_whitened_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v12_whitened_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ_row = by_name["champion_r1"]
    champ_submits = champ_row["unchecked_submits_left"]

    cands = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        beats_random = rnd is None or (
            (r["unchecked_targeted_flips"], -r["unchecked_submits_left"], -r["unchecked_bare_pytest"])
            > (rnd["unchecked_targeted_flips"], -rnd["unchecked_submits_left"], -rnd["unchecked_bare_pytest"])
        )
        ver_n, ver_d = map(int, r["verified_preserved"].split("/"))
        edc_n, edc_d = map(int, r["edit_preserved_vs_champ"].split("/"))
        loc_n, loc_d = map(int, r["localize_preserved"].split("/"))
        if (
            ver_n == ver_d
            and edc_n >= max(1, edc_d - 0)
            and loc_n >= loc_d - 1
            and r["unchecked_bare_pytest"] == 0
            and r["unchecked_targeted_flips"] >= 2
            and r["unchecked_submits_left"] <= max(1, champ_submits - 2)
            and beats_random
        ):
            cands.append(r)

    if not cands:
        print("GATE STOP: No V12 candidate met strict preservation + 0 bare-pytest + >=2 targeted flips gate!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            int(r["edit_exact_match"].split("/")[0]) >= int(r["edit_exact_match"].split("/")[1]) // 2,
            r["unchecked_defect_catches"],
            r["unchecked_targeted_flips"],
            -r["unchecked_submits_left"],
            int(r["edit_exact_match"].split("/")[0]),
            int(r["edit_preserved_abs"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v12_whitened/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v12_whitened_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V12 GATE PASSED! Launching QUICK_15 wsp32_iter03 on {winner['adapter']} (sha={sha}) ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(HERE / "run_step5_and_step6.py"),
            "--mode",
            "quick15",
            "--run-name",
            "wsp32_iter03",
            "--remote-adapter",
            ad_path,
            "--sha",
            sha,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
