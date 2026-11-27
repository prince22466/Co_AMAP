"""Rehydrate fresh L4 session wsp32 from local 19-snapshot + clean_v14 caches, synthesize & screen V15 Full-Space AlphaEdit, and run gated QUICK_15."""
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from provision_and_step2 import CLI, HERE, RESEARCH, SESSION, ensure_session, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage

DISCOVERY_19 = [
    "rich_3506", "rich_3296", "rich_3067", "rich_3469", "rich_3130",
    "rich_4079", "rich_3480", "rich_3472", "rich_3105", "rich_3006",
    "fastapi_14786", "fastapi_14479", "requests_7328",
    "fastapi_15661", "fastapi_15588", "requests_7505", "requests_7502",
    "rich_4070", "rich_4006",
]


def rehydrate_wsp32() -> None:
    ensure_session()
    fast_chk = remote_exec("""
from pathlib import Path
import subprocess
r = Path('/content/wsp')
py = r / 'eval_env/bin/python'
ok = (r / 'harvest32/clean_v14/champion_clean.json').is_file() and (r / 'assets.json').is_file() and py.is_file()
if ok:
    ok = subprocess.run([str(py), '-c', 'import vllm, swegemma, adk_submission'], capture_output=True).returncode == 0
print('FAST_READY=' + str(ok))
""", timeout=60)
    if "FAST_READY=True" in fast_chk:
        print("=== Session wsp32 already hydrated; uploading latest synth_v15_alphaedit.py only ===", flush=True)
        upload_verified(HERE / "remote_synth_v15_alphaedit.py", "/content/wsp/synth_v15_alphaedit.py")
        return
    print("=== 1. Uploading setup bundle & cached 19-snapshot + clean_v14 archives ===", flush=True)
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
        "synthesize_targeted_dispatch_v3.py": HERE / "synthesize_targeted_dispatch_v3.py",
        "extract_v14_clean.py": HERE / "remote_extract_v14_clean.py",
        "synth_v15_alphaedit.py": HERE / "remote_synth_v15_alphaedit.py",
    }
    for name in (
        "prepare_colab.py", "prepare_evaluator.py", "fetch_public_data.py",
        "run_evaluator.py", "decision_cases.py", "native_replays.py",
        "verified_states.py", "harvest_decisions.py", "rank_columns.py",
    ):
        files[name] = wsp31 / name

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "wsp_setup_v15.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for arc, src in files.items():
                zf.write(src, arc)
        upload_verified(bundle, "/content/wsp_setup_v15.zip")

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

    print("=== 2. Extracting archives & launching prepare_colab.py ===", flush=True)
    remote_exec("""
import os, subprocess, sys, zipfile
from pathlib import Path

root = Path('/content/wsp')
root.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/wsp_setup_v15.zip') as z:
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

    print("=== 3. Staging public competition snapshots & wheelhouse via stage_public_data_v2.py ===", flush=True)
    subprocess.run([sys.executable, str(HERE / "stage_public_data_v2.py"), SESSION, *DISCOVERY_19, "--with-wheels"], check=True)

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


def write_screen_v15() -> None:
    remote_exec(r"""
import subprocess
from pathlib import Path
subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)
Path('/content/wsp/screen_v15_alphaedit.py').write_text('''
import json, os, re, shutil, tempfile, zipfile
from pathlib import Path
from extract_v14_clean import parse_gemma4_call
from screen_native_v2 import DATA, ROOT, execute_generated_check, load_returned_patches
from decision_cases_v2 import TEST_CMD

IS_BARE_PYTEST = re.compile(r"^(?:python(?:3)?\s+-m\s+)?pytest(?:\s+-[qvx]+)*$")

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

    cdir = ROOT / "harvest32/clean_v14"
    vdir = ROOT / "harvest32/v15_alphaedit"
    out_dir = ROOT / "harvest32/screen_v15_alphaedit"
    out_dir.mkdir(parents=True, exist_ok=True)

    ad_dirs = sorted((vdir / "adapters").iterdir())
    for d in ad_dirs:
        if d.name.endswith("_candidate"):
            verify_compliance(d)

    all_cases = json.loads((cdir / "clean_cases.json").read_text())
    champ_clean = {r["prompt_sha"]: r for r in json.loads((cdir / "champion_clean.json").read_text())}

    selected = []
    for c in all_cases:
        nat_act = champ_clean.get(c["prompt_sha"], {}).get("action")
        nat_cmd = ((champ_clean.get(c["prompt_sha"], {}).get("arguments") or {}).get("command") or "").strip()
        rl = c["clean_role"]
        bucket = None
        if rl in ("imm_edit_ok", "post_edit_prem_submit") and (nat_act == "submit_patch" or IS_BARE_PYTEST.match(nat_cmd)):
            bucket = "b1_prem_or_bare"
        elif rl == "imm_edit_err" and nat_act in ("edit_file", "submit_patch"):
            bucket = "b2_edit_err_loop"
        elif rl == "imm_read_full" and nat_act == "edit_file":
            bucket = "b3_read_to_edit"
        elif rl == "imm_test_pass" and nat_act == "submit_patch":
            bucket = "b4_verified_submit"
        elif rl == "turn1_init" and nat_act in ("read_file", "run_command"):
            bucket = "b5_turn1_init"
        if bucket:
            selected.append(dict(c, bucket=bucket))

    from collections import Counter
    print("Clean 5-Bucket Screen Suite:", Counter(c["bucket"] for c in selected), flush=True)

    # Order champion_r1 and V15_hybrid_zero FIRST!
    adapters = {"champion_r1": ROOT / "experiments/discovery02/input/adapters/wsp_r1"}
    if (vdir / "adapters/V15_hybrid_zero/adapter_config.json").is_file():
        adapters["V15_hybrid_zero"] = vdir / "adapters/V15_hybrid_zero"
    for d in ad_dirs:
        if d.name not in adapters and (d / "adapter_config.json").is_file():
            adapters[d.name] = d

    runs = [p for p in (ROOT / "experiments/discovery02", ROOT / "experiments/discovery03", ROOT / "experiments/discovery04") if p.is_dir()]
    returned_patches = load_returned_patches(runs)
    tasks_map = {r["instance_id"]: r for r in map(json.loads, (DATA / "tasks.jsonl").read_text().splitlines())}
    assets = json.loads((ROOT / "assets.json").read_text())

    engine = LLM(
        model=assets["model"], dtype="bfloat16", max_model_len=32768,
        max_num_seqs=8, max_num_batched_tokens=2048, enable_lora=True,
        max_loras=1, max_lora_rank=_round_up_vllm_lora_rank(1),
        gpu_memory_utilization=0.82, scheduler_reserve_full_isl=False, tensor_parallel_size=1,
    )
    tok = engine.get_tokenizer()
    params = SamplingParams(
        temperature=0.0, top_p=0.95, max_tokens=4096,
        stop_token_ids=[tok.convert_tokens_to_ids(t) for t in ("<tool_call|>", "<turn|>")],
        skip_special_tokens=False,
    )
    payload = [{"prompt_token_ids": c["tokens"]} for c in selected]

    summary_rows = []
    champ_b3_tails = {}

    for idx, (label, ad_path) in enumerate(adapters.items(), 1):
        save_p = out_dir / f"{label}.json"
        if save_p.is_file():
            records = json.loads(save_p.read_text())
        else:
            req = LoRARequest(label, idx, str(ad_path))
            batch = engine.generate(payload, params, lora_request=req, use_tqdm=False)
            records = []
            for c, r in zip(selected, batch, strict=True):
                txt = tok.decode(r.outputs[0].token_ids, skip_special_tokens=False)
                act, args = parse_gemma4_call(txt)
                cmd = (args.get("command") or "").strip() if act == "run_command" else ""
                is_bare = bool(IS_BARE_PYTEST.match(cmd)) if cmd else False
                is_targ_test = bool(act == "run_command" and not is_bare and TEST_CMD.search(cmd))
                defect_catch = False
                if is_targ_test and c["issue"] in returned_patches and c["issue"] in tasks_map:
                    ex = execute_generated_check(c["issue"], cmd, returned_patches[c["issue"]], tasks_map[c["issue"]]["patch"])
                    defect_catch = bool(ex.get("discriminating_defect_catch"))
                has_leak = any(tk in txt for tk in ("<|tool_call><|tool_call>",)) or (act is None and "<|" in txt.split("<channel|>")[-1])
                valid_edit = bool(act == "edit_file" and args.get("filepath") and args.get("old_string") and args.get("new_string") and args.get("old_string") != args.get("new_string") and not has_leak)
                records.append({
                    "issue": c["issue"], "bucket": c["bucket"], "prompt_sha": c["prompt_sha"],
                    "action": act, "command": cmd, "is_bare_pytest": is_bare,
                    "is_targ_test": is_targ_test, "defect_catch": defect_catch,
                    "valid_edit": valid_edit, "syntax_leak": has_leak, "text_tail": txt[-400:],
                })
            save_p.write_text(json.dumps(records, indent=2))

        if label == "champion_r1":
            champ_b3_tails = {x["prompt_sha"]: x["text_tail"] for x in records if x["bucket"] == "b3_read_to_edit"}

        b1 = [x for x in records if x["bucket"] == "b1_prem_or_bare"]
        b2 = [x for x in records if x["bucket"] == "b2_edit_err_loop"]
        b3 = [x for x in records if x["bucket"] == "b3_read_to_edit"]
        b4 = [x for x in records if x["bucket"] == "b4_verified_submit"]
        b5 = [x for x in records if x["bucket"] == "b5_turn1_init"]

        agg = {
            "adapter": label,
            "b1_total": len(b1),
            "b1_submits_left": sum(1 for x in b1 if x["action"] == "submit_patch"),
            "b1_bare_pytest": sum(1 for x in b1 if x["is_bare_pytest"]),
            "b1_targeted_tests": sum(1 for x in b1 if x["is_targ_test"]),
            "b1_defect_catches": sum(1 for x in b1 if x["defect_catch"]),
            "b2_err_to_read_flips": f"{sum(1 for x in b2 if x['action'] == 'read_file')}/{len(b2)}",
            "b3_valid_edit_kept": f"{sum(1 for x in b3 if x['valid_edit'])}/{len(b3)}",
            "b3_exact_edit_match": f"{sum(1 for x in b3 if champ_b3_tails.get(x['prompt_sha']) == x['text_tail'])}/{len(b3)}",
            "b4_verified_kept": f"{sum(1 for x in b4 if x['action'] == 'submit_patch')}/{len(b4)}",
            "b5_turn1_kept": f"{sum(1 for x in b5 if x['action'] in ('read_file', 'run_command') and not x['is_bare_pytest'])}/{len(b5)}",
            "syntax_leaks": sum(1 for x in records if x["syntax_leak"]),
        }
        summary_rows.append(agg)
        print("V15_SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
''')
""")


def main():
    rehydrate_wsp32()
    write_screen_v15()

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 4. Synthesizing V15 Full-Space AlphaEdit Multi-Circuit Adapters ===", flush=True)
    run_remote_stage("synth_v15_alphaedit", f"{py_base} -u /content/wsp/synth_v15_alphaedit.py", timeout_sec=900)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v15_alphaedit/manifest.json", str(HERE / "v15_alphaedit_manifest.json")], check=True)
    print((HERE / "v15_alphaedit_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 5. Screening V15 Adapters on Clean 5-Bucket Native vLLM Suite ===", flush=True)
    run_remote_stage("screen_v15_alphaedit", f"{py_eval} -u /content/wsp/screen_v15_alphaedit.py", timeout_sec=3600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v15_alphaedit/summary.json", str(HERE / "screen_v15_alphaedit_summary.json")], check=True)
    print("=== V15 ALPHAEDIT SUMMARY ===", flush=True)
    print((HERE / "screen_v15_alphaedit_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v15_alphaedit_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ = by_name["champion_r1"]
    champ_b3 = int(champ["b3_valid_edit_kept"].split("/")[0])

    cands = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        b2_n = int(r["b2_err_to_read_flips"].split("/")[0])
        b3_n, _ = map(int, r["b3_valid_edit_kept"].split("/"))
        b4_n, b4_d = map(int, r["b4_verified_kept"].split("/"))
        b5_n, b5_d = map(int, r["b5_turn1_kept"].split("/"))

        score_cand = (-r["b1_submits_left"], r["b1_targeted_tests"] + b2_n, -r["b1_bare_pytest"])
        score_champ = (-champ["b1_submits_left"], champ["b1_targeted_tests"] + int(champ["b2_err_to_read_flips"].split("/")[0]), -champ["b1_bare_pytest"])
        score_rnd = (
            (-rnd["b1_submits_left"], rnd["b1_targeted_tests"] + int(rnd["b2_err_to_read_flips"].split("/")[0]), -rnd["b1_bare_pytest"])
            if rnd else (-99, -1, -99)
        )

        if (
            r["syntax_leaks"] <= 1
            and b4_n == b4_d
            and b3_n >= champ_b3
            and b5_n >= max(1, b5_d - 1)
            and r["b1_bare_pytest"] <= 1
            and r["b1_submits_left"] <= 4
            and score_cand > score_champ
            and score_cand > score_rnd
        ):
            cands.append(r)

    if not cands:
        print("GATE STOP: Reviewing V15 summary before choosing QUICK_15 candidate!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            r["syntax_leaks"] == 0,
            r["b1_defect_catches"],
            -r["b1_submits_left"],
            r["b1_targeted_tests"] + int(r["b2_err_to_read_flips"].split("/")[0]),
            int(r["b3_exact_edit_match"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v15_alphaedit/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v15_alphaedit_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V15 GATE PASSED! Launching QUICK_15 wsp32_iter04 on {winner['adapter']} (sha={sha}) ===", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", str(HERE / "run_step5_and_step6.py"),
            "--mode", "quick15",
            "--run-name", "wsp32_iter04",
            "--remote-adapter", ad_path,
            "--sha", sha,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
