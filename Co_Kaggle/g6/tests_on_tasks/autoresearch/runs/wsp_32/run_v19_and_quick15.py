"""Compile V19 Decoupled Best-Per-Layer Surgical `down_proj` r=1 Adapters (V17 L39/L40 + V18 L41), screen in vLLM, and run QUICK_15 wsp32_iter05."""
import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, ensure_session, remote_exec
from run_step3_and_step4 import run_remote_stage


def compile_and_write_v19() -> None:
    out = remote_exec(r"""
import json, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, '/content/wsp')
import torch
from safetensors.torch import load_file, save_file
from recipe import compile_explicit

subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)
root = Path('/content/wsp')
out_dir = root / 'harvest32/v19_decoupled'
shutil.rmtree(out_dir, ignore_errors=True)
out_dir.mkdir(parents=True, exist_ok=True)
ad_root = out_dir / 'adapters'

f17 = load_file(str(root / 'harvest32/v17_surgical/fac_V17_donly_39_40_41_mid.safetensors'))
f18 = load_file(str(root / 'harvest32/v18_sweetspot/fac_V18_d_39_40_41_firm.safetensors'))

variants = [
    # 1. Pure L39 (V17_mid: defect_catches=2!) + L41 (V18_firm: b4=4/4, cmdq-nullspaced, leaks=0!) — NO L40!
    ('V19_39v17_41v18_clean', {
        '39.A': f17['39.A'], '39.B': (0.96 * f17['39.B'].float()).bfloat16(),
        '41.A': f18['41.A'], '41.B': (1.08 * f18['41.B'].float()).bfloat16(),
    }),
    # 2. Balanced 3-layer: V17_mid L39 (0.96x) + gentle V17_mid L40 (0.80x -> FileEditError->read_file) + V18_firm L41 (1.08x)
    ('V19_39v17_40v17_41v18_bal', {
        '39.A': f17['39.A'], '39.B': (0.96 * f17['39.B'].float()).bfloat16(),
        '40.A': f17['40.A'], '40.B': (0.80 * f17['40.B'].float()).bfloat16(),
        '41.A': f18['41.A'], '41.B': (1.08 * f18['41.B'].float()).bfloat16(),
    }),
    # 3. Pure L41 ONLY (100% mathematically zero KV-cache drift on every non-final token!)
    ('V19_41v18_only', {
        '41.A': f18['41.A'], '41.B': (1.10 * f18['41.B'].float()).bfloat16(),
    }),
]

compiled = []
for name, ft in variants:
    fp = out_dir / f'fac_{name}.safetensors'
    save_file(ft, str(fp))
    for ctrl in ('candidate', 'random'):
        inf = compile_explicit(root / 'champion.zip', fp, ad_root / f'{name}_{ctrl}', ctrl, rank=1)
        compiled.append({'adapter': f'{name}_{ctrl}', 'sha256': inf['adapter_sha256'], 'control': ctrl, 'layers': inf['active_layers']})

(out_dir / 'manifest.json').write_text(json.dumps({'compiled': compiled}, indent=2))
print('COMPILED_V19:', len(compiled))

# Reuse V18 screen script pointing to v19_decoupled (and reusing champion_r1.json from v18 to save time!)
s18 = (root / 'harvest32/screen_v18_sweetspot/champion_r1.json')
assert s18.is_file(), 'Expected cached V18 champion_r1.json'
scr_code = Path('/content/wsp/screen_v18_sweetspot.py').read_text()
scr_code = scr_code.replace('harvest32/v18_sweetspot', 'harvest32/v19_decoupled')
scr_code = scr_code.replace('harvest32/screen_v18_sweetspot', 'harvest32/screen_v19_decoupled')
scr_code = scr_code.replace('V18_SCREEN_RESULT', 'V19_SCREEN_RESULT')
scr_code = scr_code.replace(
    'for idx, (label, ad_path) in enumerate(adapters.items(), 1):\n        req = LoRARequest(label, idx, str(ad_path))',
    'for idx, (label, ad_path) in enumerate(adapters.items(), 1):\n'
    '        if label == "champion_r1":\n'
    '            records = json.loads(Path("/content/wsp/harvest32/screen_v18_sweetspot/champion_r1.json").read_text())\n'
    '            (out_dir / "champion_r1.json").write_text(json.dumps(records, indent=2))\n'
    '            champ_map = {x["prompt_sha"]: x for x in records}\n'
    '            continue\n'
    '        req = LoRARequest(label, idx, str(ad_path))'
)
Path('/content/wsp/screen_v19_decoupled.py').write_text(scr_code)
""", timeout=120)
    print(out, flush=True)


def main():
    ensure_session()
    compile_and_write_v19()
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v19_decoupled/manifest.json", str(HERE / "v19_decoupled_manifest.json")], check=True)

    py_eval = "/content/wsp/eval_env/bin/python"
    print("=== Screening V19 Decoupled Surgical Adapters (6 adapters, ~10 mins) ===", flush=True)
    run_remote_stage("screen_v19_decoupled", f"{py_eval} -u /content/wsp/screen_v19_decoupled.py", timeout_sec=2400)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v19_decoupled/summary.json", str(HERE / "screen_v19_decoupled_summary.json")], check=True)
    print("=== V19 DECOUPLED SUMMARY ===", flush=True)
    print((HERE / "screen_v19_decoupled_summary.json").read_text(encoding="utf-8"), flush=True)

    champ = json.loads((HERE / "screen_v18_sweetspot_summary.json").read_text(encoding="utf-8"))[0]
    rows = json.loads((HERE / "screen_v19_decoupled_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}

    cands = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        b2_n = int(r["b2_err_to_read_flips"].split("/")[0])
        b3_n, b3_d = map(int, r["b3_valid_edit_kept"].split("/"))
        b4_n, b4_d = map(int, r["b4_verified_kept"].split("/"))
        b5_n, b5_d = map(int, r["b5_turn1_kept"].split("/"))

        score_cand = (r["b1_defect_catches"], r["b1_file_pytest"] + b2_n, -r["b1_submits_left"], -r["b1_slow_pytest"])
        score_champ = (champ["b1_defect_catches"], champ["b1_file_pytest"] + int(champ["b2_err_to_read_flips"].split("/")[0]), -champ["b1_submits_left"], -champ["b1_slow_pytest"])
        score_rnd = (
            (rnd["b1_defect_catches"], rnd["b1_file_pytest"] + int(rnd["b2_err_to_read_flips"].split("/")[0]), -rnd["b1_submits_left"], -rnd["b1_slow_pytest"])
            if rnd else (-1, -1, -99, -99)
        )

        if (
            r["syntax_leaks"] <= 1
            and b4_n == b4_d
            and b3_n >= max(1, b3_d - 1)
            and b5_n >= max(1, b5_d - 1)
            and r["b1_submits_left"] <= 2
            and score_cand > score_champ
            and score_cand > score_rnd
        ):
            cands.append(r)

    if not cands:
        print("GATE STOP: No V19 candidate passed gate!", flush=True)
        return

    # Prioritize defect catches + zero leaks + b3_exact_args_match + file-level pytest!
    cands.sort(
        key=lambda r: (
            r["b1_defect_catches"],
            r["syntax_leaks"] == 0,
            int(r["b3_valid_edit_kept"].split("/")[0]),
            int(r["b3_exact_args_match"].split("/")[0]),
            r["b1_file_pytest"] + int(r["b2_err_to_read_flips"].split("/")[0]),
            -r["b1_submits_left"],
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v19_decoupled/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v19_decoupled_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V19 GATE PASSED! Launching QUICK_15 wsp32_iter05 on {winner['adapter']} (sha={sha}) ===", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", str(HERE / "run_step5_and_step6.py"),
            "--mode", "quick15",
            "--run-name", "wsp32_iter05",
            "--remote-adapter", ad_path,
            "--sha", sha,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
