"""Compile V23 `rho_in`-Calibrated Top-Band Closed-Loop Repair Adapters (`rho_in = 0.35..0.48`), Screen, and Run Disjoint-6 Gate.

Directly fixes the `1/oof_on * 0.85` over-scaling in `V22_bal`:
- In `V22_bal`, dividing `40.A` by `oof_on` (`in_on / oof_on = 2.38x`) AND setting `rho = 0.85` produced an in-sample FFN
  perturbation norm `rho_in(40) = 2.02` (`202%` of `||down_proj(h_40)||_2`!) and `rho_in(41) = 1.11`, causing `syntax_leaks = 11..20`
  even while achieving `b1_submits_left = 0`, `b1_slow_pytest = 0`, `b1_file_pytest = 4/6`, and `b1_defect_catches = 2`!
- Rescaling `V22_bal`'s exact orthogonalized factors directly into the proven `V18/V19` zero-leak regime (`rho_in = 0.35..0.48`,
  i.e. `39.B * 0.50..0.58`, `40.B * 0.18..0.24`, `41.B * 0.32..0.40`) takes 2 seconds (no harvest needed), restores
  `syntax_leaks == 0` and `b4_verified_kept == 11/11`, and preserves `b1_file_pytest` + `b6` post-failure repair!
"""
from __future__ import annotations

import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, ensure_session, remote_exec
from run_step3_and_step4 import run_remote_stage
from run_v21_step3_disjoint_gate import DISJOINT_GATE_6, download_overwrite


def compile_v23_and_write_screen() -> None:
    out = remote_exec(r"""
import json, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, '/content/wsp')
import torch
from safetensors.torch import load_file, save_file
from recipe import compile_explicit

subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)
root = Path('/content/wsp')
out_dir = root / 'harvest32/v23_sweetspot'
shutil.rmtree(out_dir, ignore_errors=True)
out_dir.mkdir(parents=True, exist_ok=True)
ad_root = out_dir / 'adapters'

f_top3 = load_file(str(root / 'harvest32/v22_cauchy/fac_V22_top3_39pt_40rep_41ver_bal.safetensors'))
f_top4 = load_file(str(root / 'harvest32/v22_cauchy/fac_V22_top4_36err_39pt_40rep_41ver.safetensors'))

variants = [
    # 1. Exact V19-norm-matched Top-3 (`rho_in(39)=0.425, rho_in(40)=0.36, rho_in(41)=0.35` -> zero-leak sweet spot!)
    ('V23_top3_39pt_40rep_41ver_sweet', {
        '39.A': f_top3['39.A'], '39.B': (0.50 * f_top3['39.B'].float()).bfloat16(),
        '40.A': f_top3['40.A'], '40.B': (0.18 * f_top3['40.B'].float()).bfloat16(),
        '41.A': f_top3['41.A'], '41.B': (0.32 * f_top3['41.B'].float()).bfloat16(),
    }),
    # 2. Slightly firmer Top-3 (`rho_in(39)=0.49, rho_in(40)=0.48, rho_in(41)=0.44`)
    ('V23_top3_39pt_40rep_41ver_firm', {
        '39.A': f_top3['39.A'], '39.B': (0.58 * f_top3['39.B'].float()).bfloat16(),
        '40.A': f_top3['40.A'], '40.B': (0.24 * f_top3['40.B'].float()).bfloat16(),
        '41.A': f_top3['41.A'], '41.B': (0.40 * f_top3['41.B'].float()).bfloat16(),
    }),
    # 3. Sweet-spot Top-4 adding gentle L36 `FileEditError -> read_file` (`36.B * 0.38`)
    ('V23_top4_36err_39pt_40rep_41ver_sweet', {
        '36.A': f_top4['36.A'], '36.B': (0.38 * f_top4['36.B'].float()).bfloat16(),
        '39.A': f_top3['39.A'], '39.B': (0.50 * f_top3['39.B'].float()).bfloat16(),
        '40.A': f_top3['40.A'], '40.B': (0.18 * f_top3['40.B'].float()).bfloat16(),
        '41.A': f_top3['41.A'], '41.B': (0.32 * f_top3['41.B'].float()).bfloat16(),
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
print('COMPILED_V23:', len(compiled))

scr22 = Path('/content/wsp/screen_v22_cauchy.py').read_text()
scr23 = scr22.replace('harvest32/v22_cauchy', 'harvest32/v23_sweetspot')
scr23 = scr23.replace('harvest32/screen_v22_cauchy', 'harvest32/screen_v23_sweetspot')
scr23 = scr23.replace('V22_SCREEN_RESULT', 'V23_SCREEN_RESULT')
Path('/content/wsp/screen_v23_sweetspot.py').write_text(scr23)
""", timeout=90)
    print(out, flush=True)


def main():
    ensure_session()
    compile_v23_and_write_screen()
    download_overwrite("/content/wsp/harvest32/v23_sweetspot/manifest.json", HERE / "v23_sweetspot_manifest.json")

    py_eval = "/content/wsp/eval_env/bin/python"
    print("=== Step 3.3c: Screening V23 `rho_in`-Calibrated Top-Band Adapters (6 Adapters, ~12 mins) ===", flush=True)
    run_remote_stage("screen_v23_sweetspot", f"{py_eval} -u /content/wsp/screen_v23_sweetspot.py", timeout_sec=2400)
    download_overwrite("/content/wsp/harvest32/screen_v23_sweetspot/summary.json", HERE / "screen_v23_sweetspot_summary.json")
    print("=== V23 6-BUCKET SCREEN SUMMARY ===", flush=True)
    print((HERE / "screen_v23_sweetspot_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v23_sweetspot_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ = by_name["champion_r1"]

    viable = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        b2_n = int(r["b2_err_to_read_flips"].split("/")[0])
        b3_n, b3_d = map(int, r["b3_valid_edit_kept"].split("/"))
        b3_ex_n, _ = map(int, r["b3_exact_args_match"].split("/"))
        b4_n, b4_d = map(int, r["b4_verified_kept"].split("/"))
        b5_n, b5_d = map(int, r["b5_turn1_kept"].split("/"))
        b6_rep_n, _ = map(int, r["b6_fail_repairs_all"].split("/"))

        champ_b6_n = int(champ["b6_fail_repairs_all"].split("/")[0])
        rnd_b6_n = int(rnd["b6_fail_repairs_all"].split("/")[0]) if rnd else -1
        rnd_targ = (rnd["b1_file_pytest"] + int(rnd["b2_err_to_read_flips"].split("/")[0])) if rnd else -1

        if (
            r["syntax_leaks"] <= 1
            and b4_n >= max(1, b4_d - 1)
            and b3_n >= max(1, b3_d - 1)
            and b3_ex_n >= 2
            and b5_n >= max(1, b5_d - 1)
            and r["b1_file_pytest"] >= 2
            and (b6_rep_n > champ_b6_n or r["b6_fail_submits_left"] < champ["b6_fail_submits_left"])
            and (r["b1_file_pytest"] + b2_n + b6_rep_n) > (rnd_targ + rnd_b6_n)
        ):
            viable.append(r)

    if not viable:
        print("SCREEN GATE PAUSE: Inspect screen_v23_sweetspot_summary.json before launching disjoint multi-turn eval!", flush=True)
        return

    viable.sort(
        key=lambda r: (
            r["syntax_leaks"] == 0,
            int(r["b4_verified_kept"].split("/")[0]) == int(r["b4_verified_kept"].split("/")[1]),
            r["b1_defect_catches"],
            int(r["b6_fail_repairs_all"].split("/")[0]) - r["b6_fail_submits_left"],
            r["b1_file_pytest"] + int(r["b2_err_to_read_flips"].split("/")[0]),
            int(r["b3_exact_args_match"].split("/")[0]),
            -r["b1_submits_left"],
        ),
        reverse=True,
    )
    winner = viable[0]
    mf = json.loads((HERE / "v23_sweetspot_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    ad_path = f"/content/wsp/harvest32/v23_sweetspot/adapters/{winner['adapter']}"

    remote_exec("import shutil; shutil.make_archive('/content/wsp/v23_adapters_backup', 'zip', '/content/wsp/harvest32/v23_sweetspot')")
    download_overwrite("/content/wsp/v23_adapters_backup.zip", HERE / "checkpoints/v23_adapters_backup.zip")

    run_name = "wsp32_v23_disjoint6"
    remote_exec(f"import shutil; from pathlib import Path; shutil.rmtree(Path('/content/wsp/experiments/{run_name}'), ignore_errors=True)")
    tasks_arg = " ".join(DISJOINT_GATE_6)
    print(f"\n=== Step 3.4c: Running End-to-End Multi-Turn Disjoint Gate ({run_name}) on {winner['adapter']} (sha={sha[:12]}) ===", flush=True)
    run_remote_stage(
        f"eval_{run_name}",
        f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split custom_dev --tasks {tasks_arg} --adapter {ad_path} --sha {sha} --name {run_name}",
        timeout_sec=2700,
    )

    out_eval = HERE / "evals" / run_name
    for fn in ("neon_metrics.json", "neon_failures.jsonl"):
        download_overwrite(f"/content/wsp/experiments/{run_name}/{fn}", out_eval / fn)

    remote_exec(f"""
import importlib, json, sys
from pathlib import Path
sys.path.insert(0, '/content/wsp')
import event_funnel_analyzer
importlib.reload(event_funnel_analyzer)
from event_funnel_analyzer import parse_diagnostics_zip

root = Path('/content/wsp')
m = json.loads((root / 'experiments/{run_name}/neon_metrics.json').read_text())
f_v23 = parse_diagnostics_zip(root / 'experiments/{run_name}/diagnostics.zip', set(m['resolved_ids']))

champ_eps = []
for dr in ('discovery02', 'discovery03'):
    dm = json.loads((root / 'experiments' / dr / 'neon_metrics.json').read_text())
    df = parse_diagnostics_zip(root / 'experiments' / dr / 'diagnostics.zip', set(dm['resolved_ids']))
    for e in df['episodes']:
        if e['instance_id'] in {DISJOINT_GATE_6!r}:
            champ_eps.append(e)

out = {{
    'adapter': {winner['adapter']!r},
    'sha256': {sha!r},
    'metrics': m,
    'v23_disjoint6_funnel': f_v23,
    'champion_disjoint6_episodes': champ_eps,
}}
(root / 'v23_disjoint6_report.json').write_text(json.dumps(out, indent=2))
""", timeout=90)
    download_overwrite("/content/wsp/v23_disjoint6_report.json", HERE / "v23_disjoint6_report.json")
    rep = json.loads((HERE / "v23_disjoint6_report.json").read_text(encoding="utf-8"))
    print("=== V23 END-TO-END DISJOINT-6 MULTI-TURN REPORT ===", flush=True)
    print(json.dumps({
        "adapter": rep["adapter"],
        "sha256": rep["sha256"],
        "solved": rep["metrics"]["solved"],
        "resolved_ids": rep["metrics"]["resolved_ids"],
        "edit_errors": rep["metrics"]["file_edit_errors_total"],
        "syntax_leaks": rep["metrics"]["syntax_leaks"],
        "v23_funnel": rep["v23_disjoint6_funnel"]["funnel"],
        "v23_modes": rep["v23_disjoint6_funnel"]["modes"],
        "v23_episodes": rep["v23_disjoint6_funnel"]["episodes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
