"""Step 3 Orchestrator for V22 Top-Band Pure-WSP Closed-Loop Repair Cascade (`L39[pytest] + L40[repair] + L41[verify]`).

Reuses `/content/wsp/harvest32/clean_v21/real_replays.json` AND cached `champion_r1.json` from `screen_v21_fsm`,
screens the 6 V22 adapters across all 6 behavioral buckets, and runs End-to-End Multi-Turn Disjoint Gate (`wsp32_v22_disjoint6`)!
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, ensure_session, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage
from run_v21_step3_disjoint_gate import DISJOINT_GATE_6, download_overwrite


def write_remote_screen_v22() -> None:
    remote_exec(r"""
import subprocess
from pathlib import Path
subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)

scr21 = Path('/content/wsp/screen_v21_fsm.py').read_text()
scr22 = scr21.replace('harvest32/v21_fsm', 'harvest32/v22_cauchy')
scr22 = scr22.replace('harvest32/screen_v21_fsm', 'harvest32/screen_v22_cauchy')
scr22 = scr22.replace('V21_SCREEN_RESULT', 'V22_SCREEN_RESULT')
# Reuse cached `champion_r1.json` from `screen_v21_fsm` on the exact same `clean_v21` pool!
scr22 = scr22.replace(
    'for idx, (label, ad_path) in enumerate(adapters.items(), 1):\n        req = LoRARequest(label, idx, str(ad_path))',
    'for idx, (label, ad_path) in enumerate(adapters.items(), 1):\n'
    '        if label == "champion_r1" and Path("/content/wsp/harvest32/screen_v21_fsm/champion_r1.json").is_file():\n'
    '            records = json.loads(Path("/content/wsp/harvest32/screen_v21_fsm/champion_r1.json").read_text())\n'
    '            (out_dir / "champion_r1.json").write_text(json.dumps(records, indent=2))\n'
    '            champ_map = {x["prompt_sha"]: x for x in records}\n'
    '            b1 = [x for x in records if x["true_role"] in ("true_imm_edit_ok", "true_imm_bad_test_path") and (x["action"] == "submit_patch" or x["is_bare_pytest"])]\n'
    '            b2 = [x for x in records if x["true_role"] == "true_imm_edit_err" and x["action"] in ("edit_file", "submit_patch")]\n'
    '            b3 = [x for x in records if x["true_role"] == "true_imm_read_full" and x["valid_edit"]]\n'
    '            b4 = [x for x in records if x["true_role"] == "true_imm_test_pass" and x["action"] == "submit_patch"]\n'
    '            b5 = [x for x in records if x["true_role"] == "turn1_init" and x["action"] in ("read_file", "run_command")]\n'
    '            b6_all = [x for x in records if x["true_role"] == "true_imm_real_test_fail"]\n'
    '            b6_bad = [x for x in b6_all if x["action"] in ("submit_patch", "run_command", None)]\n'
    '            summary_rows.append({"adapter": "champion_r1", "b1_total": len(b1), "b1_submits_left": sum(1 for x in b1 if x["action"]=="submit_patch"), "b1_slow_pytest": sum(1 for x in b1 if x["is_bare_pytest"]), "b1_file_pytest": sum(1 for x in b1 if x["is_targ_test"]), "b1_defect_catches": sum(1 for x in b1 if x["defect_catch"]), "b2_err_to_read_flips": f"0/{len(b2)}", "b3_valid_edit_kept": f"{len(b3)}/{len(b3)}", "b3_exact_args_match": f"{len(b3)}/{len(b3)}", "b4_verified_kept": f"{len(b4)}/{len(b4)}", "b5_turn1_kept": f"{sum(1 for x in b5 if not x[\'is_bare_pytest\'])}/{len(b5)}", "b6_fail_repairs_all": f"{sum(1 for x in b6_all if x[\'action\'] in (\'edit_file\', \'read_file\'))}/{len(b6_all)}", "b6_fail_submits_left": sum(1 for x in b6_all if x["action"]=="submit_patch"), "b6_fail_flips": f"0/{len(b6_bad)}", "syntax_leaks": sum(1 for x in records if x["syntax_leak"]), "sample_cmds": [x["command"] for x in b1 if x["action"]=="run_command"][:6]})\n'
    '            continue\n'
    '        req = LoRARequest(label, idx, str(ad_path))'
)
Path('/content/wsp/screen_v22_cauchy.py').write_text(scr22)
""", timeout=60)


def main():
    ensure_session()
    upload_verified(HERE / "remote_synth_v22_cauchy.py", "/content/wsp/remote_synth_v22_cauchy.py")
    write_remote_screen_v22()

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== Step 3.2b: Synthesizing V22 Top-Band (`L39+L40[repair]+L41[verify]`) Cauchy-Optimal Adapters ===", flush=True)
    run_remote_stage("synth_v22_cauchy", f"{py_base} -u /content/wsp/remote_synth_v22_cauchy.py", timeout_sec=1200)
    download_overwrite("/content/wsp/harvest32/v22_cauchy/manifest.json", HERE / "v22_cauchy_manifest.json")
    print((HERE / "v22_cauchy_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== Step 3.3b: Screening V22 Candidates vs Random Controls (6 Adapters, Reusing Champion Cache) ===", flush=True)
    run_remote_stage("screen_v22_cauchy", f"{py_eval} -u /content/wsp/screen_v22_cauchy.py", timeout_sec=2400)
    download_overwrite("/content/wsp/harvest32/screen_v22_cauchy/summary.json", HERE / "screen_v22_cauchy_summary.json")
    print("=== V22 6-BUCKET SCREEN SUMMARY ===", flush=True)
    print((HERE / "screen_v22_cauchy_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v22_cauchy_summary.json").read_text(encoding="utf-8"))
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
            and b3_ex_n >= 3
            and b5_n >= max(1, b5_d - 1)
            and r["b1_file_pytest"] >= 2
            and b6_rep_n > champ_b6_n
            and (r["b1_file_pytest"] + b2_n + b6_rep_n) > (rnd_targ + rnd_b6_n)
        ):
            viable.append(r)

    if not viable:
        print("SCREEN GATE PAUSE: Inspect screen_v22_cauchy_summary.json before launching disjoint multi-turn eval!", flush=True)
        return

    viable.sort(
        key=lambda r: (
            r["syntax_leaks"] == 0,
            int(r["b4_verified_kept"].split("/")[0]) == int(r["b4_verified_kept"].split("/")[1]),
            r["b1_defect_catches"],
            int(r["b6_fail_repairs_all"].split("/")[0]) - r["b6_fail_submits_left"],
            r["b1_file_pytest"],
            int(r["b3_exact_args_match"].split("/")[0]),
            -r["b1_submits_left"],
        ),
        reverse=True,
    )
    winner = viable[0]
    mf = json.loads((HERE / "v22_cauchy_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    ad_path = f"/content/wsp/harvest32/v22_cauchy/adapters/{winner['adapter']}"

    remote_exec("import shutil; shutil.make_archive('/content/wsp/v22_adapters_backup', 'zip', '/content/wsp/harvest32/v22_cauchy')")
    download_overwrite("/content/wsp/v22_adapters_backup.zip", HERE / "checkpoints/v22_adapters_backup.zip")

    run_name = "wsp32_v22_disjoint6"
    remote_exec(f"import shutil; from pathlib import Path; shutil.rmtree(Path('/content/wsp/experiments/{run_name}'), ignore_errors=True)")
    tasks_arg = " ".join(DISJOINT_GATE_6)
    print(f"\n=== Step 3.4b: Running End-to-End Multi-Turn Disjoint Gate ({run_name}) on {winner['adapter']} (sha={sha[:12]}) ===", flush=True)
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
f_v22 = parse_diagnostics_zip(root / 'experiments/{run_name}/diagnostics.zip', set(m['resolved_ids']))

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
    'v22_disjoint6_funnel': f_v22,
    'champion_disjoint6_episodes': champ_eps,
}}
(root / 'v22_disjoint6_report.json').write_text(json.dumps(out, indent=2))
""", timeout=90)
    download_overwrite("/content/wsp/v22_disjoint6_report.json", HERE / "v22_disjoint6_report.json")
    rep = json.loads((HERE / "v22_disjoint6_report.json").read_text(encoding="utf-8"))
    print("=== V22 END-TO-END DISJOINT-6 MULTI-TURN REPORT ===", flush=True)
    print(json.dumps({
        "adapter": rep["adapter"],
        "sha256": rep["sha256"],
        "solved": rep["metrics"]["solved"],
        "resolved_ids": rep["metrics"]["resolved_ids"],
        "edit_errors": rep["metrics"]["file_edit_errors_total"],
        "syntax_leaks": rep["metrics"]["syntax_leaks"],
        "v22_funnel": rep["v22_disjoint6_funnel"]["funnel"],
        "v22_modes": rep["v22_disjoint6_funnel"]["modes"],
        "v22_episodes": rep["v22_disjoint6_funnel"]["episodes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
