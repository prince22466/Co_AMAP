"""Synthesize surgical V3b blends (exact ML2_35_41_hi + gentle ab33/ab39), screen, and evaluate top 2 finalists vs champion on HOLDOUT."""
import hashlib
import json
import subprocess
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, remote_exec
from run_step3_and_step4 import run_remote_stage

CHAMP_SHA = "8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815"


def main():
    # 1. Build V3b surgical blends on remote from exact ML2_late_35_41_hi + gentle ab33 (0.5x) / ab39 (0.55x)
    remote_exec("""
import json
from pathlib import Path
from safetensors.torch import load_file, save_file

hdir = Path('/content/wsp/harvest32')
ml2 = load_file(str(hdir / 'multilayer_candidates/ML2_late_35_41_hi.safetensors'))
v3_33 = load_file(str(hdir / 'v3_candidates/V3_tp35_41_ab33_mid.safetensors'))
v3_39 = load_file(str(hdir / 'v3_candidates/V3_tp35_41_ab33_39_hi.safetensors'))

out = hdir / 'v3b_candidates'
out.mkdir(parents=True, exist_ok=True)

# Blend 1: exact ML2(35,41) + 0.50 * ab33 (rho_ab = 0.08)
t1 = {
    '35.A': ml2['35.A'], '35.B': ml2['35.B'],
    '41.A': ml2['41.A'], '41.B': ml2['41.B'],
    '33.A': v3_33['33.A'], '33.B': (v3_33['33.B'].float() * 0.50).bfloat16(),
}
save_file(t1, str(out / 'V3b_exact35_41_ab33_r008.safetensors'))

# Blend 2: exact ML2(35,41) + 0.45 * ab39 (rho_ab = 0.10, downstream of 35!)
t2 = {
    '35.A': ml2['35.A'], '35.B': ml2['35.B'],
    '41.A': ml2['41.A'], '41.B': ml2['41.B'],
    '39.A': v3_39['39.A'], '39.B': (v3_39['39.B'].float() * 0.45).bfloat16(),
}
save_file(t2, str(out / 'V3b_exact35_41_ab39_r010.safetensors'))

manifest = {
    'candidates': [
        {'name': 'V3b_exact35_41_ab33_r008', 'layers': [33, 35, 41]},
        {'name': 'V3b_exact35_41_ab39_r010', 'layers': [35, 39, 41]},
    ]
}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

# Create v3b screen script reusing screen_v3 logic on v3b_candidates -> screen_v3b_r1
code = Path('/content/wsp/screen_v3.py').read_text()
code = code.replace('v3_candidates', 'v3b_candidates').replace('screen_v3_r1', 'screen_v3b_r1')
Path('/content/wsp/screen_v3b.py').write_text(code)
print('Prepared v3b_candidates and screen_v3b.py')
""")

    py_eval = "/content/wsp/eval_env/bin/python"
    print("=== 1. Screening 2 surgical V3b blends in native vLLM ===", flush=True)
    run_remote_stage(
        "screen_v3b",
        f"{py_eval} -u /content/wsp/screen_v3b.py "
        f"--harvest-dir /content/wsp/harvest32 "
        f"--runs /content/wsp/experiments/discovery02 /content/wsp/experiments/discovery03",
        timeout_sec=1200,
    )
    subprocess.run(
        [CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v3b_r1/summary.json", str(HERE / "screen_v3b_summary.json")],
        check=True,
    )
    v3b_sum = json.loads((HERE / "screen_v3b_summary.json").read_text(encoding="utf-8"))
    print("V3B SUMMARY:", json.dumps(v3b_sum, indent=2), flush=True)

    # 2. Pick Best V3/V3b anti-bare-pytest candidate alongside ML2_late_35_41_hi_candidate!
    v3_sum = json.loads((HERE / "screen_v3_summary.json").read_text(encoding="utf-8"))
    pool = []
    for row in v3b_sum:
        if row["adapter"].endswith("_candidate"):
            pool.append((row, f"/content/wsp/harvest32/screen_v3b_r1/adapters/{row['adapter']}"))
    for row in v3_sum:
        if row["adapter"].endswith("_candidate"):
            pool.append((row, f"/content/wsp/harvest32/screen_v3_r1/adapters/{row['adapter']}"))

    viable = [
        (r, p) for r, p in pool
        if r["verified_preserved"] == "3/3" and r["localize_preserved"] == "13/13"
    ]
    viable.sort(
        key=lambda item: (
            item[0]["edit_preserved_vs_champ"] == "4/4",
            item[0]["unchecked_bare_pytest"] == 0,
            -item[0]["unchecked_submits_left"],
            item[0]["unchecked_targeted_flips"],
            int(item[0]["edit_preserved_abs"].split("/")[0]),
        ),
        reverse=True,
    )
    best_v3_row, best_v3_path = viable[0]
    ml2_path = "/content/wsp/harvest32/screen_multilayer_r1/adapters/ML2_late_35_41_hi_candidate"

    # Fetch exact SHA-256 of both finalist adapters on remote!
    sha_out = remote_exec(f"""
import hashlib
from pathlib import Path
for tag, p in (('ml2', {ml2_path!r}), ('v3best', {best_v3_path!r})):
    h = hashlib.sha256((Path(p) / 'adapter_model.safetensors').read_bytes()).hexdigest()
    print(f'ADAPTER_SHA_{{tag}}={{h}}')
""")
    shas = {}
    for line in sha_out.splitlines():
        if line.startswith("ADAPTER_SHA_"):
            k, v = line.split("=", 1)
            shas[k.removeprefix("ADAPTER_SHA_")] = v.strip()

    print(f"Selected HOLDOUT lineup:\n  1. holdout_champ (sha={CHAMP_SHA[:12]})\n  2. holdout_ml2_35_41_hi ({ml2_path}, sha={shas['ml2'][:12]})\n  3. holdout_v3_best ({best_v3_row['adapter']}, sha={shas['v3best'][:12]})", flush=True)

    # 3. Run HOLDOUT (rich_3043, rich_3064, rich_2725) on all 3!
    holdout_runs = [
        ("holdout_champ", None, CHAMP_SHA),
        ("holdout_ml2_35_41_hi", ml2_path, shas["ml2"]),
        ("holdout_v3_best", best_v3_path, shas["v3best"]),
    ]

    all_holdout_metrics = {}
    for rname, rpath, rsha in holdout_runs:
        chk = remote_exec(f"from pathlib import Path; print('EXISTS=' + str(Path('/content/wsp/experiments/{rname}/neon_metrics.json').is_file()))")
        if "EXISTS=False" in chk:
            ad_flag = f"--adapter {rpath}" if rpath else ""
            print(f"\n=== Running HOLDOUT evaluation: {rname} ===", flush=True)
            run_remote_stage(
                f"eval_{rname}",
                f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split holdout {ad_flag} --sha {rsha} --name {rname}",
                timeout_sec=1800,
            )
        out_local = HERE / "evals" / rname
        out_local.mkdir(parents=True, exist_ok=True)
        for fn in ("neon_metrics.json", "neon_failures.jsonl"):
            subprocess.run([CLI, "download", "-s", SESSION, f"/content/wsp/experiments/{rname}/{fn}", str(out_local / fn)], check=True)
        m = json.loads((out_local / "neon_metrics.json").read_text(encoding="utf-8"))
        m["adapter_path"] = rpath or "champion.zip"
        all_holdout_metrics[rname] = m
        print(f"HOLDOUT RESULT [{rname}]:", json.dumps(m, indent=2), flush=True)

    (HERE / "holdout_comparison.json").write_text(json.dumps(all_holdout_metrics, indent=2), encoding="utf-8")
    print("\n=== SAVED holdout_comparison.json ===", flush=True)


if __name__ == "__main__":
    main()
