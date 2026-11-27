"""Execute Step 1 (collect wsp32_iter07 + 5-stage funnel on iter04/05/06/07) & Step 2 (2 pinned champion repeats + reproducibility table)."""
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from event_funnel_analyzer import parse_diagnostics_zip
from provision_and_step2 import CLI, HERE, RESEARCH, SESSION, ensure_session, remote_exec, upload_verified
from run_step5_and_step6 import CHAMPION_SOLVES, eval_on_split
from wait_iter07_and_iter08 import wait_for_running_iter07

CHAMP_SHA = "8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815"


def run_remote_funnel_all() -> dict:
    upload_verified(HERE / "event_funnel_analyzer.py", "/content/wsp/event_funnel_analyzer.py")
    remote_exec(r"""
import json, sys
from pathlib import Path
sys.path.insert(0, '/content/wsp')
from event_funnel_analyzer import parse_diagnostics_zip

root = Path('/content/wsp')
report = {}
for rname in ('discovery02', 'discovery03', 'discovery04', 'wsp32_iter05', 'wsp32_iter06', 'wsp32_iter07', 'wsp32_champ_rep2', 'wsp32_champ_rep3'):
    mp = root / 'experiments' / rname / 'neon_metrics.json'
    if not mp.is_file():
        continue
    m = json.loads(mp.read_text())
    dz = Path(m.get('diagnostics_zip', ''))
    if not dz.is_file():
        dz = root / 'experiments' / rname / 'diagnostics.zip'
    if dz.is_file():
        report[rname] = parse_diagnostics_zip(dz, set(m['resolved_ids']))
(root / 'remote_funnels.json').write_text(json.dumps(report, indent=2))
print('PARSED_REMOTE_FUNNELS:', list(report.keys()))
""", timeout=120)
    out_p = HERE / "remote_funnels.json"
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/remote_funnels.json", str(out_p)], check=True)
    return json.loads(out_p.read_text(encoding="utf-8"))


def main():
    ensure_session()

    # 1. Local iter_04 funnel (Kaggle 4x L4 reference)
    iter04_dir = RESEARCH / "runs/iter_04"
    iter04_m = json.loads((iter04_dir / "neon_metrics.json").read_text(encoding="utf-8"))
    zips = sorted((iter04_dir / "diagnostic_runs").iterdir())
    local_funnels = {}
    if zips:
        local_funnels["iter_04_kaggle_4xL4"] = parse_diagnostics_zip(zips[0], set(iter04_m["resolved_ids"]))
        print("Local iter_04 5-Stage Funnel:", json.dumps(local_funnels["iter_04_kaggle_4xL4"]["funnel"], indent=2), flush=True)
        print("Local iter_04 Terminal Modes:", json.dumps(local_funnels["iter_04_kaggle_4xL4"]["modes"], indent=2), flush=True)

    # 2. Finish collecting wsp32_iter07
    mf20 = json.loads((HERE / "v20_cascade_manifest.json").read_text(encoding="utf-8"))
    sha7 = next(x["sha256"] for x in mf20["compiled"] if x["adapter"] == "V20_casc_39_40_41_sweet_candidate")
    if not (HERE / "evals/wsp32_iter07/neon_metrics.json").is_file():
        m7 = wait_for_running_iter07(sha7)
    else:
        m7 = json.loads((HERE / "evals/wsp32_iter07/neon_metrics.json").read_text(encoding="utf-8"))

    # If wsp32_iter07 already hit >=7/15 with 0 lost champion solves, stop right away!
    if m7["solved"] >= 7 and not (CHAMPION_SOLVES - set(m7["resolved_ids"])) and m7["syntax_leaks"] <= 1:
        run_remote_funnel_all()
        return

    # 3. Run 2 pinned unmodified champion repeats (`wsp32_champ_rep2`, `wsp32_champ_rep3`) alongside `wsp32_champ_cal`!
    for rep_name in ("wsp32_champ_rep2", "wsp32_champ_rep3"):
        if not (HERE / "evals" / rep_name / "neon_metrics.json").is_file():
            print(f"\n=== Running pinned unmodified champion repeat: {rep_name} ===", flush=True)
            eval_on_split("15", rep_name, None, CHAMP_SHA, timeout_sec=4500)

    # 4. Compute unified 5-stage funnel & champion reproducibility summary!
    remote_f = run_remote_funnel_all()
    all_funnels = {**local_funnels, **remote_f}

    champ_runs = ["wsp32_champ_cal", "wsp32_champ_rep2", "wsp32_champ_rep3"]
    solve_counts = Counter()
    per_run_solves = {}
    for rn in champ_runs:
        mp = HERE / "evals" / rn / "neon_metrics.json"
        if mp.is_file():
            rids = json.loads(mp.read_text(encoding="utf-8"))["resolved_ids"]
            per_run_solves[rn] = rids
            for t in rids:
                solve_counts[t] += 1

    summary = {
        "pinned_colab_champion_runs": per_run_solves,
        "colab_solve_frequency_out_of_3": dict(solve_counts.most_common()),
        "kaggle_iter04_solves": iter04_m["resolved_ids"],
        "funnels_by_run": {k: {"funnel": v["funnel"], "modes": v["modes"]} for k, v in all_funnels.items()},
        "detailed_episodes": {k: v["episodes"] for k, v in all_funnels.items()},
    }
    out_summary = HERE / "champion_reproducibility_and_funnel.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("=== CHAMPION REPRODUCIBILITY & 5-STAGE FUNNEL SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "detailed_episodes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
