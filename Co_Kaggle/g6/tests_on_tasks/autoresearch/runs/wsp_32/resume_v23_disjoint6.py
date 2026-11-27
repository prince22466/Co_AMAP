"""Resume monitoring already-running `eval_wsp32_v23_disjoint6` on Colab after local harness restart and collect `v23_disjoint6_report.json`."""
from __future__ import annotations

import json

from provision_and_step2 import HERE, ensure_session, remote_exec
from run_step3_and_step4 import run_remote_stage
from run_v21_step3_disjoint_gate import DISJOINT_GATE_6, download_overwrite


def main():
    ensure_session()
    winner_name = "V23_top3_39pt_40rep_41ver_sweet_candidate"
    mf = json.loads((HERE / "v23_sweetspot_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner_name)
    ad_path = f"/content/wsp/harvest32/v23_sweetspot/adapters/{winner_name}"
    py_eval = "/content/wsp/eval_env/bin/python"
    run_name = "wsp32_v23_disjoint6"
    tasks_arg = " ".join(DISJOINT_GATE_6)

    # `run_remote_stage` checks `/content/wsp/eval_wsp32_v23_disjoint6.pid` and attaches cleanly if alive (or reads `.rc` if already done)!
    done_chk = remote_exec(f"""
from pathlib import Path
rcf = Path('/content/wsp/eval_{run_name}.rc')
print('ALREADY_DONE_RC=' + (rcf.read_text().strip() if rcf.is_file() else 'NONE'))
""")
    if "ALREADY_DONE_RC=0" not in done_chk:
        run_remote_stage(
            f"eval_{run_name}",
            f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split custom_dev --tasks {tasks_arg} --adapter {ad_path} --sha {sha} --name {run_name}",
            timeout_sec=2700,
        )
    else:
        print(f"[eval_{run_name}] Already completed with rc=0!", flush=True)

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
    'adapter': {winner_name!r},
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
