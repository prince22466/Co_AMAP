"""Run V11 Hybrid Pareto (Pure down_proj h0-Subspace + Low-Dim Balanced GeGLU) synthesis, screen, and gated QUICK_15 iter03."""
import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def main():
    upload_verified(HERE / "remote_synth_v11_hybrid_pareto.py", "/content/wsp/synth_v11_hybrid.py")

    remote_exec("""
import subprocess
from pathlib import Path
subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)
code = Path('/content/wsp/screen_v5_geglu.py').read_text()
code = code.replace('v5_geglu', 'v11_hybrid')
Path('/content/wsp/screen_v11_hybrid.py').write_text(code)
""")

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 1. Synthesizing V11 Hybrid Pareto Adapters ===", flush=True)
    run_remote_stage("synth_v11_hybrid", f"{py_base} -u /content/wsp/synth_v11_hybrid.py", timeout_sec=600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v11_hybrid/manifest.json", str(HERE / "v11_hybrid_manifest.json")], check=True)
    print((HERE / "v11_hybrid_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 2. Screening V11 Hybrid Pareto Adapters in Native vLLM ===", flush=True)
    run_remote_stage("screen_v11_hybrid", f"{py_eval} -u /content/wsp/screen_v11_hybrid.py", timeout_sec=3000)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v11_hybrid/summary.json", str(HERE / "screen_v11_hybrid_summary.json")], check=True)

    remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_v11_hybrid')
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
Path('/content/wsp/harvest32/screen_v11_hybrid/unchecked_details.json').write_text(json.dumps(out, indent=2))
""")
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v11_hybrid/unchecked_details.json", str(HERE / "screen_v11_hybrid_unchecked_details.json")], check=True)
    print("=== V11 HYBRID PARETO SUMMARY ===", flush=True)
    print((HERE / "screen_v11_hybrid_summary.json").read_text(encoding="utf-8"), flush=True)
    print("=== V11 HYBRID PARETO UNCHECKED DETAILS ===", flush=True)
    print((HERE / "screen_v11_hybrid_unchecked_details.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v11_hybrid_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}

    cands = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        beats_random = rnd is None or (
            (r["unchecked_targeted_flips"], -r["unchecked_submits_left"], -r["unchecked_bare_pytest"])
            > (rnd["unchecked_targeted_flips"], -rnd["unchecked_submits_left"], -rnd["unchecked_bare_pytest"])
        )
        if (
            r["verified_preserved"] == "3/3"
            and r["edit_preserved_vs_champ"] == "4/4"
            and r["localize_preserved"] == "13/13"
            and r["unchecked_bare_pytest"] == 0
            and r["unchecked_submits_left"] <= 2
            and r["unchecked_targeted_flips"] >= 2
            and beats_random
        ):
            cands.append(r)

    if not cands:
        print("GATE STOP: No V11 candidate met strict 4/4 champ-edit + 3/3 verified + 13/13 localize + 0 bare-pytest + >=2 targeted flips gate!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            r["unchecked_defect_catches"],
            int(r["edit_exact_match"].split("/")[0]) >= 9,
            r["unchecked_targeted_flips"],
            -r["unchecked_submits_left"],
            int(r["edit_exact_match"].split("/")[0]),
            int(r["edit_preserved_abs"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v11_hybrid/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v11_hybrid_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V11 GATE PASSED! Launching QUICK_15 wsp32_iter03 on {winner['adapter']} (sha={sha}) ===", flush=True)
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
