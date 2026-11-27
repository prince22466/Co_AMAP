"""Run V7 Bit-Exact Per-Token GeGLU synthesis, native vLLM screen, and gated QUICK_15 iter02."""
import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def main():
    upload_verified(HERE / "remote_synth_v7_bitexact_geglu.py", "/content/wsp/synth_v7_geglu.py")

    remote_exec("""
from pathlib import Path
code = Path('/content/wsp/screen_v5_geglu.py').read_text()
code = code.replace('v5_geglu', 'v7_geglu')
Path('/content/wsp/screen_v7_geglu.py').write_text(code)
""")

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 1. Synthesizing V7 Bit-Exact Per-Token GeGLU Transistors ===", flush=True)
    run_remote_stage("synth_v7_geglu", f"{py_base} -u /content/wsp/synth_v7_geglu.py", timeout_sec=600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v7_geglu/manifest.json", str(HERE / "v7_geglu_manifest.json")], check=True)
    print((HERE / "v7_geglu_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 2. Screening V7 Bit-Exact Per-Token GeGLU Transistors in Native vLLM ===", flush=True)
    run_remote_stage("screen_v7_geglu", f"{py_eval} -u /content/wsp/screen_v7_geglu.py", timeout_sec=2400)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v7_geglu/summary.json", str(HERE / "screen_v7_geglu_summary.json")], check=True)

    remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_v7_geglu')
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
Path('/content/wsp/harvest32/screen_v7_geglu/unchecked_details.json').write_text(json.dumps(out, indent=2))
""")
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v7_geglu/unchecked_details.json", str(HERE / "screen_v7_geglu_unchecked_details.json")], check=True)
    print("=== V7 BIT-EXACT GEGLU SUMMARY ===", flush=True)
    print((HERE / "screen_v7_geglu_summary.json").read_text(encoding="utf-8"), flush=True)
    print("=== V7 BIT-EXACT GEGLU UNCHECKED DETAILS ===", flush=True)
    print((HERE / "screen_v7_geglu_unchecked_details.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v7_geglu_summary.json").read_text(encoding="utf-8"))
    cands = [
        r for r in rows
        if r["adapter"].endswith("_candidate")
        and r["verified_preserved"] == "3/3"
        and r["edit_preserved_vs_champ"] == "4/4"
        and r["localize_preserved"] == "13/13"
        and (r["unchecked_submits_left"] <= 2 or r["unchecked_targeted_flips"] >= 2)
    ]
    if not cands:
        print("GATE STOP: No V7 candidate met strict 4/4 edit + 3/3 verified + unchecked flip gate!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            r["unchecked_defect_catches"],
            int(r["edit_exact_match"].split("/")[0]) >= 8,
            r["unchecked_bare_pytest"] == 0,
            r["unchecked_targeted_flips"],
            -r["unchecked_submits_left"],
            int(r["edit_exact_match"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v7_geglu/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v7_geglu_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V7 GATE PASSED! Launching QUICK_15 wsp32_iter02 on {winner['adapter']} (sha={sha}) ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(HERE / "run_step5_and_step6.py"),
            "--mode",
            "quick15",
            "--run-name",
            "wsp32_iter02",
            "--remote-adapter",
            ad_path,
            "--sha",
            sha,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
