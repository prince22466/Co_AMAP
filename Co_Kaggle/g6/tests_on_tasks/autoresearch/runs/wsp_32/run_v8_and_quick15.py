"""Run V8 Pure-L41 Causal-Readout GeGLU + Gentle Interpolation screen and launch gated QUICK_15 iter02."""
import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def main():
    upload_verified(HERE / "remote_synth_v8_pure41.py", "/content/wsp/synth_v8_pure41.py")

    remote_exec("""
from pathlib import Path
code = Path('/content/wsp/screen_v5_geglu.py').read_text()
code = code.replace('v5_geglu', 'v8_pure41')
Path('/content/wsp/screen_v8_pure41.py').write_text(code)
""")

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 1. Synthesizing V8 Pure-L41 Causal-Readout GeGLU + Gentle Interpolations ===", flush=True)
    run_remote_stage("synth_v8_pure41", f"{py_base} -u /content/wsp/synth_v8_pure41.py", timeout_sec=600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v8_pure41/manifest.json", str(HERE / "v8_pure41_manifest.json")], check=True)
    print((HERE / "v8_pure41_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 2. Screening V8 adapters in Native vLLM ===", flush=True)
    run_remote_stage("screen_v8_pure41", f"{py_eval} -u /content/wsp/screen_v8_pure41.py", timeout_sec=2400)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v8_pure41/summary.json", str(HERE / "screen_v8_pure41_summary.json")], check=True)

    remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_v8_pure41')
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
Path('/content/wsp/harvest32/screen_v8_pure41/unchecked_details.json').write_text(json.dumps(out, indent=2))
""")
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v8_pure41/unchecked_details.json", str(HERE / "screen_v8_pure41_unchecked_details.json")], check=True)
    print("=== V8 PURE41 SUMMARY ===", flush=True)
    print((HERE / "screen_v8_pure41_summary.json").read_text(encoding="utf-8"), flush=True)
    print("=== V8 PURE41 UNCHECKED DETAILS ===", flush=True)
    print((HERE / "screen_v8_pure41_unchecked_details.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v8_pure41_summary.json").read_text(encoding="utf-8"))
    # Require 4/4 champion edit preservation, 3/3 verified, 13/13 localize, and strictly better unchecked flips than zero/champion!
    cands = [
        r for r in rows
        if r["adapter"].endswith("_candidate")
        and r["verified_preserved"] == "3/3"
        and r["edit_preserved_vs_champ"] == "4/4"
        and r["localize_preserved"] == "13/13"
        and (r["unchecked_submits_left"] <= 3 and (r["unchecked_targeted_flips"] >= 1 or r["unchecked_bare_pytest"] == 0))
    ]
    if not cands:
        print("GATE STOP: No V8 candidate met 4/4 edit + 3/3 verified + unchecked improvement gate!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            r["unchecked_defect_catches"],
            int(r["edit_exact_match"].split("/")[0]) >= 9,
            r["unchecked_targeted_flips"],
            -r["unchecked_bare_pytest"],
            -r["unchecked_submits_left"],
            int(r["edit_exact_match"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v8_pure41/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v8_pure41_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V8 GATE PASSED! Launching QUICK_15 wsp32_iter02 on {winner['adapter']} (sha={sha}) ===", flush=True)
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
