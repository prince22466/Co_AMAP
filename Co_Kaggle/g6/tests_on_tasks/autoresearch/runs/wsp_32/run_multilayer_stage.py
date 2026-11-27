"""Upload and execute multi-layer coherent r=1 synthesis + native vLLM screening on Colab wsp32."""
import json
import subprocess
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def main():
    for fname in ("synthesize_multilayer_v2.py", "screen_multilayer_v2.py"):
        upload_verified(HERE / fname, f"/content/wsp/{fname}")

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 1. Synthesizing multi-layer r=1 disjoint candidates (prefix_end+thought + tool_pre) ===", flush=True)
    run_remote_stage(
        "synth_ml32",
        f"{py_base} -u /content/wsp/synthesize_multilayer_v2.py "
        f"--harvest-dir /content/wsp/harvest32 "
        f"--native-replays /content/wsp/harvest32/native/champion.json",
        timeout_sec=900,
    )
    subprocess.run(
        [CLI, "download", "-s", SESSION, "/content/wsp/harvest32/multilayer_candidates/manifest.json", str(HERE / "multilayer_manifest.json")],
        check=True,
    )
    print("=== MULTILAYER MANIFEST ===", flush=True)
    mf = json.loads((HERE / "multilayer_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({k: mf[k] for k in ("selected_mid", "selected_late", "candidates")}, indent=2), flush=True)

    print("=== 2. Screening multi-layer r=1 candidates vs champion_r1, zero, and random controls ===", flush=True)
    run_remote_stage(
        "screen_ml32",
        f"{py_eval} -u /content/wsp/screen_multilayer_v2.py "
        f"--harvest-dir /content/wsp/harvest32 "
        f"--runs /content/wsp/experiments/discovery02 /content/wsp/experiments/discovery03",
        timeout_sec=1800,
    )
    subprocess.run(
        [CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_multilayer_r1/summary.json", str(HERE / "screen_multilayer_summary.json")],
        check=True,
    )

    # Also pull detailed unchecked decisions for candidates that flip >= 2 unchecked cases!
    det = remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_multilayer_r1')
for p in sorted(d.glob('*.json')):
    if p.name == 'summary.json':
        continue
    rows = json.loads(p.read_text())
    unch = [r for r in rows if r.get('role') == 'unchecked']
    if sum(1 for r in unch if r.get('flipped_to_targeted_check')) >= 1 or p.stem == 'champion_r1':
        out[p.stem] = [
            {
                'issue': r['issue'],
                'action': r.get('action'),
                'command': r.get('command'),
                'defect_catch': r.get('discriminating_defect_catch'),
                'agent_exit': r.get('on_returned_patch_exit'),
                'ref_exit': r.get('on_reference_fix_exit'),
            }
            for r in unch
        ]
Path('/content/wsp/harvest32/screen_multilayer_r1/unchecked_details.json').write_text(json.dumps(out, indent=2))
print('Saved unchecked_details.json for', len(out), 'adapters')
""")
    print(det, flush=True)
    subprocess.run(
        [CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_multilayer_r1/unchecked_details.json", str(HERE / "screen_multilayer_unchecked_details.json")],
        check=True,
    )

    print("\n=== MULTI-LAYER NATIVE SCREEN SUMMARY ===", flush=True)
    print((HERE / "screen_multilayer_summary.json").read_text(encoding="utf-8"), flush=True)
    print("\n=== UNCHECKED CHECK EXECUTION DETAILS ===", flush=True)
    print((HERE / "screen_multilayer_unchecked_details.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
