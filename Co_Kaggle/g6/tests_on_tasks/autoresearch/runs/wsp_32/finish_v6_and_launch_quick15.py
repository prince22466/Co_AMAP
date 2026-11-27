"""Wait for the final adapter of screen_v6_l41, download V6 summary, and run gated QUICK_15 wsp32_iter02."""
import json
import subprocess
import sys
import time

from provision_and_step2 import CLI, HERE, SESSION, remote_exec


def main():
    deadline = time.monotonic() + 450
    while time.monotonic() < deadline:
        st = remote_exec("""
from pathlib import Path
rc_p = Path('/content/wsp/screen_v6_l41.rc')
if rc_p.is_file() and rc_p.read_text().strip() != '':
    print('V6_DONE_RC=' + rc_p.read_text().strip())
""", timeout=60)
        if "V6_DONE_RC=" in st:
            print(st.strip(), flush=True)
            break
        time.sleep(15)

    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v6_l41/summary.json", str(HERE / "screen_v6_l41_summary.json")], check=True)
    remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_v6_l41')
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
Path('/content/wsp/harvest32/screen_v6_l41/unchecked_details.json').write_text(json.dumps(out, indent=2))
""")
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v6_l41/unchecked_details.json", str(HERE / "screen_v6_l41_unchecked_details.json")], check=True)

    print("=== V6 L41 SUMMARY ===", flush=True)
    print((HERE / "screen_v6_l41_summary.json").read_text(encoding="utf-8"), flush=True)
    print("=== V6 L41 UNCHECKED DETAILS ===", flush=True)
    print((HERE / "screen_v6_l41_unchecked_details.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v6_l41_summary.json").read_text(encoding="utf-8"))
    cands = [
        r for r in rows
        if r["adapter"].endswith("_candidate")
        and r["verified_preserved"] == "3/3"
        and r["edit_preserved_vs_champ"] == "4/4"
        and r["localize_preserved"] == "13/13"
        and (r["unchecked_submits_left"] <= 2 or r["unchecked_targeted_flips"] >= 2)
    ]
    if not cands:
        print("GATE STOP: No V6 L41 candidate met strict 4/4 edit + 3/3 verified + unchecked flip gate!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            r["unchecked_defect_catches"],
            int(r["edit_exact_match"].split("/")[0]) >= 6,
            r["unchecked_targeted_flips"],
            -r["unchecked_bare_pytest"],
            -r["unchecked_submits_left"],
            int(r["edit_exact_match"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v6_l41/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v6_l41_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V6 GATE PASSED! Launching QUICK_15 wsp32_iter02 on {winner['adapter']} (sha={sha}) ===", flush=True)
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
