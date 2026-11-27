"""Attach to already-running screen_v12_whitened on wsp32 (or resume cleanly), inspect summary, and launch gated QUICK_15 iter03."""
import json
import subprocess
import sys
import time
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, remote_exec
from run_step3_and_step4 import run_remote_stage


def wait_existing_or_resume_screen() -> None:
    log_path = "/content/wsp/screen_v12_whitened.log"
    done_path = "/content/wsp/screen_v12_whitened.rc"
    py_eval = "/content/wsp/eval_env/bin/python"

    st0 = remote_exec(f"""
import subprocess
from pathlib import Path
done = Path({done_path!r})
running = subprocess.run(['pgrep', '-f', 'screen_v12_whitened.py'], capture_output=True, text=True).stdout.strip()
print('DONE=' + (done.read_text().strip() if done.is_file() else 'NONE'))
print('PIDS=' + running)
""")
    print("Initial remote screen status:\n" + st0, flush=True)

    if "DONE=0" not in st0:
        if "PIDS=" in st0 and st0.split("PIDS=")[1].splitlines()[0].strip():
            print("Attaching to live running screen_v12_whitened.py...", flush=True)
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                st = remote_exec(f"""
from pathlib import Path
done = Path({done_path!r})
log = Path({log_path!r}).read_text(errors='replace')[-1400:] if Path({log_path!r}).exists() else ''
print('TAIL:', log)
if done.is_file() and done.read_text().strip() != '':
    print('STAGE_RC=' + done.read_text().strip())
""", timeout=120)
                if "STAGE_RC=" in st:
                    rc = int(next(l for l in st.splitlines() if l.startswith("STAGE_RC=")).split("=")[1])
                    if rc != 0:
                        raise RuntimeError(f"screen_v12_whitened failed (rc={rc}):\n{st}")
                    print("[screen_v12_whitened] COMPLETED cleanly!", flush=True)
                    break
                lines = [l for l in st.splitlines() if l.strip() and l != "WSP_OK"]
                if lines:
                    print(f"[screen_v12_whitened] {lines[-1][:140]}", flush=True)
                time.sleep(25)
            else:
                raise TimeoutError("screen_v12_whitened timed out")
        else:
            print("Resuming screen_v12_whitened.py (skips already-saved adapter .json files!)...", flush=True)
            run_remote_stage("screen_v12_whitened", f"{py_eval} -u /content/wsp/screen_v12_whitened.py", timeout_sec=3600)


def main():
    wait_existing_or_resume_screen()

    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v12_whitened/summary.json", str(HERE / "screen_v12_whitened_summary.json")], check=True)

    remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_v12_whitened')
for p in sorted(d.glob('*.json')):
    if p.name == 'summary.json':
        continue
    rows = json.loads(p.read_text())
    unch = [r for r in rows if r.get('role') == 'unchecked']
    if p.stem.endswith('_candidate') or p.stem == 'champion_r1' or 'zero' in p.stem:
        out[p.stem] = [
            {
                'issue': r['issue'], 'action': r.get('action'), 'command': r.get('command'),
                'is_bare_pytest': r.get('is_bare_pytest'), 'defect_catch': r.get('discriminating_defect_catch'),
                'agent_exit': r.get('on_returned_patch_exit'), 'ref_exit': r.get('on_reference_fix_exit'),
            }
            for r in unch
        ]
Path('/content/wsp/harvest32/screen_v12_whitened/unchecked_details.json').write_text(json.dumps(out, indent=2))
""")
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v12_whitened/unchecked_details.json", str(HERE / "screen_v12_whitened_unchecked_details.json")], check=True)
    print("=== V12 WHITENED SUMMARY ===", flush=True)
    print((HERE / "screen_v12_whitened_summary.json").read_text(encoding="utf-8"), flush=True)
    print("=== V12 WHITENED UNCHECKED DETAILS ===", flush=True)
    print((HERE / "screen_v12_whitened_unchecked_details.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v12_whitened_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ_row = by_name["champion_r1"]

    cands = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        beats_champ = (
            (r["unchecked_targeted_flips"], -r["unchecked_submits_left"], -r["unchecked_bare_pytest"])
            > (champ_row["unchecked_targeted_flips"], -champ_row["unchecked_submits_left"], -champ_row["unchecked_bare_pytest"])
        )
        beats_random = rnd is None or (
            (r["unchecked_targeted_flips"], -r["unchecked_submits_left"], -r["unchecked_bare_pytest"], int(r["edit_exact_match"].split("/")[0]))
            > (rnd["unchecked_targeted_flips"], -rnd["unchecked_submits_left"], -rnd["unchecked_bare_pytest"], int(rnd["edit_exact_match"].split("/")[0]))
        )
        ver_n, ver_d = map(int, r["verified_preserved"].split("/"))
        edc_n, edc_d = map(int, r["edit_preserved_vs_champ"].split("/"))
        loc_n, loc_d = map(int, r["localize_preserved"].split("/"))
        if (
            ver_n == ver_d
            and edc_n >= edc_d - 1
            and loc_n >= loc_d - 1
            and r["unchecked_bare_pytest"] == 0
            and r["unchecked_targeted_flips"] >= max(2, champ_row["unchecked_targeted_flips"] + 1)
            and beats_champ
            and beats_random
        ):
            cands.append(r)

    if not cands:
        print("GATE STOP: No V12 candidate met strict preservation + 0 bare-pytest + targeted flip improvement over champ & random!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            int(r["edit_preserved_vs_champ"].split("/")[0]) == int(r["edit_preserved_vs_champ"].split("/")[1]),
            int(r["edit_exact_match"].split("/")[0]) >= (int(r["edit_exact_match"].split("/")[1]) * 2) // 3,
            r["unchecked_defect_catches"],
            r["unchecked_targeted_flips"],
            -r["unchecked_submits_left"],
            int(r["edit_exact_match"].split("/")[0]),
            int(r["edit_preserved_abs"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v12_whitened/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v12_whitened_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V12 GATE PASSED! Launching QUICK_15 wsp32_iter03 on {winner['adapter']} (sha={sha}) ===", flush=True)
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
