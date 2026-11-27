"""Wait for already-running screen_v20_cascade to finish, download summary.json, and run gated wsp32_iter07."""
import json
import subprocess
import sys
import time

from provision_and_step2 import CLI, HERE, SESSION, ensure_session, remote_exec


def main():
    ensure_session()
    print("=== Waiting for already-running screen_v20_cascade to complete ===", flush=True)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        st = remote_exec("""
from pathlib import Path
done = Path('/content/wsp/screen_v20_cascade.rc')
log = Path('/content/wsp/screen_v20_cascade.log').read_text(errors='replace')[-1400:] if Path('/content/wsp/screen_v20_cascade.log').exists() else ''
print('TAIL:', log)
if done.is_file() and done.read_text().strip() != '':
    print('STAGE_RC=' + done.read_text().strip())
""", timeout=90)
        if "STAGE_RC=" in st:
            rc = int(next(l for l in st.splitlines() if l.startswith("STAGE_RC=")).split("=")[1])
            if rc != 0:
                raise RuntimeError(f"screen_v20_cascade failed (rc={rc}):\n{st}")
            print("[screen_v20_cascade] COMPLETED cleanly!", flush=True)
            break
        lines = [l for l in st.splitlines() if l.strip() and l != "WSP_OK"]
        if lines:
            print(f"[screen_v20_cascade] {lines[-1][:140]}", flush=True)
        time.sleep(15)
    else:
        raise TimeoutError("screen_v20_cascade timed out")

    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v20_cascade/summary.json", str(HERE / "screen_v20_cascade_summary.json")], check=True)
    print("=== V20 CASCADE SUMMARY ===", flush=True)
    print((HERE / "screen_v20_cascade_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v20_cascade_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ = by_name["champion_r1"]

    cands = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        b2_n = int(r["b2_err_to_read_flips"].split("/")[0])
        b3_n, b3_d = map(int, r["b3_valid_edit_kept"].split("/"))
        b4_n, b4_d = map(int, r["b4_verified_kept"].split("/"))
        b5_n, b5_d = map(int, r["b5_turn1_kept"].split("/"))

        score_cand = (r["b1_defect_catches"], r["b1_file_pytest"] + b2_n, -r["b1_submits_left"], -r["b1_slow_pytest"])
        score_champ = (champ["b1_defect_catches"], champ["b1_file_pytest"] + int(champ["b2_err_to_read_flips"].split("/")[0]), -champ["b1_submits_left"], -champ["b1_slow_pytest"])
        score_rnd = (
            (rnd["b1_defect_catches"], rnd["b1_file_pytest"] + int(rnd["b2_err_to_read_flips"].split("/")[0]), -rnd["b1_submits_left"], -rnd["b1_slow_pytest"])
            if rnd else (-1, -1, -99, -99)
        )

        if (
            r["syntax_leaks"] <= 1
            and b4_n == b4_d
            and b3_n >= max(1, b3_d - 1)
            and b5_n >= max(1, b5_d - 1)
            and r["b1_submits_left"] < champ["b1_submits_left"]
            and score_cand > score_champ
            and score_cand > score_rnd
        ):
            cands.append(r)

    if not cands:
        print("GATE STOP: Reviewing V20 summary before launching QUICK_15!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            r["syntax_leaks"] == 0,
            int(r["b2_err_to_read_flips"].split("/")[0]) >= 2,
            r["b1_defect_catches"],
            r["b1_file_pytest"] + int(r["b2_err_to_read_flips"].split("/")[0]),
            -r["b1_submits_left"],
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v20_cascade/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v20_cascade_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V20 GATE PASSED! Launching QUICK_15 wsp32_iter07 on {winner['adapter']} (sha={sha}) ===", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", str(HERE / "run_step5_and_step6.py"),
            "--mode", "quick15",
            "--run-name", "wsp32_iter07",
            "--remote-adapter", ad_path,
            "--sha", sha,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
