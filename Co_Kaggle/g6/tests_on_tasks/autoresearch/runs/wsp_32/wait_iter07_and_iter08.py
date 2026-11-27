"""Attach to the live wsp32_iter07 QUICK_15 evaluation on wsp32, download results, and run wsp32_iter08 (V20_firm) if <7/15."""
import json
import subprocess
import sys
import time

from provision_and_step2 import CLI, HERE, RESEARCH, SESSION, ensure_session, remote_exec
from run_step5_and_step6 import CHAMPION_SOLVES, package_verified_submission


def wait_for_running_iter07(expected_sha: str, timeout_sec: int = 5400) -> dict:
    # Remove the duplicate-retry stale .rc file ONLY if wsp32_iter07 is still actively running!
    remote_exec("""
import subprocess
from pathlib import Path
ps = subprocess.run(['pgrep', '-af', 'name wsp32_iter07'], capture_output=True, text=True).stdout.strip()
rc = Path('/content/wsp/eval_wsp32_iter07.rc')
if ps and rc.is_file() and rc.read_text().strip() == '1':
    rc.unlink()
    print('Cleared stale duplicate-retry rc=1 while live wsp32_iter07 is running:', ps.splitlines()[0])
else:
    print('Status check ps=', bool(ps), 'rc_exists=', rc.is_file())
""", timeout=60)

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        st = remote_exec("""
import subprocess
from pathlib import Path
ps = subprocess.run(['pgrep', '-af', 'name wsp32_iter07'], capture_output=True, text=True).stdout.strip()
nm = Path('/content/wsp/experiments/wsp32_iter07/neon_metrics.json')
log = Path('/content/wsp/eval_wsp32_iter07.log')
tail = log.read_text(errors='replace').replace('\\x00', '')[-600:] if log.is_file() else ''
print('TAIL:', tail.splitlines()[-1] if tail.splitlines() else '')
print('METRICS_READY=' + str(nm.is_file()))
print('ALIVE=' + str(bool(ps)))
""", timeout=90)
        for ln in st.splitlines():
            if ln.startswith("TAIL:"):
                print(f"[wsp32_iter07] {ln[5:].strip()[:130]}", flush=True)
        if "METRICS_READY=True" in st and "ALIVE=False" in st:
            print("[wsp32_iter07] COMPLETED cleanly!", flush=True)
            break
        if "METRICS_READY=False" in st and "ALIVE=False" in st:
            raise RuntimeError(f"wsp32_iter07 exited without writing neon_metrics.json:\n{st}")
        time.sleep(25)
    else:
        raise TimeoutError("wsp32_iter07 timed out")

    out_local = HERE / "evals/wsp32_iter07"
    out_local.mkdir(parents=True, exist_ok=True)
    for fn in ("neon_metrics.json", "neon_failures.jsonl"):
        subprocess.run([CLI, "download", "-s", SESSION, f"/content/wsp/experiments/wsp32_iter07/{fn}", str(out_local / fn)], check=True)
    m = json.loads((out_local / "neon_metrics.json").read_text(encoding="utf-8"))
    assert m["adapter_sha256"] == expected_sha
    assert m["enable_lora"] is True
    resolved = set(m["resolved_ids"])
    m["retained_champion_solves"] = sorted(CHAMPION_SOLVES & resolved)
    m["lost_champion_solves"] = sorted(CHAMPION_SOLVES - resolved)
    m["newly_solved"] = sorted(resolved - CHAMPION_SOLVES)
    print("QUICK_15 METRICS (wsp32_iter07):", json.dumps(m, indent=2), flush=True)
    if m["solved"] >= 7 and not m["lost_champion_solves"] and m["syntax_leaks"] <= 1:
        local_ad = out_local / "adapter"
        local_ad.mkdir(parents=True, exist_ok=True)
        rem_ad = "/content/wsp/harvest32/v20_cascade/adapters/V20_casc_39_40_41_sweet_candidate"
        for fn in ("adapter_config.json", "adapter_model.safetensors"):
            subprocess.run([CLI, "download", "-s", SESSION, f"{rem_ad}/{fn}", str(local_ad / fn)], check=True)
        kept_zip = RESEARCH / "kept_submissions" / f"submission_wsp_wsp32_iter07_{m['solved']}of15_{expected_sha[:12]}.zip"
        pkg = package_verified_submission(local_ad, kept_zip)
        print("ARCHIVED WINNING SUBMISSION:", json.dumps(pkg, indent=2), flush=True)
    return m


def main():
    ensure_session()
    # Also ensure v20_adapters_backup.zip is saved locally!
    bkp = HERE / "checkpoints/v20_adapters_backup.zip"
    if not bkp.is_file():
        remote_exec("""
import zipfile
from pathlib import Path
out_dir = Path('/content/wsp/harvest32/v20_cascade')
with zipfile.ZipFile('/content/v20_adapters_backup.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zb:
    for p in out_dir.rglob('*'):
        if p.is_file():
            zb.write(p, p.relative_to(out_dir))
""", timeout=90)
        subprocess.run([CLI, "download", "-s", SESSION, "/content/v20_adapters_backup.zip", str(bkp)], check=True)

    mf = json.loads((HERE / "v20_cascade_manifest.json").read_text(encoding="utf-8"))
    shas = {x["adapter"]: x["sha256"] for x in mf["compiled"]}

    m7 = wait_for_running_iter07(shas["V20_casc_39_40_41_sweet_candidate"])
    if m7["solved"] >= 7 and not m7["lost_champion_solves"] and m7["syntax_leaks"] <= 1:
        print("=== TARGET >= 7/15 ACHIEVED ON wsp32_iter07! ===", flush=True)
        return

    print("\n=== Launching wsp32_iter08 on V20_casc_39_40_41_firm_candidate (b2_err_to_read=3/8, b3_exact=5/6, leaks=0) ===", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", str(HERE / "run_step5_and_step6.py"),
            "--mode", "quick15",
            "--run-name", "wsp32_iter08",
            "--remote-adapter", "/content/wsp/harvest32/v20_cascade/adapters/V20_casc_39_40_41_firm_candidate",
            "--sha", shas["V20_casc_39_40_41_firm_candidate"],
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
