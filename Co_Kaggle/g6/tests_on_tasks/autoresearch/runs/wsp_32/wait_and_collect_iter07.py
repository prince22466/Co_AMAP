"""Wait for active wsp32_iter07 QUICK_15 run to finish, collect metrics, and run wsp32_iter08 (firm cascade) if < 7/15."""
import json
import subprocess
import sys
import time
from pathlib import Path

from provision_and_step2 import CLI, HERE, RESEARCH, SESSION, ensure_session, remote_exec
from run_step5_and_step6 import CHAMPION_SOLVES, package_verified_submission


def collect_live_iter07(expected_sha: str, timeout_sec: int = 4500) -> dict:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        st = remote_exec("""
import subprocess
from pathlib import Path
ps = subprocess.run(['pgrep', '-af', 'wsp32_iter07'], capture_output=True, text=True).stdout.strip()
mp = Path('/content/wsp/experiments/wsp32_iter07/neon_metrics.json')
log = Path('/content/wsp/eval_wsp32_iter07.log')
tail = ''
if log.is_file():
    raw = log.read_bytes()[-1800:].replace(b'\\x00', b'').decode('utf-8', errors='replace')
    lines = [l for l in raw.splitlines() if l.strip()]
    tail = lines[-1][:140] if lines else ''
print('TAIL:', tail)
print('HAS_METRICS=' + str(mp.is_file()))
print('ALIVE=' + str(bool(ps)))
""", timeout=90)
        if "HAS_METRICS=True" in st and "ALIVE=False" in st:
            break
        if "ALIVE=False" in st and "HAS_METRICS=False" in st:
            raise RuntimeError(f"wsp32_iter07 exited without neon_metrics.json:\n{st}")
        for l in st.splitlines():
            if l.startswith("TAIL:"):
                print(f"[wsp32_iter07] {l}", flush=True)
        time.sleep(25)
    else:
        raise TimeoutError("Timed out waiting for wsp32_iter07")

    out_local = HERE / "evals/wsp32_iter07"
    out_local.mkdir(parents=True, exist_ok=True)
    for fn in ("neon_metrics.json", "neon_failures.jsonl"):
        subprocess.run([CLI, "download", "-s", SESSION, f"/content/wsp/experiments/wsp32_iter07/{fn}", str(out_local / fn)], check=True)
    m = json.loads((out_local / "neon_metrics.json").read_text(encoding="utf-8"))
    assert m["adapter_sha256"] == expected_sha
    resolved = set(m["resolved_ids"])
    m["retained_champion_solves"] = sorted(CHAMPION_SOLVES & resolved)
    m["lost_champion_solves"] = sorted(CHAMPION_SOLVES - resolved)
    m["newly_solved"] = sorted(resolved - CHAMPION_SOLVES)
    print("wsp32_iter07 METRICS:", json.dumps(m, indent=2), flush=True)

    if m["solved"] >= 7 and not m["lost_champion_solves"] and m["syntax_leaks"] <= 1:
        local_ad = out_local / "adapter"
        local_ad.mkdir(parents=True, exist_ok=True)
        ad_rem = "/content/wsp/harvest32/v20_cascade/adapters/V20_casc_39_40_41_sweet_candidate"
        for fn in ("adapter_config.json", "adapter_model.safetensors"):
            subprocess.run([CLI, "download", "-s", SESSION, f"{ad_rem}/{fn}", str(local_ad / fn)], check=True)
        kept_zip = RESEARCH / "kept_submissions" / f"submission_wsp_wsp32_iter07_{m['solved']}of15_{expected_sha[:12]}.zip"
        print("ARCHIVED WINNING SUBMISSION:", json.dumps(package_verified_submission(local_ad, kept_zip), indent=2), flush=True)
    return m


def main():
    ensure_session()
    mf = json.loads((HERE / "v20_cascade_manifest.json").read_text(encoding="utf-8"))
    shas = {x["adapter"]: x["sha256"] for x in mf["compiled"]}

    m7 = collect_live_iter07(shas["V20_casc_39_40_41_sweet_candidate"])
    if m7["solved"] >= 7 and not m7["lost_champion_solves"] and m7["syntax_leaks"] <= 1:
        print("=== TARGET >= 7/15 REACHED ON wsp32_iter07! ===", flush=True)
        return

    print("\n=== Launching QUICK_15 wsp32_iter08 on V20_casc_39_40_41_firm_candidate ===", flush=True)
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
