"""Download completed discovery03/ranking/geglu reports and wait for running screen_native32 to finish."""
import json
import subprocess
import time
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, remote_exec


def main():
    for rem, loc in (
        ("/content/wsp/experiments/discovery03/neon_metrics.json", HERE / "discovery03_metrics.json"),
        ("/content/wsp/experiments/discovery03/neon_failures.jsonl", HERE / "discovery03_failures.jsonl"),
        ("/content/wsp/harvest32/candidates/ranking.json", HERE / "ranking_v2.json"),
        ("/content/wsp/harvest32/geglu_candidates/geglu_report.json", HERE / "geglu_report_v2.json"),
    ):
        subprocess.run([CLI, "download", "-s", SESSION, rem, str(loc)], check=False)

    print("=== DISCOVERY03 METRICS ===", flush=True)
    if (HERE / "discovery03_metrics.json").is_file():
        print((HERE / "discovery03_metrics.json").read_text(), flush=True)

    print("=== GEGLU REPORT V2 ===", flush=True)
    if (HERE / "geglu_report_v2.json").is_file():
        print((HERE / "geglu_report_v2.json").read_text(), flush=True)

    print("=== RANKING V2 CANDIDATES ===", flush=True)
    if (HERE / "ranking_v2.json").is_file():
        rk = json.loads((HERE / "ranking_v2.json").read_text())
        print(json.dumps(rk["candidates"], indent=2), flush=True)

    # Wait for detached screen_native32 to complete
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        st = remote_exec("""
from pathlib import Path
rc_p = Path('/content/wsp/screen_native32.rc')
log = Path('/content/wsp/screen_native32.log').read_text(errors='replace')[-2000:] if Path('/content/wsp/screen_native32.log').exists() else ''
print(log)
if rc_p.is_file() and rc_p.read_text().strip() != '':
    print('SCREEN_DONE_RC=' + rc_p.read_text().strip())
""", timeout=90)
        for line in st.splitlines():
            if line.startswith("SCREEN_RESULT "):
                print(" ", line, flush=True)
        if "SCREEN_DONE_RC=" in st:
            rc = int(next(l for l in st.splitlines() if l.startswith("SCREEN_DONE_RC=")).split("=")[1])
            print("screen_native32 finished with rc=", rc, flush=True)
            if rc != 0:
                print(st[-2000:], flush=True)
            break
        time.sleep(20)

    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_r1/summary.json", str(HERE / "screen_r1_summary.json")], check=True)
    print("\n=== FULL NATIVE vLLM SCREEN SUMMARY ===", flush=True)
    print((HERE / "screen_r1_summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
