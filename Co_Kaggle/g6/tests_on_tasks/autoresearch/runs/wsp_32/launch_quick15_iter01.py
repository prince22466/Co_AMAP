"""Fetch exact SHA-256 of V4_ml2_35_41_g080_candidate and launch QUICK_15 benchmark (wsp32_iter01)."""
import subprocess
import sys

from provision_and_step2 import HERE, remote_exec


def main():
    ad_path = "/content/wsp/harvest32/screen_v4_r1/adapters/V4_ml2_35_41_g080_candidate"
    out = remote_exec(
        f"import hashlib; from pathlib import Path; "
        f"print('SHA=' + hashlib.sha256((Path({ad_path!r})/'adapter_model.safetensors').read_bytes()).hexdigest())"
    )
    sha = next(l.split("=")[1].strip() for l in out.splitlines() if l.startswith("SHA="))
    print(f"Launching QUICK_15 wsp32_iter01 on {ad_path} (sha={sha})", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(HERE / "run_step5_and_step6.py"),
            "--mode",
            "quick15",
            "--run-name",
            "wsp32_iter01",
            "--remote-adapter",
            ad_path,
            "--sha",
            sha,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
