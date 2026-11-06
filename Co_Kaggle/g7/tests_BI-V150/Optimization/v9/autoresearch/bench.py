#!/usr/bin/env python3
"""Remote-native v9 server / immutable provided half-run helper.

Run ONLY on the authorized BI-V150 host at /root/Optimization/v9.
No SSH, rsync, other Optimization versions or external source edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from uuid import uuid4

V9 = Path("/root/Optimization/v9")
HERE = Path(__file__).resolve().parents[1]
WORK = V9 / ".autoresearch-work"
BENCH = Path("/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py")
MODEL = Path("/workspace/MiniCPM5-2B")
PID_FILE = WORK / "serve-owner.json"
PORT = 9031
CASES = "[[4096,1024,64,256],[16384,1024,64,128]]"

SERVE = [
    "vllm", "serve", str(MODEL),
    "--port", str(PORT),
    "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    "--served-model-name", "minicpm",
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "131072",
]

BENCHMARK = [
    "python3", str(BENCH),
    "--model", str(MODEL),
    "--served-model-name", "minicpm",
    "--port", str(PORT),
    "--test-cases", CASES,
]


def require_v9() -> None:
    if HERE != V9:
        raise SystemExit(
            f"Refusing non-v9 execution: script is under {HERE}, required {V9}"
        )
    if not (V9 / "FlagGems/src/flag_gems").is_dir():
        raise SystemExit("v9 FlagGems/src missing")
    if not (V9 / "vllm-plugin-FL/vllm_fl").is_dir():
        raise SystemExit("v9 vllm-plugin-FL missing")
    if not MODEL.is_dir():
        raise SystemExit(f"Model missing: {MODEL}")


def serving_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        (str(V9 / "FlagGems/src"), str(V9 / "vllm-plugin-FL"), env.get("PYTHONPATH", ""))
    )
    env["VLLM_PLUGINS"] = "fl"
    for key in ("VLLM_FL_USE_FLAGGEMS_ATTN", "VLLM_ATTENTION_BACKEND", "VLLM_BATCH_INVARIANT"):
        env.pop(key, None)
    return env


def start_ticks(pid: int) -> str:
    """Linux /proc stat field 22; guard against stale PID reuse."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[19]


def owner() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        record = json.loads(PID_FILE.read_text())
        pid = int(record["pid"])
        recorded_ticks = str(record["start_ticks"])
        if start_ticks(pid) != recorded_ticks:
            raise ValueError("PID reused")
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        if not (
            b"vllm" in cmdline
            and b"serve" in cmdline
            and str(MODEL).encode() in cmdline
            and os.getpgid(pid) == pid
        ):
            raise ValueError("Tracked process identity does not match")
        return pid
    except (ProcessLookupError, FileNotFoundError):
        PID_FILE.unlink(missing_ok=True)
        return None
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise SystemExit(f"Unsafe owner PID state, refusing to signal: {exc}") from exc


def ensure_port_free() -> None:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=1):
            raise SystemExit(f"Port {PORT} occupied. Do not disturb another server.")
    except (ConnectionRefusedError, TimeoutError, OSError):
        pass


def start() -> None:
    require_v9()
    WORK.mkdir(exist_ok=True)
    if owner() is not None:
        raise SystemExit("Tracked v9 vLLM server is running; stop it first.")
    ensure_port_free()
    with (WORK / "serve.log").open("ab", buffering=0) as log:
        child = subprocess.Popen(
            SERVE, cwd=V9, env=serving_env(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    record = {
        "pid": child.pid, "start_ticks": start_ticks(child.pid),
        "cwd": str(V9), "command": SERVE, "source_dirs": [
            str(V9 / "FlagGems/src"), str(V9 / "vllm-plugin-FL")
        ]
    }
    PID_FILE.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Started v9-owned server PID {child.pid}; log: {WORK / 'serve.log'}")


def ready(timeout: float = 900.0) -> None:
    require_v9()
    deadline = time.monotonic() + timeout
    while True:
        if owner() is None:
            raise SystemExit("No live v9-owned server: check serve.log")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/v1/models", timeout=5
            ) as response:
                if response.status == 200:
                    print(f"v9 server ready on {PORT}")
                    return
        except (OSError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise SystemExit("v9 server readiness timeout; inspect serve.log")
        time.sleep(3)


def stop() -> None:
    require_v9()
    pid = owner()
    if pid is None:
        print("No live tracked v9 server.")
        return
    os.killpg(pid, signal.SIGTERM)
    # Avoid killing anything else and do not silently send SIGKILL.
    for _ in range(30):
        try:
            if start_ticks(pid) != json.loads(PID_FILE.read_text())["start_ticks"]:
                break
        except (OSError, FileNotFoundError, ProcessLookupError):
            break
        time.sleep(0.25)
    PID_FILE.unlink(missing_ok=True)
    print(f"Sent SIGTERM to owned v9 server group {pid}")


def git_sha() -> str | None:
    """Remote v9 may be a standalone code copy, not a Git worktree."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(V9), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "-C", str(V9), "status", "--porcelain", "--", str(V9)],
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (subprocess.CalledProcessError, OSError):
        return None


def source_hashes() -> dict[str, str]:
    paths = (
        "vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py",
        "vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py",
        "FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm.py",
        "FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/linear.py",
    )
    return {
        name: hashlib.sha256((V9 / name).read_bytes()).hexdigest()
        for name in paths if (V9 / name).is_file()
    }


def run_tests() -> None:
    require_v9()
    subprocess.run(
        [
            "python3", "-m", "pytest", "-q", "-s", "--noconftest",
            "vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py",
            "vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py",
        ],
        cwd=V9, env=serving_env(), check=True,
    )


def benchmark() -> None:
    require_v9()
    if not BENCH.is_file():
        raise SystemExit(f"Provided READ-ONLY benchmark missing: {BENCH}")
    ready()
    run_dir = WORK / "bench_runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    )
    run_dir.mkdir(parents=True)
    benchmark_hash = hashlib.sha256(BENCH.read_bytes()).hexdigest()
    env = serving_env()
    with (run_dir / "benchmark.log").open("wb") as logfile:
        proc = subprocess.run(
            BENCHMARK, cwd=run_dir, env=env,
            stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
        )
    results = run_dir / "benchmark_results"
    raw = sorted(results.glob("raw_runs_*.csv"))
    summ = sorted(results.glob("summary_*.csv"))
    manifest = {
        "kind": "development_half_run",
        "remote_v9": str(V9),
        "v9_commit_sha": git_sha(),
        "v9_dirty": git_dirty(),
        "v9_kernel_file_sha256": source_hashes(),
        "server_owner_pid": owner(),
        "benchmark_path": str(BENCH),
        "benchmark_sha256_before": benchmark_hash,
        "benchmark_sha256_after": hashlib.sha256(BENCH.read_bytes()).hexdigest(),
        "command": BENCHMARK,
        "cwd": str(run_dir),
        "cases": CASES,
        "script_runs": 2,
        "script_skip_first": 0,
        "returncode": proc.returncode,
        "raw_csv": [str(x) for x in raw],
        "summary_csv": [str(x) for x in summ],
        "stdout": str(run_dir / "benchmark.log"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run: {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Original half-run CSVs: raw={raw} summary={summ}")
    if manifest["benchmark_sha256_before"] != manifest["benchmark_sha256_after"]:
        raise SystemExit("Provided benchmark changed DURING run; reject evidence")
    if proc.returncode or len(raw) != 1 or len(summ) != 1:
        raise SystemExit(f"Benchmark failed or missing expected CSVs; see {run_dir}")
    # The provided script can return code 0 while skipping failed summary rows.
    from evaluate import summary, validate_raw
    validate_raw(raw[0])
    summary(summ[0])
    print("Both 4K/16K cases and two complete successful raw runs per case verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "ready", "stop", "tests", "benchmark"))
    action = parser.parse_args().action
    if action == "start":
        start()
    elif action == "ready":
        ready()
    elif action == "stop":
        stop()
    elif action == "tests":
        run_tests()
    elif action == "benchmark":
        benchmark()


if __name__ == "__main__":
    main()
