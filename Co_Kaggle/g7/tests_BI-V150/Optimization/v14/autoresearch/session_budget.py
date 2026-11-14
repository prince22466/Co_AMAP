#!/usr/bin/env python3
"""v12 budget guard; fail closed and run the unchanged full two-case half-run.

A zero exit code is not proof of successful raw CSV rows or reproducible gains.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

if __package__:
    from .checkpoint import ROOT, checkpoint
else:
    from checkpoint import ROOT, checkpoint

STATE = ROOT / ".autoresearch-work" / "session_budget.json"
BENCHMARK = Path("/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py")


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def start_session(duration_minutes: float | None, deadline_utc: str | None,
                  now: float | None = None) -> dict:
    if (duration_minutes is None) == (deadline_utc is None):
        raise ValueError("Provide exactly one of duration or UTC deadline")
    now = time.time() if now is None else now
    if duration_minutes is not None:
        if duration_minutes <= 0:
            raise ValueError("Duration must be positive")
        deadline = now + duration_minutes * 60
    else:
        parsed = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError("Deadline must include a timezone")
        deadline = parsed.timestamp()
    if deadline <= now:
        raise ValueError("Deadline already passed")
    return {
        "started_epoch": now, "deadline_epoch": deadline,
        "baseline_run_minutes": 105.0,  # observed half-run duration
        "per_run_margin_minutes": 20.0,
        "preservation_reserve_minutes": 25.0,
        "observed_runs_minutes": [],
        "deadline_provenance": "operator-confirmed",
    }


def read_state(path: Path = STATE) -> dict:
    if not path.exists():
        raise ValueError("Unknown GPU deadline; operator must run init first")
    data = json.loads(path.read_text())
    if data.get("deadline_epoch", 0) <= data.get("started_epoch", 0):
        raise ValueError("Invalid deadline in budget state")
    return data


def decision(state: dict, runs: int = 1, now: float | None = None) -> dict:
    if runs < 1:
        raise ValueError("Runs must be positive")
    now = time.time() if now is None else now
    remaining = max(0.0, (state["deadline_epoch"] - now) / 60)
    durations = state.get("observed_runs_minutes", [])
    estimate = max([state["baseline_run_minutes"], *durations[-5:]])
    per_run = estimate + state["per_run_margin_minutes"]
    required = runs * per_run + state["preservation_reserve_minutes"]
    return {
        "remaining_minutes": round(remaining, 2),
        "estimate_per_run_minutes": round(per_run, 2),
        "preservation_reserve_minutes": state["preservation_reserve_minutes"],
        "runs_requested": runs, "required_minutes": round(required, 2),
        "admitted": remaining >= required,
        "reason": "admitted" if remaining >= required else
                  "not enough verified time; use short diagnostics/checkpoints only",
    }


def execute_half(state: dict, state_path: Path, push: bool = False) -> int:
    verdict = decision(state)
    if not verdict["admitted"]:
        print(json.dumps(verdict, indent=2), file=sys.stderr)
        return 2
    if not BENCHMARK.is_file():
        raise FileNotFoundError(f"External benchmark missing: {BENCHMARK}")
    with urlopen("http://127.0.0.1:9031/health", timeout=5) as result:
        if result.status != 200:
            raise RuntimeError("vLLM server is not healthy")
    root = ROOT / ".autoresearch-work" / "bench_runs"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"guard_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    run_dir.mkdir()
    sha = hashlib.sha256(BENCHMARK.read_bytes()).hexdigest()
    cmd = [
        sys.executable, str(BENCHMARK),
        "--model", "/workspace/MiniCPM5-2B",
        "--served-model-name", "minicpm", "--port", "9031",
        "--test-cases", "[[4096,1024,64,256],[16384,1024,64,128]]",
    ]
    record = {
        "run_dir": str(run_dir), "benchmark_sha256": sha, "command": cmd,
        "started_epoch": time.time(), "status": "STARTED_UNVERIFIED",
    }
    save(run_dir / "run.json", record)
    # Publish before an expensive run so a credits timeout does not erase its intent.
    checkpoint(ROOT, "Starting unchanged v12 half-run",
               f"Manifest: {run_dir / 'run.json'}; benchmark SHA256={sha}",
               "After run, validate all four SUCCESS rows and persist raw evidence",
               "provisional", verdict["remaining_minutes"], push)
    # A slow git push or stalled preflight may have consumed the allocation.
    # Re-admit immediately before launch; never start inside the safety reserve.
    refreshed = decision(state)
    if not refreshed["admitted"]:
        record.update({"status": "CANCELLED_NOT_STARTED", "finished_epoch": time.time()})
        save(run_dir / "run.json", record)
        checkpoint(ROOT, "Benchmark cancelled after preflight deadline recheck",
                   f"Run manifest: {run_dir / 'run.json'}; remaining={refreshed['remaining_minutes']}",
                   "Preserve current evidence; do not start a half-run", "blocked",
                   refreshed["remaining_minutes"], push)
        print(json.dumps(refreshed, indent=2), file=sys.stderr)
        return 2
    hard_seconds = int(state["deadline_epoch"] - time.time()
                       - 60 * state["preservation_reserve_minutes"])
    if hard_seconds <= 0:
        return 2
    timed_out = False
    with (run_dir / "benchmark.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(cmd, cwd=run_dir, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=hard_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = 124
    elapsed = (time.time() - record["started_epoch"]) / 60
    record.update({
        "finished_epoch": time.time(), "elapsed_minutes": round(elapsed, 2),
        "returncode": returncode,
        "status": "TIMED_OUT_INCOMPLETE" if timed_out else
                  ("EXIT_ZERO_CSV_UNVERIFIED" if returncode == 0 else "FAILED_INCOMPLETE"),
    })
    save(run_dir / "run.json", record)
    state.setdefault("observed_runs_minutes", []).append(elapsed)
    save(state_path, state)
    csv_files = sorted(
        p for p in run_dir.rglob("*.csv")
        if p.name.startswith(("raw_runs", "summary"))
    )
    checkpoint(ROOT, f"Half-run ended: {record['status']}",
               f"Elapsed={elapsed:.1f} min; rc={returncode}; raw/log: {run_dir}; SHA256={sha}",
               "Validate raw CSV and independent A/B; back up GPU traces off-instance",
               "provisional" if returncode == 0 else "blocked",
               decision(state)["remaining_minutes"], push, csv_files)
    print(json.dumps(record, indent=2))
    return 0 if returncode == 0 else (124 if timed_out else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--duration-minutes", type=float)
    init.add_argument("--deadline-utc")
    admit = sub.add_parser("admit")
    admit.add_argument("--runs", type=int, default=1)
    sub.add_parser("status")
    rec = sub.add_parser("record")
    rec.add_argument("--elapsed-minutes", type=float, required=True)
    runner = sub.add_parser("run-half")
    runner.add_argument("--push-checkpoint", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "init":
            state = start_session(args.duration_minutes, args.deadline_utc)
            save(args.state, state)
            print(json.dumps(decision(state), indent=2))
            return 0
        state = read_state(args.state)
        if args.command in ("status", "admit"):
            report = decision(state, getattr(args, "runs", 1))
            print(json.dumps(report, indent=2))
            return 0 if args.command == "status" or report["admitted"] else 2
        if args.command == "record":
            if args.elapsed_minutes <= 0:
                raise ValueError("Elapsed minutes must be positive")
            state.setdefault("observed_runs_minutes", []).append(args.elapsed_minutes)
            save(args.state, state)
            print(json.dumps(decision(state), indent=2))
            return 0
        return execute_half(state, args.state, args.push_checkpoint)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Budget guard failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
