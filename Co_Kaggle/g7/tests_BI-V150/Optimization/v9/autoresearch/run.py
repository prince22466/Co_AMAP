#!/usr/bin/env python3
"""Bounded Codex research supervisor. No remote GPU work occurs until launched."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

V9 = Path(__file__).resolve().parents[1]
WORK = V9 / ".autoresearch-work"
REPORT = WORK / "round-report.json"
PR_PATTERN = re.compile(r"^https://github\.com/prince22466/Co_AMAP/pull/[1-9][0-9]*$")


def preflight() -> Path:
    for tool in ("codex", "ssh", "rsync", "gh", "git"):
        if not shutil.which(tool):
            raise SystemExit(f"Missing required program: {tool}")
    if not os.getenv("V9_SSH_HOST"):
        raise SystemExit("Set V9_SSH_HOST to your authorized ~/.ssh/config host alias")
    root = subprocess.check_output(
        ["git", "-C", str(V9), "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if not root:
        raise SystemExit("v9 must be inside a Git checkout")
    return Path(root)


def run_codex(round_id: int, seconds: float, repo_root: Path) -> bool:
    WORK.mkdir(parents=True, exist_ok=True)
    REPORT.unlink(missing_ok=True)
    log_path = WORK / f"codex-round-{round_id:03d}.log"
    prompt = (
        "You are one bounded turn in the MiniCPM5-2B BI-V150 autoresearch loop. "
        f"This is ROUND {round_id}; remaining allowed wall time for this turn "
        f"is at most {int(seconds)} seconds. Read AGENTS.md, "
        "autoresearch/GOAL.md and autoresearch/README.md first. "
        "Only experiment on the authorized V9_SSH_HOST using the local "
        "autoresearch/remote.py bridge. Follow the ONE-hypothesis cycle; "
        "run official full workload and math_500 on the real machine. "
        "Never edit benchmarks or violate competition restrictions. "
        "Accept only a scorer-passing, demonstrably causal serving improvement. "
        "If accepted, commit kernel/tests/findings, push and OPEN a GitHub PR "
        "immediately even below target, then prepare to continue on the best "
        "branch. Update tracked autoresearch/EXPERIMENTS.md. "
        f"Write .autoresearch-work/round-report.json with round={round_id} "
        "and status, gate_report, pr_url, notes exactly as specified in GOAL.md. "
        "If resources or auth are unavailable, mark blocked; do not invent results. "
        "Do not spin on multiple long hypotheses this round."
    )
    command = [
        "codex", "exec", "--sandbox", "workspace-write",
        "-c", "sandbox_workspace_write.network_access=true",
        "-C", str(V9), "--add-dir", str(repo_root), "-",
    ]
    print(f"Round {round_id}: Codex starting, log={log_path}", flush=True)
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            command, cwd=V9, stdin=subprocess.PIPE, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            proc.communicate(input=prompt.encode("utf-8"), timeout=seconds)
        except subprocess.TimeoutExpired:
            print(f"Round {round_id} timed out; terminating local Codex group", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return False
    if proc.returncode != 0:
        print(f"Codex exited {proc.returncode}; see {log_path}", flush=True)
        return False
    return True


def check_report(round_id: int) -> str:
    if not REPORT.exists():
        raise ValueError("Codex did not write the required round-report.json")
    obj = json.loads(REPORT.read_text())
    if obj.get("round") != round_id:
        raise ValueError("Stale or wrong round report")
    status = obj.get("status")
    if status not in ("improved", "goal_reached", "no_gain", "blocked"):
        raise ValueError(f"Unknown round status: {status!r}")
    if status in ("improved", "goal_reached"):
        pr = obj.get("pr_url")
        if not isinstance(pr, str) or PR_PATTERN.fullmatch(pr) is None:
            raise ValueError("Improvement claimed without a valid checkpoint PR URL")
        rel = obj.get("gate_report")
        if not isinstance(rel, str):
            raise ValueError("Improvement claimed without gate_report")
        location = (V9 / rel).resolve()
        if not location.is_relative_to(WORK.resolve()):
            raise ValueError("Gate report must reside in the excluded research workdir")
        evidence = json.loads(location.read_text())
        if evidence.get("passed") is not True:
            raise ValueError("Improvement claimed without passing evidence gate")
        if status == "goal_reached" and evidence.get("target_met") is not True:
            raise ValueError("Goal claimed but official target not met")
        try:
            observed = subprocess.check_output(
                ["gh", "pr", "view", pr, "--json", "url,state"], text=True, timeout=30,
            )
            observed_pr = json.loads(observed)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
            raise ValueError(f"GitHub cannot verify checkpoint PR: {exc}") from exc
        if observed_pr.get("url") != pr or observed_pr.get("state") not in ("OPEN", "MERGED"):
            raise ValueError("PR URL/state did not match the claimed checkpoint")
        print("Verified checkpoint PR:", pr, flush=True)
        if evidence.get("target_met") is True:
            status = "goal_reached"
    print(f"Round {round_id}: {status} — {obj.get('notes', '')}", flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, required=True, help="hard overall wall-clock budget")
    parser.add_argument("--round-minutes", type=float, default=90,
                        help="hard limit for each Codex exec turn")
    args = parser.parse_args()
    if args.hours <= 0 or args.round_minutes <= 0:
        parser.error("Both budgets must be >0")
    repo = preflight()
    WORK.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600
    round_id = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining < 60:
            print("Global time budget exhausted; preserve best accepted PR and logs")
            return 0
        round_id += 1
        if not run_codex(round_id, min(remaining, args.round_minutes * 60), repo):
            print("Stopped after Codex timeout or failure; inspect log and remote owned server")
            return 2
        try:
            status = check_report(round_id)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print("Invalid/unverified round outcome:", exc, file=sys.stderr)
            return 3
        if status == "goal_reached":
            print("Verified milestone reached; supervisor stopping")
            return 0
        if status == "blocked":
            print("Blocked; stop instead of consuming GPU or fabrication")
            return 4


if __name__ == "__main__":
    raise SystemExit(main())
