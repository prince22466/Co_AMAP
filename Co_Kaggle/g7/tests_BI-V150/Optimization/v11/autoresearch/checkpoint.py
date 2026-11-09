#!/usr/bin/env python3
"""Preserve a research checkpoint in tracked notes; optionally push its PR branch.

Pushing does not create a PR. Raw profiler/benchmark files remain outside Git
and need separate durable backup before the rented instance disappears.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def checkpoint(root: Path, summary: str, evidence: str, next_step: str,
               status: str = "provisional",
               remaining_minutes: float | None = None,
               push: bool = False) -> Path:
    """Append evidence without rewriting history and refresh the handoff."""
    if not summary.strip() or not evidence.strip() or not next_step.strip():
        raise ValueError("summary, evidence, and next_step are mandatory")
    if status not in ("hypothesis", "provisional", "verified", "rejected", "blocked"):
        raise ValueError("Unknown status")
    directory = root / "autoresearch"
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "EXPERIMENTS.md"
    handoff = directory / "HANDOFF.md"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        source = git(root, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        source = "not available"
    left = f"{remaining_minutes:.1f} minutes" if remaining_minutes is not None else "unknown"
    entry = (
        f"\n### Research checkpoint {timestamp}: {status}\n\n"
        f"- HEAD before checkpoint: {source}\n"
        f"- Budget remaining: {left}\n"
        f"- Finding: {summary.strip()}\n"
        f"- Evidence and artifact paths: {evidence.strip()}\n"
        f"- Next action: {next_step.strip()}\n"
        "- Verification status is researcher-reported; this tool does NOT certify "
        "benchmark validity, model accuracy, or independently replicated gains.\n"
    )
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(entry)
        stream.flush()
    handoff.write_text(
        "# v11 latest research handoff\n\n"
        f"Updated UTC: {timestamp}\n\n"
        f"Status: {status} (researcher-reported)\n\n"
        f"Source HEAD before checkpoint: {source}\n\n"
        f"Remaining GPU time: {left}\n\n"
        f"Finding: {summary.strip()}\n\n"
        f"Evidence/provenance: {evidence.strip()}\n\n"
        f"Next action: {next_step.strip()}\n\n"
        "Incomplete half-runs are never verified results. "
        "Ignored raw artifacts must be backed up outside the rented instance. "
        "This handoff does not verify accuracy or establish a causal speedup.\n",
        encoding="utf-8",
    )
    if push:
        branch = git(root, "branch", "--show-current")
        if not branch or branch in ("main", "master"):
            raise RuntimeError("Refusing push from detached HEAD or main/master")
        git(root, "add", "--", "autoresearch/EXPERIMENTS.md", "autoresearch/HANDOFF.md")
        # --only prevents committing unrelated staged experiments/source edits.
        git(root, "commit", "--only", "-m", "docs(v11): preserve research checkpoint",
            "--", "autoresearch/EXPERIMENTS.md", "autoresearch/HANDOFF.md")
        git(root, "push", "origin", "HEAD:refs/heads/" + branch)
    return handoff


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--next-step", required=True)
    p.add_argument("--status", default="provisional",
                   choices=["hypothesis", "provisional", "verified", "rejected", "blocked"])
    p.add_argument("--remaining-minutes", type=float)
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    try:
        out = checkpoint(ROOT, args.summary, args.evidence, args.next_step,
                         args.status, args.remaining_minutes, args.push)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"Checkpoint failed; inspect tracked local notes: {exc}\n")
    print(f"Checkpoint saved at {out}")
    print("PR branch pushed" if args.push else "Local checkpoint only; --push publishes it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
