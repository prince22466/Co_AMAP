#!/usr/bin/env python3
"""Preserve a research checkpoint in tracked notes; optionally push its PR branch.

Pushing does not create a PR. Raw profiler/benchmark files remain outside Git
and need separate durable backup before the rented instance disappears.
"""
from __future__ import annotations

import argparse
import hashlib
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
               push: bool = False, csv_paths: list[Path] | None = None) -> Path:
    """Append evidence without rewriting history and refresh the handoff."""
    if not summary.strip() or not evidence.strip() or not next_step.strip():
        raise ValueError("summary, evidence, and next_step are mandatory")
    if status not in ("hypothesis", "provisional", "verified", "rejected", "blocked"):
        raise ValueError("Unknown status")
    directory = root / "autoresearch"
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "EXPERIMENTS.md"
    handoff = directory / "HANDOFF.md"
    # Preserve bounded, non-sensitive benchmark metrics on the PR branch.
    # Do not accidentally commit complete prompts, model data, logs or traces.
    preserved = []
    evidence_paths = []
    allowed = (root / ".autoresearch-work").resolve()
    for csv in csv_paths or []:
        source_path = csv.resolve()
        try:
            source_path.relative_to(allowed)
        except ValueError as exc:
            raise ValueError(f"CSV outside ignored work tree: {csv}") from exc
        if (not source_path.is_file()
                or not source_path.name.startswith(("raw_runs", "summary"))
                or source_path.suffix.lower() != ".csv"
                or source_path.stat().st_size > 1_000_000):
            raise ValueError(f"Only raw_runs/summary CSV <=1 MB are allowed: {csv}")
        raw = source_path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        stored = directory / "evidence" / (
            f"{source_path.parent.parent.name}_{sha256[:12]}_{source_path.name}"
        )
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_bytes(raw)
        preserved.append(f"{stored.relative_to(root)} (sha256 {sha256})")
        evidence_paths.append(str(stored.relative_to(root)))
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        source = git(root, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        source = "not available"
    left = f"{remaining_minutes:.1f} minutes" if remaining_minutes is not None else "unknown"
    csv_details = "; ".join(preserved) if preserved else "none (external backup required)"
    entry = (
        f"\n### Research checkpoint {timestamp}: {status}\n\n"
        f"- HEAD before checkpoint: {source}\n"
        f"- Budget remaining: {left}\n"
        f"- Finding: {summary.strip()}\n"
        f"- Evidence and artifact paths: {evidence.strip()}\n"
        f"- PR-preserved small benchmark CSVs: {csv_details}\n"
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
        f"Small CSV evidence committed when --push works: {csv_details}\n\n"
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
        changed_paths = ["autoresearch/EXPERIMENTS.md", "autoresearch/HANDOFF.md", *evidence_paths]
        git(root, "add", "--", *changed_paths)
        # --only prevents committing unrelated staged experiments/source edits.
        git(root, "commit", "--only", "-m", "docs(v11): preserve research checkpoint",
            "--", *changed_paths)
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
    p.add_argument("--csv", type=Path, action="append", default=[],
                   help="Optional raw_runs/summary CSV <=1MB from .autoresearch-work")
    args = p.parse_args()
    try:
        out = checkpoint(ROOT, args.summary, args.evidence, args.next_step,
                         args.status, args.remaining_minutes, args.push, args.csv)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"Checkpoint failed; inspect tracked local notes: {exc}\n")
    print(f"Checkpoint saved at {out}")
    print("PR branch pushed" if args.push else "Local checkpoint only; --push publishes it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
