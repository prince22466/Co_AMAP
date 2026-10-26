"""Verify the 15 independent patches with SWE-gemma's actual Phase-2 verifier.

Run INSIDE an environment where the official swegemma, adk-submission,
competition snapshots, Docker image and wheels are installed. This script
does not run an LLM, read gold reference patches, or modify submission agents.

Example:
  python verify_official_phase2.py --tasks /kaggle/input/.../tasks.jsonl \
      --snapshots /kaggle/input/.../snapshots \
      --wheels /kaggle/input/.../wheels \
      --submission-dir /kaggle/input/.../sample_submission \
      --output /kaggle/working/g6-first15-official-phase2
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path


async def run(args: argparse.Namespace) -> dict:
    from adk_submission import ModelRegistry
    from swegemma.config import EvalConfig
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness.verification import verify_task
    from swegemma.models import load_tasks
    from swegemma.sandbox import ContainerConfig, ContainerManager

    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    task_rows = {task.instance_id: task for task in load_tasks(args.tasks)}
    missing = sorted(set(manifest) - set(task_rows))
    if missing:
        raise ValueError(f"Missing official task rows: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)

    config_kwargs = dict(
        tasks_path=args.tasks,
        snapshots_dir=args.snapshots,
        submission_dir=args.submission_dir,
        results_dir=args.output,
        models=ModelRegistry(),  # No model calls are performed.
        timeout_seconds=args.timeout_seconds,
    )
    if args.wheels is not None:
        config_kwargs["wheels_dir"] = args.wheels
    cfg = EvalConfig(**config_kwargs)
    container = ContainerManager(ContainerConfig(
        image=cfg.image, timeout_seconds=cfg.harness.command_timeout_seconds
    ))
    records = []
    try:
        for task_id, spec in manifest.items():
            task = task_rows[task_id]
            if task.repo != spec["repo"] or task.base_commit != spec["base_commit"]:
                raise ValueError(f"Task metadata mismatch for {task_id}")
            patch_file = root / spec["candidate_patch"]
            patch = patch_file.read_text(encoding="utf-8")
            if not patch.strip():
                raise ValueError(f"Empty candidate patch: {task_id}")
            snapshot, base_snapshot, snapshot_patch = resolve_task_snapshot_paths(
                cfg.snapshots_dir, task.instance_id, task.repo
            )
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)
            row = {"id": task_id, "repo": task.repo, "base_commit": task.base_commit,
                   "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}
            for stage, agent_patch in (
                ("baseline", ""), ("candidate", patch),
            ):
                try:
                    result = await verify_task(
                        container, cfg, task, snapshot,
                        base_snapshot_path=base_snapshot,
                        patch_path=snapshot_patch,
                        agent_patch=agent_patch,
                        start_time=time.perf_counter(),
                    )
                    row[stage] = {
                        "resolved": result.resolved,
                        "test_exit_code": result.test_exit_code,
                        "error": str(result.error) if result.error else None,
                    }
                except Exception as exc:
                    row[stage] = {
                        "resolved": None, "test_exit_code": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            row["fix_confirmed"] = (
                row["baseline"]["resolved"] is False
                and row["candidate"]["resolved"] is True
            )
            records.append(row)
            (args.output / "per_task.json").write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(row), flush=True)
    finally:
        container.close()

    confirmed = sum(r["fix_confirmed"] for r in records)
    verified = sum(r["candidate"]["resolved"] is True for r in records)
    summary = {
        "task_count": len(records),
        "baseline_fail_candidate_pass": confirmed,
        "candidate_resolved": verified,
        "all_15_confirmed": len(records) == 15 and confirmed == 15,
        "environment": "official SWE-gemma Phase 2 (when run in official runtime)",
        "note": "Patch-only verification; NOT a GPT-6 tool-call or Kaggle hidden-score run.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    arguments = parser.parse_args()
    verdict = asyncio.run(run(arguments))
    print(json.dumps(verdict, indent=2))
    raise SystemExit(0 if verdict["all_15_confirmed"] else 1)
