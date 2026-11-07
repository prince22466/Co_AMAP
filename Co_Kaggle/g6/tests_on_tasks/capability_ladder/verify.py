"""Grade L1/L2 patches with official swegemma phase-2 verification.

Run after the GPU probe on a machine with official wheels and Docker.
No model calls. All reference patches remain within the verifier controls.
Never report 'resolved' unless the baseline fails and the reference passes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time

from probe import read_tasks


class VerificationBlocked(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight_result(result: object) -> bool:
    """A meaningful failing original test is required, not a broken setup."""
    output = getattr(result, "test_output", "") or ""
    return (getattr(result, "resolved", None) is False
            and getattr(result, "test_exit_code", None) == 1
            and not getattr(result, "error", None)
            and bool(output.strip())
            and not re.search(
                r"ModuleNotFoundError|ImportError|ERROR collecting|"
                r"ERROR at setup|no tests ran|recursive dependency involving fixture",
                output, re.I,
            ))


def reference_control(result: object) -> bool:
    return (getattr(result, "resolved", None) is True
            and getattr(result, "test_exit_code", None) == 0
            and not getattr(result, "error", None))


def load_probes(path: Path, tasks: dict) -> list[dict]:
    rows = []
    used = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("instance_id")
        if not isinstance(key, str) or key in used or key not in tasks:
            raise VerificationBlocked("Missing/duplicate/unknown task ID")
        if row.get("level") not in ("reasoning", "tool"):
            raise VerificationBlocked("Unexpected probe level")
        if row.get("resolved") is not None:
            raise VerificationBlocked("Probe unexpectedly claims resolution")
        if not isinstance(row.get("patch"), str):
            raise VerificationBlocked("Missing patch string")
        if row.get("edit_valid") is not (row["status"] == "edit_applied_to_pristine"):
            raise VerificationBlocked("Probe edit/status disagreement")
        if row.get("edit_valid"):
            patch = row["patch"]
            if (len(patch) > 20_000 or not patch.startswith("--- a/")
                    or hashlib.sha256(patch.encode()).hexdigest()
                    != row.get("patch_sha256")):
                raise VerificationBlocked("Patch integrity mismatch")
        elif row["patch"]:
            raise VerificationBlocked("Invalid edit with nonempty patch")
        used.add(key)
        rows.append(row)
    if not rows:
        raise VerificationBlocked("Empty probes")
    return rows


async def verify_one(manager, config, task, snapshot, base_snapshot,
                     snapshot_patch, patch: str, verify_func) -> object:
    return await verify_func(
        manager, config, task, snapshot,
        base_snapshot_path=base_snapshot, patch_path=snapshot_patch,
        agent_patch=patch, start_time=time.perf_counter(),
    )


async def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise VerificationBlocked("Output directory exists; do not overwrite")
    task_rows = read_tasks(args.tasks)
    probes = load_probes(args.probes, task_rows)
    # The model input task set must match exactly the verification dataset.
    model_manifest_path = args.probes.parent / "manifest.json"
    if not model_manifest_path.is_file():
        raise VerificationBlocked("Missing generator manifest; task parity unknown")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if (model_manifest.get("tasks_sha256") != sha(args.tasks)
            or model_manifest.get("task_ids")
            != [row["instance_id"] for row in probes]
            or model_manifest.get("model") != "gemma-4-e4b-it-qat-w4a16-ct"
            or model_manifest.get("level") != probes[0]["level"]):
        raise VerificationBlocked("Probe task/model manifest mismatch")
    # Import only at verification time so synthetic CPU tests need no official wheels.
    from adk_submission import ModelRegistry
    from swegemma.config import EvalConfig
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness.verification import verify_task
    from swegemma.models import load_tasks
    from swegemma.sandbox import ContainerConfig, ContainerManager

    official_tasks = {task.instance_id: task for task in load_tasks(args.tasks)}
    if set(row["instance_id"] for row in probes) - official_tasks.keys():
        raise VerificationBlocked("Official task loader disagrees with probes")
    if not args.submission_dir.is_dir():
        raise VerificationBlocked("Missing S29 submission package directory")
    args.output.mkdir(parents=True)
    cfg = EvalConfig(
        tasks_path=args.tasks, snapshots_dir=args.snapshots,
        submission_dir=args.submission_dir, results_dir=args.output / "runtime",
        models=ModelRegistry(), timeout_seconds=300,
    )
    versions = {}
    for pkg in ("swegemma", "adk-submission", "adk-eval-core"):
        versions[pkg] = importlib.metadata.version(pkg)
    method_file = Path(verify_task.__code__.co_filename)
    manifest = {
        "scope": "official phase-2 verifier; no model inference",
        "probe_input_sha256": sha(args.probes),
        "generator_manifest_sha256": sha(model_manifest_path),
        "generator_level": model_manifest["level"],
        "generator_seed": model_manifest.get("seed"),
        "served_model": model_manifest.get("served_model"),
        "tasks_sha256": sha(args.tasks),
        "source_verifier_sha256": sha(method_file),
        "versions": versions,
        "task_ids": [r["instance_id"] for r in probes],
        "controls": "baseline must fail meaningfully, oracle patch must pass",
        "note": "Research ceiling; oracle source location is not valid Kaggle agent context.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    sandbox = ContainerManager(ContainerConfig(
        image=cfg.image, timeout_seconds=cfg.harness.command_timeout_seconds,
    ))
    try:
        with (args.output / "verified.jsonl").open("x", encoding="utf-8") as out:
            for probe in probes:
                key = probe["instance_id"]
                task = official_tasks[key]
                row = {
                    "instance_id": key, "level": probe["level"],
                    "edit_valid": probe["edit_valid"],
                    "status": "blocked_verifier", "scorable": False,
                    "resolved": None, "baseline_resolved": None,
                    "reference_resolved": None, "error": None,
                }
                if probe["status"] in ("blocked_input", "model_or_transport_error",
                                       "not_started"):
                    row["status"] = "blocked_generation"
                    row["error"] = probe.get("error") or "Generation unavailable"
                elif not (isinstance(task.patch, str) and task.patch.strip()
                          and isinstance(task.test_patch, str)
                          and task.test_patch.strip()):
                    row["error"] = "Missing reference or test patch for verifier control"
                else:
                    try:
                        snapshot, base, snap_patch = resolve_task_snapshot_paths(
                            cfg.snapshots_dir, key, task.repo)
                        if not snapshot.is_file():
                            raise VerificationBlocked("Missing official snapshot")
                        negative = await verify_one(
                            sandbox, cfg, task, snapshot, base, snap_patch, "", verify_task)
                        positive = await verify_one(
                            sandbox, cfg, task, snapshot, base, snap_patch,
                            task.patch, verify_task)
                        row["baseline_resolved"] = getattr(negative, "resolved", None)
                        row["reference_resolved"] = getattr(positive, "resolved", None)
                        if not preflight_result(negative):
                            raise VerificationBlocked("Baseline did not fail meaningfully")
                        if not reference_control(positive):
                            raise VerificationBlocked("Reference patch did not verify")
                        row["scorable"] = True
                        if probe["edit_valid"]:
                            candidate = await verify_one(
                                sandbox, cfg, task, snapshot, base, snap_patch,
                                probe["patch"], verify_task)
                            if not isinstance(candidate.resolved, bool):
                                raise VerificationBlocked("Missing candidate verdict")
                            # A candidate patch rejected by git apply or tests is
                            # an agent failure, not a broken verifier. The empty
                            # and reference controls already proved the harness.
                            row["resolved"] = candidate.resolved
                            row["status"] = "resolved" if candidate.resolved else "unresolved"
                            row["test_exit_code"] = getattr(candidate, "test_exit_code", None)
                            row["candidate_error"] = getattr(candidate, "error", None)
                        else:
                            row["resolved"] = False
                            row["status"] = "invalid_edit"
                    except Exception as exc:
                        row["scorable"] = False
                        row["resolved"] = None
                        row["status"] = "blocked_verifier"
                        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(key, row["status"], flush=True)
    finally:
        sandbox.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True,
                        help="Unmodified submission_29 directory; verifier creates no agent")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
