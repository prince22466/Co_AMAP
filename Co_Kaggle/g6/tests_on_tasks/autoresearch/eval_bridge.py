"""eval_bridge.py — Weight-Space Programming (WSP) Compiler & Kaggle CLI Bridge.

Locked infrastructure script per tests_on_tasks/autoresearch/program.md.
Compiles recipe.py into base_submission/adapters/wsp_r1/{adapter_config.json,adapter_model.safetensors},
validates the submission directory structure, and orchestrates Kaggle CLI dataset/kernel evaluation
plus automatic archiving of winning (>= 7/15) submissions into kept_submissions/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

AUTORESEARCH_DIR = Path(__file__).resolve().parent
BASE_SUBMISSION_DIR = AUTORESEARCH_DIR / "base_submission"
ADAPTER_DIR = BASE_SUBMISSION_DIR / "adapters" / "wsp_r1"
KEPT_SUBMISSIONS_DIR = AUTORESEARCH_DIR / "kept_submissions"
RUNS_DIR = AUTORESEARCH_DIR / "runs"
KERNEL_DIR = AUTORESEARCH_DIR / "kaggle_eval_kernel"

DATASET_SLUG = "prince22466/g6-autoresearch-sub-v2"
KERNEL_SLUG = "prince22466/gemma-4-developer-agent-testing"

ALLOWED_ARCHIVE_FILES = frozenset({
    "agent.yaml",
    "eval_config.yaml",
    "instruction.md",
    "adapters/wsp_r1/adapter_config.json",
    "adapters/wsp_r1/adapter_model.safetensors",
})


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_recipe_compile() -> dict:
    """Execute recipe.py using Python 3.11 (with torch + safetensors) and return manifest."""
    py_exe = Path("D:/Python311/python.exe")
    if not py_exe.exists():
        py_exe = Path(sys.executable)
    cmd = [str(py_exe), str(AUTORESEARCH_DIR / "recipe.py")]
    proc = subprocess.run(cmd, cwd=str(AUTORESEARCH_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"recipe.py failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    safetensors_path = ADAPTER_DIR / "adapter_model.safetensors"
    config_path = ADAPTER_DIR / "adapter_config.json"
    if not safetensors_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("recipe.py did not produce adapter_model.safetensors and adapter_config.json")
    adapter_sha = compute_sha256(safetensors_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "adapter_sha256": adapter_sha,
        "adapter_short_sha": adapter_sha[:12],
        "rank": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "layers_to_transform": cfg.get("layers_to_transform"),
        "target_modules": cfg.get("target_modules"),
        "safetensors_bytes": safetensors_path.stat().st_size,
        "recipe_stdout": proc.stdout.strip(),
    }


def package_clean_submission_zip(output_zip: Path) -> dict:
    """Package ONLY competition-valid files into output_zip and verify entries."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(ALLOWED_ARCHIVE_FILES):
            src = BASE_SUBMISSION_DIR / rel
            if not src.is_file():
                raise FileNotFoundError(f"Missing required submission file: {src}")
            zf.write(src, arcname=rel)
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = set(zf.namelist())
        if names != ALLOWED_ARCHIVE_FILES:
            raise ValueError(f"Invalid archive members: {sorted(names)} != {sorted(ALLOWED_ARCHIVE_FILES)}")
        for forbidden in ("autoresearch_run.json", "dataset-metadata.json"):
            if forbidden in names:
                raise ValueError(f"Forbidden metadata file {forbidden} leaked into submission zip!")
    return {
        "zip_path": str(output_zip),
        "zip_sha256": compute_sha256(output_zip),
        "zip_bytes": output_zip.stat().st_size,
    }


def stage_and_upload_dataset(iter_id: str, split: int, adapter_sha256: str) -> None:
    """Upload base_submission + autoresearch_run.json to prince22466/g6-autoresearch-sub."""
    with tempfile.TemporaryDirectory(prefix="g6_wsp_ds_") as tmp:
        stage_dir = Path(tmp)
        for rel in sorted(ALLOWED_ARCHIVE_FILES):
            src = BASE_SUBMISSION_DIR / rel
            dst = stage_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        run_meta = {
            "iter_id": iter_id,
            "split": split,
            "expected_adapter_sha256": adapter_sha256,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (stage_dir / "autoresearch_run.json").write_text(
            json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
        )
        ds_meta = {
            "title": "g6-autoresearch-sub-v2",
            "id": DATASET_SLUG,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (stage_dir / "dataset-metadata.json").write_text(
            json.dumps(ds_meta, indent=2) + "\n", encoding="utf-8"
        )

        # Check if dataset already exists on Kaggle
        chk = subprocess.run(
            ["kaggle", "datasets", "status", DATASET_SLUG],
            capture_output=True, text=True
        )
        if chk.returncode == 0 and "ready" in (chk.stdout + chk.stderr).lower():
            cmd = [
                "kaggle", "datasets", "version",
                "-p", str(stage_dir),
                "-m", f"{iter_id}: split={split} sha={adapter_sha256[:12]}",
                "--dir-mode", "zip",
            ]
        else:
            cmd = [
                "kaggle", "datasets", "create",
                "-p", str(stage_dir),
                "--dir-mode", "zip",
            ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Kaggle dataset upload failed:\n{res.stdout}\n{res.stderr}")
        print(res.stdout.strip())

        # Poll until dataset status is ready
        for _ in range(30):
            st = subprocess.run(
                ["kaggle", "datasets", "status", DATASET_SLUG],
                capture_output=True, text=True
            )
            if "ready" in (st.stdout + st.stderr).lower():
                break
            time.sleep(5)


def push_and_wait_kernel(iter_id: str, accelerator: str, poll_seconds: int = 30) -> Path:
    """Push kaggle_eval_kernel with specified accelerator, poll until COMPLETE, and pull outputs."""
    v12_nb = AUTORESEARCH_DIR.parent / "gemma-4-developer-agent-testing_v12_wsp.ipynb"
    shutil.copy2(v12_nb, KERNEL_DIR / "gemma-4-developer-agent-testing_v12_wsp.ipynb")

    meta_path = KERNEL_DIR / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["machine_shape"] = accelerator
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    push_cmd = [
        "kaggle", "kernels", "push",
        "-p", str(KERNEL_DIR),
        "--accelerator", accelerator,
    ]
    res = subprocess.run(push_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kaggle kernels push failed:\n{res.stdout}\n{res.stderr}")
    print(res.stdout.strip())

    t0 = time.time()
    while True:
        time.sleep(poll_seconds)
        st = subprocess.run(
            ["kaggle", "kernels", "status", KERNEL_SLUG],
            capture_output=True, text=True
        )
        out = (st.stdout + st.stderr).strip()
        elapsed = int(time.time() - t0)
        print(f"[{elapsed}s] {out}")
        if "COMPLETE" in out:
            break
        if "ERROR" in out or "CANCEL" in out:
            out_dir = RUNS_DIR / iter_id
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(out_dir)])
            raise RuntimeError(f"Kaggle kernel ended with status: {out}")

    out_dir = RUNS_DIR / iter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pull = subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(out_dir)],
        capture_output=True, text=True
    )
    if pull.returncode != 0:
        raise RuntimeError(f"kaggle kernels output failed:\n{pull.stdout}\n{pull.stderr}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="WSP Compiler & Kaggle CLI Evaluation Bridge")
    parser.add_argument("--iter", required=True, help="Iteration identifier (e.g., iter_01)")
    parser.add_argument("--split", type=int, default=15, choices=(15, 30), help="Evaluation task split (15 or 30)")
    parser.add_argument("--accelerator", default="NvidiaL4", help="Kaggle accelerator machine shape (default: NvidiaL4)")
    parser.add_argument("--compile-only", action="store_true", help="Only run recipe.py and validate local package without pushing to Kaggle")
    parser.add_argument("--package-zip", type=str, default=None, help="Optional output path to package a clean submission_x.zip locally")
    args = parser.parse_args()

    manifest = run_recipe_compile()
    print(json.dumps(manifest, indent=2))

    if args.package_zip:
        pkg_info = package_clean_submission_zip(Path(args.package_zip))
        print("Packaged clean submission ZIP:", json.dumps(pkg_info, indent=2))

    if args.compile_only:
        return

    stage_and_upload_dataset(args.iter, args.split, manifest["adapter_sha256"])
    out_dir = push_and_wait_kernel(args.iter, args.accelerator)

    metrics_path = out_dir / "neon_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"neon_metrics.json not found in {out_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("adapter_sha256") != manifest["adapter_sha256"]:
        raise RuntimeError(
            f"Adapter SHA mismatch! Compiled={manifest['adapter_sha256']}, evaluated={metrics.get('adapter_sha256')}"
        )
    if not metrics.get("enable_lora"):
        raise RuntimeError("Evaluated run had enable_lora != True!")

    solved = int(metrics.get("solved", 0))
    if args.split == 15 and solved >= 7:
        short_sha = manifest["adapter_short_sha"]
        kept_zip = KEPT_SUBMISSIONS_DIR / f"submission_wsp_{args.iter}_{solved}of15_{short_sha}.zip"
        if not kept_zip.exists():
            pkg = package_clean_submission_zip(kept_zip)
            metrics["kept_zip"] = pkg["zip_path"]
            print(f"TARGET >= 7/15 PASSED! Archived winning submission to: {kept_zip}")

    print("Evaluation metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
