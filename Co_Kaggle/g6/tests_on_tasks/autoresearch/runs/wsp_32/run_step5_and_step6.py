"""Step 5 & Step 6: Holdout screen, QUICK_15 benchmark, clean archive packaging, and QUICK_30 confirmation."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from provision_and_step2 import CLI, HERE, RESEARCH, SESSION, remote_exec
from run_step3_and_step4 import run_remote_stage

CHAMPION_SOLVES = {"rich_4077", "rich_3882", "rich_3905", "rich_3518", "rich_3454"}
ALLOWED_ARCHIVE_FILES = frozenset({
    "agent.yaml",
    "eval_config.yaml",
    "instruction.md",
    "adapters/wsp_r1/adapter_config.json",
    "adapters/wsp_r1/adapter_model.safetensors",
})

QUICK_15_IDX = [24, 14, 17, 19, 21, 22, 66, 67, 72, 75, 107, 113, 71, 106, 117]
QUICK_30_EXTRA_IDX = [16, 13, 95, 68, 69, 74, 100, 108, 109, 114, 115, 118, 119, 98, 127]


def get_task_ids(indices: list[int]) -> list[str]:
    tasks_path = RESEARCH.parents[1] / "competition_material/competition_data/tasks.jsonl"
    tasks = [json.loads(l) for l in tasks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [tasks[i]["instance_id"] for i in indices]


def package_verified_submission(adapter_dir: Path, output_zip: Path) -> dict:
    """Package ONLY the 5 competition-compliant files usingsubmission_40.zip's frozen agent/prompt."""
    champ_zip = RESEARCH.parents[1] / "submission_nb/artifacts/submission_40.zip"
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(champ_zip, "r") as zc, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for base_file in ("agent.yaml", "eval_config.yaml", "instruction.md"):
            zout.writestr(base_file, zc.read(base_file))
        for ad_file in ("adapter_config.json", "adapter_model.safetensors"):
            zout.write(adapter_dir / ad_file, arcname=f"adapters/wsp_r1/{ad_file}")

    with zipfile.ZipFile(output_zip, "r") as zf:
        assert zf.testzip() is None, "Corrupt submission zip"
        names = set(zf.namelist())
        assert names == ALLOWED_ARCHIVE_FILES, f"Unexpected archive files: {names}"
        ad_sha = hashlib.sha256(zf.read("adapters/wsp_r1/adapter_model.safetensors")).hexdigest()
    return {
        "zip_path": str(output_zip),
        "zip_bytes": output_zip.stat().st_size,
        "zip_sha256": hashlib.sha256(output_zip.read_bytes()).hexdigest(),
        "adapter_sha256": ad_sha,
    }


def eval_on_split(split: str, run_name: str, remote_adapter_dir: str | None, expected_sha: str, timeout_sec: int) -> dict:
    py_eval = "/content/wsp/eval_env/bin/python"
    ad_arg = f"--adapter {remote_adapter_dir}" if remote_adapter_dir else ""
    run_remote_stage(
        f"eval_{run_name}",
        f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split {split} {ad_arg} --sha {expected_sha} --name {run_name}",
        timeout_sec=timeout_sec,
    )
    out_local = HERE / "evals" / run_name
    out_local.mkdir(parents=True, exist_ok=True)
    for fn in ("neon_metrics.json", "neon_failures.jsonl"):
        subprocess.run([CLI, "download", "-s", SESSION, f"/content/wsp/experiments/{run_name}/{fn}", str(out_local / fn)], check=True)
    metrics = json.loads((out_local / "neon_metrics.json").read_text(encoding="utf-8"))
    assert metrics["adapter_sha256"] == expected_sha, f"SHA mismatch: {metrics['adapter_sha256']} != {expected_sha}"
    assert metrics["enable_lora"] is True
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("holdout", "quick15", "quick30"), required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--remote-adapter", default=None, help="Remote adapter directory on /content/wsp, or omit for champion")
    p.add_argument("--sha", required=True)
    args = p.parse_args()

    if args.mode == "holdout":
        m = eval_on_split("holdout", args.run_name, args.remote_adapter, args.sha, timeout_sec=1800)
        print("HOLDOUT METRICS:", json.dumps(m, indent=2), flush=True)
    elif args.mode == "quick15":
        q15_ids = get_task_ids(QUICK_15_IDX)
        subprocess.run([sys.executable, str(HERE / "stage_public_data_v2.py"), SESSION, *q15_ids], check=True)
        m = eval_on_split("15", args.run_name, args.remote_adapter, args.sha, timeout_sec=4500)
        resolved = set(m["resolved_ids"])
        retained = sorted(CHAMPION_SOLVES & resolved)
        lost = sorted(CHAMPION_SOLVES - resolved)
        gained = sorted(resolved - CHAMPION_SOLVES)
        m["retained_champion_solves"] = retained
        m["lost_champion_solves"] = lost
        m["newly_solved"] = gained
        print("QUICK_15 METRICS:", json.dumps(m, indent=2), flush=True)
        if m["solved"] >= 7 and not lost and m["syntax_leaks"] <= 1:
            local_ad = HERE / "evals" / args.run_name / "adapter"
            local_ad.mkdir(parents=True, exist_ok=True)
            for fn in ("adapter_config.json", "adapter_model.safetensors"):
                subprocess.run([CLI, "download", "-s", SESSION, f"{args.remote_adapter}/{fn}", str(local_ad / fn)], check=True)
            kept_zip = RESEARCH / "kept_submissions" / f"submission_wsp_{args.run_name}_{m['solved']}of15_{args.sha[:12]}.zip"
            pkg = package_verified_submission(local_ad, kept_zip)
            print("ARCHIVED WINNING SUBMISSION:", json.dumps(pkg, indent=2), flush=True)
    elif args.mode == "quick30":
        q30_ids = get_task_ids(QUICK_15_IDX + QUICK_30_EXTRA_IDX)
        subprocess.run([sys.executable, str(HERE / "stage_public_data_v2.py"), SESSION, *q30_ids], check=True)
        m = eval_on_split("30", args.run_name, args.remote_adapter, args.sha, timeout_sec=9000)
        q15_set = set(get_task_ids(QUICK_15_IDX))
        extra_set = set(get_task_ids(QUICK_30_EXTRA_IDX))
        res_set = set(m["resolved_ids"])
        m["original_15_solved"] = len(res_set & q15_set)
        m["original_15_ids"] = sorted(res_set & q15_set)
        m["additional_15_solved"] = len(res_set & extra_set)
        m["additional_15_ids"] = sorted(res_set & extra_set)
        (HERE / "evals" / args.run_name / "neon_metrics_split_report.json").write_text(json.dumps(m, indent=2))
        print("QUICK_30 SPLIT REPORT:", json.dumps(m, indent=2), flush=True)


if __name__ == "__main__":
    main()
