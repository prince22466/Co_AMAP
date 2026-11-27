"""Screen compiled candidates vs champion, zero, and random controls in native vLLM + execute generated checks."""
import argparse
import json
import os
import re
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

from recipe import compile_explicit

ROOT = Path("/content/wsp")
DATA = ROOT / "input/competitions/gemma-4-developer-agent"


def load_returned_patches(runs: list[Path]) -> dict[str, str]:
    patches = {}
    for run in runs:
        with zipfile.ZipFile(run / "diagnostics.zip") as z:
            for name in z.namelist():
                m = re.search(r"/patches/([^/]+)_returned\.patch$", name)
                if m:
                    patches[m[1]] = z.read(name).decode("utf-8", errors="replace")
    return patches


def execute_generated_check(issue: str, command: str, agent_patch: str, ref_patch: str) -> dict:
    """Execute the model's generated command on (a) the agent's returned patch and (b) the reference fix."""
    snap = DATA / "snapshots" / f"{issue}.tgz"
    if not snap.is_file() or not command.strip():
        return {"executed": False}
    py_bin = str(ROOT / "eval_env/bin")
    env = dict(os.environ, PATH=f"{py_bin}:" + os.environ["PATH"], PYTHONNOUSERSITE="1")

    def run_on_patch(patch_text: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            with tarfile.open(snap) as arc:
                arc.extractall(work, filter="data")
            env_local = dict(env, PYTHONPATH=f"{work / 'src'}:{work}" if (work / "src").is_dir() else str(work))
            if patch_text.strip():
                ap = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    input=patch_text if patch_text.endswith("\n") else patch_text + "\n",
                    text=True, cwd=work, capture_output=True,
                )
                if ap.returncode != 0:
                    return {"patch_applied": False, "stderr": ap.stderr[-400:]}
            try:
                p = subprocess.run(
                    ["/bin/bash", "-c", command],
                    cwd=work, env=env_local, text=True, capture_output=True, timeout=25,
                )
                return {
                    "patch_applied": True,
                    "exit_code": p.returncode,
                    "stdout": p.stdout[-500:],
                    "stderr": p.stderr[-500:],
                }
            except subprocess.TimeoutExpired:
                return {"patch_applied": True, "exit_code": -9, "timeout": True}

    on_agent = run_on_patch(agent_patch)
    on_ref = run_on_patch(ref_patch)
    exposes_defect = (
        on_agent.get("patch_applied") is True
        and on_agent.get("exit_code", 0) != 0
        and on_ref.get("patch_applied") is True
        and on_ref.get("exit_code") == 0
    )
    return {
        "executed": True,
        "command": command,
        "on_returned_patch_exit": on_agent.get("exit_code"),
        "on_reference_fix_exit": on_ref.get("exit_code"),
        "discriminating_defect_catch": exposes_defect,
        "agent_out_tail": (on_agent.get("stdout", "") + "\n" + on_agent.get("stderr", "")).strip()[-350:],
    }


def grade_generation(txt: str, case: dict, returned_patches: dict[str, str], tasks_map: dict[str, dict]) -> dict:
    m = re.search(r"<\|tool_call>call:(\w+)\{", txt)
    action = m[1] if m else None
    has_leak = any(tok in txt for tok in ("<|tool_call><|tool_call>",)) or (action is None and "<|" in txt.split("<channel|>")[-1])
    role = case["role"]
    if role == "unchecked":
        cm = re.search(r'call:run_command\{command:<\|"\|>(.*?)<\|"\|>\}', txt, re.S)
        cmd = cm[1].strip() if cm else ""
        is_bare_pytest = bool(re.fullmatch(r"(?:python(?:3)?\s+-m\s+)?pytest(?:\s+-[qvx]+)*", cmd))
        exec_info = {}
        if action == "run_command" and cmd and not is_bare_pytest:
            issue = case["issue"]
            exec_info = execute_generated_check(
                issue, cmd, returned_patches.get(issue, ""), tasks_map.get(issue, {}).get("patch", "")
            )
        return {
            "action": action,
            "command": cmd,
            "is_bare_pytest": is_bare_pytest,
            "flipped_to_targeted_check": bool(action == "run_command" and cmd and not is_bare_pytest),
            **exec_info,
        }
    if role == "verified":
        return {"action": action, "preserved": action == "submit_patch"}
    if role == "edit":
        fields = {k: v for k, v in re.findall(r'(filepath|old_string|new_string):<\|"\|>(.*?)<\|"\|>', txt, re.S)}
        valid = action == "edit_file" and len(fields) == 3 and bool(fields.get("old_string")) and not has_leak
        return {"action": action, "preserved": valid}
    if role == "localize":
        return {"action": action, "preserved": action in ("read_file", "run_command") and not has_leak}
    return {"action": action}


def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    p = argparse.ArgumentParser()
    p.add_argument("--harvest-dir", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--bucket-rank", type=int, choices=(1, 2), required=True)
    p.add_argument("--extra-adapters-dir", type=Path)
    args = p.parse_args()

    hdir = args.harvest_dir
    out_dir = hdir / f"screen_r{args.bucket_rank}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_list = [
        c for c in json.loads((hdir / "candidates/ranking.json").read_text())["candidates"]
        if c["rank_export"] == args.bucket_rank
    ]
    extra_paths = []
    if args.extra_adapters_dir and args.extra_adapters_dir.is_dir():
        for ad in sorted(args.extra_adapters_dir.iterdir()):
            if (ad / "adapter_config.json").is_file():
                r_ad = json.loads((ad / "adapter_config.json").read_text())["r"]
                if r_ad == args.bucket_rank:
                    extra_paths.append(ad)
    if not cand_list and not extra_paths:
        print(f"No candidates with rank_export={args.bucket_rank}", flush=True)
        (out_dir / "summary.json").write_text(json.dumps({"candidates": []}))
        return

    # Select up to 1 case per (issue, role) across unchecked, verified, edit, localize
    all_cases = json.loads((hdir / "cases.json").read_text())
    selected = []
    seen = set()
    for c in all_cases:
        if c["role"] not in ("unchecked", "verified", "edit", "localize"):
            continue
        key = (c["issue"], c["role"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(c)
    print(f"Screening {len(cand_list)} candidates (r={args.bucket_rank}) on {len(selected)} native cases...", flush=True)

    # Compile zero, candidate, and random adapters
    adapters = {}
    if args.bucket_rank == 1:
        champ_dir = ROOT / "experiments/discovery02/input/adapters/wsp_r1"
        adapters["champion_r1"] = champ_dir
    zero_layers = set()
    for c in cand_list:
        fac = hdir / "candidates" / f"{c['name']}.safetensors"
        for ctrl in ("zero", "candidate", "random"):
            if ctrl == "zero":
                if c["layer"] in zero_layers:
                    continue
                zero_layers.add(c["layer"])
                label = f"L{c['layer']}_r{args.bucket_rank}_zero"
            else:
                label = f"{c['name']}_{ctrl}"
            dest = out_dir / "adapters" / label
            if not dest.exists():
                compile_explicit(ROOT / "champion.zip", fac, dest, ctrl, rank=args.bucket_rank)
            adapters[label] = dest
    for ad in extra_paths:
        adapters[ad.name] = ad

    returned_patches = load_returned_patches(args.runs)
    tasks_map = {
        r["instance_id"]: r
        for r in map(json.loads, (DATA / "tasks.jsonl").read_text().splitlines())
    }

    assets = json.loads((ROOT / "assets.json").read_text())
    engine = LLM(
        model=assets["model"],
        dtype="bfloat16",
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=_round_up_vllm_lora_rank(args.bucket_rank),
        gpu_memory_utilization=0.80,
        scheduler_reserve_full_isl=False,
        tensor_parallel_size=1,
    )
    tokenizer = engine.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=1024,
        stop_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call|>", "<turn|>")],
        skip_special_tokens=False,
    )
    payload = [{"prompt_token_ids": c["tokens"]} for c in selected]

    summary_rows = []
    for idx, (label, ad_path) in enumerate(adapters.items(), 1):
        save_p = out_dir / f"{label}.json"
        if save_p.is_file():
            records = json.loads(save_p.read_text())
        else:
            req = LoRARequest(label, idx, str(ad_path))
            batch = engine.generate(payload, params, lora_request=req, use_tqdm=False)
            records = []
            for case, r in zip(selected, batch, strict=True):
                txt = tokenizer.decode(r.outputs[0].token_ids, skip_special_tokens=False)
                graded = grade_generation(txt, case, returned_patches, tasks_map)
                records.append({
                    "issue": case["issue"],
                    "role": case["role"],
                    "observed": case["action"],
                    "generated_tokens": len(r.outputs[0].token_ids),
                    "text_tail": txt[-400:],
                    **graded,
                })
            save_p.write_text(json.dumps(records, indent=2))

        unch = [x for x in records if x["role"] == "unchecked"]
        ver = [x for x in records if x["role"] == "verified"]
        ed = [x for x in records if x["role"] == "edit"]
        loc = [x for x in records if x["role"] == "localize"]
        agg = {
            "adapter": label,
            "unchecked_total": len(unch),
            "unchecked_targeted_flips": sum(1 for x in unch if x.get("flipped_to_targeted_check")),
            "unchecked_defect_catches": sum(1 for x in unch if x.get("discriminating_defect_catch")),
            "unchecked_bare_pytest": sum(1 for x in unch if x.get("is_bare_pytest")),
            "verified_preserved": f"{sum(1 for x in ver if x.get('preserved'))}/{len(ver)}",
            "edit_preserved": f"{sum(1 for x in ed if x.get('preserved'))}/{len(ed)}",
            "localize_preserved": f"{sum(1 for x in loc if x.get('preserved'))}/{len(loc)}",
        }
        summary_rows.append(agg)
        print("SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
