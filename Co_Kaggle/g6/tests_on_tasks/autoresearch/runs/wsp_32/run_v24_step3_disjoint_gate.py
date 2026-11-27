"""Step 3 Orchestrator for V24 Pure `L39 + L41` Matched-History Pure-WSP Adapter (ZERO `L40`).

1. Synthesizes V24 (`remote_synth_v24_matched39_41.py`) across all 19 disjoint snapshots (including previously-skipped
   `fastapi_14479`, `rich_3506`, `requests_7502` `edit_ok` turns + matched premature-submit history pairs!).
2. Screens V24 on an expanded pool that EXPLICITLY includes all genuine `UNCHECKED_SUBMIT` turns from `discovery02/03/04`
   (`b1_prem_flips` measures real `submit_patch -> run_command` flips on `fastapi_14479`, `fastapi_14786`, `rich_3506`, etc.!).
3. Runs End-to-End Multi-Turn Disjoint Gate (`wsp32_v24_disjoint6`) and inspects the 5-stage causal funnel!
"""
from __future__ import annotations

import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, ensure_session, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage
from run_v21_step3_disjoint_gate import DISJOINT_GATE_6, download_overwrite


def write_remote_screen_v24() -> None:
    remote_exec(r"""
import subprocess
from pathlib import Path
subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)

Path('/content/wsp/screen_v24_matched.py').write_text('''
import json, os, re, shutil, tempfile, zipfile
from collections import Counter
from pathlib import Path
from build_v21_disjoint_corpus import DISCOVERY_19, parse_gemma4_call, TEST_CMD
from screen_native_v2 import DATA, ROOT, execute_generated_check, load_returned_patches

def is_slow_suite_pytest(cmd: str) -> bool:
    if not cmd:
        return False
    c = cmd.strip()
    if "pytest" not in c:
        return False
    return (".py" not in c) and (" -k " not in c)

def verify_compliance(ad_dir: Path):
    import adk_submission
    from swegemma.config import build_submission_limits
    val_fn = getattr(adk_submission, "validate_directory")
    limits, _ = build_submission_limits()
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "sub"
        with zipfile.ZipFile(ROOT / "champion.zip") as z:
            z.extractall(sub)
        for fn in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(ad_dir / fn, sub / "adapters/wsp_r1" / fn)
        val_fn(sub, limits=limits)
    print("COMPLIANCE_PASS:", ad_dir.name, flush=True)

def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    vdir = ROOT / "harvest32/v24_matched39_41"
    out_dir = ROOT / "harvest32/screen_v24_matched"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ad_dirs = sorted((vdir / "adapters").iterdir())
    for d in ad_dirs:
        if d.name.endswith("_candidate"):
            verify_compliance(d)

    assets = json.loads((ROOT / "assets.json").read_text())
    tok_hf = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tasks_all = [json.loads(s) for s in (DATA / "tasks.jsonl").read_text().splitlines() if s.strip()]
    tasks_map = {r["instance_id"]: r for r in tasks_all}

    all_replays = list(json.loads((ROOT / "harvest32/clean_v21/real_replays.json").read_text()))
    seen_sha = {r["prompt_sha"] for r in all_replays}

    # Include ALL missed `edit_ok` prompts from discovery02/03/04 so `fastapi_14479`, `rich_3506`, `requests_7502` are in b1!
    disc_ps = [tasks_map[i]["problem_statement"] for i in DISCOVERY_19]
    disc_needles = {}
    for iid in DISCOVERY_19:
        cands = [ln.strip() for ln in tasks_map[iid]["problem_statement"].splitlines() if len(ln.strip()) >= 8]
        uniq = [ln for ln in cands if sum(1 for o in disc_ps if ln in o) == 1] or cands
        disc_needles[iid] = max(uniq, key=len)

    for rn in ("discovery02", "discovery03", "discovery04"):
        rp = ROOT / "experiments" / rn / "replay_tokens.jsonl"
        if not rp.is_file():
            continue
        for ln in rp.read_text().splitlines():
            if not ln.strip():
                continue
            o = json.loads(ln)
            if o["request_sha"] in seen_sha:
                continue
            txt_tail = tok_hf.decode(o["tokens"][-900:], skip_special_tokens=False)
            last_tr_pos = txt_tail.rfind("<|tool_response>response:")
            if last_tr_pos < 0:
                continue
            tail_after_tr = txt_tail[last_tr_pos:]
            if tail_after_tr.startswith("<|tool_response>response:edit_file{") and '"status": "ok"' in tail_after_tr:
                end_tr = tail_after_tr.find("<tool_response|>")
                if end_tr > 0 and "<|turn>user" not in tail_after_tr[end_tr:] and "<|tool_call>" not in tail_after_tr[end_tr:]:
                    txt_full = tok_hf.decode(o["tokens"], skip_special_tokens=False)
                    if "<|tool_response>response:submit_patch{" in txt_full:
                        continue
                    iss = next((iid for iid, ndl in disc_needles.items() if ndl in txt_full), None)
                    if iss in DISCOVERY_19:
                        all_replays.append({
                            "issue": iss, "true_role": "true_imm_edit_ok",
                            "tokens": o["tokens"], "prompt_sha": o["request_sha"],
                        })
                        seen_sha.add(o["request_sha"])

    pool = []
    per_role = Counter()
    role_caps = {
        "true_imm_edit_ok": 28,
        "true_imm_bad_test_path": 4,
        "true_imm_real_test_fail": 16,
        "true_imm_read_full": 12,
        "true_imm_test_pass": 14,
        "turn1_init": 12,
    }
    for c in all_replays:
        rl = c["true_role"]
        if rl in role_caps and per_role[rl] < role_caps[rl]:
            per_role[rl] += 1
            pool.append(c)

    adapters = {"champion_r1": ROOT / "experiments/discovery02/input/adapters/wsp_r1"}
    for d in ad_dirs:
        if (d / "adapter_config.json").is_file():
            adapters[d.name] = d

    runs = [p for p in (ROOT / "experiments/discovery02", ROOT / "experiments/discovery03", ROOT / "experiments/discovery04") if p.is_dir()]
    returned_patches = load_returned_patches(runs)

    engine = LLM(
        model=assets["model"], dtype="bfloat16", max_model_len=32768,
        max_num_seqs=8, max_num_batched_tokens=2048, enable_lora=True,
        max_loras=1, max_lora_rank=_round_up_vllm_lora_rank(1),
        gpu_memory_utilization=0.82, scheduler_reserve_full_isl=False, tensor_parallel_size=1,
    )
    tok = engine.get_tokenizer()
    params = SamplingParams(
        temperature=0.0, top_p=0.95, max_tokens=4096,
        stop_token_ids=[tok.convert_tokens_to_ids(t) for t in ("<tool_call|>", "<turn|>")],
        skip_special_tokens=False,
    )
    payload = [{"prompt_token_ids": c["tokens"]} for c in pool]

    summary_rows = []
    champ_map = {}

    for idx, (label, ad_path) in enumerate(adapters.items(), 1):
        req = LoRARequest(label, idx, str(ad_path))
        batch = engine.generate(payload, params, lora_request=req, use_tqdm=False)
        records = []
        for c, r in zip(pool, batch, strict=True):
            txt = tok.decode(r.outputs[0].token_ids, skip_special_tokens=False)
            act, args = parse_gemma4_call(txt)
            cmd = (args.get("command") or "").strip() if act == "run_command" else ""
            is_slow = is_slow_suite_pytest(cmd)
            is_file_test = bool(act == "run_command" and not is_slow and "run_command" not in cmd and TEST_CMD.search(cmd))
            defect_catch = False
            if is_file_test and c["issue"] in returned_patches and c["issue"] in tasks_map:
                ex = execute_generated_check(c["issue"], cmd, returned_patches[c["issue"]], tasks_map[c["issue"]]["patch"])
                defect_catch = bool(ex.get("discriminating_defect_catch"))
            has_leak = any(tk in txt for tk in ("<|tool_call><|tool_call>",)) or (act is None and "<|" in txt.split("<channel|>")[-1])
            valid_edit = bool(act == "edit_file" and args.get("filepath") and args.get("old_string") and args.get("new_string") and args.get("old_string") != args.get("new_string") and not has_leak)
            records.append({
                "issue": c["issue"], "true_role": c["true_role"], "prompt_sha": c["prompt_sha"],
                "action": act, "args": args, "command": cmd, "is_bare_pytest": is_slow,
                "is_targ_test": is_file_test, "defect_catch": defect_catch,
                "valid_edit": valid_edit, "syntax_leak": has_leak,
            })
        (out_dir / f"{label}.json").write_text(json.dumps(records, indent=2))
        if label == "champion_r1":
            champ_map = {x["prompt_sha"]: x for x in records}

        b1_prem = [x for x in records if x["true_role"] in ("true_imm_edit_ok", "true_imm_bad_test_path") and champ_map[x["prompt_sha"]]["action"] == "submit_patch"]
        b1_all = [x for x in records if x["true_role"] in ("true_imm_edit_ok", "true_imm_bad_test_path") and (
            champ_map[x["prompt_sha"]]["action"] == "submit_patch" or champ_map[x["prompt_sha"]]["is_bare_pytest"]
        )]
        b3 = [x for x in records if x["true_role"] == "true_imm_read_full" and champ_map[x["prompt_sha"]]["valid_edit"]]
        b4 = [x for x in records if x["true_role"] == "true_imm_test_pass" and champ_map[x["prompt_sha"]]["action"] == "submit_patch"]
        b5 = [x for x in records if x["true_role"] == "turn1_init" and champ_map[x["prompt_sha"]]["action"] in ("read_file", "run_command")]
        b6_all = [x for x in records if x["true_role"] == "true_imm_real_test_fail"]

        agg = {
            "adapter": label,
            "b1_prem_total": len(b1_prem),
            "b1_prem_flipped_to_run": sum(1 for x in b1_prem if x["action"] == "run_command"),
            "b1_submits_left": sum(1 for x in b1_all if x["action"] == "submit_patch"),
            "b1_repeat_edits": sum(1 for x in b1_all if x["action"] == "edit_file"),
            "b1_slow_pytest": sum(1 for x in b1_all if x["is_bare_pytest"]),
            "b1_file_pytest": sum(1 for x in b1_all if x["is_targ_test"]),
            "b1_defect_catches": sum(1 for x in b1_all if x["defect_catch"]),
            "b3_valid_edit_kept": f"{sum(1 for x in b3 if x['valid_edit'])}/{len(b3)}",
            "b3_exact_args_match": f"{sum(1 for x in b3 if x['valid_edit'] and champ_map[x['prompt_sha']]['args'] == x['args'])}/{len(b3)}",
            "b4_verified_kept": f"{sum(1 for x in b4 if x['action'] == 'submit_patch')}/{len(b4)}",
            "b5_turn1_kept": f"{sum(1 for x in b5 if x['action'] in ('read_file', 'run_command') and not x['is_bare_pytest'])}/{len(b5)}",
            "b6_natural_repairs_kept": f"{sum(1 for x in b6_all if x['action'] in ('edit_file', 'read_file'))}/{len(b6_all)}",
            "syntax_leaks": sum(1 for x in records if x["syntax_leak"]),
            "sample_cmds": [x["command"] for x in b1_all if x["action"] == "run_command"][:8],
        }
        summary_rows.append(agg)
        print("V24_SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
''')
""", timeout=60)


def main():
    ensure_session()
    upload_verified(HERE / "remote_synth_v24_matched39_41.py", "/content/wsp/remote_synth_v24_matched39_41.py")
    write_remote_screen_v24()

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== Step 3.2d: Synthesizing V24 Pure `L39+L41` Matched-History Global Detector ===", flush=True)
    run_remote_stage("synth_v24_matched", f"{py_base} -u /content/wsp/remote_synth_v24_matched39_41.py", timeout_sec=1200)
    download_overwrite("/content/wsp/harvest32/v24_matched39_41/manifest.json", HERE / "v24_matched_manifest.json")
    print((HERE / "v24_matched_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== Step 3.3d: Screening V24 Pure `L39+L41` vs Champion & Random Controls ===", flush=True)
    run_remote_stage("screen_v24_matched", f"{py_eval} -u /content/wsp/screen_v24_matched.py", timeout_sec=2400)
    download_overwrite("/content/wsp/harvest32/screen_v24_matched/summary.json", HERE / "screen_v24_matched_summary.json")
    print("=== V24 SCREEN SUMMARY ===", flush=True)
    print((HERE / "screen_v24_matched_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v24_matched_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ = by_name["champion_r1"]

    viable = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        b3_n, b3_d = map(int, r["b3_valid_edit_kept"].split("/"))
        b3_ex_n, _ = map(int, r["b3_exact_args_match"].split("/"))
        b4_n, b4_d = map(int, r["b4_verified_kept"].split("/"))
        b5_n, b5_d = map(int, r["b5_turn1_kept"].split("/"))

        if (
            r["syntax_leaks"] <= 1
            and r["b1_repeat_edits"] == 0
            and b4_n >= max(1, b4_d - 1)
            and b3_n == b3_d
            and b3_ex_n >= max(3, b3_d - 2)
            and b5_n >= max(1, b5_d - 1)
            and r["b1_prem_flipped_to_run"] > 0
            and r["b1_file_pytest"] > (rnd["b1_file_pytest"] if rnd else 0)
        ):
            viable.append(r)

    if not viable:
        print("SCREEN GATE PAUSE: Inspect screen_v24_matched_summary.json before launching disjoint multi-turn eval!", flush=True)
        return

    viable.sort(
        key=lambda r: (
            r["syntax_leaks"] == 0,
            int(r["b4_verified_kept"].split("/")[0]) == int(r["b4_verified_kept"].split("/")[1]),
            r["b1_prem_flipped_to_run"],
            r["b1_defect_catches"],
            r["b1_file_pytest"],
            int(r["b3_exact_args_match"].split("/")[0]),
            -r["b1_submits_left"],
        ),
        reverse=True,
    )
    winner = viable[0]
    mf = json.loads((HERE / "v24_matched_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    ad_path = f"/content/wsp/harvest32/v24_matched39_41/adapters/{winner['adapter']}"

    remote_exec("import shutil; shutil.make_archive('/content/wsp/v24_adapters_backup', 'zip', '/content/wsp/harvest32/v24_matched39_41')")
    download_overwrite("/content/wsp/v24_adapters_backup.zip", HERE / "checkpoints/v24_adapters_backup.zip")

    run_name = "wsp32_v24_disjoint6"
    remote_exec(f"import shutil; from pathlib import Path; shutil.rmtree(Path('/content/wsp/experiments/{run_name}'), ignore_errors=True)")
    tasks_arg = " ".join(DISJOINT_GATE_6)
    print(f"\n=== Step 3.4d: Running End-to-End Multi-Turn Disjoint Gate ({run_name}) on {winner['adapter']} (sha={sha[:12]}) ===", flush=True)
    run_remote_stage(
        f"eval_{run_name}",
        f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split custom_dev --tasks {tasks_arg} --adapter {ad_path} --sha {sha} --name {run_name}",
        timeout_sec=2700,
    )

    out_eval = HERE / "evals" / run_name
    for fn in ("neon_metrics.json", "neon_failures.jsonl"):
        download_overwrite(f"/content/wsp/experiments/{run_name}/{fn}", out_eval / fn)

    remote_exec(f"""
import importlib, json, sys
from pathlib import Path
sys.path.insert(0, '/content/wsp')
import event_funnel_analyzer
importlib.reload(event_funnel_analyzer)
from event_funnel_analyzer import parse_diagnostics_zip

root = Path('/content/wsp')
m = json.loads((root / 'experiments/{run_name}/neon_metrics.json').read_text())
f_v24 = parse_diagnostics_zip(root / 'experiments/{run_name}/diagnostics.zip', set(m['resolved_ids']))
(root / 'v24_disjoint6_report.json').write_text(json.dumps({{
    'adapter': {winner['adapter']!r},
    'sha256': {sha!r},
    'metrics': m,
    'v24_disjoint6_funnel': f_v24,
}}, indent=2))
""", timeout=90)
    download_overwrite("/content/wsp/v24_disjoint6_report.json", HERE / "v24_disjoint6_report.json")
    rep = json.loads((HERE / "v24_disjoint6_report.json").read_text(encoding="utf-8"))
    print("=== V24 END-TO-END DISJOINT-6 MULTI-TURN REPORT ===", flush=True)
    print(json.dumps({
        "adapter": rep["adapter"],
        "sha256": rep["sha256"],
        "solved": rep["metrics"]["solved"],
        "resolved_ids": rep["metrics"]["resolved_ids"],
        "edit_errors": rep["metrics"]["file_edit_errors_total"],
        "syntax_leaks": rep["metrics"]["syntax_leaks"],
        "v24_funnel": rep["v24_disjoint6_funnel"]["funnel"],
        "v24_modes": rep["v24_disjoint6_funnel"]["modes"],
        "v24_episodes": rep["v24_disjoint6_funnel"]["episodes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
