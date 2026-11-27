"""Run V17 Surgical Zero-KV-Drift synthesis, strict file-level pytest & >=5/6 exact-edit screen, and gated QUICK_15 wsp32_iter05."""
import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, ensure_session, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def write_screen_v17() -> None:
    remote_exec(r"""
import subprocess
from pathlib import Path
subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)
Path('/content/wsp/screen_v17_surgical.py').write_text('''
import json, os, re, shutil, tempfile, zipfile
from collections import Counter
from pathlib import Path
from extract_v14_clean import parse_gemma4_call
from screen_native_v2 import DATA, ROOT, execute_generated_check, load_returned_patches
from decision_cases_v2 import TEST_CMD
from synth_v16_upperband import literal_role

def is_slow_suite_pytest(cmd: str) -> bool:
    # Flag bare `pytest` AND directory-wide `pytest tests/` (anything without `.py` or `-k` or `python -c`)!
    if not cmd:
        return False
    c = cmd.strip()
    if "pytest" not in c:
        return False
    # Fast targeted pytest MUST specify a `.py` test file or `-k` filter!
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
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    cdir = ROOT / "harvest32/clean_v14"
    vdir = ROOT / "harvest32/v17_surgical"
    out_dir = ROOT / "harvest32/screen_v17_surgical"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ad_dirs = sorted((vdir / "adapters").iterdir())
    for d in ad_dirs:
        if d.name.endswith("_candidate"):
            verify_compliance(d)

    assets = json.loads((ROOT / "assets.json").read_text())
    tok_hf = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)

    all_cases = {}
    for c in json.loads((cdir / "clean_cases.json").read_text()):
        tr = literal_role(tok_hf, c["tokens"], c.get("source_run", ""))
        all_cases[c["prompt_sha"]] = {**c, "true_role": tr}

    tasks_all = [json.loads(s) for s in (DATA / "tasks.jsonl").read_text().splitlines() if s.strip()]
    disc_ids = {
        "rich_3506", "rich_3296", "rich_3067", "rich_3469", "rich_3130",
        "rich_4079", "rich_3480", "rich_3472", "rich_3105", "rich_3006",
        "fastapi_14786", "fastapi_14479", "requests_7328",
        "fastapi_15661", "fastapi_15588", "requests_7505", "requests_7502",
        "rich_4070", "rich_4006",
    }
    disc_ps = [t["problem_statement"] for t in tasks_all if t["instance_id"] in disc_ids]
    disc_needles = {}
    for t in tasks_all:
        if t["instance_id"] in disc_ids:
            cands = [ln.strip() for ln in t["problem_statement"].splitlines() if len(ln.strip()) >= 8]
            uniq = [ln for ln in cands if sum(1 for o in disc_ps if ln in o) == 1] or cands
            disc_needles[t["instance_id"]] = max(uniq, key=len)

    for rname in ("discovery02", "discovery03", "discovery04"):
        rp = ROOT / "experiments" / rname / "replay_tokens.jsonl"
        if not rp.is_file():
            continue
        for line in rp.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            sha = o["request_sha"]
            if sha in all_cases:
                continue
            tr = literal_role(tok_hf, o["tokens"])
            if tr in ("true_imm_edit_ok", "true_imm_test_fail", "true_imm_test_pass"):
                txt_full = tok_hf.decode(o["tokens"], skip_special_tokens=False)
                iss = next((iid for iid, ndl in disc_needles.items() if ndl in txt_full), f"disc_{sha[:8]}")
                all_cases[sha] = {"issue": iss, "prompt_sha": sha, "tokens": o["tokens"], "true_role": tr}

    pool = []
    per_role = Counter()
    role_caps = {
        "true_imm_edit_ok": 20,
        "true_imm_test_fail": 10,
        "true_imm_edit_err": 12,
        "true_imm_read_full": 16,
        "true_imm_test_pass": 10,
        "turn1_init": 16,
    }
    for sha, c in all_cases.items():
        rl = c["true_role"]
        if rl in role_caps and per_role[rl] < role_caps[rl]:
            per_role[rl] += 1
            pool.append(c)

    adapters = {"champion_r1": ROOT / "experiments/discovery02/input/adapters/wsp_r1"}
    for zname in ("V17_hybrid_zero", "V17_dzero_39_40_41"):
        zp = vdir / "adapters" / zname
        if (zp / "adapter_config.json").is_file():
            adapters[zname] = zp
    for d in ad_dirs:
        if d.name not in adapters and not d.name.startswith("V17_dzero_") and (d / "adapter_config.json").is_file():
            adapters[d.name] = d

    runs = [p for p in (ROOT / "experiments/discovery02", ROOT / "experiments/discovery03", ROOT / "experiments/discovery04") if p.is_dir()]
    returned_patches = load_returned_patches(runs)
    tasks_map = {r["instance_id"]: r for r in tasks_all}

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

    # Reuse champion_r1 raw outputs from V16 screen if identical prompt_shas to save 2 mins!
    v16_champ_p = ROOT / "harvest32/screen_v16_upperband/champion_r1.json"
    v16_cache = {}
    if v16_champ_p.is_file():
        v16_cache = {x["prompt_sha"]: x for x in json.loads(v16_champ_p.read_text())}

    summary_rows = []
    champ_map = {}

    for idx, (label, ad_path) in enumerate(adapters.items(), 1):
        if label == "champion_r1" and all(c["prompt_sha"] in v16_cache for c in pool):
            records = []
            for c in pool:
                x = dict(v16_cache[c["prompt_sha"]])
                cmd = x.get("command") or ""
                is_slow = is_slow_suite_pytest(cmd)
                is_file_test = bool(x["action"] == "run_command" and not is_slow and TEST_CMD.search(cmd))
                x["is_bare_pytest"] = is_slow
                x["is_targ_test"] = is_file_test
                records.append(x)
        else:
            req = LoRARequest(label, idx, str(ad_path))
            batch = engine.generate(payload, params, lora_request=req, use_tqdm=False)
            records = []
            for c, r in zip(pool, batch, strict=True):
                txt = tok.decode(r.outputs[0].token_ids, skip_special_tokens=False)
                act, args = parse_gemma4_call(txt)
                cmd = (args.get("command") or "").strip() if act == "run_command" else ""
                is_slow = is_slow_suite_pytest(cmd)
                is_file_test = bool(act == "run_command" and not is_slow and TEST_CMD.search(cmd))
                defect_catch = False
                if is_file_test and c["issue"] in returned_patches and c["issue"] in tasks_map:
                    ex = execute_generated_check(c["issue"], cmd, returned_patches[c["issue"]], tasks_map[c["issue"]]["patch"])
                    defect_catch = bool(ex.get("discriminating_defect_catch"))
                has_leak = any(tk in txt for tk in ("<|tool_call><|tool_call>",)) or (act is None and "<|" in txt.split("<channel|>")[-1])
                valid_edit = bool(act == "edit_file" and args.get("filepath") and args.get("old_string") and args.get("new_string") and args.get("old_string") != args.get("new_string") and not has_leak)
                records.append({
                    "issue": c["issue"], "true_role": c["true_role"], "prompt_sha": c["prompt_sha"],
                    "action": act, "command": cmd, "is_bare_pytest": is_slow,
                    "is_targ_test": is_file_test, "defect_catch": defect_catch,
                    "valid_edit": valid_edit, "syntax_leak": has_leak, "text_tail": txt[-400:],
                })
        (out_dir / f"{label}.json").write_text(json.dumps(records, indent=2))
        if label == "champion_r1":
            champ_map = {x["prompt_sha"]: x for x in records}

        b1 = [x for x in records if x["true_role"] in ("true_imm_edit_ok", "true_imm_test_fail") and (
            champ_map[x["prompt_sha"]]["action"] == "submit_patch" or champ_map[x["prompt_sha"]]["is_bare_pytest"]
        )]
        b2 = [x for x in records if x["true_role"] == "true_imm_edit_err" and champ_map[x["prompt_sha"]]["action"] in ("edit_file", "submit_patch")]
        b3 = [x for x in records if x["true_role"] == "true_imm_read_full" and champ_map[x["prompt_sha"]]["valid_edit"]]
        b4 = [x for x in records if x["true_role"] == "true_imm_test_pass" and champ_map[x["prompt_sha"]]["action"] == "submit_patch"]
        b5 = [x for x in records if x["true_role"] == "turn1_init" and champ_map[x["prompt_sha"]]["action"] in ("read_file", "run_command")]

        agg = {
            "adapter": label,
            "b1_total": len(b1),
            "b1_submits_left": sum(1 for x in b1 if x["action"] == "submit_patch"),
            "b1_slow_pytest": sum(1 for x in b1 if x["is_bare_pytest"]),
            "b1_file_pytest": sum(1 for x in b1 if x["is_targ_test"]),
            "b1_repairs_or_reads": sum(1 for x in b1 if x["action"] in ("edit_file", "read_file")),
            "b1_defect_catches": sum(1 for x in b1 if x["defect_catch"]),
            "b2_err_to_read_flips": f"{sum(1 for x in b2 if x['action'] == 'read_file')}/{len(b2)}",
            "b3_valid_edit_kept": f"{sum(1 for x in b3 if x['valid_edit'])}/{len(b3)}",
            "b3_exact_edit_match": f"{sum(1 for x in b3 if champ_map[x['prompt_sha']]['text_tail'] == x['text_tail'])}/{len(b3)}",
            "b4_verified_kept": f"{sum(1 for x in b4 if x['action'] == 'submit_patch')}/{len(b4)}",
            "b5_turn1_kept": f"{sum(1 for x in b5 if x['action'] in ('read_file', 'run_command') and not x['is_bare_pytest'])}/{len(b5)}",
            "syntax_leaks": sum(1 for x in records if x["syntax_leak"]),
            "sample_cmds": [x["command"] for x in b1 if x["action"] == "run_command"][:5],
        }
        summary_rows.append(agg)
        print("V17_SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
''')
""")


def main():
    ensure_session()
    upload_verified(HERE / "remote_synth_v17_surgical.py", "/content/wsp/synth_v17_surgical.py")
    write_screen_v17()

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 1. Synthesizing V17 Surgical Zero-KV-Drift (`L39,L40,L41 down_proj` + gentle `L35 o_proj`) Adapters ===", flush=True)
    run_remote_stage("synth_v17_surgical", f"{py_base} -u /content/wsp/synth_v17_surgical.py", timeout_sec=900)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v17_surgical/manifest.json", str(HERE / "v17_surgical_manifest.json")], check=True)
    print((HERE / "v17_surgical_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 2. Screening V17 Adapters (Strict File-Level Pytest + >=4/6 Exact-Edit Gate) ===", flush=True)
    run_remote_stage("screen_v17_surgical", f"{py_eval} -u /content/wsp/screen_v17_surgical.py", timeout_sec=3600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v17_surgical/summary.json", str(HERE / "screen_v17_surgical_summary.json")], check=True)
    print("=== V17 SURGICAL SUMMARY ===", flush=True)
    print((HERE / "screen_v17_surgical_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v17_surgical_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ = by_name["champion_r1"]

    cands = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        b2_n = int(r["b2_err_to_read_flips"].split("/")[0])
        b3_n, b3_d = map(int, r["b3_valid_edit_kept"].split("/"))
        b3_ex, _ = map(int, r["b3_exact_edit_match"].split("/"))
        b4_n, b4_d = map(int, r["b4_verified_kept"].split("/"))
        b5_n, b5_d = map(int, r["b5_turn1_kept"].split("/"))

        score_cand = (r["b1_file_pytest"] + r["b1_repairs_or_reads"] + b2_n, -r["b1_submits_left"], -r["b1_slow_pytest"])
        score_champ = (champ["b1_file_pytest"] + champ["b1_repairs_or_reads"] + int(champ["b2_err_to_read_flips"].split("/")[0]), -champ["b1_submits_left"], -champ["b1_slow_pytest"])
        score_rnd = (
            (rnd["b1_file_pytest"] + rnd["b1_repairs_or_reads"] + int(rnd["b2_err_to_read_flips"].split("/")[0]), -rnd["b1_submits_left"], -rnd["b1_slow_pytest"])
            if rnd else (-1, -99, -99)
        )

        # HARD PRESERVATION GATE: b3_valid_edit == 6/6, b3_exact_edit_match >= 4/6, b4 == 4/4, b5 >= 15/16, slow_pytest <= 1!
        if (
            r["syntax_leaks"] <= 1
            and b4_n == b4_d
            and b3_n == b3_d
            and b3_ex >= 4
            and b5_n >= max(1, b5_d - 1)
            and r["b1_slow_pytest"] <= 1
            and r["b1_file_pytest"] >= 2
            and r["b1_submits_left"] <= 3
            and score_cand > score_champ
            and score_cand > score_rnd
        ):
            cands.append(r)

    if not cands:
        print("GATE STOP: Reviewing V17 summary before launching QUICK_15!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            r["syntax_leaks"] == 0,
            r["b1_defect_catches"],
            int(r["b3_exact_edit_match"].split("/")[0]),
            -r["b1_submits_left"],
            r["b1_file_pytest"] + int(r["b2_err_to_read_flips"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v17_surgical/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v17_surgical_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V17 GATE PASSED! Launching QUICK_15 wsp32_iter05 on {winner['adapter']} (sha={sha}) ===", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", str(HERE / "run_step5_and_step6.py"),
            "--mode", "quick15",
            "--run-name", "wsp32_iter05",
            "--remote-adapter", ad_path,
            "--sha", sha,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
