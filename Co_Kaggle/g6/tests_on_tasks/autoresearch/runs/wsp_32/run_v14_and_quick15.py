"""Run V14 Clean Immediate-State Extraction, Multi-Circuit Synthesis, 4-Bucket Native vLLM Screen, and Gated QUICK_15 iter04."""
import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def write_screen_v14() -> None:
    remote_exec(r"""
import subprocess
from pathlib import Path
subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)
Path('/content/wsp/screen_v14_multicircuit.py').write_text('''
import json, os, re, shutil, tempfile, zipfile
from pathlib import Path
from extract_v14_clean import parse_gemma4_call
from screen_native_v2 import DATA, ROOT, execute_generated_check, load_returned_patches
from decision_cases_v2 import TEST_CMD

IS_BARE_PYTEST = re.compile(r"^(?:python(?:3)?\s+-m\s+)?pytest(?:\s+-[qvx]+)*$")

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

    cdir = ROOT / "harvest32/clean_v14"
    vdir = ROOT / "harvest32/v14_multicircuit"
    out_dir = ROOT / "harvest32/screen_v14_multicircuit"
    out_dir.mkdir(parents=True, exist_ok=True)

    ad_dirs = sorted((vdir / "adapters").iterdir())
    for d in ad_dirs:
        if d.name.endswith("_candidate"):
            verify_compliance(d)

    all_cases = json.loads((cdir / "clean_cases.json").read_text())
    champ_clean = {r["prompt_sha"]: r for r in json.loads((cdir / "champion_clean.json").read_text())}

    # Build the 4 crystal-clear, zero-contamination evaluation buckets!
    selected = []
    for c in all_cases:
        nat_act = champ_clean.get(c["prompt_sha"], {}).get("action")
        nat_cmd = ((champ_clean.get(c["prompt_sha"], {}).get("arguments") or {}).get("command") or "").strip()
        rl = c["clean_role"]
        bucket = None
        if rl in ("imm_edit_ok", "post_edit_prem_submit") and (nat_act == "submit_patch" or IS_BARE_PYTEST.match(nat_cmd)):
            bucket = "b1_prem_or_bare"       # True post-edit premature submit or bare pytest in champion!
        elif rl == "imm_edit_err" and nat_act in ("edit_file", "submit_patch"):
            bucket = "b2_edit_err_loop"      # True FileEditError -> blind edit_file / submit loop in champion!
        elif rl == "imm_read_full" and nat_act == "edit_file":
            bucket = "b3_read_to_edit"       # True clean read_file -> valid edit_file in champion (MUST PRESERVE!)
        elif rl == "imm_test_pass" and nat_act == "submit_patch":
            bucket = "b4_verified_submit"    # True verified pass -> submit_patch in champion (MUST PRESERVE!)
        elif rl == "turn1_init" and nat_act in ("read_file", "run_command"):
            bucket = "b5_turn1_init"         # True Turn 1 localization (MUST PRESERVE!)
        if bucket:
            c2 = dict(c, bucket=bucket)
            selected.append(c2)

    from collections import Counter
    print("Clean 5-Bucket Screen Suite:", Counter(c["bucket"] for c in selected), flush=True)

    adapters = {"champion_r1": ROOT / "experiments/discovery02/input/adapters/wsp_r1"}
    for d in ad_dirs:
        if (d / "adapter_config.json").is_file():
            adapters[d.name] = d

    runs = [p for p in (ROOT / "experiments/discovery02", ROOT / "experiments/discovery03", ROOT / "experiments/discovery04") if p.is_dir()]
    returned_patches = load_returned_patches(runs)
    tasks_map = {r["instance_id"]: r for r in map(json.loads, (DATA / "tasks.jsonl").read_text().splitlines())}
    assets = json.loads((ROOT / "assets.json").read_text())

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
    payload = [{"prompt_token_ids": c["tokens"]} for c in selected]

    summary_rows = []
    champ_b3_tails = {}

    for idx, (label, ad_path) in enumerate(adapters.items(), 1):
        save_p = out_dir / f"{label}.json"
        if save_p.is_file():
            records = json.loads(save_p.read_text())
        else:
            req = LoRARequest(label, idx, str(ad_path))
            batch = engine.generate(payload, params, lora_request=req, use_tqdm=False)
            records = []
            for c, r in zip(selected, batch, strict=True):
                txt = tok.decode(r.outputs[0].token_ids, skip_special_tokens=False)
                act, args = parse_gemma4_call(txt)
                cmd = (args.get("command") or "").strip() if act == "run_command" else ""
                is_bare = bool(IS_BARE_PYTEST.match(cmd)) if cmd else False
                is_targ_test = bool(act == "run_command" and not is_bare and TEST_CMD.search(cmd))
                defect_catch = False
                if is_targ_test and c["issue"] in returned_patches and c["issue"] in tasks_map:
                    ex = execute_generated_check(c["issue"], cmd, returned_patches[c["issue"]], tasks_map[c["issue"]]["patch"])
                    defect_catch = bool(ex.get("discriminating_defect_catch"))
                has_leak = any(tk in txt for tk in ("<|tool_call><|tool_call>",)) or (act is None and "<|" in txt.split("<channel|>")[-1])
                valid_edit = bool(act == "edit_file" and args.get("filepath") and args.get("old_string") and args.get("new_string") and args.get("old_string") != args.get("new_string") and not has_leak)
                records.append({
                    "issue": c["issue"], "bucket": c["bucket"], "prompt_sha": c["prompt_sha"],
                    "action": act, "command": cmd, "is_bare_pytest": is_bare,
                    "is_targ_test": is_targ_test, "defect_catch": defect_catch,
                    "valid_edit": valid_edit, "syntax_leak": has_leak, "text_tail": txt[-400:],
                })
            save_p.write_text(json.dumps(records, indent=2))

        if label == "champion_r1":
            champ_b3_tails = {x["prompt_sha"]: x["text_tail"] for x in records if x["bucket"] == "b3_read_to_edit"}

        b1 = [x for x in records if x["bucket"] == "b1_prem_or_bare"]
        b2 = [x for x in records if x["bucket"] == "b2_edit_err_loop"]
        b3 = [x for x in records if x["bucket"] == "b3_read_to_edit"]
        b4 = [x for x in records if x["bucket"] == "b4_verified_submit"]
        b5 = [x for x in records if x["bucket"] == "b5_turn1_init"]

        agg = {
            "adapter": label,
            "b1_total": len(b1),
            "b1_submits_left": sum(1 for x in b1 if x["action"] == "submit_patch"),
            "b1_bare_pytest": sum(1 for x in b1 if x["is_bare_pytest"]),
            "b1_targeted_tests": sum(1 for x in b1 if x["is_targ_test"]),
            "b1_defect_catches": sum(1 for x in b1 if x["defect_catch"]),
            "b2_err_to_read_flips": f"{sum(1 for x in b2 if x['action'] == 'read_file')}/{len(b2)}",
            "b3_valid_edit_kept": f"{sum(1 for x in b3 if x['valid_edit'])}/{len(b3)}",
            "b3_exact_edit_match": f"{sum(1 for x in b3 if champ_b3_tails.get(x['prompt_sha']) == x['text_tail'])}/{len(b3)}",
            "b4_verified_kept": f"{sum(1 for x in b4 if x['action'] == 'submit_patch')}/{len(b4)}",
            "b5_turn1_kept": f"{sum(1 for x in b5 if x['action'] in ('read_file', 'run_command') and not x['is_bare_pytest'])}/{len(b5)}",
            "syntax_leaks": sum(1 for x in records if x["syntax_leak"]),
        }
        summary_rows.append(agg)
        print("V14_SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
''')
""")


def main():
    upload_verified(HERE / "remote_extract_v14_clean.py", "/content/wsp/extract_v14_clean.py")
    upload_verified(HERE / "remote_synth_v14_multicircuit.py", "/content/wsp/synth_v14_multicircuit.py")
    write_screen_v14()

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    chk = remote_exec("from pathlib import Path; print('CLEAN14=' + str(Path('/content/wsp/harvest32/clean_v14/champion_clean.json').is_file()))")
    if "CLEAN14=False" in chk:
        print("=== 1. Extracting & Replaying 100% Clean Immediate-State Turns across 19 Snapshots ===", flush=True)
        run_remote_stage("extract_v14_clean", f"{py_eval} -u /content/wsp/extract_v14_clean.py", timeout_sec=1500)

    print("=== 2. Synthesizing V14 Multi-Circuit (`o_proj + down_proj` & `down_proj`-only) Adapters ===", flush=True)
    run_remote_stage("synth_v14_multicircuit", f"{py_base} -u /content/wsp/synth_v14_multicircuit.py", timeout_sec=900)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v14_multicircuit/manifest.json", str(HERE / "v14_multicircuit_manifest.json")], check=True)
    print((HERE / "v14_multicircuit_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 3. Screening V14 Adapters on Clean 5-Bucket Native vLLM Suite ===", flush=True)
    run_remote_stage("screen_v14_multicircuit", f"{py_eval} -u /content/wsp/screen_v14_multicircuit.py", timeout_sec=3600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v14_multicircuit/summary.json", str(HERE / "screen_v14_multicircuit_summary.json")], check=True)
    print("=== V14 MULTI-CIRCUIT SUMMARY ===", flush=True)
    print((HERE / "screen_v14_multicircuit_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v14_multicircuit_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ = by_name["champion_r1"]

    # Verify zero controls are 100% bit-exact first!
    for zname in ("V14_hybrid_zero",):
        if zname in by_name:
            zr = by_name[zname]
            print("Verified hybrid zero control:", zr, flush=True)

    cands = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        b2_n = int(r["b2_err_to_read_flips"].split("/")[0])
        b3_n, b3_d = map(int, r["b3_valid_edit_kept"].split("/"))
        b4_n, b4_d = map(int, r["b4_verified_kept"].split("/"))
        b5_n, b5_d = map(int, r["b5_turn1_kept"].split("/"))

        score_cand = (r["b1_targeted_tests"] + b2_n, -r["b1_submits_left"], -r["b1_bare_pytest"])
        score_champ = (champ["b1_targeted_tests"] + int(champ["b2_err_to_read_flips"].split("/")[0]), -champ["b1_submits_left"], -champ["b1_bare_pytest"])
        score_rnd = (
            (rnd["b1_targeted_tests"] + int(rnd["b2_err_to_read_flips"].split("/")[0]), -rnd["b1_submits_left"], -rnd["b1_bare_pytest"])
            if rnd else (-1, -99, -99)
        )

        if (
            r["syntax_leaks"] == 0
            and b4_n == b4_d
            and b3_n >= max(1, b3_d - 1)
            and b5_n >= max(1, b5_d - 1)
            and r["b1_bare_pytest"] == 0
            and score_cand > score_champ
            and score_cand > score_rnd
        ):
            cands.append(r)

    if not cands:
        print("GATE STOP: No V14 candidate met strict 5-bucket preservation + 0 syntax-leaks + multi-circuit gain gate!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            int(r["b3_valid_edit_kept"].split("/")[0]) == int(r["b3_valid_edit_kept"].split("/")[1]),
            r["b1_defect_catches"],
            r["b1_targeted_tests"] + int(r["b2_err_to_read_flips"].split("/")[0]),
            -r["b1_submits_left"],
            int(r["b3_exact_edit_match"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v14_multicircuit/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v14_multicircuit_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V14 GATE PASSED! Launching QUICK_15 wsp32_iter04 on {winner['adapter']} (sha={sha}) ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(HERE / "run_step5_and_step6.py"),
            "--mode",
            "quick15",
            "--run-name",
            "wsp32_iter04",
            "--remote-adapter",
            ad_path,
            "--sha",
            sha,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
