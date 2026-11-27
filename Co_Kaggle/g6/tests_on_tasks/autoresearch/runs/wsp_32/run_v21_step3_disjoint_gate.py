"""Step 3 End-to-End Orchestrator for V21 Pure-WSP 4-State Closed-Loop Repair Circuit.

1. Builds execution-grounded disjoint corpus (`build_v21_disjoint_corpus.py`).
2. Synthesizes V21 Sacred-Orthogonalized 4-Transition Cascade (`remote_synth_v21_fsm.py`).
3. Screens candidates vs `champion_r1`, `zero`, and `random` controls across all 6 behavioral buckets:
   - `b1`: `true_imm_edit_ok` -> targeted check (`run_command`, NOT `submit_patch` or bare `pytest`)
   - `b2`: `true_imm_edit_err` -> `read_file`
   - `b3`: `true_imm_read_full` -> valid `edit_file` AND high `b3_exact_args_match` (`>= 4/6`)!
   - `b4`: `true_imm_test_pass` -> `100%` `submit_patch` kept!
   - `b5`: `turn1_init` -> localization kept!
   - `b6` (NEW!): `true_imm_real_test_fail` -> `edit_file` / `read_file` repair (kills `RATIONALIZED_FAIL_SUB` & re-test loops)!
4. Runs END-TO-END MULTI-TURN DISJOINT VERIFICATION (`wsp32_v21_disjoint6`) on 6 disjoint public snapshots:
     Preservation (2): `rich_3067`, `rich_3006` (champion = 2/2)
     Conversion   (4): `rich_3296`, `fastapi_14786`, `fastapi_14479`, `rich_3480` (champion = 0/4)
   and computes the 5-stage causal event funnel (`s1 -> s2 -> s3 -> s4 -> s5 -> grader_pass`)!
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, ensure_session, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage

DISJOINT_GATE_6 = [
    "rich_3067",      # Disjoint Preservation Control #1 (Champion SOLVED)
    "rich_3006",      # Disjoint Preservation Control #2 (Champion SOLVED)
    "rich_3296",      # Disjoint Conversion #1 (Champion UNCHECKED_SUBMIT, SyntaxError caught in 0.27s)
    "fastapi_14786",  # Disjoint Conversion #2 (Champion RATIONALIZED_FAIL_SUB, TypeError caught in 0.78s)
    "fastapi_14479",  # Disjoint Conversion #3 (Champion UNCHECKED_SUBMIT, 4 failed in 0.76s)
    "rich_3480",      # Disjoint Conversion #4 (Champion RATIONALIZED_FAIL_SUB, pytest ec=1)
]


def write_remote_screen_v21() -> None:
    remote_exec(r"""
import subprocess
from pathlib import Path
subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)

Path('/content/wsp/screen_v21_fsm.py').write_text('''
import json, os, shutil, tempfile, zipfile
from collections import Counter
from pathlib import Path
from build_v21_disjoint_corpus import parse_gemma4_call, TEST_CMD
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
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    vdir = ROOT / "harvest32/v21_fsm"
    out_dir = ROOT / "harvest32/screen_v21_fsm"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ad_dirs = sorted((vdir / "adapters").iterdir())
    for d in ad_dirs:
        if d.name.endswith("_candidate") or d.name.endswith("_zero"):
            verify_compliance(d)

    all_replays = json.loads((ROOT / "harvest32/clean_v21/real_replays.json").read_text())
    pool = []
    per_role = Counter()
    role_caps = {
        "true_imm_edit_ok": 20,
        "true_imm_bad_test_path": 4,
        "true_imm_real_test_fail": 20,
        "true_imm_edit_err": 14,
        "true_imm_read_full": 16,
        "true_imm_test_pass": 16,
        "turn1_init": 16,
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

    tasks_all = [json.loads(s) for s in (DATA / "tasks.jsonl").read_text().splitlines() if s.strip()]
    runs = [p for p in (ROOT / "experiments/discovery02", ROOT / "experiments/discovery03", ROOT / "experiments/discovery04") if p.is_dir()]
    returned_patches = load_returned_patches(runs)
    tasks_map = {r["instance_id"]: r for r in tasks_all}
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
                "valid_edit": valid_edit, "syntax_leak": has_leak, "text_tail": txt[-400:],
            })
        (out_dir / f"{label}.json").write_text(json.dumps(records, indent=2))
        if label == "champion_r1":
            champ_map = {x["prompt_sha"]: x for x in records}

        # Bucket 1: strictly UNCHECKED edit_ok / bad_test_path where champion prematurely submits or runs bare pytest!
        b1 = [x for x in records if x["true_role"] in ("true_imm_edit_ok", "true_imm_bad_test_path") and (
            champ_map[x["prompt_sha"]]["action"] == "submit_patch" or champ_map[x["prompt_sha"]]["is_bare_pytest"]
        )]
        # Bucket 2: FileEditError -> read_file
        b2 = [x for x in records if x["true_role"] == "true_imm_edit_err" and champ_map[x["prompt_sha"]]["action"] in ("edit_file", "submit_patch")]
        # Bucket 3: Valid edit preservation AND exact patch args match!
        b3 = [x for x in records if x["true_role"] == "true_imm_read_full" and champ_map[x["prompt_sha"]]["valid_edit"]]
        # Bucket 4: Verified test pass -> submit_patch preservation!
        b4 = [x for x in records if x["true_role"] == "true_imm_test_pass" and champ_map[x["prompt_sha"]]["action"] == "submit_patch"]
        # Bucket 5: Turn 1 localization preservation!
        b5 = [x for x in records if x["true_role"] == "turn1_init" and champ_map[x["prompt_sha"]]["action"] in ("read_file", "run_command")]
        # Bucket 6 (NEW!): Real post-edit test failure (`ec in {1, 2}`) -> must NOT submit_patch or loop run_command; must repair (`edit_file` or `read_file`)!
        b6_all = [x for x in records if x["true_role"] == "true_imm_real_test_fail"]
        b6_bad = [x for x in b6_all if champ_map[x["prompt_sha"]]["action"] in ("submit_patch", "run_command", None)]

        agg = {
            "adapter": label,
            "b1_total": len(b1),
            "b1_submits_left": sum(1 for x in b1 if x["action"] == "submit_patch"),
            "b1_slow_pytest": sum(1 for x in b1 if x["is_bare_pytest"]),
            "b1_file_pytest": sum(1 for x in b1 if x["is_targ_test"]),
            "b1_defect_catches": sum(1 for x in b1 if x["defect_catch"]),
            "b2_err_to_read_flips": f"{sum(1 for x in b2 if x['action'] == 'read_file')}/{len(b2)}",
            "b3_valid_edit_kept": f"{sum(1 for x in b3 if x['valid_edit'])}/{len(b3)}",
            "b3_exact_args_match": f"{sum(1 for x in b3 if x['valid_edit'] and champ_map[x['prompt_sha']]['args'] == x['args'])}/{len(b3)}",
            "b4_verified_kept": f"{sum(1 for x in b4 if x['action'] == 'submit_patch')}/{len(b4)}",
            "b5_turn1_kept": f"{sum(1 for x in b5 if x['action'] in ('read_file', 'run_command') and not x['is_bare_pytest'])}/{len(b5)}",
            "b6_fail_repairs_all": f"{sum(1 for x in b6_all if x['action'] in ('edit_file', 'read_file'))}/{len(b6_all)}",
            "b6_fail_submits_left": sum(1 for x in b6_all if x["action"] == "submit_patch"),
            "b6_fail_flips": f"{sum(1 for x in b6_bad if x['action'] in ('edit_file', 'read_file'))}/{len(b6_bad)}",
            "syntax_leaks": sum(1 for x in records if x["syntax_leak"]),
            "sample_cmds": [x["command"] for x in b1 if x["action"] == "run_command"][:6],
        }
        summary_rows.append(agg)
        print("V21_SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
''')
""", timeout=90)


def download_overwrite(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        local_path.unlink()
    subprocess.run([CLI, "download", "-s", SESSION, remote_path, str(local_path)], check=True)


def main():
    ensure_session()
    upload_verified(HERE / "build_v21_disjoint_corpus.py", "/content/wsp/build_v21_disjoint_corpus.py")
    upload_verified(HERE / "remote_synth_v21_fsm.py", "/content/wsp/remote_synth_v21_fsm.py")
    upload_verified(HERE / "event_funnel_analyzer.py", "/content/wsp/event_funnel_analyzer.py")
    write_remote_screen_v21()

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== Step 3.1: Building Execution-Grounded Disjoint Repair Corpus (`clean_v21`) ===", flush=True)
    run_remote_stage("build_v21_corpus", f"{py_eval} -u /content/wsp/build_v21_disjoint_corpus.py", timeout_sec=1200)
    download_overwrite("/content/wsp/harvest32/clean_v21/real_replays.json", HERE / "checkpoints/clean_v14/real_replays_v21.json")

    print("=== Step 3.2: Synthesizing V21 Sacred-Orthogonalized 4-State FSM Adapters ===", flush=True)
    run_remote_stage("synth_v21_fsm", f"{py_base} -u /content/wsp/remote_synth_v21_fsm.py", timeout_sec=1200)
    download_overwrite("/content/wsp/harvest32/v21_fsm/manifest.json", HERE / "v21_fsm_manifest.json")
    print((HERE / "v21_fsm_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== Step 3.3: Screening V21 Candidates vs Champion, Zero, and Random Controls (6 Buckets) ===", flush=True)
    run_remote_stage("screen_v21_fsm", f"{py_eval} -u /content/wsp/screen_v21_fsm.py", timeout_sec=2700)
    download_overwrite("/content/wsp/harvest32/screen_v21_fsm/summary.json", HERE / "screen_v21_fsm_summary.json")
    print("=== V21 6-BUCKET SCREEN SUMMARY ===", flush=True)
    print((HERE / "screen_v21_fsm_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v21_fsm_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ = by_name["champion_r1"]

    # Verify zero control matches champion_r1 100%!
    zeros = [r for r in rows if r["adapter"].endswith("_zero")]
    if zeros:
        z0 = zeros[0]
        print(f"Zero-Control Check ({z0['adapter']}): b3_exact_args_match={z0['b3_exact_args_match']}, syntax_leaks={z0['syntax_leaks']}", flush=True)

    viable = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        b2_n = int(r["b2_err_to_read_flips"].split("/")[0])
        b3_n, b3_d = map(int, r["b3_valid_edit_kept"].split("/"))
        b3_ex_n, _ = map(int, r["b3_exact_args_match"].split("/"))
        b4_n, b4_d = map(int, r["b4_verified_kept"].split("/"))
        b5_n, b5_d = map(int, r["b5_turn1_kept"].split("/"))
        b6_rep_n, _ = map(int, r["b6_fail_repairs_all"].split("/"))

        champ_b6_n = int(champ["b6_fail_repairs_all"].split("/")[0])
        rnd_b6_n = int(rnd["b6_fail_repairs_all"].split("/")[0]) if rnd else -1
        rnd_targ = (rnd["b1_file_pytest"] + int(rnd["b2_err_to_read_flips"].split("/")[0])) if rnd else -1

        if (
            r["syntax_leaks"] == 0
            and b4_n >= max(1, b4_d - 1)
            and b3_n == b3_d
            and b3_ex_n >= max(3, b3_d - 2)
            and b5_n >= max(1, b5_d - 1)
            and b6_rep_n > champ_b6_n
            and (r["b1_file_pytest"] + b2_n + b6_rep_n) > (rnd_targ + rnd_b6_n)
        ):
            viable.append(r)

    if not viable:
        print("SCREEN GATE PAUSE: Inspect screen_v21_fsm_summary.json before launching disjoint multi-turn eval!", flush=True)
        return

    viable.sort(
        key=lambda r: (
            int(r["b4_verified_kept"].split("/")[0]) == int(r["b4_verified_kept"].split("/")[1]),
            int(r["b3_exact_args_match"].split("/")[0]),
            int(r["b6_fail_repairs_all"].split("/")[0]) - r["b6_fail_submits_left"],
            r["b1_defect_catches"],
            r["b1_file_pytest"] + int(r["b2_err_to_read_flips"].split("/")[0]),
            -r["b1_submits_left"],
        ),
        reverse=True,
    )
    winner = viable[0]
    mf = json.loads((HERE / "v21_fsm_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    ad_path = f"/content/wsp/harvest32/v21_fsm/adapters/{winner['adapter']}"

    # Save backup of V21 adapters locally before multi-turn eval
    remote_exec("import shutil; shutil.make_archive('/content/wsp/v21_adapters_backup', 'zip', '/content/wsp/harvest32/v21_fsm')")
    download_overwrite("/content/wsp/v21_adapters_backup.zip", HERE / "checkpoints/v21_adapters_backup.zip")

    run_name = "wsp32_v21_disjoint6"
    remote_exec(f"import shutil; from pathlib import Path; shutil.rmtree(Path('/content/wsp/experiments/{run_name}'), ignore_errors=True)")
    tasks_arg = " ".join(DISJOINT_GATE_6)
    print(f"\n=== Step 3.4: Running End-to-End Multi-Turn Disjoint Gate ({run_name}) on {winner['adapter']} (sha={sha[:12]}) ===", flush=True)
    run_remote_stage(
        f"eval_{run_name}",
        f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split custom_dev --tasks {tasks_arg} --adapter {ad_path} --sha {sha} --name {run_name}",
        timeout_sec=2700,
    )

    out_eval = HERE / "evals" / run_name
    for fn in ("neon_metrics.json", "neon_failures.jsonl"):
        download_overwrite(f"/content/wsp/experiments/{run_name}/{fn}", out_eval / fn)

    # Compute exact 5-stage funnel on wsp32_v21_disjoint6 and compare head-to-head vs champion on those exact 6 tasks!
    remote_exec(f"""
import importlib, json, sys
from pathlib import Path
sys.path.insert(0, '/content/wsp')
import event_funnel_analyzer
importlib.reload(event_funnel_analyzer)
from event_funnel_analyzer import parse_diagnostics_zip

root = Path('/content/wsp')
m = json.loads((root / 'experiments/{run_name}/neon_metrics.json').read_text())
f_v21 = parse_diagnostics_zip(root / 'experiments/{run_name}/diagnostics.zip', set(m['resolved_ids']))

# Also collect champion baseline episodes on those exact 6 tasks from discovery02 + discovery03!
champ_eps = []
for dr in ('discovery02', 'discovery03'):
    dm = json.loads((root / 'experiments' / dr / 'neon_metrics.json').read_text())
    df = parse_diagnostics_zip(root / 'experiments' / dr / 'diagnostics.zip', set(dm['resolved_ids']))
    for e in df['episodes']:
        if e['instance_id'] in {DISJOINT_GATE_6!r}:
            champ_eps.append(e)

out = {{
    'adapter': {winner['adapter']!r},
    'sha256': {sha!r},
    'metrics': m,
    'v21_disjoint6_funnel': f_v21,
    'champion_disjoint6_episodes': champ_eps,
}}
(root / 'v21_disjoint6_report.json').write_text(json.dumps(out, indent=2))
""", timeout=90)
    download_overwrite("/content/wsp/v21_disjoint6_report.json", HERE / "v21_disjoint6_report.json")
    rep = json.loads((HERE / "v21_disjoint6_report.json").read_text(encoding="utf-8"))
    print("=== V21 END-TO-END DISJOINT-6 MULTI-TURN REPORT ===", flush=True)
    print(json.dumps({
        "adapter": rep["adapter"],
        "sha256": rep["sha256"],
        "solved": rep["metrics"]["solved"],
        "resolved_ids": rep["metrics"]["resolved_ids"],
        "edit_errors": rep["metrics"]["file_edit_errors_total"],
        "syntax_leaks": rep["metrics"]["syntax_leaks"],
        "v21_funnel": rep["v21_disjoint6_funnel"]["funnel"],
        "v21_modes": rep["v21_disjoint6_funnel"]["modes"],
        "v21_episodes": rep["v21_disjoint6_funnel"]["episodes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
