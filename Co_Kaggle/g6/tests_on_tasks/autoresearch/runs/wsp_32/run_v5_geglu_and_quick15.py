"""Run V5 GeGLU Exact-Cancellation Transistor synthesis, compliance preflight, native vLLM screen, and gated QUICK_15 iter02."""
import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def main():
    upload_verified(HERE / "remote_synth_v5_geglu.py", "/content/wsp/synth_v5_geglu.py")

    # Write native vLLM screener for v5_geglu/adapters (also checks exact edit action+filepath+args match vs champion_r1!)
    remote_exec("""
from pathlib import Path
Path('/content/wsp/screen_v5_geglu.py').write_text('''
import argparse, json, os, shutil, tempfile, zipfile
from pathlib import Path
from screen_native_v2 import DATA, ROOT, grade_generation, load_returned_patches

def verify_geglu_compliance(ad_dir: Path):
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

    hdir = Path("/content/wsp/harvest32")
    v5_dir = hdir / "v5_geglu"
    out_dir = hdir / "screen_v5_geglu"
    out_dir.mkdir(parents=True, exist_ok=True)

    ad_dirs = sorted((v5_dir / "adapters").iterdir())
    verify_geglu_compliance(next(d for d in ad_dirs if d.name.endswith("_candidate")))

    adapters = {"champion_r1": ROOT / "experiments/discovery02/input/adapters/wsp_r1"}
    for d in ad_dirs:
        if (d / "adapter_config.json").is_file():
            adapters[d.name] = d

    prev_champ = hdir / "screen_multilayer_r1/champion_r1.json"
    if prev_champ.is_file() and not (out_dir / "champion_r1.json").is_file():
        (out_dir / "champion_r1.json").write_text(prev_champ.read_text())

    all_cases = json.loads((hdir / "cases.json").read_text())
    selected, seen = [], set()
    for c in all_cases:
        if c["role"] in ("unchecked", "verified", "edit", "localize"):
            key = (c["issue"], c["role"])
            if key not in seen:
                seen.add(key)
                selected.append(c)

    runs = [Path("/content/wsp/experiments/discovery02"), Path("/content/wsp/experiments/discovery03")]
    returned_patches = load_returned_patches(runs)
    tasks_map = {r["instance_id"]: r for r in map(json.loads, (DATA / "tasks.jsonl").read_text().splitlines())}
    assets = json.loads((ROOT / "assets.json").read_text())

    engine = LLM(
        model=assets["model"], dtype="bfloat16", max_model_len=32768,
        max_num_seqs=8, max_num_batched_tokens=2048, enable_lora=True,
        max_loras=1, max_lora_rank=_round_up_vllm_lora_rank(1),
        gpu_memory_utilization=0.80, scheduler_reserve_full_isl=False, tensor_parallel_size=1,
    )
    tokenizer = engine.get_tokenizer()
    params = SamplingParams(
        temperature=0.0, top_p=0.95, max_tokens=2048,
        stop_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call|>", "<turn|>")],
        skip_special_tokens=False,
    )
    payload = [{"prompt_token_ids": c["tokens"]} for c in selected]

    summary_rows = []
    champ_edit_tails = {}
    champ_edit_ok = set()

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
                    "issue": case["issue"], "role": case["role"], "observed": case["action"],
                    "generated_tokens": len(r.outputs[0].token_ids), "text_tail": txt[-400:], **graded,
                })
            save_p.write_text(json.dumps(records, indent=2))

        if label == "champion_r1":
            champ_edit_ok = {x["issue"] for x in records if x["role"] == "edit" and x.get("preserved")}
            champ_edit_tails = {x["issue"]: x["text_tail"] for x in records if x["role"] == "edit"}

        unch = [x for x in records if x["role"] == "unchecked"]
        ver = [x for x in records if x["role"] == "verified"]
        ed = [x for x in records if x["role"] == "edit"]
        ed_vs_c = [x for x in ed if x["issue"] in champ_edit_ok]
        exact_ed_match = sum(1 for x in ed if champ_edit_tails.get(x["issue"]) == x["text_tail"])
        loc = [x for x in records if x["role"] == "localize"]
        agg = {
            "adapter": label,
            "unchecked_total": len(unch),
            "unchecked_submits_left": sum(1 for x in unch if x.get("action") == "submit_patch"),
            "unchecked_targeted_flips": sum(1 for x in unch if x.get("flipped_to_targeted_check")),
            "unchecked_defect_catches": sum(1 for x in unch if x.get("discriminating_defect_catch")),
            "unchecked_bare_pytest": sum(1 for x in unch if x.get("is_bare_pytest")),
            "verified_preserved": f"{sum(1 for x in ver if x.get('preserved'))}/{len(ver)}",
            "edit_preserved_abs": f"{sum(1 for x in ed if x.get('preserved'))}/{len(ed)}",
            "edit_preserved_vs_champ": f"{sum(1 for x in ed_vs_c if x.get('preserved'))}/{len(ed_vs_c)}",
            "edit_exact_match": f"{exact_ed_match}/{len(ed)}",
            "localize_preserved": f"{sum(1 for x in loc if x.get('preserved'))}/{len(loc)}",
        }
        summary_rows.append(agg)
        print("V5_SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
''')
""")

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 1. Synthesizing V5 GeGLU Exact-Cancellation Transistors on {39, 40, 41} ===", flush=True)
    run_remote_stage("synth_v5_geglu", f"{py_base} -u /content/wsp/synth_v5_geglu.py", timeout_sec=600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v5_geglu/manifest.json", str(HERE / "v5_geglu_manifest.json")], check=True)
    print((HERE / "v5_geglu_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 2. Screening V5 GeGLU Transistors + Compliance Check in Native vLLM ===", flush=True)
    run_remote_stage("screen_v5_geglu", f"{py_eval} -u /content/wsp/screen_v5_geglu.py", timeout_sec=1800)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v5_geglu/summary.json", str(HERE / "screen_v5_geglu_summary.json")], check=True)

    remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_v5_geglu')
for p in sorted(d.glob('*.json')):
    if p.name == 'summary.json':
        continue
    rows = json.loads(p.read_text())
    unch = [r for r in rows if r.get('role') == 'unchecked']
    if p.stem.endswith('_candidate') or p.stem == 'champion_r1':
        out[p.stem] = [
            {
                'issue': r['issue'], 'action': r.get('action'), 'command': r.get('command'),
                'is_bare_pytest': r.get('is_bare_pytest'), 'defect_catch': r.get('discriminating_defect_catch'),
                'agent_exit': r.get('on_returned_patch_exit'), 'ref_exit': r.get('on_reference_fix_exit'),
            }
            for r in unch
        ]
Path('/content/wsp/harvest32/screen_v5_geglu/unchecked_details.json').write_text(json.dumps(out, indent=2))
""")
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v5_geglu/unchecked_details.json", str(HERE / "screen_v5_geglu_unchecked_details.json")], check=True)
    print("=== V5 GEGLU SUMMARY ===", flush=True)
    print((HERE / "screen_v5_geglu_summary.json").read_text(encoding="utf-8"), flush=True)
    print("=== V5 GEGLU UNCHECKED DETAILS ===", flush=True)
    print((HERE / "screen_v5_geglu_unchecked_details.json").read_text(encoding="utf-8"), flush=True)

    # 3. Automatic strict gate for QUICK_15 iter02!
    rows = json.loads((HERE / "screen_v5_geglu_summary.json").read_text(encoding="utf-8"))
    cands = [
        r for r in rows
        if r["adapter"].endswith("_candidate")
        and r["verified_preserved"] == "3/3"
        and r["edit_preserved_vs_champ"] == "4/4"
        and r["localize_preserved"] == "13/13"
        and r["unchecked_submits_left"] <= 2
    ]
    if not cands:
        print("GATE STOP: No V5 GeGLU candidate met strict 4/4 edit + 3/3 verified + <=2 submits_left gate; inspect summary before QUICK_15!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            r["unchecked_defect_catches"],
            r["unchecked_bare_pytest"] == 0,
            r["unchecked_targeted_flips"],
            -r["unchecked_submits_left"],
            int(r["edit_exact_match"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v5_geglu/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v5_geglu_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== GATE PASSED! Launching QUICK_15 wsp32_iter02 on {winner['adapter']} (sha={sha}) ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(HERE / "run_step5_and_step6.py"),
            "--mode",
            "quick15",
            "--run-name",
            "wsp32_iter02",
            "--remote-adapter",
            ad_path,
            "--sha",
            sha,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
