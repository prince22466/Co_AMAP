"""Run v3 targeted-dispatch + anti-bare-pytest synthesis and native vLLM screening on Colab wsp32."""
import json
import subprocess
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def main():
    # Write a lightweight runner wrapper for screen_multilayer_v2 pointing to v3_candidates -> screen_v3_r1
    screen_v3_code = """\"\"\"Screen v3 r=1 disjoint candidates vs champion_r1, zero, and random controls.\"\"\"
import argparse, json, os
from pathlib import Path
from recipe import compile_explicit
from screen_native_v2 import DATA, ROOT, grade_generation, load_returned_patches

def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    p = argparse.ArgumentParser()
    p.add_argument("--harvest-dir", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    args = p.parse_args()

    hdir = args.harvest_dir
    cand_dir = hdir / "v3_candidates"
    out_dir = hdir / "screen_v3_r1"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_list = json.loads((cand_dir / "manifest.json").read_text())["candidates"]
    all_cases = json.loads((hdir / "cases.json").read_text())
    selected, seen = [], set()
    for c in all_cases:
        if c["role"] not in ("unchecked", "verified", "edit", "localize"):
            continue
        key = (c["issue"], c["role"])
        if key not in seen:
            seen.add(key)
            selected.append(c)

    adapters = {"champion_r1": ROOT / "experiments/discovery02/input/adapters/wsp_r1"}
    seen_zero = set()
    for c in cand_list:
        fac = cand_dir / f"{c['name']}.safetensors"
        lset = tuple(sorted(c["layers"]))
        if lset not in seen_zero:
            seen_zero.add(lset)
            zlbl = f"V3_zero_{'_'.join(map(str, lset))}"
            zdest = out_dir / "adapters" / zlbl
            if not zdest.exists():
                compile_explicit(ROOT / "champion.zip", fac, zdest, "zero", rank=1)
            adapters[zlbl] = zdest
        for ctrl in ("candidate", "random"):
            lbl = f"{c['name']}_{ctrl}"
            dest = out_dir / "adapters" / lbl
            if not dest.exists():
                compile_explicit(ROOT / "champion.zip", fac, dest, ctrl, rank=1)
            adapters[lbl] = dest

    # Reuse champion_r1.json from screen_multilayer_r1 if present to save 60s!
    prev_champ = hdir / "screen_multilayer_r1/champion_r1.json"
    if prev_champ.is_file() and not (out_dir / "champion_r1.json").is_file():
        (out_dir / "champion_r1.json").write_text(prev_champ.read_text())

    returned_patches = load_returned_patches(args.runs)
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
    champ_edit_ok = None
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

        unch = [x for x in records if x["role"] == "unchecked"]
        ver = [x for x in records if x["role"] == "verified"]
        ed = [x for x in records if x["role"] == "edit"]
        ed_vs_c = [x for x in ed if champ_edit_ok is None or x["issue"] in champ_edit_ok]
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
            "localize_preserved": f"{sum(1 for x in loc if x.get('preserved'))}/{len(loc)}",
        }
        summary_rows.append(agg)
        print("V3_SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
"""
    (HERE / "screen_v3.py").write_text(screen_v3_code, encoding="utf-8")

    for fname in ("synthesize_targeted_dispatch_v3.py", "screen_v3.py"):
        upload_verified(HERE / fname, f"/content/wsp/{fname}")

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 1. Synthesizing v3 targeted-dispatch + anti-bare-pytest r=1 candidates ===", flush=True)
    run_remote_stage(
        "synth_v3",
        f"{py_base} -u /content/wsp/synthesize_targeted_dispatch_v3.py "
        f"--harvest-dir /content/wsp/harvest32 "
        f"--native-replays /content/wsp/harvest32/native/champion.json",
        timeout_sec=900,
    )
    subprocess.run(
        [CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v3_candidates/manifest.json", str(HERE / "v3_manifest.json")],
        check=True,
    )
    print((HERE / "v3_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 2. Screening v3 candidates in native vLLM ===", flush=True)
    run_remote_stage(
        "screen_v3",
        f"{py_eval} -u /content/wsp/screen_v3.py "
        f"--harvest-dir /content/wsp/harvest32 "
        f"--runs /content/wsp/experiments/discovery02 /content/wsp/experiments/discovery03",
        timeout_sec=2700,
    )
    subprocess.run(
        [CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v3_r1/summary.json", str(HERE / "screen_v3_summary.json")],
        check=True,
    )

    remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_v3_r1')
for p in sorted(d.glob('*.json')):
    if p.name == 'summary.json':
        continue
    rows = json.loads(p.read_text())
    unch = [r for r in rows if r.get('role') == 'unchecked']
    if p.stem.endswith('_candidate') or p.stem == 'champion_r1':
        out[p.stem] = [
            {
                'issue': r['issue'],
                'action': r.get('action'),
                'command': r.get('command'),
                'is_bare_pytest': r.get('is_bare_pytest'),
                'defect_catch': r.get('discriminating_defect_catch'),
                'agent_exit': r.get('on_returned_patch_exit'),
                'ref_exit': r.get('on_reference_fix_exit'),
                'out_tail': r.get('agent_out_tail', '')[-180:],
            }
            for r in unch
        ]
Path('/content/wsp/harvest32/screen_v3_r1/unchecked_details.json').write_text(json.dumps(out, indent=2))
""")
    subprocess.run(
        [CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v3_r1/unchecked_details.json", str(HERE / "screen_v3_unchecked_details.json")],
        check=True,
    )
    print("\n=== V3 SCREEN SUMMARY ===", flush=True)
    print((HERE / "screen_v3_summary.json").read_text(encoding="utf-8"), flush=True)
    print("\n=== V3 UNCHECKED DETAILS ===", flush=True)
    print((HERE / "screen_v3_unchecked_details.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
