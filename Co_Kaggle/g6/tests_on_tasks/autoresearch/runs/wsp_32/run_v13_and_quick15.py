"""Run V13 Sparse Naturally-Silent Base-Neuron synthesis, 4096-token native vLLM screen (including reproduced unchecked cases!), and gated QUICK_15 iter04."""
import json
import subprocess
import sys

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def main():
    upload_verified(HERE / "remote_synth_v13_sparse_natural.py", "/content/wsp/synth_v13_sparse_natural.py")

    # Note: Screen with max_tokens=4096 (matching agent.yaml!) AND keep ALL reproduced `submit_patch` unchecked cases!
    remote_exec("""
import subprocess
from pathlib import Path
subprocess.run(['pkill', '-9', '-f', 'vllm'], check=False)
Path('/content/wsp/screen_v13_sparse_natural.py').write_text('''
import json, os, shutil, tempfile, zipfile
from pathlib import Path
from screen_native_v2 import DATA, ROOT, grade_generation, load_returned_patches

def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    hdir = Path("/content/wsp/harvest32")
    vdir = hdir / "v13_sparse_natural"
    out_dir = hdir / "screen_v13_sparse_natural"
    out_dir.mkdir(parents=True, exist_ok=True)

    adapters = {"champion_r1": ROOT / "experiments/discovery02/input/adapters/wsp_r1"}
    for d in sorted((vdir / "adapters").iterdir()):
        if (d / "adapter_config.json").is_file():
            adapters[d.name] = d

    all_cases = json.loads((hdir / "cases.json").read_text())
    native = {r["prompt_sha"]: r for r in json.loads((hdir / "native/champion.json").read_text())}
    selected, seen = [], set()
    # For unchecked: prefer turns where native champion actually emitted submit_patch or bare pytest!
    for c in sorted(all_cases, key=lambda x: (x["role"] == "unchecked" and native.get(x["prompt_sha"], {}).get("action") != "submit_patch")):
        if c["role"] in ("unchecked", "verified", "edit", "localize"):
            key = (c["issue"], c["role"])
            if key not in seen:
                seen.add(key)
                selected.append(c)

    runs = [p for p in (
        Path("/content/wsp/experiments/discovery02"),
        Path("/content/wsp/experiments/discovery03"),
        Path("/content/wsp/experiments/discovery04"),
    ) if p.is_dir()]
    returned_patches = load_returned_patches(runs)
    tasks_map = {r["instance_id"]: r for r in map(json.loads, (DATA / "tasks.jsonl").read_text().splitlines())}
    assets = json.loads((ROOT / "assets.json").read_text())

    engine = LLM(
        model=assets["model"], dtype="bfloat16", max_model_len=32768,
        max_num_seqs=8, max_num_batched_tokens=2048, enable_lora=True,
        max_loras=1, max_lora_rank=_round_up_vllm_lora_rank(1),
        gpu_memory_utilization=0.85, scheduler_reserve_full_isl=False, tensor_parallel_size=1,
    )
    tokenizer = engine.get_tokenizer()
    params = SamplingParams(
        temperature=0.0, top_p=0.95, max_tokens=4096,
        stop_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call|>", "<turn|>")],
        skip_special_tokens=False,
    )
    payload = [{"prompt_token_ids": c["tokens"]} for c in selected]

    summary_rows = []
    champ_edit_tails = {}
    champ_edit_ok = set()
    champ_ver_ok = set()

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
            champ_ver_ok = {x["issue"] for x in records if x["role"] == "verified" and x.get("preserved")}

        unch = [x for x in records if x["role"] == "unchecked"]
        ver = [x for x in records if x["role"] == "verified" and (not champ_ver_ok or x["issue"] in champ_ver_ok)]
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
        print("V13_SCREEN_RESULT", json.dumps(agg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
''')
""")

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 1. Synthesizing V13 Sparse Naturally-Silent Base-Neuron Readouts ===", flush=True)
    run_remote_stage("synth_v13_sparse_natural", f"{py_base} -u /content/wsp/synth_v13_sparse_natural.py", timeout_sec=900)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v13_sparse_natural/manifest.json", str(HERE / "v13_sparse_natural_manifest.json")], check=True)
    print((HERE / "v13_sparse_natural_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 2. Screening V13 Adapters in Native vLLM (max_tokens=4096) ===", flush=True)
    run_remote_stage("screen_v13_sparse_natural", f"{py_eval} -u /content/wsp/screen_v13_sparse_natural.py", timeout_sec=3600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v13_sparse_natural/summary.json", str(HERE / "screen_v13_sparse_natural_summary.json")], check=True)
    print("=== V13 SPARSE NATURAL SUMMARY ===", flush=True)
    print((HERE / "screen_v13_sparse_natural_summary.json").read_text(encoding="utf-8"), flush=True)

    rows = json.loads((HERE / "screen_v13_sparse_natural_summary.json").read_text(encoding="utf-8"))
    by_name = {r["adapter"]: r for r in rows}
    champ_row = by_name["champion_r1"]

    cands = []
    for r in rows:
        if not r["adapter"].endswith("_candidate"):
            continue
        rnd = by_name.get(r["adapter"].replace("_candidate", "_random"))
        beats_champ = (
            (r["unchecked_targeted_flips"], -r["unchecked_bare_pytest"], -r["unchecked_submits_left"])
            > (champ_row["unchecked_targeted_flips"], -champ_row["unchecked_bare_pytest"], -champ_row["unchecked_submits_left"])
        )
        beats_random = rnd is None or (
            (r["unchecked_targeted_flips"], -r["unchecked_bare_pytest"], int(r["edit_exact_match"].split("/")[0]))
            > (rnd["unchecked_targeted_flips"], -rnd["unchecked_bare_pytest"], int(rnd["edit_exact_match"].split("/")[0]))
        )
        ver_n, ver_d = map(int, r["verified_preserved"].split("/"))
        edc_n, edc_d = map(int, r["edit_preserved_vs_champ"].split("/"))
        ex_n, ex_d = map(int, r["edit_exact_match"].split("/"))
        if (
            ver_n == ver_d
            and edc_n == edc_d
            and ex_n >= (ex_d // 2)
            and r["unchecked_bare_pytest"] == 0
            and beats_champ
            and beats_random
        ):
            cands.append(r)

    if not cands:
        print("GATE STOP: No V13 candidate met 100% champ-edit + >=50% exact-edit + 0 bare-pytest + flip improvement gate!", flush=True)
        return

    cands.sort(
        key=lambda r: (
            int(r["edit_exact_match"].split("/")[0]) >= (int(r["edit_exact_match"].split("/")[1]) * 2) // 3,
            r["unchecked_defect_catches"],
            r["unchecked_targeted_flips"],
            int(r["edit_exact_match"].split("/")[0]),
        ),
        reverse=True,
    )
    winner = cands[0]
    ad_path = f"/content/wsp/harvest32/v13_sparse_natural/adapters/{winner['adapter']}"
    mf = json.loads((HERE / "v13_sparse_natural_manifest.json").read_text(encoding="utf-8"))
    sha = next(x["sha256"] for x in mf["compiled"] if x["adapter"] == winner["adapter"])
    print(f"\n=== V13 GATE PASSED! Launching QUICK_15 wsp32_iter04 on {winner['adapter']} (sha={sha}) ===", flush=True)
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
