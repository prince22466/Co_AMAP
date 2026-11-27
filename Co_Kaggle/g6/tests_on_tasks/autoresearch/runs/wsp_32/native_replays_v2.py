"""Replay exact development token IDs (discovery02 + discovery03) through competition vLLM."""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path("/content/wsp")
DATA = ROOT / "input/competitions/gemma-4-developer-agent"
QUICK_30_IDX = {24, 14, 17, 19, 21, 22, 66, 67, 72, 75, 107, 113, 71, 106, 117,
                16, 13, 95, 68, 69, 74, 100, 108, 109, 114, 115, 118, 119, 98, 127}
HOLDOUT_IDS = {"rich_3043", "rich_3064", "rich_2725"}


def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", choices=("unchecked", "check", "verified", "edit", "localize"))
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True, parents=True)
    tasks = [json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines() if l.strip()]
    forbidden_ids = {tasks[i]["instance_id"] for i in QUICK_30_IDX} | HOLDOUT_IDS
    forbidden_snaps = {(tasks[i]["repo"], tasks[i]["base_commit"]) for i in QUICK_30_IDX}

    cases = json.loads(args.cases.read_text())
    for c in cases:
        assert c["issue"] not in forbidden_ids, f"Forbidden benchmark/holdout case {c['issue']}"
        if "repo" in c and "snapshot" in c:
            assert (c["repo"], c["snapshot"]) not in forbidden_snaps, f"Snapshot overlap on {c['issue']}"

    selected = []
    seen_non_post = set()
    for c in cases:
        if args.roles and c["role"] not in args.roles:
            continue
        if c["role"] in ("edit", "localize"):
            key = (c["issue"], c["role"])
            if key in seen_non_post:
                continue
            seen_non_post.add(key)
        selected.append(c)

    original = ROOT / "experiments/discovery02/input/adapters/wsp_r1"
    assets = json.loads((ROOT / "assets.json").read_text())
    engine = LLM(
        model=assets["model"],
        dtype="bfloat16",
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=_round_up_vllm_lora_rank(1),
        gpu_memory_utilization=0.80,
        scheduler_reserve_full_isl=False,
        tensor_parallel_size=1,
    )
    tokenizer = engine.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=2048,
        stop_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call|>", "<turn|>")],
        skip_special_tokens=False,
    )

    output = args.output / "champion.json"
    results = []
    payload = [{"prompt_token_ids": c["tokens"]} for c in selected]
    req = LoRARequest("champion", 1, str(original))
    batch = engine.generate(payload, params, lora_request=req, use_tqdm=False)
    for case, r in zip(selected, batch, strict=True):
        gen = r.outputs[0]
        txt = tokenizer.decode(gen.token_ids, skip_special_tokens=False)
        m = re.search(r"<\|tool_call>call:(\w+)\{", txt)
        action = m[1] if m else None
        results.append(dict(
            issue=case["issue"],
            role=case["role"],
            prompt_sha=case["prompt_sha"],
            observed=case["action"],
            action=action,
            text=txt,
            tokens=list(gen.token_ids),
            prompt_tokens=len(case["tokens"]),
            generated_tokens=len(gen.token_ids),
            finish_reason=gen.finish_reason,
        ))
        print("champion", case["issue"], case["role"], case["action"], "->", action, len(gen.token_ids), flush=True)
    output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
