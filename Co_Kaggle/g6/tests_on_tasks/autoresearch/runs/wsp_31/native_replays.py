"""Replay exact development token IDs through the competition vLLM and retain native continuations."""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path("/content/wsp")


def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--adapters",type=Path)
    parser.add_argument("--verified",type=Path)
    parser.add_argument("--roles",nargs="+",choices=("unchecked","check","verified","edit"))
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    cases=json.loads(args.cases.read_text())
    if args.verified:
        cases+=json.loads(args.verified.read_text())
    # Only issue-disjoint development cases are accepted. The benchmark has its own frozen runner.
    assert all(c["issue"] in {"rich_3506","rich_3296","rich_3067","rich_3469","rich_3130"} for c in cases)
    selected=[]
    edited=set()
    for c in cases:
        if args.roles and c["role"] not in args.roles: continue
        if c["role"]=="edit":
            if c["issue"] in edited: continue
            edited.add(c["issue"])
        selected.append(c)
    original=ROOT/"experiments/discovery02/input/adapters/wsp_r1"
    adapters={"champion":original}
    if args.adapters:
        adapters={p.name:p for p in sorted(args.adapters.iterdir()) if (p/"adapter_config.json").is_file()}
        adapters={"champion_same_engine": original, **adapters}
    rank=max(json.loads((path/"adapter_config.json").read_text())["r"] for path in adapters.values())
    model_path=ROOT/"input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"
    engine=LLM(model=str(model_path),dtype="bfloat16",max_model_len=32768,
               max_num_seqs=8,max_num_batched_tokens=2048,
               enable_lora=True,max_loras=1,max_lora_rank=_round_up_vllm_lora_rank(rank),
               gpu_memory_utilization=0.90,scheduler_reserve_full_isl=False,tensor_parallel_size=1)
    tokenizer=engine.get_tokenizer()
    params=SamplingParams(temperature=0.0,top_p=.95,max_tokens=4096,
                          stop_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call|>","<turn|>")],
                          skip_special_tokens=False)
    for idx,(label,adapter) in enumerate(adapters.items(),1):
        output=args.output/(label+".json")
        if output.exists(): continue
        results=[]
        payload=[{"prompt_token_ids":case["tokens"]} for case in selected]
        adapter_request=LoRARequest(label,idx,str(adapter))
        if args.adapters:
            batch=engine.generate(payload,params,lora_request=adapter_request,use_tqdm=False)
        else:
            batch=(engine.generate([prompt],params,lora_request=adapter_request,use_tqdm=False)[0] for prompt in payload)
        for case,r in zip(selected,batch,strict=True):
            ids=case["tokens"]
            gen=r.outputs[0]
            txt=tokenizer.decode(gen.token_ids,skip_special_tokens=False)
            match=re.search(r"<\|tool_call>call:(\w+)\{",txt)
            action=match[1] if match else None
            results.append(dict(issue=case["issue"],role=case["role"],prompt_sha=case["prompt_sha"],
                                observed=case["action"],action=action,text=txt,tokens=list(gen.token_ids),
                                prompt_tokens=len(ids),generated_tokens=len(gen.token_ids),finish_reason=gen.finish_reason))
            print(label,case["issue"],case["role"],action,len(gen.token_ids),flush=True)
            tmp=output.with_suffix(".partial")
            tmp.write_text(json.dumps(results))
        tmp.replace(output)


if __name__=="__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD","spawn")
    main()
