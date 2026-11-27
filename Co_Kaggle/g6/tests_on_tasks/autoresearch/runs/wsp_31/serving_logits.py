"""Read the full next-token logits through vLLM at each archive's real rank bucket."""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/content/wsp")


def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from recipe import compile_explicit
    p=argparse.ArgumentParser()
    p.add_argument("control",choices=("champion","zero","paired"))
    p.add_argument("--cases",type=Path)
    args=p.parse_args()
    out=ROOT/"logit_preflight"
    out.mkdir(exist_ok=True)
    champion=out/"champion/adapters/wsp_r1"
    zero=out/"zero"
    if args.control=="champion":
        with zipfile.ZipFile(ROOT/"champion.zip") as z:
            for n in ("adapter_config.json","adapter_model.safetensors"):
                z.extract("adapters/wsp_r1/"+n,out/"champion")
    if args.control in ("zero","paired") and not zero.is_dir():
        compile_explicit(ROOT/"champion.zip",None,zero,"zero")
    adapters={"champion":champion,"zero":zero}
    if args.control=="paired":
        # Compare this engine with the preceding rank-one champion process.
        if not (out/"champion-0.npy").is_file():
            raise ValueError("Run the champion rank bucket on these same prefixes first")
        del adapters["champion"]
        adapters.update({p.name:p for p in (ROOT/"harvest03/native04/screen/adapters").glob("*_zero")})
        adapters["base"]=None
    else:
        adapters={args.control:adapters[args.control]}
    buckets={_round_up_vllm_lora_rank(json.loads((d/"adapter_config.json").read_text())["r"])
             for d in adapters.values() if d is not None}
    if len(buckets)!=1: raise RuntimeError("Run each actual serving rank bucket separately")
    assets=json.loads((ROOT/"assets.json").read_text())
    engine=LLM(model=assets["model"],enable_lora=True,max_loras=1,max_lora_rank=buckets.pop(),
               max_model_len=32768,max_num_seqs=8,max_num_batched_tokens=2048,
               gpu_memory_utilization=0.90,dtype="bfloat16",scheduler_reserve_full_isl=False,
               tensor_parallel_size=1,logprobs_mode="raw_logits",max_logprobs=-1)
    tokenizer=engine.get_tokenizer()
    prefixes=[tokenizer.encode("After editing a Python package, check its public interface.")]
    if args.cases:
        kept=set()
        for c in json.loads(args.cases.read_text()):
            if c["role"] not in kept:
                prefixes.append(c["tokens"])
                kept.add(c["role"])
    for idx,(label,adapter) in enumerate(adapters.items(),1):
        for i,ids in enumerate(prefixes):
            request=LoRARequest(label,idx,str(adapter)) if adapter is not None else None
            result=engine.generate([{"prompt_token_ids":ids}],
                                   SamplingParams(temperature=0,max_tokens=1,logprobs=-1,detokenize=False),
                                   lora_request=request,use_tqdm=False)[0]
            mapping=result.outputs[0].logprobs[0]
            keys=sorted(mapping)
            assert keys==list(range(len(keys))) and len(keys)>=len(tokenizer)
            logits=np.array([mapping[k].logprob for k in keys],dtype=np.float32)
            np.save(out/f"{label}-{i}.npy",logits)
        print("Read raw logits",label,flush=True)
    prefix_ids=[hashlib.sha256(np.asarray(ids,dtype="<i4").tobytes()).hexdigest() for ids in prefixes]
    if args.control=="champion":
        (out/"prefixes.json").write_text(json.dumps(prefix_ids))
    if args.control in ("zero","paired"):
        if prefix_ids!=json.loads((out/"prefixes.json").read_text()):
            raise ValueError("Champion and zero snapshots have different token inputs")
        rows=[]
        for label in adapters.keys()-{"champion"}:
            for i,ids in enumerate(prefixes):
                original=np.load(out/f"champion-{i}.npy")
                padded=np.load(out/f"{label}-{i}.npy")
                rows.append({"adapter":label,"prefix_sha256":hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()).hexdigest(),
                             "tokens":len(ids),"vocab":len(original),"identical":bool(np.array_equal(original,padded)),
                             "max_abs":float(np.max(np.abs(original-padded))),"same_argmax":int(original.argmax())==int(padded.argmax())})
        (out/"comparison.json").write_text(json.dumps(rows,indent=2))
        print(rows,flush=True)


if __name__=="__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD","spawn")
    main()
