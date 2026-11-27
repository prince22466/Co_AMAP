"""Compare the four compiled edits and matched controls on observed discovery decisions."""
import argparse
import json
import re
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_decisions import ROOT, attach_adapter, replay
from recipe import compile_explicit


def sample_states(cases):
    seen = set()
    for c in cases:
        key = (c["snapshot"],c["role"])
        if c["role"] not in ("unchecked","verified","edit") or key in seen:
            continue
        seen.add(key)
        yield c


def grade(text, action, case):
    if case["role"] == "unchecked":
        m = re.search(r'call:run_command\{command:<\|"\|>(.*?)<\|"\|>\}', text, re.S)
        command = m[1] if m else ""
        pytest=bool(re.search(r"\bpytest\s+[^;|&]*tests?/",command))
        direct=bool(re.match(r"python(?:3)?\s+-c\s",command) and "assert " in command)
        return {"action":action,"check_command":command,"test_requested":pytest or direct}
    if case["role"] == "edit":
        fields = {k:v for k,v in re.findall(r'(filepath|old_string|new_string):<\|"\|>(.*?)<\|"\|>',text,re.S)}
        return {"action":action,"valid":action=="edit_file" and len(fields)==3 and bool(fields.get("old_string"))}
    return {"action":action,"valid":action=="submit_patch"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    out=args.output
    candidates=json.loads((out/"candidates/ranking.json").read_text())["candidates"]
    if not candidates:
        raise ValueError("No discovery fold-supported candidates")
    cases=list(sample_states(json.loads((out/"cases.json").read_text())))
    assets=json.loads((ROOT/"assets.json").read_text())
    tokenizer=AutoTokenizer.from_pretrained(assets["model"],local_files_only=True)
    tokenizer.truncation_side="left"
    model=AutoModelForCausalLM.from_pretrained(assets["model"],local_files_only=True,
                                               dtype=torch.bfloat16,device_map="cuda").eval()
    layers=getattr(model.model,"language_model",model.model).layers
    with zipfile.ZipFile(ROOT/"champion.zip") as z:
        champion=load(z.read("adapters/wsp_r1/adapter_model.safetensors"))
    screen={}
    def measure(name,tensors):
        handles=attach_adapter(layers,tensors)
        results=[]
        try:
            for c in cases:
                text,action=replay(model,tokenizer,c["tokens"],256)
                result=grade(text,action,c)
                results.append(dict(issue=c["issue"],role=c["role"],prompt_sha=c["prompt_sha"],
                                    observed=c["action"],text=text,**result))
        finally:
            for h in handles: h.remove()
        screen[name]=results
        (out/"screen.json").write_text(json.dumps(screen,indent=2))
        print(name,[(x["issue"],x["role"],x["action"]) for x in results],flush=True)
    measure("champion",champion)
    for c in candidates:
        name=c["name"]
        factors=out/"candidates"/(name+".safetensors")
        for control in ("zero","candidate","random"):
            label=name+"_"+control
            adapter=out/"adapters"/label
            if not adapter.exists():
                compile_explicit(ROOT/"champion.zip",factors,adapter,control)
            measure(label,load_file(str(adapter/"adapter_model.safetensors")))


if __name__=="__main__":
    main()
