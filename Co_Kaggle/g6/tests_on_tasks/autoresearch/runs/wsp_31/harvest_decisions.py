"""Replay champion decisions and harvest matched thought and tool-dispatch windows."""
import argparse
import collections
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision_cases import read_cases
from rank_columns import shortlist

ROOT = Path("/content/wsp")
DATA = ROOT / "input/competitions/gemma-4-developer-agent"


def attach_adapter(layers, tensors):
    handles = []
    for key, row in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        layer = int(key.split(".layers.")[1].split(".")[0])
        a = row.cuda()
        b = tensors[key.replace("lora_A", "lora_B")].cuda()
        def add(module, inputs, output, a=a, b=b):
            return output + (inputs[0] @ a.T) @ b.T
        handles.append(layers[layer].mlp.down_proj.register_forward_hook(add))
    return handles


def proposal_test(case):
    # Resolve a conventional source-corresponding test from the unpatched public snapshot.
    if case["issue"] == "rich_3506" and case["filepath"] == "rich/segment.py":
        # Discovery-only reproducer. Verified fail on 4101991898, pass with its published fix.
        return ('python -c "from rich.segment import Segment; s=Segment(chr(0x6c34)*2+\'aaaaa\'); '
                'a,b=s.split_cells(3); assert (a.cell_length,b.cell_length)==(3,s.cell_length-3)"')
    path = Path(case["filepath"] or "")
    if path.suffix != ".py":
        return None
    target = "tests/test_" + path.stem.lstrip("_") + ".py"
    with tarfile.open(DATA/"snapshots"/(case["issue"] + ".tgz")) as archive:
        if any(m.name.removeprefix("./") == target and m.isfile() for m in archive):
            return "python -m pytest -q " + target
    return None


def replay(model, tokenizer, prefix, max_new):
    ids = torch.tensor([prefix], device=model.device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             do_sample=False, max_new_tokens=max_new, use_cache=True, prefill_chunk_size=512,
                             eos_token_id=[tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call|>", "<turn|>")])
    text = tokenizer.decode(out[0,ids.shape[1]:], skip_special_tokens=False)
    match = re.search(r"<\|tool_call>call:(\w+)\{", text)
    return text, match.group(1) if match else None


def contrast_suffixes(tokenizer, generated, command):
    # A matched counterfactual uses the same repair history. Only the continuation differs.
    # Use the real thought if present. If absent, label the matched thought pair as teacher-forced.
    start, end, dispatch = "<|channel>thought\n", "\n<channel|>", "<|tool_call>call:"
    thought = re.search(r"<\|channel>thought\n(.*?)\s*<channel\|>", generated, re.S)
    is_submit = "call:submit_patch{" in generated
    unchecked = "The change is complete and the requested behavior has been implemented."
    useful = "I should run a focused test of the public interface before finishing."
    if thought and thought[1].strip():
        if is_submit:
            unchecked = thought[1].strip()
        else:
            useful = thought[1].strip()
    encoded = lambda s: tokenizer.encode(s, add_special_tokens=False)
    for role, reasoning, call in [
        ("unchecked", unchecked, "submit_patch{}"),
        ("useful", useful, 'run_command{command:<|"|>' + command + '<|"|>}')]:
        header = encoded(start)
        body = encoded(reasoning)[:96]
        thought_end = header + body
        before_call = thought_end + encoded(end)
        tool_tokens = encoded(dispatch + call + "<tool_call|>")
        suffix = before_call + tool_tokens
        yield role, suffix, {"thought": (len(header),len(header)+min(8,len(body))),
                             "tool": (len(before_call),len(before_call)+min(10,len(tool_tokens)))}


def summarize(model, layers, prefix, suffix, windows):
    # Preserve exactly the same post-edit prefix on both branches by reserving suffix space
    # before calling this function. Positions below refer to this specific forward pass.
    tokens = prefix + suffix
    if len(tokens) > 4096:
        raise ValueError("Continuation exceeds the 4096-token harvest bound")
    bins = {name: {} for name in windows}
    hooks = []
    for i, layer in enumerate(layers):
        def record(module, inputs, out, index=i):
            for name,(a,b) in windows.items():
                span = slice(len(prefix)+a,len(prefix)+b)
                bins[name][index] = (inputs[0][0,span].float().mean(0).cpu(),
                                     out[0,span].float().mean(0).cpu())
        hooks.append(layer.mlp.down_proj.register_forward_hook(record))
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([tokens],device=model.device),use_cache=False,logits_to_keep=1)
    finally:
        for h in hooks:
            h.remove()
    return {name: (torch.stack([data[i][0] for i in range(len(layers))]),
                   torch.stack([data[i][1] for i in range(len(layers))])) for name,data in bins.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-replays", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = read_cases(args.run)
    counts = collections.Counter(c["role"] for c in cases)
    print("Observed decision states", dict(counts), flush=True)
    (args.output/"cases.json").write_text(json.dumps(cases))
    assets = json.loads((ROOT/"assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"],local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(assets["model"], dtype=torch.bfloat16,
                                                 device_map="cuda",local_files_only=True).eval()
    lm = getattr(model.model,"language_model",model.model)
    layers = lm.layers
    assert len(layers) == 42
    with zipfile.ZipFile(ROOT/"champion.zip") as z:
        blob = z.read("adapters/wsp_r1/adapter_model.safetensors")
        assert hashlib.sha256(blob).hexdigest() == "8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815"
        champ = load(blob)
    attach_adapter(layers,champ)
    with torch.inference_mode():
        model(input_ids=torch.tensor([[tokenizer.bos_token_id]],device=model.device),
              use_cache=False,logits_to_keep=1)
    rows, log = [], []
    native = {r["prompt_sha"]:r for r in json.loads(args.native_replays.read_text())} if args.native_replays else {}
    for issue in sorted({c["issue"] for c in cases}):
        # One successful contrast per snapshot keeps issues equally weighted.
        options = [c for c in cases if c["issue"] == issue and c["role"] in ("unchecked","check")]
        for case in sorted(options,key=lambda c:(c["role"]!="unchecked",len(c["tokens"]))):
            prefix = case["tokens"]
            command = proposal_test(case)
            if not command:
                continue
            print("Replaying",issue,len(prefix),case["action"],flush=True)
            if args.native_replays:
                actual = native.get(case["prompt_sha"])
                if not actual or actual["prompt_tokens"] != len(prefix):
                    continue
                text, action = actual["text"], actual["action"]
            else:
                text, action = replay(model,tokenizer,prefix,512)
            log.append(dict(issue=issue,observed=case["action"],full_replay=action,
                            full_tokens=len(prefix),backend="vllm" if args.native_replays else "transformers",
                            sample=text[-400:]))
            (args.output/"replays.json").write_text(json.dumps(log,indent=2))
            if action != case["action"]:
                continue
            if args.native_replays and action == "run_command":
                cmd = re.search(r'call:run_command\{command:<\|"\|>(.*?)<\|"\|>',text,re.S)
                if not cmd or cmd[1].strip() != case["arguments"].get("command","").strip():
                    continue
            # Measure only the specified left-truncated window after validating the full request.
            prefix = prefix[-3904:]
            branches = list(contrast_suffixes(tokenizer,text,command))
            if any(len(suffix)+len(prefix)>4096 for _,suffix,_ in branches):
                continue
            packed = {}
            for role,suffix,windows in branches:
                for window,(h,y) in summarize(model,layers,prefix,suffix,windows).items():
                    record = packed.setdefault(window, dict(issue=issue, snapshot=case["snapshot"],
                                                           split="discovery", window=window))
                    record[role+"_h"],record[role+"_y"] = h,y
            rows.extend(packed.values())
            log.append(dict(issue=issue,action=action,case_sha=case["prompt_sha"],command=command,
                            prefix_tokens=len(prefix),method="observed replay plus teacher-forced matched branches"))
            print("Harvested",issue,action,len(prefix),flush=True)
            torch.save(rows,args.output/"summaries.pt")
            break
        (args.output/"replays.json").write_text(json.dumps(log,indent=2))
    report=shortlist(rows,layers,champ,args.output/"candidates")
    print("Candidate count",len(report["candidates"]),flush=True)


if __name__=="__main__":
    main()
