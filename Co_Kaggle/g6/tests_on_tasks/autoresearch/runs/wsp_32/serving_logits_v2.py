"""Validate serving raw next-token logits & competition compliance on Colab L4 (Step 2 of HANDOFF.md)."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path("/content/wsp")


def run_compliance_check():
    """Verify submission_40.zip and compiled explicit r=1 / r=2 archives pass official competition validators."""
    import adk_submission
    from adk_submission import discover_adapters
    from adk_submission.discovery import discover_declared_models
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, build_submission_limits

    validate_fn = getattr(adk_submission, "validate_directory", None)
    if validate_fn is None:
        for submod in ("validation", "validator", "limits"):
            try:
                mod = __import__(f"adk_submission.{submod}", fromlist=["validate_directory"])
                validate_fn = getattr(mod, "validate_directory", None)
                if validate_fn:
                    break
            except ImportError:
                pass

    limits, gen_constraints = build_submission_limits()
    allowed_models = {"gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct",
                      "gemma-4-e4b-it-qat-w4a16-ct", "gemma-4-e2b-it-qat-w4a16-ct"}
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        from recipe import compile_explicit, D_IN, D_OUT
        fac = tmp_path / "f35.safetensors"
        row = torch.zeros(D_IN, dtype=torch.bfloat16); row[0] = 0.125
        col = torch.full((D_OUT,), 0.01, dtype=torch.bfloat16)
        save_file({"35.A": row, "35.B": col}, str(fac))

        for label, rank in (("champion_r1", 0), ("explicit_r1_disjoint", 1), ("explicit_r2_union", 2)):
            sub_dir = tmp_path / label
            with zipfile.ZipFile(ROOT / "champion.zip") as z:
                z.extractall(sub_dir)
            if rank in (1, 2):
                ad_out = tmp_path / f"ad_{label}"
                compile_explicit(ROOT / "champion.zip", fac, ad_out, "candidate", rank=rank)
                shutil.copy2(ad_out / "adapter_config.json", sub_dir / "adapters/wsp_r1/adapter_config.json")
                shutil.copy2(ad_out / "adapter_model.safetensors", sub_dir / "adapters/wsp_r1/adapter_model.safetensors")

            if validate_fn is not None:
                validate_fn(sub_dir, limits=limits)
            declared = discover_declared_models(sub_dir)
            assert len(declared) == 1 and next(iter(declared)) in allowed_models
            adapters = discover_adapters(str(sub_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
            total_bytes = sum(p.stat().st_size for p in sub_dir.rglob("*") if p.is_file())
            assert total_bytes < limits.max_total_size_bytes
            exts = {p.suffix.lower() for p in sub_dir.rglob("*") if p.is_file()}
            assert exts.issubset({e if e.startswith(".") else "." + e for e in limits.allowed_file_extensions})
            results[label] = {
                "declared_models": sorted(declared),
                "adapter_names": sorted(getattr(adapters, "adapters", {}).keys()),
                "rank": json.loads((sub_dir / "adapters/wsp_r1/adapter_config.json").read_text())["r"],
                "total_bytes": total_bytes,
                "validate_directory_fn": bool(validate_fn),
                "compliance": "PASS",
            }
    (ROOT / "compliance_preflight.json").write_text(json.dumps(results, indent=2))
    print("COMPLIANCE_OK", json.dumps(results), flush=True)


def run_logits(bucket: str, cases_path: Path):
    from adk_submission.server import _round_up_vllm_lora_rank
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from recipe import compile_explicit, D_IN, D_OUT

    out = ROOT / "logit_preflight"
    out.mkdir(exist_ok=True)
    champion_dir = out / "champion/adapters/wsp_r1"
    if not champion_dir.is_dir():
        with zipfile.ZipFile(ROOT / "champion.zip") as z:
            for n in ("adapter_config.json", "adapter_model.safetensors"):
                z.extract("adapters/wsp_r1/" + n, out / "champion")

    if bucket == "rank1":
        # Prepare rank-1 disjoint zero controls on L35 and L23
        for l_idx in (23, 35):
            zdir = out / f"L{l_idx}_r1_zero"
            if not zdir.is_dir():
                fac = out / f"_dummy_{l_idx}.safetensors"
                save_file({f"{l_idx}.A": torch.zeros(D_IN, dtype=torch.bfloat16),
                           f"{l_idx}.B": torch.zeros(D_OUT, dtype=torch.bfloat16)}, str(fac))
                compile_explicit(ROOT / "champion.zip", fac, zdir, "zero", rank=1)
                fac.unlink()
        adapters = {
            "champion": champion_dir,
            "L23_r1_zero": out / "L23_r1_zero",
            "L35_r1_zero": out / "L35_r1_zero",
        }
        max_rank = 1
    elif bucket == "rank2":
        zero_r2 = out / "zero_r2"
        if not zero_r2.is_dir():
            compile_explicit(ROOT / "champion.zip", None, zero_r2, "zero", rank=2)
        adapters = {"zero_r2": zero_r2}
        for p in sorted((ROOT / "harvest03/native04/screen/adapters").glob("*_zero")):
            adapters[p.name] = p
        adapters["base"] = None
        max_rank = _round_up_vllm_lora_rank(2)
    else:
        raise ValueError(bucket)

    assets = json.loads((ROOT / "assets.json").read_text())
    engine = LLM(
        model=assets["model"],
        enable_lora=True,
        max_loras=1,
        max_lora_rank=max_rank,
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.80,
        dtype="bfloat16",
        scheduler_reserve_full_isl=False,
        tensor_parallel_size=1,
        logprobs_mode="raw_logits",
        max_logprobs=-1,
    )
    tokenizer = engine.get_tokenizer()
    prefixes = [tokenizer.encode("After editing a Python package, check its public interface.")]
    kept = set()
    for c in json.loads(cases_path.read_text()):
        if c["role"] not in kept:
            prefixes.append(c["tokens"])
            kept.add(c["role"])

    for idx, (label, adapter) in enumerate(adapters.items(), 1):
        for i, ids in enumerate(prefixes):
            req = LoRARequest(label, idx, str(adapter)) if adapter is not None else None
            res = engine.generate(
                [{"prompt_token_ids": ids}],
                SamplingParams(temperature=0, max_tokens=1, logprobs=-1, detokenize=False),
                lora_request=req,
                use_tqdm=False,
            )[0]
            mapping = res.outputs[0].logprobs[0]
            keys = sorted(mapping)
            assert keys == list(range(len(keys))) and len(keys) >= len(tokenizer)
            logits = np.array([mapping[k].logprob for k in keys], dtype=np.float32)
            np.save(out / f"{label}-{i}.npy", logits)
        print("Saved raw logits:", label, flush=True)

    prefix_ids = [hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest() for ids in prefixes]
    if bucket == "rank1":
        (out / "prefixes.json").write_text(json.dumps(prefix_ids))
    else:
        assert prefix_ids == json.loads((out / "prefixes.json").read_text()), "Prefix mismatch!"

    # Compare everything saved so far against champion
    rows = []
    for npy in sorted(out.glob("*-0.npy")):
        label = npy.name[:-len("-0.npy")]
        if label == "champion":
            continue
        for i, ids in enumerate(prefixes):
            orig = np.load(out / f"champion-{i}.npy")
            other = np.load(out / f"{label}-{i}.npy")
            diff = np.abs(orig - other)
            top2_orig = np.partition(orig, -2)[-2:]
            margin = float(top2_orig[-1] - top2_orig[-2])
            rows.append({
                "adapter": label,
                "case_idx": i,
                "tokens": len(ids),
                "identical": bool(np.array_equal(orig, other)),
                "max_abs": float(np.max(diff)),
                "mean_abs": float(np.mean(diff)),
                "top1_margin": margin,
                "same_argmax": int(orig.argmax()) == int(other.argmax()),
                "orig_top1": int(orig.argmax()),
                "other_top1": int(other.argmax()),
            })
    (out / "comparison.json").write_text(json.dumps(rows, indent=2))
    print("COMPARISON_SUMMARY", json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("compliance", "rank1", "rank2"))
    p.add_argument("--cases", type=Path, default=ROOT / "harvest03/native04/columns/cases.json")
    args = p.parse_args()
    if args.action == "compliance":
        run_compliance_check()
    else:
        run_logits(args.action, args.cases)
