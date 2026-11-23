"""recipe.py — Weight-Space Programming (WSP) Global-Attention-Anchored Rank-1 Compiler.

THIS IS THE ONLY MUTABLE FILE IN PHASE B (per tests_on_tasks/autoresearch/program.md).
Directly programs PEFT LoRA weights (`lora_A`, `lora_B`) of
`base_submission/adapters/wsp_r1/{adapter_config.json,adapter_model.safetensors}`.

Iter 23 Hypothesis:
Gemma 4 E4B has sliding_window=512 on 35/42 layers; ONLY global full-attention layers
[5, 11, 17, 23, 29, 35, 41] can attend across a 5k-12k token coding trajectory.
Adding L29 (global full-attention) jumped iter_03 (4/15) -> iter_04 (5/15, leaks=0).
Now extend iter_04's exact [29,31,34,37,38] with adjacent global full-attention layers
L23 (gain=0.10) and L35 (gain=0.12) in the proven Rank-1 edit_file_loop circuit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

# ==============================================================================
# GEOMETRIC KNOBS (Iteration 23: iter_04 + Global Full-Attention Layers L23, L35)
# ==============================================================================
MODE = "global_attn_anchored_r1"
BASE_MODEL_NAME = "google/gemma-4-e4b-it-qat-w4a16-ct"
KEY_PREFIX_FMT = "base_model.model.model.language_model.layers.{layer}.mlp.down_proj"
D_IN = 10240
D_OUT = 2560

RANK = 1
TOP_K = 64

GAINS = {
    23: 0.10,  # Global full-attention layer
    29: 0.14,  # Global full-attention layer (iter_04 exact)
    31: 0.16,  # iter_04 exact
    34: 0.12,  # iter_04 exact
    35: 0.12,  # Global full-attention layer
    37: 0.16,  # iter_04 exact
    38: 0.14,  # iter_04 exact
}

ACTIVE_LAYERS = sorted(GAINS.keys())


def sparsify_topk(v: torch.Tensor, top_k: int) -> torch.Tensor:
    v_f32 = v.detach().to(torch.float32).clone()
    if 0 < top_k < v_f32.numel():
        _, idx = torch.topk(v_f32.abs(), k=top_k, largest=True, sorted=False)
        mask = torch.zeros_like(v_f32)
        mask[idx] = 1.0
        v_f32 = v_f32 * mask
    norm = v_f32.norm()
    if norm > 1e-8:
        v_f32 = v_f32 / norm
    return v_f32


def compile_adapter() -> dict:
    root = Path(__file__).resolve().parent
    basis = torch.load(root / "basis_cache.pt", map_location="cpu", weights_only=True)
    assert basis["d_in"] == D_IN and basis["d_out"] == D_OUT

    tensors: dict[str, torch.Tensor] = {}
    layer_norms: dict[int, float] = {}

    for layer in ACTIVE_LAYERS:
        entry = basis["faults"]["edit_file_loop"]["layers"][layer]
        v_sparse = sparsify_topk(entry["v"].to(torch.float32), TOP_K)
        u_dir = entry["u"].to(torch.float32)
        u_dir = u_dir / (u_dir.norm() + 1e-8)
        gain = float(GAINS[layer])

        lora_A = v_sparse.unsqueeze(0).to(torch.bfloat16).contiguous()
        lora_B = (gain * u_dir).unsqueeze(1).to(torch.bfloat16).contiguous()
        layer_norms[layer] = round(lora_B.float().norm().item(), 5)

        prefix = KEY_PREFIX_FMT.format(layer=layer)
        tensors[f"{prefix}.lora_A.weight"] = lora_A
        tensors[f"{prefix}.lora_B.weight"] = lora_B

    out_dir = root / "base_submission" / "adapters" / "wsp_r1"
    out_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = out_dir / "adapter_model.safetensors"
    save_file(tensors, str(safetensors_path))

    adapter_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": BASE_MODEL_NAME,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": "layers",
        "layers_to_transform": ACTIVE_LAYERS,
        "loftq_config": {},
        "lora_alpha": RANK,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": RANK,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ["down_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    config_path = out_dir / "adapter_config.json"
    config_path.write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")

    sha = hashlib.sha256(safetensors_path.read_bytes()).hexdigest()
    summary = {
        "mode": MODE,
        "rank": RANK,
        "lora_alpha": RANK,
        "active_layers": ACTIVE_LAYERS,
        "layer_b_norms": layer_norms,
        "adapter_sha256": sha,
        "safetensors_bytes": safetensors_path.stat().st_size,
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    compile_adapter()
