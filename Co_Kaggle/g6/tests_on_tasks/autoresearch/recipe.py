"""recipe.py — Weight-Space Programming (WSP) Negative LASER Dominant-Mode Dampening + iter_04 Compiler.

THIS IS THE ONLY MUTABLE FILE IN PHASE B (per tests_on_tasks/autoresearch/program.md).
Directly programs PEFT LoRA weights (`lora_A`, `lora_B`) of
`base_submission/adapters/wsp_r1/{adapter_config.json,adapter_model.safetensors}`.

Iter 30 Hypothesis (Opposite Sign of iter_22):
  - Positive spectral boost (+0.35 * W_top32 in iter_22) amplified greedy shortcut modes (edit_err=139).
  - iter_04 succeeded (5/15, edit_err=68, leaks=0) via NEGATIVE rank-1 dampening (-g (u u^T) W_down).
  - Test NEGATIVE spectral dampening (-0.15 * U_{:8} S_{:8} V_{:8}^T) on disjoint late MLP layers
    [33, 35, 36, 40] to dampen overconfident greedy-collapse modes under temperature=0.0, while
    preserving iter_04's exact Rank-1 edit_file_loop on [29, 31, 34, 37, 38].
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

# ==============================================================================
# GEOMETRIC KNOBS (Iteration 30: Negative LASER Rank-8 Dampening + exact iter_04)
# ==============================================================================
MODE = "neg_laser_r8_plus_iter04"
BASE_MODEL_NAME = "google/gemma-4-e4b-it-qat-w4a16-ct"
KEY_PREFIX_FMT = "base_model.model.model.language_model.layers.{layer}.mlp.down_proj"
D_IN = 10240
D_OUT = 2560

RANK = 8

# Exact iter_04 mlp.down_proj circuit (5/15, edit_err=68, leaks=0 — UNTOUCHED)
ITER04_GAINS = {
    29: 0.14,
    31: 0.16,
    34: 0.12,
    37: 0.16,
    38: 0.14,
}
ITER04_TOP_K = 64

# Disjoint-layer NEGATIVE spectral dampening of top-8 dominant shortcut modes
NEG_LASER_ALPHAS = {
    33: -0.15,
    35: -0.18,
    36: -0.18,
    40: -0.12,
}

ACTIVE_LAYERS = sorted(set(ITER04_GAINS.keys()) | set(NEG_LASER_ALPHAS.keys()))


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
    svd_cache = torch.load(root / "laser_svd_cache.pt", map_location="cpu", weights_only=True)
    assert basis["d_in"] == D_IN and basis["d_out"] == D_OUT

    tensors: dict[str, torch.Tensor] = {}
    layer_info: dict[str, str] = {}

    for layer in ACTIVE_LAYERS:
        lora_A = torch.zeros((RANK, D_IN), dtype=torch.float32)
        lora_B = torch.zeros((D_OUT, RANK), dtype=torch.float32)

        if layer in ITER04_GAINS:
            entry = basis["faults"]["edit_file_loop"]["layers"][layer]
            v_sparse = sparsify_topk(entry["v"].to(torch.float32), ITER04_TOP_K)
            u_dir = entry["u"].to(torch.float32)
            u_dir = u_dir / (u_dir.norm() + 1e-8)
            gain = float(ITER04_GAINS[layer])
            lora_A[0, :] = v_sparse
            lora_B[:, 0] = gain * u_dir
            layer_info[f"L{layer}"] = f"iter04(g={gain})"
        elif layer in NEG_LASER_ALPHAS:
            sv = svd_cache["layers"][layer]
            U = sv["U"][:, :RANK].to(torch.float32)
            S = sv["S"][:RANK].to(torch.float32)
            V = sv["V"][:, :RANK].to(torch.float32)
            alpha = float(NEG_LASER_ALPHAS[layer])
            lora_A[:, :] = V.T
            lora_B[:, :] = alpha * (U * S.unsqueeze(0))
            layer_info[f"L{layer}"] = f"neg_laser_r{RANK}(a={alpha}, ||B||={lora_B.norm().item():.2f})"

        prefix = KEY_PREFIX_FMT.format(layer=layer)
        tensors[f"{prefix}.lora_A.weight"] = lora_A.to(torch.bfloat16).contiguous()
        tensors[f"{prefix}.lora_B.weight"] = lora_B.to(torch.bfloat16).contiguous()

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
        "layer_info": layer_info,
        "adapter_sha256": sha,
        "safetensors_bytes": safetensors_path.stat().st_size,
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    compile_adapter()
