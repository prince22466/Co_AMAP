"""recipe.py — Weight-Space Programming (WSP) LASER Spectral Denoising + iter_04 Compiler.

THIS IS THE ONLY MUTABLE FILE IN PHASE B (per tests_on_tasks/autoresearch/program.md).
Directly programs PEFT LoRA weights (`lora_A`, `lora_B`) of
`base_submission/adapters/wsp_r1/{adapter_config.json,adapter_model.safetensors}`
combining two disjoint-layer pure-WSP mechanisms on `mlp.down_proj`:
  1. Proven iter_04 Rank-1 `edit_file_loop` circuit on [29, 31, 34, 37, 38] (untouched).
  2. LASER (Layer-Selective Rank Reduction, ICLR 2024) Rank-32 spectral amplification
     (\Delta W = \alpha * U_{:32} \Sigma_{:32} V_{:32}^T) on disjoint late MLP layers [33, 35, 36, 40].
     Because Gemma 4 applies post_feedforward_layernorm (RMSNorm) immediately after down_proj,
     boosting W_{top-32} by (1 + \alpha) suppresses the 4-bit QAT noise tail by 1/(1 + \alpha).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

# ==============================================================================
# GEOMETRIC KNOBS (Iteration 22: Disjoint-Layer LASER Rank-32 + exact iter_04)
# ==============================================================================
MODE = "laser_r32_plus_iter04"
BASE_MODEL_NAME = "google/gemma-4-e4b-it-qat-w4a16-ct"
KEY_PREFIX_FMT = "base_model.model.model.language_model.layers.{layer}.mlp.down_proj"
D_IN = 10240
D_OUT = 2560

RANK = 32

# Exact iter_04 mlp.down_proj circuit (5/15, edit_err=68, leaks=0 — UNTOUCHED)
ITER04_GAINS = {
    29: 0.14,
    31: 0.16,
    34: 0.12,
    37: 0.16,
    38: 0.14,
}
ITER04_TOP_K = 64

# Disjoint-layer LASER Spectral Denoising (top-32 singular values boosted by alpha)
# L35 has highest spectral concentration in the model (top64_energy=0.081, s1=3.62)
LASER_ALPHAS = {
    33: 0.30,
    35: 0.35,
    36: 0.35,
    40: 0.25,
}

ACTIVE_LAYERS = sorted(set(ITER04_GAINS.keys()) | set(LASER_ALPHAS.keys()))


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
        elif layer in LASER_ALPHAS:
            sv = svd_cache["layers"][layer]
            U = sv["U"][:, :RANK].to(torch.float32)   # [2560, 32]
            S = sv["S"][:RANK].to(torch.float32)      # [32]
            V = sv["V"][:, :RANK].to(torch.float32)   # [10240, 32]
            alpha = float(LASER_ALPHAS[layer])
            # B @ A = alpha * U @ diag(S) @ V^T
            lora_A[:, :] = V.T                        # [32, 10240]
            lora_B[:, :] = alpha * (U * S.unsqueeze(0))  # [2560, 32]
            layer_info[f"L{layer}"] = f"laser_r{RANK}(a={alpha}, ||B||={lora_B.norm().item():.2f})"

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
