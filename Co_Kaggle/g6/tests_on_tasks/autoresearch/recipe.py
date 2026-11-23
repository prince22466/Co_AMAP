"""recipe.py — Weight-Space Programming (WSP) BF16 Quantization-Residual Recovery + iter_04 Compiler.

THIS IS THE ONLY MUTABLE FILE IN PHASE B (per tests_on_tasks/autoresearch/program.md).
Directly programs PEFT LoRA weights (`lora_A`, `lora_B`) of
`base_submission/adapters/wsp_r1/{adapter_config.json,adapter_model.safetensors}`.

Iter 24 Paradigm: Exact BF16 Quantization-Residual Recovery (LoftQ/SVD Weight Programming)
  - Base model on Kaggle is 4-bit INT4 QAT (`gemma-4-e4b-it-qat-w4a16-ct`), which has
    ~10.8% directional error vs full BF16 `google/gemma-4-E4B-it`.
  - Compute scale-aligned residual E = c* * W_bf16 - dequant(W_int4) (preserving QAT
    layernorm calibration) and factorize E via Rank-64 SVD into BF16 LoRA (B A = U_r S_r V_r^T)
    across BOTH `self_attn.o_proj` and `mlp.down_proj` on [23,29,31,34,35,36,37,38,40,41].
  - Preserve iter_04's exact Rank-1 `edit_file_loop` circuit in Slot 0 of down_proj [29,31,34,37,38].
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

# ==============================================================================
# GEOMETRIC KNOBS (Iteration 24: Rank-64 BF16 Residual Recovery + iter_04)
# ==============================================================================
MODE = "bf16_residual_r64_plus_iter04"
BASE_MODEL_NAME = "google/gemma-4-e4b-it-qat-w4a16-ct"
DOWN_PREFIX_FMT = "base_model.model.model.language_model.layers.{layer}.mlp.down_proj"
OPROJ_PREFIX_FMT = "base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
D_IN_DOWN = 10240
D_OUT = 2560

RANK = 64
RESIDUAL_SCALE = 1.0  # 1.0 = exact scale-aligned BF16 residual recovery

ACTIVE_LAYERS = [23, 29, 31, 34, 35, 36, 37, 38, 40, 41]

ITER04_GAINS = {
    29: 0.14,
    31: 0.16,
    34: 0.12,
    37: 0.16,
    38: 0.14,
}
ITER04_TOP_K = 64


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
    res_cache = torch.load(root / "bf16_residual_cache.pt", map_location="cpu", weights_only=True)
    assert basis["d_in"] == D_IN_DOWN and basis["d_out"] == D_OUT

    tensors: dict[str, torch.Tensor] = {}
    layer_summary: dict[str, str] = {}

    for layer in ACTIVE_LAYERS:
        # 1. self_attn.o_proj: full Rank-64 BF16 quantization-residual recovery
        oe = res_cache["o_proj"][layer]
        d_in_o = int(oe["d_in"])
        Uo = oe["U"][:, :RANK].to(torch.float32)
        So = oe["S"][:RANK].to(torch.float32)
        Vo = oe["V"][:, :RANK].to(torch.float32)
        lora_A_o = Vo.T.contiguous()                                # [64, d_in_o]
        lora_B_o = (RESIDUAL_SCALE * Uo * So.unsqueeze(0)).contiguous()  # [2560, 64]

        op = OPROJ_PREFIX_FMT.format(layer=layer)
        tensors[f"{op}.lora_A.weight"] = lora_A_o.to(torch.bfloat16)
        tensors[f"{op}.lora_B.weight"] = lora_B_o.to(torch.bfloat16)

        # 2. mlp.down_proj: BF16 residual recovery + exact iter_04 in Slot 0 on [29,31,34,37,38]
        de = res_cache["down_proj"][layer]
        lora_A_d = torch.zeros((RANK, D_IN_DOWN), dtype=torch.float32)
        lora_B_d = torch.zeros((D_OUT, RANK), dtype=torch.float32)

        if layer in ITER04_GAINS:
            entry = basis["faults"]["edit_file_loop"]["layers"][layer]
            v_sparse = sparsify_topk(entry["v"].to(torch.float32), ITER04_TOP_K)
            u_dir = entry["u"].to(torch.float32)
            u_dir = u_dir / (u_dir.norm() + 1e-8)
            lora_A_d[0, :] = v_sparse
            lora_B_d[:, 0] = float(ITER04_GAINS[layer]) * u_dir

            rk = RANK - 1
            Ud = de["U"][:, :rk].to(torch.float32)
            Sd = de["S"][:rk].to(torch.float32)
            Vd = de["V"][:, :rk].to(torch.float32)
            lora_A_d[1:, :] = Vd.T
            lora_B_d[:, 1:] = RESIDUAL_SCALE * Ud * Sd.unsqueeze(0)
            layer_summary[f"L{layer}"] = f"o_r64+d_iter04+r63"
        else:
            Ud = de["U"][:, :RANK].to(torch.float32)
            Sd = de["S"][:RANK].to(torch.float32)
            Vd = de["V"][:, :RANK].to(torch.float32)
            lora_A_d[:, :] = Vd.T
            lora_B_d[:, :] = RESIDUAL_SCALE * Ud * Sd.unsqueeze(0)
            layer_summary[f"L{layer}"] = f"o_r64+d_r64"

        dp = DOWN_PREFIX_FMT.format(layer=layer)
        tensors[f"{dp}.lora_A.weight"] = lora_A_d.to(torch.bfloat16).contiguous()
        tensors[f"{dp}.lora_B.weight"] = lora_B_d.to(torch.bfloat16).contiguous()

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
        "target_modules": ["down_proj", "o_proj"],
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
        "target_modules": ["down_proj", "o_proj"],
        "layers": layer_summary,
        "adapter_sha256": sha,
        "safetensors_bytes": safetensors_path.stat().st_size,
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    compile_adapter()
