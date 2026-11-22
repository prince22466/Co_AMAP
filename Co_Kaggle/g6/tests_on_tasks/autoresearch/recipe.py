"""recipe.py — Weight-Space Programming (WSP) Multi-Module Low-Rank Adapter Compiler.

THIS IS THE ONLY MUTABLE FILE IN PHASE B (per tests_on_tasks/autoresearch/program.md).
Directly programs PEFT LoRA weights (`lora_A`, `lora_B`) of
`base_submission/adapters/wsp_r1/{adapter_config.json,adapter_model.safetensors}`
across BOTH:
  - `mlp.down_proj` (proven iter_04 edit_file_loop circuit on [29, 31, 34, 37, 38])
  - `self_attn.o_proj` (DLA + AlphaEdit null-space premature_submit interceptor on [28, 31, 36])
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

# ==============================================================================
# GEOMETRIC KNOBS (Iteration 21: Gentle 0.25x AlphaEdit o_proj + exact iter_04 down_proj)
# ==============================================================================
MODE = "alphaedit_multimodule_gentle_r1"
BASE_MODEL_NAME = "google/gemma-4-e4b-it-qat-w4a16-ct"
DOWN_PREFIX_FMT = "base_model.model.model.language_model.layers.{layer}.mlp.down_proj"
OPROJ_PREFIX_FMT = "base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
D_IN_DOWN = 10240
D_OUT = 2560

RANK = 1

# Exact iter_04 mlp.down_proj circuit (5/15, edit_err=68, leaks=0 — UNTOUCHED)
DOWN_PROJ_GAINS = {
    29: 0.14,
    31: 0.16,
    34: 0.12,
    37: 0.16,
    38: 0.14,
}
DOWN_PROJ_TOP_K = 64

# DLA + AlphaEdit Null-Space self_attn.o_proj circuit scaled to iter_04's safe norm regime (eff_norm 0.05-0.14)
# Top 3 positive Leave-Task-Out CV layers: L36 (gap=+0.146), L31 (gap=+0.063), L28 (gap=+0.027)
OPROJ_PREM_GAINS = {
    28: 0.30,
    31: 0.40,
    36: 0.55,
}

ACTIVE_LAYERS = sorted(set(DOWN_PROJ_GAINS.keys()) | set(OPROJ_PREM_GAINS.keys()))


def sparsify_topk(v: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the top_k largest-magnitude entries of v and unit-normalize."""
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
    ae = torch.load(root / "alphaedit_basis.pt", map_location="cpu", weights_only=True)
    assert basis["d_in"] == D_IN_DOWN and basis["d_out"] == D_OUT

    u_run_minus_sub = ae["u_run_minus_sub"].to(torch.float32)
    u_run_minus_sub = u_run_minus_sub / (u_run_minus_sub.norm() + 1e-8)

    tensors: dict[str, torch.Tensor] = {}
    layer_norms: dict[str, float] = {}

    for layer in ACTIVE_LAYERS:
        # 1. mlp.down_proj (exact iter_04 on [29,31,34,37,38], exact zeros on other layers)
        if layer in DOWN_PROJ_GAINS:
            entry = basis["faults"]["edit_file_loop"]["layers"][layer]
            v_sparse = sparsify_topk(entry["v"].to(torch.float32), DOWN_PROJ_TOP_K)
            u_dir = entry["u"].to(torch.float32)
            u_dir = u_dir / (u_dir.norm() + 1e-8)
            gain = float(DOWN_PROJ_GAINS[layer])
            lora_A_d = v_sparse.unsqueeze(0).to(torch.bfloat16).contiguous()
            lora_B_d = (gain * u_dir).unsqueeze(1).to(torch.bfloat16).contiguous()
            layer_norms[f"L{layer}.down_proj"] = round(lora_B_d.float().norm().item(), 5)
        else:
            lora_A_d = torch.zeros((RANK, D_IN_DOWN), dtype=torch.bfloat16)
            lora_B_d = torch.zeros((D_OUT, RANK), dtype=torch.bfloat16)

        dp = DOWN_PREFIX_FMT.format(layer=layer)
        tensors[f"{dp}.lora_A.weight"] = lora_A_d
        tensors[f"{dp}.lora_B.weight"] = lora_B_d

        # 2. self_attn.o_proj (AlphaEdit null-space prem_submit interceptor on [28,31,36], zeros elsewhere)
        o_entry = ae["o_proj"][layer]
        d_in_o = int(o_entry["d_in"])
        if layer in OPROJ_PREM_GAINS:
            vp = o_entry["v_prem"].to(torch.float32) * float(o_entry["scale_prem"])
            gain_o = float(OPROJ_PREM_GAINS[layer])
            lora_A_o = vp.unsqueeze(0).to(torch.bfloat16).contiguous()
            lora_B_o = (gain_o * u_run_minus_sub).unsqueeze(1).to(torch.bfloat16).contiguous()
            layer_norms[f"L{layer}.o_proj"] = round((lora_B_o.float().norm() * lora_A_o.float().norm()).item(), 5)
        else:
            lora_A_o = torch.zeros((RANK, d_in_o), dtype=torch.bfloat16)
            lora_B_o = torch.zeros((D_OUT, RANK), dtype=torch.bfloat16)

        op = OPROJ_PREFIX_FMT.format(layer=layer)
        tensors[f"{op}.lora_A.weight"] = lora_A_o
        tensors[f"{op}.lora_B.weight"] = lora_B_o

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
        "effective_norms": layer_norms,
        "adapter_sha256": sha,
        "safetensors_bytes": safetensors_path.stat().st_size,
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    compile_adapter()
