"""recipe.py — Weight-Space Programming (WSP) Thought-Start AlphaEdit + iter_04 Compiler.

THIS IS THE ONLY MUTABLE FILE IN PHASE B (per tests_on_tasks/autoresearch/program.md).
Directly programs PEFT LoRA weights (`lora_A`, `lora_B`) of
`base_submission/adapters/wsp_r1/{adapter_config.json,adapter_model.safetensors}`.

Iter 25 Breakthrough (Verified Live via Autoregressive Generation on GPU):
  - `mlp.down_proj` on [29, 31, 34, 37, 38]: exact iter_04 `edit_file_loop` (5/15, edit_err=68, leaks=0).
  - `self_attn.o_proj` on DISJOINT mid-network layers [16, 19, 21, 32]:
    Thought-Start (`<|thought>`) AlphaEdit Null-Space detector (leave-task-out CV gaps +0.310, +0.230,
    +0.226, +0.218 — 4.6x stronger on premature_submit than valid_submit!) steering along
    `u_thought = (pytest/test/verify/check/run) - (submit/done/complete/finished/ready)`.
    Verified live on GPU to flip requests_7205 from `submit_patch` -> `run_command(python3 test)`
    while preserving `submit_patch` on valid post-pytest `rich_3454`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

# ==============================================================================
# GEOMETRIC KNOBS (Iteration 25: Thought-Start AlphaEdit o_proj [16,19,21,32] + iter_04 down_proj)
# ==============================================================================
MODE = "thought_start_alphaedit_r1"
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

# Thought-Start AlphaEdit self_attn.o_proj on top-4 leave-task-out CV layers (100% disjoint from iter_04!)
# Live GPU generation verified: g=3.0 flipped requests_7205 submit_patch -> run_command while preserving rich_3454!
OPROJ_THOUGHT_GAINS = {
    16: 3.0,  # CV gap +0.310 (oof_prem=0.395 vs oof_val=0.085)
    19: 2.6,  # CV gap +0.230
    21: 2.6,  # CV gap +0.226
    32: 2.4,  # CV gap +0.218
}

ACTIVE_LAYERS = sorted(set(DOWN_PROJ_GAINS.keys()) | set(OPROJ_THOUGHT_GAINS.keys()))


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
    tb = torch.load(root / "thought_basis.pt", map_location="cpu", weights_only=True)
    assert basis["d_in"] == D_IN_DOWN and basis["d_out"] == D_OUT

    u_thought = tb["u_thought"].to(torch.float32)
    u_thought = u_thought / (u_thought.norm() + 1e-8)

    tensors: dict[str, torch.Tensor] = {}
    eff_norms: dict[str, float] = {}

    for layer in ACTIVE_LAYERS:
        # 1. mlp.down_proj: exact iter_04 on [29,31,34,37,38], zeros elsewhere
        if layer in DOWN_PROJ_GAINS:
            entry = basis["faults"]["edit_file_loop"]["layers"][layer]
            v_sparse = sparsify_topk(entry["v"].to(torch.float32), DOWN_PROJ_TOP_K)
            u_dir = entry["u"].to(torch.float32)
            u_dir = u_dir / (u_dir.norm() + 1e-8)
            gain = float(DOWN_PROJ_GAINS[layer])
            lora_A_d = v_sparse.unsqueeze(0).to(torch.bfloat16).contiguous()
            lora_B_d = (gain * u_dir).unsqueeze(1).to(torch.bfloat16).contiguous()
            eff_norms[f"L{layer}.down_proj"] = round(lora_B_d.float().norm().item(), 5)
        else:
            lora_A_d = torch.zeros((RANK, D_IN_DOWN), dtype=torch.bfloat16)
            lora_B_d = torch.zeros((D_OUT, RANK), dtype=torch.bfloat16)

        dp = DOWN_PREFIX_FMT.format(layer=layer)
        tensors[f"{dp}.lora_A.weight"] = lora_A_d
        tensors[f"{dp}.lora_B.weight"] = lora_B_d

        # 2. self_attn.o_proj: Thought-Start AlphaEdit on [16,19,21,32], zeros elsewhere
        oe = tb["o_proj"][layer]
        d_in_o = int(oe["d_in"])
        if layer in OPROJ_THOUGHT_GAINS:
            vp = oe["v"].to(torch.float32) * float(oe["scale"])
            gain_o = float(OPROJ_THOUGHT_GAINS[layer])
            lora_A_o = vp.unsqueeze(0).to(torch.bfloat16).contiguous()
            lora_B_o = (gain_o * u_thought).unsqueeze(1).to(torch.bfloat16).contiguous()
            eff_norms[f"L{layer}.o_proj"] = round((lora_B_o.float().norm() * lora_A_o.float().norm()).item(), 5)
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
        "effective_norms": eff_norms,
        "adapter_sha256": sha,
        "safetensors_bytes": safetensors_path.stat().st_size,
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    compile_adapter()
