r"""recipe.py — Weight-Space Programming (WSP) Low-Rank Adapter Compiler.

THIS IS THE ONLY MUTABLE FILE IN PHASE B (per tests_on_tasks/autoresearch/program.md).
Directly programs the PEFT LoRA weights (`lora_A`, `lora_B`) of
`base_submission/adapters/wsp_r1/{adapter_config.json,adapter_model.safetensors}`
targeting selected `mlp.down_proj` residual-stream write heads:

    \Delta W_l = B_l A_l = \sum_{i=1}^r \lambda_{l,i} u_{l,i} v_{l,i}^T  (with \alpha = r)

- Rows of lora_A (v_{l,i} in R^{10240}): Top-k sparsified MLP neuron detectors from
  contrastive failure-vs-healthy basis_cache.pt (conditional trigger gate).
- Columns of lora_B (\lambda_{l,i} u_{l,i} in R^{2560}): Orthogonalized residual-stream
  steering directions pushing out of localized failure modes (`edit_file_loop`, `premature_submit`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
import torch
from safetensors.torch import load, load_file, save_file

# ==============================================================================
# GEOMETRIC KNOBS (Iteration 1: MODE="basis_r1", RANK=1, primary="edit_file_loop")
# ==============================================================================
MODE = "basis_r1"
CHAMPION_SHA = "8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815"
BASE_MODEL_NAME = "google/gemma-4-e4b-it-qat-w4a16-ct"
KEY_PREFIX_FMT = "base_model.model.model.language_model.layers.{layer}.mlp.down_proj"
D_IN = 10240
D_OUT = 2560

RANK = 1
ORTHOGONALIZE_B = True

# Iter 4: shift window slightly earlier into middle-late band [29, 31, 34, 37, 38] and scale gains 1.35x
ACTIVE_LAYERS = [29, 31, 34, 37, 38]

# Slot configurations (Slot 0 active when RANK >= 1; Slot 1 active when RANK >= 2)
SLOTS = [
    {
        "fault": "edit_file_loop",
        "top_k": 64,
        "sign": 1.0,
        "gains": {
            29: 0.14,
            31: 0.16,
            34: 0.12,
            37: 0.16,
            38: 0.14,
            39: 0.09,
        },
    },
    {
        "fault": "premature_submit",
        "top_k": 128,
        "sign": 1.0,
        "gains": {
            31: 0.06,
            34: 0.06,
            37: 0.08,
            38: 0.07,
            39: 0.09,
        },
    },
]


def sparsify_topk(v: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the top_k largest-magnitude entries of v in R^{D_IN} and unit-normalize."""
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


def compile_explicit(champion_zip: Path, factors_path: Path | None, output: Path,
                     control: str = "candidate", seed: int = 31, rank: int = 2) -> dict:
    """Copy iter_04 verbatim into slot zero and write the supplied BF16 edit (rank 2, or rank 1 on disjoint layers)."""
    if control not in {"candidate", "zero", "random"}:
        raise ValueError(f"Unknown control {control!r}")
    if rank not in (1, 2):
        raise ValueError(f"Unsupported explicit rank {rank!r}; expected 1 or 2")
    with zipfile.ZipFile(champion_zip) as archive:
        prefix = "adapters/wsp_r1/"
        blob = archive.read(prefix + "adapter_model.safetensors")
        config = json.loads(archive.read(prefix + "adapter_config.json"))
    if hashlib.sha256(blob).hexdigest() != CHAMPION_SHA or (config["r"], config["lora_alpha"]) != (1, 1):
        raise ValueError("Expected the archived iter_04 rank-one champion")
    champion = load(blob)
    factors = load_file(str(factors_path)) if factors_path is not None else {}
    if not factors and control != "zero":
        raise ValueError("An edit requires at least one layer with explicit A and B vectors")
    edit_layers = {int(key.split(".")[0]) for key in factors}
    if any(layer < 0 or layer >= 42 for layer in edit_layers):
        raise ValueError("Layer outside Gemma E4B")
    if set(factors) != {f"{layer}.{side}" for layer in edit_layers for side in ("A", "B")}:
        raise ValueError("Expected one <layer>.A and one <layer>.B vector per layer")
    champ_layers = set(config["layers_to_transform"])
    if rank == 1 and (edit_layers & champ_layers):
        raise ValueError(f"rank=1 explicit export requires disjoint edit layers; overlap={sorted(edit_layers & champ_layers)}")
    layers = sorted(champ_layers | edit_layers)
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in layers:
        prefix = KEY_PREFIX_FMT.format(layer=layer)
        a_key, b_key = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
        a = torch.zeros(rank, D_IN, dtype=torch.bfloat16)
        b = torch.zeros(D_OUT, rank, dtype=torch.bfloat16)
        if layer in champ_layers:
            a[:1], b[:, :1] = champion[a_key], champion[b_key]
        if layer in edit_layers:
            row = factors[f"{layer}.A"].to(torch.bfloat16)
            col = factors[f"{layer}.B"].to(torch.bfloat16)
            if row.shape != (D_IN,) or col.shape != (D_OUT,):
                raise ValueError(f"Wrong explicit factor dimensions on layer {layer}")
            if not torch.isfinite(row).all() or not torch.isfinite(col).all():
                raise ValueError(f"Nonfinite factors on layer {layer}")
            if control == "random":
                # Permuting the BF16 values preserves sparsity and both factor norms exactly.
                row = row[torch.randperm(D_IN, generator=generator)]
                col = col[torch.randperm(D_OUT, generator=generator)]
                col = col * (2 * torch.randint(2, (D_OUT,), generator=generator) - 1)
            if control != "zero":
                slot = 0 if rank == 1 else 1
                a[slot], b[:, slot] = row, col
        tensors[a_key], tensors[b_key] = a, b
    config.update(r=rank, lora_alpha=rank, layers_to_transform=layers)
    # Publish both files together. Never overwrite an archived or previously compiled result.
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "adapter"
        stage.mkdir()
        weights = stage / "adapter_model.safetensors"
        save_file(tensors, str(weights))
        restored = load_file(str(weights))
        if restored.keys() != tensors.keys() or any(not torch.equal(t, restored[k]) for k, t in tensors.items()):
            raise RuntimeError("Saved explicit factors did not round-trip")
        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        (stage / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf8")
        stage.rename(output)
    result = dict(mode="explicit_factors", rank=rank, lora_alpha=rank, control=control,
                  active_layers=layers, adapter_sha256=sha)
    print(json.dumps(result))
    return result


def compile_adapter() -> dict:
    root = Path(__file__).resolve().parent
    basis = torch.load(root / "basis_cache.pt", map_location="cpu", weights_only=True)
    assert basis["d_in"] == D_IN and basis["d_out"] == D_OUT, "Dimension mismatch against basis_cache.pt"

    active_slots = SLOTS[:RANK]
    assert len(active_slots) == RANK, f"Expected {RANK} slot configs, got {len(active_slots)}"

    tensors: dict[str, torch.Tensor] = {}
    layer_norms: dict[int, list[float]] = {}

    for layer in sorted(ACTIVE_LAYERS):
        A_rows = []
        B_cols = []
        u_prev_unit = []

        for slot_idx, slot in enumerate(active_slots):
            fault = slot["fault"]
            layer_entry = basis["faults"][fault]["layers"][layer]
            v_raw = layer_entry["v"].to(torch.float32)
            u_raw = layer_entry["u"].to(torch.float32)

            # Row of lora_A: top-k sparsified MLP neuron detector gate
            v_sparse = sparsify_topk(v_raw, int(slot["top_k"]))

            # Column of lora_B: unit steering direction, optionally Gram-Schmidt orthogonalized
            u_dir = u_raw / (u_raw.norm() + 1e-8)
            if ORTHOGONALIZE_B and u_prev_unit:
                for u_p in u_prev_unit:
                    u_dir = u_dir - torch.dot(u_dir, u_p) * u_p
                u_dir = u_dir / (u_dir.norm() + 1e-8)
            u_prev_unit.append(u_dir)

            gain = float(slot["gains"].get(layer, 0.0)) * float(slot.get("sign", 1.0))
            b_col = gain * u_dir

            A_rows.append(v_sparse)
            B_cols.append(b_col)
            layer_norms.setdefault(layer, []).append(round(b_col.norm().item(), 5))

        # lora_A.weight: shape [r, D_IN], BF16
        lora_A = torch.stack(A_rows, dim=0).to(torch.bfloat16).contiguous()
        # lora_B.weight: shape [D_OUT, r], BF16
        lora_B = torch.stack(B_cols, dim=1).to(torch.bfloat16).contiguous()

        assert lora_A.shape == (RANK, D_IN), f"Unexpected lora_A shape: {lora_A.shape}"
        assert lora_B.shape == (D_OUT, RANK), f"Unexpected lora_B shape: {lora_B.shape}"

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
        "layers_to_transform": sorted(ACTIVE_LAYERS),
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
        "active_layers": sorted(ACTIVE_LAYERS),
        "layer_b_norms": layer_norms,
        "adapter_sha256": sha,
        "safetensors_bytes": safetensors_path.stat().st_size,
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("basis_r1", "explicit_factors"), default=MODE)
    parser.add_argument("--champion-zip", type=Path, default=root.parents[1] / "submission_nb/artifacts/submission_40.zip")
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control", choices=("candidate", "zero", "random"), default="candidate")
    parser.add_argument("--rank", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.mode == "explicit_factors":
        if args.output is None:
            parser.error("--output is required for explicit factors")
        compile_explicit(args.champion_zip, args.factors, args.output, args.control, rank=args.rank)
    else:
        if args.output or args.factors or args.control != "candidate" or args.rank != 2:
            parser.error("--output, --factors, --control and --rank require --mode explicit_factors")
        compile_adapter()
