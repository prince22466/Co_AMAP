"""Produce sparse signed column edits from paired, issue-weighted FFN summaries."""
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


def fit(dh, dy, weight, a, b):
    direction = dy.mean(0)
    direction = direction / direction.norm().clamp_min(1e-10)
    projected = weight.T @ direction
    if a is not None:
        projected += a.T @ (b.T @ direction)
    scores = dh.mean(0) * projected
    return direction, projected, scores


def strongest(scores, k):
    indices = scores.topk(min(k, scores.numel())).indices
    return indices[scores[indices] > 0]


def shortlist(rows, layers, champion, out_dir):
    """rows contain one mean h/y tensor per snapshot, state and matched window."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings, edits = [], {}
    for layer_index, layer in enumerate(layers):
        weight = layer.mlp.down_proj.weight.detach().float().cpu()
        key = f"base_model.model.model.language_model.layers.{layer_index}.mlp.down_proj.lora_A.weight"
        a = champion[key].float() if key in champion else None
        b = champion[key.replace("lora_A", "lora_B")].float() if key in champion else None
        for window in ("thought", "tool"):
            paired = [r for r in rows if r["window"] == window and r["split"] == "discovery"]
            if len({r["snapshot"] for r in paired}) != len(paired):
                raise ValueError("Aggregate repeated prefixes within each training snapshot first")
            if len(paired) < 3:
                continue
            hu = torch.stack([r["unchecked_h"][layer_index].float() for r in paired])
            yu = torch.stack([r["unchecked_y"][layer_index].float() for r in paired])
            dh = hu - torch.stack([r["useful_h"][layer_index].float() for r in paired])
            dy = yu - torch.stack([r["useful_y"][layer_index].float() for r in paired])
            fold_scores = {1: [], 16: []}
            for i in range(len(paired)):
                train = torch.arange(len(paired)) != i
                u, proj, scores = fit(dh[train], dy[train], weight, a, b)
                for k in fold_scores:
                    indices = strongest(scores, k)
                    # Both projection and mask come exclusively from the other snapshots.
                    held = (dh[i, indices] * proj[indices]).sum()
                    alignment = float(dy[i] @ u)
                    fold_scores[k].append(float(held) if alignment > 0 else 0.)
            u, proj, scores = fit(dh, dy, weight, a, b)
            rec = dict(layer=layer_index, window=window, n_snapshots=len(paired), folds=fold_scores)
            rankings.append(rec)
            if min(fold_scores[16]) <= 0 or min(fold_scores[1]) <= 0:
                continue
            # Compare relative effects across layers with different FFN-output scales.
            rec["relative_support"] = sum(fold_scores[16]) / (len(paired) * max(float(yu.norm(dim=1).mean()), 1e-8))
            candidates = []
            for k in (1, 16):
                indices = strongest(scores, k)
                row = torch.zeros_like(proj)
                row[indices] = proj[indices]
                signal = hu @ row
                rho = (0.25 * signal.abs() / yu.norm(dim=1).clamp_min(1e-8)).tolist()
                gains = [0.25]
                if k == 16 and max(rho) > 0.15:
                    gains.append(0.25 / math.sqrt(len(indices)))
                for gain in gains:
                    candidates.append((k, gain, row.bfloat16(), (-gain * u).bfloat16(), rho, signal.tolist()))
            edits[(layer_index, window)] = candidates
    winners = []
    for rec in sorted(rankings, key=lambda r: r.get("relative_support", -1), reverse=True):
        layer, window = rec["layer"], rec["window"]
        if (layer, window) not in edits or any(r["layer"] == layer for r in winners):
            continue
        for k, gain, a, b, rho, signal in edits[layer, window]:
            name = f"L{layer}_{window}_k{k}_g{gain:.4f}"
            save_file({f"{layer}.A": a, f"{layer}.B": b}, str(out_dir / (name + ".safetensors")))
            winners.append(dict(name=name, layer=layer, k=k, gain=gain, window=window,
                                unchecked_activation=signal, discovery_rho=[x * gain / .25 for x in rho]))
        if len({r["layer"] for r in winners}) == 2:
            break
    report = {"layer_ranking": rankings, "candidates": winners}
    (out_dir / "ranking.json").write_text(json.dumps(report, indent=2))
    return report
