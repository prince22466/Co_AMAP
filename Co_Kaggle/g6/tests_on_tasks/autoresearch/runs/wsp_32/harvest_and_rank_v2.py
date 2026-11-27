"""Harvest genuine & paired FFN states and rank sign-consistent, preservation-selective sparse edits."""
import argparse
import collections
import hashlib
import json
import math
import re
import tarfile
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision_cases_v2 import read_cases_from_runs

ROOT = Path("/content/wsp")
DATA = ROOT / "input/competitions/gemma-4-developer-agent"
CHAMP_LAYERS = {29, 31, 34, 37, 38}
CANDIDATE_BAND = list(range(17, 42))  # Global-attention & downstream FFN band [17..41] per WSP_CONTINUATION.md


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


def resolve_targeted_check(case: dict) -> str | None:
    """Resolve a fast, targeted module pytest or Python check (NEVER bare pytest!) from the snapshot."""
    issue = case["issue"]
    fpath = case.get("filepath") or ""
    if issue == "rich_3506" and fpath == "rich/segment.py":
        return (
            'python3 -c "from rich.segment import Segment; s=Segment(chr(0x6c34)*2+\'aaaaa\'); '
            'a,b=s.split_cells(3); assert (a.cell_length,b.cell_length)==(3,s.cell_length-3)"'
        )
    path = Path(fpath)
    if path.suffix != ".py":
        return None
    stem = path.stem.lstrip("_")
    snap_tgz = DATA / "snapshots" / f"{issue}.tgz"
    if not snap_tgz.is_file():
        return None
    with tarfile.open(snap_tgz) as archive:
        members = {m.name.removeprefix("./") for m in archive if m.isfile()}
    # Check exact or prefix test file matches
    candidates = [
        f"tests/test_{stem}.py",
        f"tests/test_{path.parent.name}_{stem}.py",
    ]
    for cand in candidates:
        if cand in members:
            if issue.startswith("requests_"):
                return f"PYTHONPATH=src pytest -q {cand}"
            return f"pytest -q {cand}"
    # Fallback: find any test file containing stem
    matches = sorted(m for m in members if m.startswith("tests/") and m.endswith(".py") and stem in Path(m).stem)
    if matches:
        target = matches[0]
        return (f"PYTHONPATH=src pytest -q {target}" if issue.startswith("requests_") else f"pytest -q {target}")
    return f'python3 -c "import {fpath[:-3].replace("/", ".")}; print(\'import ok\')"'


def build_branches(tokenizer, generated_text: str, command: str):
    """Build matched continuations with leak-free windows:
    - prefix_end: exact `<|channel>thought\\n` end token (ZERO teacher-forcing difference!)
    - thought: first 8 tokens of reasoning body
    - tool_pre: `<channel|><|tool_call>call:` stopping strictly at `:` BEFORE tool name!
    """
    start, end, dispatch = "<|channel>thought\n", "\n<channel|>", "<|tool_call>call:"
    thought_m = re.search(r"<\|channel>thought\n(.*?)\s*<channel\|>", generated_text, re.S)
    is_submit = "call:submit_patch{" in generated_text
    unchecked_reason = "The edit is applied cleanly and matches the requested change. I will submit the patch now."
    useful_reason = f"I should verify the modified behavior before submitting by running `{command}`."
    if thought_m and thought_m[1].strip():
        if is_submit:
            unchecked_reason = thought_m[1].strip()
        else:
            useful_reason = thought_m[1].strip()

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    header = enc(start)
    dispatch_toks = enc(dispatch)  # `<|tool_call>call:` — ends BEFORE `submit_patch` vs `run_command`!
    out = {}
    for role, reasoning, call in (
        ("unchecked", unchecked_reason, "submit_patch{}<tool_call|>"),
        ("useful", useful_reason, f'run_command{{command:<|"|>{command}<|"|>}}<tool_call|>'),
    ):
        body = enc(reasoning)[:96]
        before_call = header + body + enc(end)
        pre_choice_end = len(before_call) + len(dispatch_toks)
        suffix = before_call + dispatch_toks + enc(call)
        windows = {
            "prefix_end": (max(0, len(header) - 2), len(header)),
            "thought": (len(header), len(header) + min(8, max(1, len(body)))),
            "tool_pre": (len(before_call), pre_choice_end),
        }
        out[role] = (suffix, windows)
    return out


def harvest_forward(model, layers, prefix: list[int], suffix: list[int], windows: dict[str, tuple[int, int]]):
    tokens = prefix + suffix
    assert len(tokens) <= 4096, f"Tokens {len(tokens)} > 4096"
    bins = {w: {} for w in windows}
    hooks = []
    for idx, layer in enumerate(layers):
        def record(mod, inp, out, i=idx):
            for wname, (a, b) in windows.items():
                sp = slice(len(prefix) + a, len(prefix) + b)
                bins[wname][i] = (
                    inp[0][0, sp].float().mean(0).cpu(),
                    out[0, sp].float().mean(0).cpu(),
                )
        hooks.append(layer.mlp.down_proj.register_forward_hook(record))
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([tokens], device=model.device), use_cache=False, logits_to_keep=1)
    finally:
        for h in hooks:
            h.remove()
    return {
        w: (
            torch.stack([bins[w][i][0] for i in range(len(layers))]),
            torch.stack([bins[w][i][1] for i in range(len(layers))]),
        )
        for w in windows
    }


def fit_selective(
    hu: torch.Tensor,      # [N_u, D_IN] genuine + paired unchecked states
    yu: torch.Tensor,      # [N_u, D_OUT]
    dh: torch.Tensor,      # [N_pair, D_IN] unchecked - useful
    dy: torch.Tensor,      # [N_pair, D_OUT] unchecked - useful
    h_pres: torch.Tensor,  # [N_pres, D_IN] verified + edit + localize states
    w_eff: torch.Tensor,   # [D_OUT, D_IN] exact effective down_proj weight
    k: int,
    alpha_pres: float = 0.6,
    lam_ridge: float = 1.5,
):
    """Sign-consistent column selection + Experiment A sparse preservation ridge projection."""
    u = dy.mean(0)
    u = u / u.norm().clamp_min(1e-10)
    proj = w_eff.T @ u  # [D_IN]

    # Per-column signed contribution on unchecked states & paired contrast
    z_unch = hu * proj.unsqueeze(0)          # [N_u, D_IN]
    dz_pair = dh * proj.unsqueeze(0)         # [N_pair, D_IN]
    z_pres = h_pres * proj.unsqueeze(0)      # [N_pres, D_IN]

    pos_frac = (z_unch > 0).float().mean(0)  # [D_IN]
    mean_unch = z_unch.mean(0)
    mean_dz = dz_pair.mean(0)
    rms_pres = z_pres.pow(2).mean(0).sqrt()

    # Enforce strict sign consistency (>80% positive on unchecked AND positive contrast)
    valid = (pos_frac >= 0.80) & (mean_unch > 0) & (mean_dz > 0)
    score = torch.where(
        valid,
        torch.minimum(mean_unch, mean_dz) - alpha_pres * rms_pres,
        torch.full_like(mean_unch, -1e9),
    )

    top_idx = score.topk(min(k, score.numel())).indices
    sel = top_idx[score[top_idx] > 0]
    if sel.numel() == 0:
        return u, torch.zeros_like(proj), sel

    # Experiment A sparse ridge projection on selected support S (WSP_CONTINUATION.md line 81)
    proj_s = proj[sel]  # [|S|]
    if sel.numel() > 1 and h_pres.shape[0] > 0 and lam_ridge > 0:
        hp_s = h_pres[:, sel]  # [N_pres, |S|]
        # Scale-invariant ridge normalized by mean column energy on unchecked
        col_scale = hu[:, sel].pow(2).mean(0).clamp_min(1e-6)
        gram = (hp_s.T @ hp_s) / max(1, h_pres.shape[0])
        reg = torch.diag(col_scale) + lam_ridge * gram
        a_s = torch.linalg.solve(reg, col_scale * proj_s)
        # Verify ridge did not flip unchecked sign; fallback to raw proj_s if it over-rotated
        if ((hu[:, sel] @ a_s) > 0).float().mean().item() < 0.80:
            a_s = proj_s
    else:
        a_s = proj_s

    row = torch.zeros_like(proj)
    row[sel] = a_s
    return u, row, sel


def rank_selective_candidates(paired_rows: list[dict], pres_rows: list[dict], layers, champ_tensors, out_dir: Path, allow_r2: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings = []
    candidates = []

    for l_idx in CANDIDATE_BAND:
        if not allow_r2 and l_idx in CHAMP_LAYERS:
            continue
        w_base = layers[l_idx].mlp.down_proj.weight.detach().float().cpu()
        akey = f"base_model.model.model.language_model.layers.{l_idx}.mlp.down_proj.lora_A.weight"
        if akey in champ_tensors:
            a_c = champ_tensors[akey].float().cpu()
            b_c = champ_tensors[akey.replace("lora_A", "lora_B")].float().cpu()
            w_eff = w_base + (b_c @ a_c)
        else:
            w_eff = w_base

        # Preservation states at prefix_end (verified + edit + localize)
        h_pres_list = [r["h"][l_idx].float() for r in pres_rows]
        h_pres = torch.stack(h_pres_list) if h_pres_list else torch.zeros(1, w_eff.shape[1])

        for window in ("thought", "tool_pre"):
            sub = [r for r in paired_rows if r["window"] == window]
            if len(sub) < 3:
                continue
            # Also collect ONLY genuine observed unchecked states in sub for strict sign gating
            gen_mask = [i for i, r in enumerate(sub) if r["observed_role"] == "unchecked"]
            hu_all = torch.stack([r["unchecked_h"][l_idx].float() for r in sub])
            yu_all = torch.stack([r["unchecked_y"][l_idx].float() for r in sub])
            dh_all = hu_all - torch.stack([r["useful_h"][l_idx].float() for r in sub])
            dy_all = yu_all - torch.stack([r["useful_y"][l_idx].float() for r in sub])

            n = len(sub)
            oof_stats = {}
            for k in (1, 16):
                oof_unch_act = []
                oof_dy_align = []
                for i in range(n):
                    tr = torch.arange(n) != i
                    hu_tr = hu_all[tr]
                    u_tr, row_tr, sel_tr = fit_selective(
                        hu_tr, yu_all[tr], dh_all[tr], dy_all[tr], h_pres, w_eff, k=k
                    )
                    if sel_tr.numel() == 0:
                        oof_unch_act.append(-1.0)
                        oof_dy_align.append(-1.0)
                    else:
                        oof_unch_act.append(float(hu_all[i] @ row_tr))
                        oof_dy_align.append(float(dy_all[i] @ u_tr))
                pos_oof = sum(1 for x, a in zip(oof_unch_act, oof_dy_align) if x > 0 and a > 0) / n
                gen_pos = (
                    sum(1 for idx in gen_mask if oof_unch_act[idx] > 0) / len(gen_mask)
                    if gen_mask else pos_oof
                )
                oof_stats[k] = {
                    "oof_unch_act": [round(x, 4) for x in oof_unch_act],
                    "oof_dy_align": [round(x, 4) for x in oof_dy_align],
                    "oof_pos_rate": round(pos_oof, 3),
                    "genuine_unchecked_pos_rate": round(gen_pos, 3),
                }

            # Full fit on all discovery snapshots
            rec = {
                "layer": l_idx,
                "is_disjoint_r1": l_idx not in CHAMP_LAYERS,
                "is_global_attn": l_idx in {17, 23, 29, 35, 41},
                "window": window,
                "n_snapshots": n,
                "n_genuine_unchecked": len(gen_mask),
                "oof": oof_stats,
            }

            # Require >= 80% positive out-of-fold activation on unchecked states AND >= 80% on genuine unchecked!
            best_pos = max(oof_stats[16]["oof_pos_rate"], oof_stats[1]["oof_pos_rate"])
            best_gen = max(oof_stats[16]["genuine_unchecked_pos_rate"], oof_stats[1]["genuine_unchecked_pos_rate"])
            if best_pos >= 0.75 and best_gen >= 0.80:
                u_full, row_16, sel_16 = fit_selective(hu_all, yu_all, dh_all, dy_all, h_pres, w_eff, k=16)
                if sel_16.numel() > 0:
                    sig_unch = (hu_all @ row_16)
                    sig_pres = (h_pres @ row_16)
                    selectivity = float(sig_unch.mean() / (sig_pres.pow(2).mean().sqrt().clamp_min(1e-6)))
                    rec["selectivity_ratio_k16"] = round(selectivity, 3)
                    rec["unch_pos_frac_full"] = round(float((sig_unch > 0).float().mean()), 3)
                    rec["pres_max_neg"] = round(float(sig_pres.min()), 4)
                    rec["pres_rms"] = round(float(sig_pres.pow(2).mean().sqrt()), 4)
                    rec["unch_mean"] = round(float(sig_unch.mean()), 4)
            rankings.append(rec)

    # Sort shortlisted layers by (genuine_pos_rate, selectivity_ratio_k16), preferring disjoint R1 global/downstream layers
    viable = [r for r in rankings if r.get("selectivity_ratio_k16", 0) > 1.2 and r.get("unch_pos_frac_full", 0) >= 0.80]
    viable.sort(
        key=lambda r: (
            r["is_disjoint_r1"],
            r["oof"][16]["genuine_unchecked_pos_rate"],
            r["oof"][16]["oof_pos_rate"],
            r.get("selectivity_ratio_k16", 0.0),
        ),
        reverse=True,
    )

    chosen_layers = []
    for rec in viable:
        l_idx, window = rec["layer"], rec["window"]
        if any(c[0] == l_idx for c in chosen_layers):
            continue
        chosen_layers.append((l_idx, window, rec))
        if len(chosen_layers) >= 3:
            break

    # Save candidate factor files with calibrated gain brackets (target rho ~ 0.06 and 0.11 per Experiment B)
    for l_idx, window, rec in chosen_layers:
        w_base = layers[l_idx].mlp.down_proj.weight.detach().float().cpu()
        akey = f"base_model.model.model.language_model.layers.{l_idx}.mlp.down_proj.lora_A.weight"
        w_eff = w_base + (champ_tensors[akey].float().cpu() @ champ_tensors[akey.replace("lora_A", "lora_B")].float().cpu()).T if akey in champ_tensors else w_base
        if akey in champ_tensors:
            w_eff = w_base + (champ_tensors[akey.replace("lora_A", "lora_B")].float().cpu() @ champ_tensors[akey].float().cpu())
        h_pres = torch.stack([r["h"][l_idx].float() for r in pres_rows])
        sub = [r for r in paired_rows if r["window"] == window]
        hu_all = torch.stack([r["unchecked_h"][l_idx].float() for r in sub])
        yu_all = torch.stack([r["unchecked_y"][l_idx].float() for r in sub])
        dh_all = hu_all - torch.stack([r["useful_h"][l_idx].float() for r in sub])
        dy_all = yu_all - torch.stack([r["useful_y"][l_idx].float() for r in sub])
        y_norm_mean = float(yu_all.norm(dim=1).mean().clamp_min(1e-6))

        for k in (1, 16):
            u_full, row_k, sel_k = fit_selective(hu_all, yu_all, dh_all, dy_all, h_pres, w_eff, k=k)
            if sel_k.numel() == 0:
                continue
            sig_unch = hu_all @ row_k
            sig_pres = h_pres @ row_k
            if float((sig_unch > 0).float().mean()) < 0.80:
                continue
            mean_act = float(sig_unch.abs().mean().clamp_min(1e-6))
            # Calibrate gains so relative FFN perturbation rho = gain * |s(h)| / ||y|| hits {0.06, 0.12} (plus 0.25 if safe)
            target_rhos = (0.06, 0.12)
            gains = sorted({round(rho_t * y_norm_mean / mean_act, 4) for rho_t in target_rhos})
            for g in gains:
                rho_vals = ((g * sig_unch.abs()) / yu_all.norm(dim=1).clamp_min(1e-6)).tolist()
                rank_export = 1 if (l_idx not in CHAMP_LAYERS) else 2
                cname = f"L{l_idx}_{window}_k{k}_r{rank_export}_g{g:.4f}"
                save_file(
                    {f"{l_idx}.A": row_k.bfloat16(), f"{l_idx}.B": (-g * u_full).bfloat16()},
                    str(out_dir / f"{cname}.safetensors"),
                )
                candidates.append({
                    "name": cname,
                    "layer": l_idx,
                    "window": window,
                    "k": int(sel_k.numel()),
                    "rank_export": rank_export,
                    "gain": g,
                    "unchecked_activation": [round(float(x), 4) for x in sig_unch.tolist()],
                    "pres_rms_activation": round(float(sig_pres.pow(2).mean().sqrt()), 4),
                    "pres_min_activation": round(float(sig_pres.min()), 4),
                    "discovery_rho": [round(x, 4) for x in rho_vals],
                })

    report = {"layer_ranking": rankings, "candidates": candidates}
    (out_dir / "ranking.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--native-replays", type=Path, required=True)
    p.add_argument("--verified-cases", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-r2", action="store_true")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cases = read_cases_from_runs(args.runs)
    if args.verified_cases and args.verified_cases.is_file():
        cases.extend(json.loads(args.verified_cases.read_text()))
    counts = collections.Counter(c["role"] for c in cases)
    print("Total extracted cases by role:", dict(counts), flush=True)
    (args.output / "cases.json").write_text(json.dumps(cases))

    native = {r["prompt_sha"]: r for r in json.loads(args.native_replays.read_text())}
    assets = json.loads((ROOT / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        assets["model"], dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    layers = getattr(model.model, "language_model", model.model).layers

    with zipfile.ZipFile(ROOT / "champion.zip") as z:
        blob = z.read("adapters/wsp_r1/adapter_model.safetensors")
        assert hashlib.sha256(blob).hexdigest() == "8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815"
        champ = load(blob)
    attach_adapter(layers, champ)

    # Warm up compressed_tensors dequantization once so W_base is materialized!
    with torch.inference_mode():
        model(input_ids=torch.tensor([[tokenizer.bos_token_id]], device=model.device), use_cache=False, logits_to_keep=1)

    paired_rows = []
    pres_rows = []
    header_toks = tokenizer.encode("<|channel>thought\n", add_special_tokens=False)

    # 1. Harvest preservation states (verified + edit + localize) at exact thought-start!
    seen_pres = set()
    for c in cases:
        if c["role"] not in ("verified", "edit", "localize"):
            continue
        key = (c["issue"], c["role"])
        if key in seen_pres:
            continue
        seen_pres.add(key)
        prefix = c["tokens"][-3904:]
        wins = {"prefix_end": (0, len(header_toks))}
        res = harvest_forward(model, layers, prefix, header_toks, wins)
        h_p, y_p = res["prefix_end"]
        pres_rows.append({"issue": c["issue"], "role": c["role"], "h": h_p, "y": y_p})
    print(f"Harvested {len(pres_rows)} preservation prefixes (verified/edit/localize).", flush=True)

    # 2. Harvest one reproduced post-edit decision contrast per distinct snapshot
    for issue in sorted({c["issue"] for c in cases}):
        opts = [c for c in cases if c["issue"] == issue and c["role"] in ("unchecked", "check")]
        for case in sorted(opts, key=lambda c: (c["role"] != "unchecked", len(c["tokens"]))):
            actual = native.get(case["prompt_sha"])
            if not actual or actual["prompt_tokens"] != len(case["tokens"]):
                continue
            if actual["action"] != case["action"]:
                continue
            cmd = resolve_targeted_check(case)
            if not cmd:
                continue
            prefix = case["tokens"][-3904:]
            branches = build_branches(tokenizer, actual["text"], cmd)
            if any(len(prefix) + len(suf) > 4096 for suf, _ in branches.values()):
                continue
            packed = {}
            for b_role, (suf, wins) in branches.items():
                hw = harvest_forward(model, layers, prefix, suf, wins)
                for wname, (h, y) in hw.items():
                    rec = packed.setdefault(
                        wname,
                        dict(
                            issue=issue,
                            snapshot=case["snapshot"],
                            observed_role=case["role"],
                            observed_action=actual["action"],
                            window=wname,
                        ),
                    )
                    rec[f"{b_role}_h"] = h
                    rec[f"{b_role}_y"] = y
            paired_rows.extend(packed.values())
            print(f"Harvested paired snapshot {issue} (observed={case['role']}/{actual['action']}, cmd={cmd})", flush=True)
            break

    torch.save({"paired": paired_rows, "pres": pres_rows}, args.output / "summaries_v2.pt")
    report = rank_selective_candidates(
        paired_rows, pres_rows, layers, champ, args.output / "candidates", allow_r2=args.allow_r2
    )
    print("Shortlisted selective candidates:", len(report["candidates"]), flush=True)
    for c in report["candidates"]:
        print(" ", c, flush=True)


if __name__ == "__main__":
    main()
