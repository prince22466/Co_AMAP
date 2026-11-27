"""Synthesize coherent multi-layer r=1 disjoint edits (prefix_end+thought + tool_pre dispatch) with window-matched preservation."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import (
    CANDIDATE_BAND,
    CHAMP_LAYERS,
    ROOT,
    attach_adapter,
    harvest_forward,
)


def build_pres_windows(tokenizer, role: str, generated_text: str | None, fallback_action: str, fallback_args: dict):
    """Build full token suffix & windows (prefix_end, thought, tool_pre, arg_body) for any preservation case."""
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    start, end, dispatch = "<|channel>thought\n", "\n<channel|>", "<|tool_call>call:"
    header = enc(start)
    dispatch_toks = enc(dispatch)

    reason = "Let me carefully inspect the current state and take the appropriate next step."
    call_str = f"{fallback_action}{{}}<tool_call|>"
    if role == "verified":
        reason = "The targeted test passed and verified the fix. I am ready to submit the patch."
        call_str = "submit_patch{}<tool_call|>"
    elif role == "edit":
        fp = fallback_args.get("filepath", "module.py")
        os_s = (fallback_args.get("old_string") or "old")[:120]
        ns_s = (fallback_args.get("new_string") or "new")[:120]
        reason = f"I have identified the exact lines in `{fp}` to update."
        call_str = f'edit_file{{filepath:<|"|>{fp}<|"|>,new_string:<|"|>{ns_s}<|"|>,old_string:<|"|>{os_s}<|"|>}}<tool_call|>'
    elif role == "localize":
        reason = "Let me read the relevant source section first."

    if generated_text:
        tm = re.search(r"<\|channel>thought\n(.*?)\s*<channel\|>", generated_text, re.S)
        if tm and tm[1].strip():
            reason = tm[1].strip()
        cm = re.search(r"<\|tool_call>call:(.*)$", generated_text, re.S)
        if cm and cm[1].strip():
            call_str = cm[1].strip()

    body = enc(reason)[:96]
    call_toks = enc(call_str)[:160]
    before_call = header + body + enc(end)
    pre_choice_end = len(before_call) + len(dispatch_toks)
    suffix = before_call + dispatch_toks + call_toks
    wins = {
        "prefix_end": (max(0, len(header) - 2), len(header)),
        "thought": (len(header), len(header) + min(8, max(1, len(body)))),
        "tool_pre": (len(before_call), pre_choice_end),
    }
    if len(call_toks) > 6:
        wins["arg_body"] = (pre_choice_end + 2, min(len(suffix), pre_choice_end + 32))
    return suffix, wins


def fit_window_selective(
    h_on: torch.Tensor,         # [N_on, D_IN] states where edit MUST fire positively
    u_dir: torch.Tensor,        # [D_OUT] unit target residual write direction (we apply -gain * u_dir)
    h_off_all: torch.Tensor,    # [N_off, D_IN] all preservation states across windows
    h_off_crit: torch.Tensor,   # [N_crit, D_IN] critical same-window preservation states (e.g. verified@tool_pre)
    w_eff: torch.Tensor,        # [D_OUT, D_IN]
    k: int = 16,
    alpha_off: float = 0.85,
    lam_ridge: float = 2.5,
):
    """Select k sign-consistent columns aligned with w_eff.T @ u_dir and orthogonalize against preservation states."""
    u = u_dir / u_dir.norm().clamp_min(1e-10)
    proj = w_eff.T @ u  # [D_IN]

    z_on = h_on * proj.unsqueeze(0)
    z_off = h_off_all * proj.unsqueeze(0)
    z_crit = h_off_crit * proj.unsqueeze(0)

    pos_frac = (z_on > 0).float().mean(0)
    min_on = z_on.min(0).values
    mean_on = z_on.mean(0)
    rms_off = z_off.pow(2).mean(0).sqrt()
    max_abs_crit = z_crit.abs().max(0).values if h_off_crit.shape[0] > 0 else torch.zeros_like(mean_on)

    # Require >= 85% positive on ON states AND positive worst-case group mean
    valid = (pos_frac >= 0.85) & (mean_on > 0) & (mean_on > alpha_off * rms_off)
    score = torch.where(
        valid,
        0.7 * mean_on + 0.3 * min_on.clamp_min(0.0) - alpha_off * rms_off - 0.35 * max_abs_crit,
        torch.full_like(mean_on, -1e9),
    )

    top_idx = score.topk(min(k, score.numel())).indices
    sel = top_idx[score[top_idx] > 0]
    if sel.numel() == 0:
        return u, torch.zeros_like(proj), sel

    proj_s = proj[sel]
    if sel.numel() > 1 and h_off_all.shape[0] > 0:
        hp_s = h_off_all[:, sel]
        hc_s = h_off_crit[:, sel] if h_off_crit.shape[0] > 0 else hp_s
        col_scale = h_on[:, sel].pow(2).mean(0).clamp_min(1e-6)
        gram = 0.5 * (hp_s.T @ hp_s) / max(1, hp_s.shape[0]) + 0.5 * (hc_s.T @ hc_s) / max(1, hc_s.shape[0])
        reg = torch.diag(col_scale) + lam_ridge * gram
        a_s = torch.linalg.solve(reg, col_scale * proj_s)
        if ((h_on[:, sel] @ a_s) > 0).float().mean().item() < 0.85:
            a_s = proj_s
    else:
        a_s = proj_s

    row = torch.zeros_like(proj)
    row[sel] = a_s
    return u, row, sel


def compute_unembed_contrast(model, tokenizer) -> torch.Tensor:
    """Compute pure weight-space unembedding direction favoring `submit_patch` over `run_command` (so -gain*u favors `run_command`)."""
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    lm_Root = getattr(model.model, "language_model", model.model)
    norm_w = lm_Root.norm.weight.detach().float().cpu()
    # In Gemma4 RMSNorm, output = x_norm * norm_w (or (1 + norm_w)); check magnitude
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

    pos_toks = [tokenizer.encode(s, add_special_tokens=False)[0] for s in ("run_command", "run")]
    neg_toks = [tokenizer.encode(s, add_special_tokens=False)[0] for s in ("submit_patch", "submit")]
    w_run = lm_w[pos_toks].mean(0)
    w_sub = lm_w[neg_toks].mean(0)
    # u points toward `submit_patch` - `run_command`, matching dy = y_unchecked - y_useful so -gain * u pushes toward `run_command`!
    u_unembed = (w_sub - w_run) * scale_w
    return u_unembed / u_unembed.norm().clamp_min(1e-10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harvest-dir", type=Path, required=True)
    p.add_argument("--native-replays", type=Path, required=True)
    args = p.parse_args()

    hdir = args.harvest_dir
    out_dir = hdir / "multilayer_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = torch.load(hdir / "summaries_v2.pt", map_location="cpu", weights_only=False)
    paired_rows = saved["paired"]

    cases = json.loads((hdir / "cases.json").read_text())
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

    with torch.inference_mode():
        model(input_ids=torch.tensor([[tokenizer.bos_token_id]], device=model.device), use_cache=False, logits_to_keep=1)

    u_unembed = compute_unembed_contrast(model, tokenizer)

    # Harvest multi-window preservation states (prefix_end, thought, tool_pre, arg_body)
    pres_multi_path = hdir / "pres_multiwindow.pt"
    if pres_multi_path.is_file():
        pres_multi = torch.load(pres_multi_path, map_location="cpu", weights_only=False)
    else:
        pres_multi = []
        seen = set()
        for c in cases:
            if c["role"] not in ("verified", "edit", "localize"):
                continue
            key = (c["issue"], c["role"])
            if key in seen:
                continue
            seen.add(key)
            act = native.get(c.get("prompt_sha", ""))
            gen_txt = act["text"] if act else None
            suf, wins = build_pres_windows(tokenizer, c["role"], gen_txt, c["action"], c.get("arguments") or {})
            pref = c["tokens"][-(4096 - len(suf)):]
            hw = harvest_forward(model, layers, pref, suf, wins)
            for wname, (h_w, y_w) in hw.items():
                pres_multi.append({
                    "issue": c["issue"],
                    "role": c["role"],
                    "window": wname,
                    "h": h_w,
                    "y": y_w,
                })
        torch.save(pres_multi, pres_multi_path)
    print(f"Loaded {len(pres_multi)} multi-window preservation states.", flush=True)

    # Disjoint r=1 layers only!
    disjoint_layers = [l for l in CANDIDATE_BAND if l not in CHAMP_LAYERS]
    mid_band = [l for l in disjoint_layers if 17 <= l <= 33]
    late_band = [l for l in disjoint_layers if 33 <= l <= 41]

    p_end_rows = [r for r in paired_rows if r["window"] == "prefix_end"]
    thought_rows = [r for r in paired_rows if r["window"] == "thought"]
    tool_rows = [r for r in paired_rows if r["window"] == "tool_pre"]

    mid_rankings = []
    mid_fits = {}

    # 1. Rank Mid-Band Layers on joint (prefix_end + thought) ON vs all-window preservation OFF
    for l_idx in mid_band:
        w_eff = layers[l_idx].mlp.down_proj.weight.detach().float().cpu()
        hp_all = torch.stack([r["h"][l_idx].float() for r in pres_multi])
        hp_crit = torch.stack([r["h"][l_idx].float() for r in pres_multi if r["window"] in ("prefix_end", "thought")])

        # Align snapshots by issue
        by_issue_p0 = {r["issue"]: r for r in p_end_rows}
        by_issue_th = {r["issue"]: r for r in thought_rows}
        issues = sorted(set(by_issue_p0) & set(by_issue_th))
        n = len(issues)

        h_p0 = torch.stack([by_issue_p0[iss]["unchecked_h"][l_idx].float() for iss in issues])
        h_th = torch.stack([by_issue_th[iss]["unchecked_h"][l_idx].float() for iss in issues])
        y_th = torch.stack([by_issue_th[iss]["unchecked_y"][l_idx].float() for iss in issues])
        dy_th = y_th - torch.stack([by_issue_th[iss]["useful_y"][l_idx].float() for iss in issues])
        gen_idx = [i for i, iss in enumerate(issues) if by_issue_th[iss]["observed_role"] == "unchecked"]

        oof_p0, oof_th = [], []
        for i in range(n):
            tr = torch.arange(n) != i
            u_tr = dy_th[tr].mean(0)
            h_on_tr = torch.cat([h_p0[tr], h_th[tr]], dim=0)
            _, row_tr, sel_tr = fit_window_selective(h_on_tr, u_tr, hp_all, hp_crit, w_eff, k=16)
            if sel_tr.numel() == 0:
                oof_p0.append(-1.0)
                oof_th.append(-1.0)
            else:
                oof_p0.append(float(h_p0[i] @ row_tr))
                oof_th.append(float(h_th[i] @ row_tr))

        u_full, row_full, sel_full = fit_window_selective(torch.cat([h_p0, h_th], dim=0), dy_th.mean(0), hp_all, hp_crit, w_eff, k=16)
        if sel_full.numel() == 0:
            continue
        sig_p0 = h_p0 @ row_full
        sig_th = h_th @ row_full
        sig_off = hp_all @ row_full
        oof_p0_pos = sum(1 for v in oof_p0 if v > 0) / n
        oof_th_pos = sum(1 for v in oof_th if v > 0) / n
        gen_p0_pos = sum(1 for idx in gen_idx if oof_p0[idx] > 0) / max(1, len(gen_idx))
        on_mean = 0.5 * float(sig_p0.mean() + sig_th.mean())
        off_rms = float(sig_off.pow(2).mean().sqrt().clamp_min(1e-6))
        sel_ratio = on_mean / off_rms

        rec = {
            "layer": l_idx,
            "is_global_attn": l_idx in {17, 23, 35, 41},
            "oof_p0_pos_rate": round(oof_p0_pos, 3),
            "oof_th_pos_rate": round(oof_th_pos, 3),
            "gen_p0_pos_rate": round(gen_p0_pos, 3),
            "p0_mean": round(float(sig_p0.mean()), 4),
            "th_mean": round(float(sig_th.mean()), 4),
            "off_rms": round(off_rms, 4),
            "off_max_abs": round(float(sig_off.abs().max()), 4),
            "selectivity": round(sel_ratio, 3),
        }
        mid_rankings.append(rec)
        mid_fits[l_idx] = {
            "u": u_full,
            "row": row_full,
            "y_norm": float(y_th.norm(dim=1).mean()),
            "on_act": on_mean,
            "rec": rec,
        }

    mid_rankings.sort(
        key=lambda r: (
            r["gen_p0_pos_rate"] >= 0.75 and r["oof_th_pos_rate"] >= 0.875,
            r["gen_p0_pos_rate"],
            r["oof_p0_pos_rate"],
            r["selectivity"],
        ),
        reverse=True,
    )
    print("Top Mid-Band (prefix_end + thought) layers:", json.dumps(mid_rankings[:6], indent=2), flush=True)

    # 2. Rank Late-Band Layers on tool_pre ON vs multi-window preservation OFF (especially verified@tool_pre & edit@tool_pre!)
    late_rankings = []
    late_fits = {}
    issues_tp = [r["issue"] for r in tool_rows]
    n_tp = len(tool_rows)
    gen_tp_idx = [i for i, r in enumerate(tool_rows) if r["observed_role"] == "unchecked"]

    for l_idx in late_band:
        w_eff = layers[l_idx].mlp.down_proj.weight.detach().float().cpu()
        hp_all = torch.stack([r["h"][l_idx].float() for r in pres_multi])
        hp_crit = torch.stack([
            r["h"][l_idx].float()
            for r in pres_multi
            if r["window"] == "tool_pre" or r["role"] == "verified"
        ])
        hp_ver_tp = torch.stack([
            r["h"][l_idx].float()
            for r in pres_multi
            if r["role"] == "verified" and r["window"] == "tool_pre"
        ])

        hu_tp = torch.stack([r["unchecked_h"][l_idx].float() for r in tool_rows])
        yu_tp = torch.stack([r["unchecked_y"][l_idx].float() for r in tool_rows])
        dy_tp = yu_tp - torch.stack([r["useful_y"][l_idx].float() for r in tool_rows])

        oof_tp = []
        for i in range(n_tp):
            tr = torch.arange(n_tp) != i
            u_tr = dy_tp[tr].mean(0)
            _, row_tr, sel_tr = fit_window_selective(hu_tp[tr], u_tr, hp_all, hp_crit, w_eff, k=16, alpha_off=1.0, lam_ridge=3.5)
            oof_tp.append(float(hu_tp[i] @ row_tr) if sel_tr.numel() > 0 else -1.0)

        u_emp, row_tp, sel_tp = fit_window_selective(hu_tp, dy_tp.mean(0), hp_all, hp_crit, w_eff, k=16, alpha_off=1.0, lam_ridge=3.5)
        if sel_tp.numel() == 0:
            continue
        sig_tp = hu_tp @ row_tp
        sig_off = hp_all @ row_tp
        sig_ver = hp_ver_tp @ row_tp

        oof_pos = sum(1 for v in oof_tp if v > 0) / n_tp
        gen_pos = sum(1 for idx in gen_tp_idx if oof_tp[idx] > 0) / max(1, len(gen_tp_idx))
        on_mean = float(sig_tp.mean())
        off_rms = float(sig_off.pow(2).mean().sqrt().clamp_min(1e-6))
        ver_max = float(sig_ver.abs().max())
        ver_ratio = on_mean / max(ver_max, 1e-5)

        rec = {
            "layer": l_idx,
            "is_global_attn": l_idx in {35, 41},
            "oof_tp_pos_rate": round(oof_pos, 3),
            "gen_tp_pos_rate": round(gen_pos, 3),
            "tp_mean": round(on_mean, 4),
            "off_rms": round(off_rms, 4),
            "verified_tp_max_abs": round(ver_max, 4),
            "unch_vs_verified_ratio": round(ver_ratio, 3),
            "selectivity": round(on_mean / off_rms, 3),
        }
        late_rankings.append(rec)

        # Blend empirical tool_pre steering with direct unembed contrast on final layers 39..41
        if l_idx >= 39:
            u_blend = 0.65 * u_emp + 0.35 * (u_unembed if torch.dot(u_emp, u_unembed) >= 0 else -u_unembed)
            u_blend = u_blend / u_blend.norm().clamp_min(1e-10)
        else:
            u_blend = u_emp
        late_fits[l_idx] = {
            "u_emp": u_emp,
            "u_blend": u_blend,
            "row": row_tp,
            "y_norm": float(yu_tp.norm(dim=1).mean()),
            "on_act": on_mean,
            "rec": rec,
        }

    late_rankings.sort(
        key=lambda r: (
            r["gen_tp_pos_rate"] >= 0.75 and r["unch_vs_verified_ratio"] >= 2.5,
            r["gen_tp_pos_rate"],
            r["unch_vs_verified_ratio"],
            r["selectivity"],
        ),
        reverse=True,
    )
    print("Top Late-Band (tool_pre vs verified@tool_pre) layers:", json.dumps(late_rankings[:6], indent=2), flush=True)

    # Select top 4 mid layers (with >=1 global-attn layer if viable) and top 2 disjoint late layers (not overlapping mid!)
    top_mid = [r["layer"] for r in mid_rankings[:4]]
    top_late = [r["layer"] for r in late_rankings if r["layer"] not in top_mid and r["unch_vs_verified_ratio"] >= 2.0][:2]
    if len(top_late) < 2:
        top_late = [r["layer"] for r in late_rankings if r["layer"] not in top_mid][:2]

    print(f"Selected Mid4={top_mid}, Late2={top_late}", flush=True)

    specs = []
    # Group 1: Mid-4 coherent (prefix_end + thought) at rho = 0.055 and 0.090 per layer
    for tag, rho in (("lo", 0.055), ("hi", 0.090)):
        tensors = {}
        cname = f"ML4_mid_{'_'.join(map(str, top_mid))}_{tag}"
        for li in top_mid:
            f = mid_fits[li]
            g = rho * f["y_norm"] / max(f["on_act"], 1e-6)
            tensors[f"{li}.A"] = f["row"].bfloat16()
            tensors[f"{li}.B"] = (-g * f["u"]).bfloat16()
        save_file(tensors, str(out_dir / f"{cname}.safetensors"))
        specs.append({"name": cname, "layers": top_mid, "rho_per_layer": rho, "family": "mid_thought"})

    # Group 2: Late-2 dispatch (tool_pre) at rho = 0.080 and 0.140 per layer
    if top_late:
        for tag, rho in (("lo", 0.080), ("hi", 0.140)):
            tensors = {}
            cname = f"ML2_late_{'_'.join(map(str, top_late))}_{tag}"
            for li in top_late:
                f = late_fits[li]
                g = rho * f["y_norm"] / max(f["on_act"], 1e-6)
                tensors[f"{li}.A"] = f["row"].bfloat16()
                tensors[f"{li}.B"] = (-g * f["u_blend"]).bfloat16()
            save_file(tensors, str(out_dir / f"{cname}.safetensors"))
            specs.append({"name": cname, "layers": top_late, "rho_per_layer": rho, "family": "late_dispatch"})

    # Group 3: Full 6-layer coherent (Mid-4 thought initiation + Late-2 tool_pre readout) at 3 calibrated gain brackets!
    if top_mid and top_late:
        for tag, rho_m, rho_l in (("lo", 0.045, 0.065), ("mid", 0.070, 0.100), ("hi", 0.100, 0.140)):
            tensors = {}
            all_l = sorted(top_mid + top_late)
            cname = f"ML6_coherent_{'_'.join(map(str, all_l))}_{tag}"
            for li in top_mid:
                f = mid_fits[li]
                g = rho_m * f["y_norm"] / max(f["on_act"], 1e-6)
                tensors[f"{li}.A"] = f["row"].bfloat16()
                tensors[f"{li}.B"] = (-g * f["u"]).bfloat16()
            for li in top_late:
                f = late_fits[li]
                g = rho_l * f["y_norm"] / max(f["on_act"], 1e-6)
                tensors[f"{li}.A"] = f["row"].bfloat16()
                tensors[f"{li}.B"] = (-g * f["u_blend"]).bfloat16()
            save_file(tensors, str(out_dir / f"{cname}.safetensors"))
            specs.append({
                "name": cname,
                "layers": all_l,
                "rho_mid": rho_m,
                "rho_late": rho_l,
                "family": "coherent_mid_plus_late",
            })

    manifest = {
        "mid_rankings": mid_rankings,
        "late_rankings": late_rankings,
        "selected_mid": top_mid,
        "selected_late": top_late,
        "candidates": specs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Synthesized {len(specs)} multi-layer r=1 disjoint factor sets!", flush=True)


if __name__ == "__main__":
    main()
