"""Synthesize V6 r=1 strictly-downstream Layer-41 natural-rectifier adapters (zero upstream perturbation to champion 29..38)."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load, load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter, harvest_forward
from recipe import compile_explicit
from synthesize_multilayer_v2 import compute_unembed_contrast
from synthesize_targeted_dispatch_v3 import find_subseq_end


def fit_l41_natural_rectifier(
    h_real_unch: torch.Tensor,   # [N_real_u, 10240] 100% REAL champion unchecked@tool_pre states!
    h_pair_unch: torch.Tensor,   # [N_pair, 10240] paired unchecked@tool_pre states
    h_off_all: torch.Tensor,     # [N_off, 10240] all ~110 multi-window preservation states
    h_ver_tp: torch.Tensor,      # [N_ver, 10240] verified@tool_pre states (MUST stay near 0!)
    h_edit_all: torch.Tensor,    # [N_ed, 10240] edit_file states across all windows (MUST stay near 0!)
    u_dir: torch.Tensor,         # [2560] target contrast direction
    w_down: torch.Tensor,        # [2560, 10240] L41 down_proj weight
    k: int = 16,
    lam_ridge: float = 6.0,
):
    """Select k naturally-rectified GeGLU columns in L41 that fire >=0 on unchecked and ~0 on edit & verified."""
    u = u_dir / u_dir.norm().clamp_min(1e-10)
    proj = w_down.T @ u  # [10240]

    z_real = h_real_unch * proj.unsqueeze(0)
    z_pair = h_pair_unch * proj.unsqueeze(0)
    z_off = h_off_all * proj.unsqueeze(0)
    z_ver = h_ver_tp * proj.unsqueeze(0)
    z_ed = h_edit_all * proj.unsqueeze(0)

    # Strict natural rectification criteria:
    # 1. >= 80% positive on REAL unchecked@tool_pre AND positive min/mean
    real_pos_frac = (z_real > 0).float().mean(0)
    pair_pos_frac = (z_pair > 0).float().mean(0)
    mean_real = z_real.mean(0)
    mean_pair = z_pair.mean(0)
    rms_off = z_off.pow(2).mean(0).sqrt()
    rms_ed = z_ed.pow(2).mean(0).sqrt()
    max_abs_ver = z_ver.abs().max(0).values
    min_off = z_off.min(0).values  # Guard against negative sign-reversal!

    on_sig = 0.65 * mean_real + 0.35 * mean_pair
    valid = (
        (real_pos_frac >= 0.80)
        & (pair_pos_frac >= 0.80)
        & (mean_real > 0)
        & (on_sig > 1.8 * rms_off)
        & (on_sig > 2.2 * rms_ed)
        # Prevent large negative excursions on preservation states (sign-reversal guard!)
        & (min_off >= -0.25 * on_sig)
    )
    score = torch.where(
        valid,
        on_sig - 1.2 * rms_ed - 0.9 * rms_off - 0.5 * max_abs_ver + 0.4 * min_off.clamp_max(0.0),
        torch.full_like(on_sig, -1e9),
    )

    top_idx = score.topk(min(k, score.numel())).indices
    sel = top_idx[score[top_idx] > 0]
    assert sel.numel() > 0, "No natural rectifier columns found on L41"

    proj_s = proj[sel]
    # Heavy preservation ridge prioritizing edit_file & verified@tool_pre orthogonality!
    h_ed_s = h_edit_all[:, sel]
    h_ver_s = h_ver_tp[:, sel]
    h_off_s = h_off_all[:, sel]
    col_scale = h_real_unch[:, sel].pow(2).mean(0).clamp_min(1e-6)
    gram = (
        0.45 * (h_ed_s.T @ h_ed_s) / max(1, h_ed_s.shape[0])
        + 0.35 * (h_ver_s.T @ h_ver_s) / max(1, h_ver_s.shape[0])
        + 0.20 * (h_off_s.T @ h_off_s) / max(1, h_off_s.shape[0])
    )
    a_s = torch.linalg.solve(torch.diag(col_scale) + lam_ridge * gram, col_scale * proj_s)
    if ((h_real_unch[:, sel] @ a_s) > 0).float().mean().item() < 0.80:
        a_s = proj_s

    row = torch.zeros_like(proj)
    row[sel] = a_s
    return u, row, sel


def main():
    hdir = Path("/content/wsp/harvest32")
    out_dir = hdir / "v6_l41"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = torch.load(hdir / "summaries_v2.pt", map_location="cpu", weights_only=False)
    paired_tp = [r for r in saved["paired"] if r["window"] == "tool_pre"]
    pres_multi = torch.load(hdir / "pres_multiwindow.pt", map_location="cpu", weights_only=False)
    ml2 = load_file(str(hdir / "multilayer_candidates/ML2_late_35_41_hi.safetensors"))

    cases = json.loads((hdir / "cases.json").read_text())
    native = {r["prompt_sha"]: r for r in json.loads((hdir / "native/champion.json").read_text())}

    assets = json.loads((ROOT / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        assets["model"], dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    lm_root = getattr(model.model, "language_model", model.model)
    layers = lm_root.layers

    with zipfile.ZipFile(ROOT / "champion.zip") as z:
        champ = load(z.read("adapters/wsp_r1/adapter_model.safetensors"))
    attach_adapter(layers, champ)
    with torch.inference_mode():
        model(input_ids=torch.tensor([[tokenizer.bos_token_id]], device=model.device), use_cache=False, logits_to_keep=1)

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    dispatch_Marker = "<|tool_call>call:"

    # 1. Harvest 100% REAL champion unchecked@tool_pre states (exact greedy text before `submit_patch` on all 5 real unchecked cases!)
    real_unch_h41 = []
    real_unch_y41 = []
    seen_u = set()
    for c in cases:
        if c["role"] != "unchecked" or c["issue"] in seen_u:
            continue
        act = native.get(c["prompt_sha"])
        if not act or not act.get("text") or dispatch_Marker not in act["text"]:
            continue
        seen_u.add(c["issue"])
        pre_call_str = act["text"].split(dispatch_Marker)[0] + dispatch_Marker
        suf_toks = enc(pre_call_str)
        disp_len = len(enc(dispatch_Marker))
        pref = c["tokens"][-(4096 - len(suf_toks)) :]
        hw = harvest_forward(model, layers, pref, suf_toks, {"real_tp": (len(suf_toks) - disp_len, len(suf_toks))})
        h_all, y_all = hw["real_tp"]
        real_unch_h41.append(h_all[41].float())
        real_unch_y41.append(y_all[41].float())
    print(f"Harvested {len(real_unch_h41)} 100% REAL unchecked@tool_pre states on L41 (issues={sorted(seen_u)}).", flush=True)

    h_real_u = torch.stack(real_unch_h41)
    y_real_u = torch.stack(real_unch_y41)
    y41_norm = float(y_real_u.norm(dim=1).mean())

    h_pair_u = torch.stack([r["unchecked_h"][41].float() for r in paired_tp])
    dy_pair = torch.stack([r["unchecked_y"][41].float() - r["useful_y"][41].float() for r in paired_tp])

    h_off_all = torch.stack([r["h"][41].float() for r in pres_multi])
    h_ver_tp = torch.stack([r["h"][41].float() for r in pres_multi if r["role"] == "verified" and r["window"] == "tool_pre"])
    h_edit_all = torch.stack([r["h"][41].float() for r in pres_multi if r["role"] == "edit"])

    w_down41 = layers[41].mlp.down_proj.weight.detach().float().cpu()
    u_unembed = compute_unembed_contrast(model, tokenizer)
    u_emp = dy_pair.mean(0)
    u_emp = u_emp / u_emp.norm().clamp_min(1e-10)
    u_blend = 0.55 * u_emp + 0.45 * (u_unembed if torch.dot(u_emp, u_unembed) >= 0 else -u_unembed)
    u_blend = u_blend / u_blend.norm().clamp_min(1e-10)

    # LOO-CV on the REAL unchecked cases!
    oof_real = []
    for i in range(len(h_real_u)):
        tr = torch.arange(len(h_real_u)) != i
        _, row_tr, _ = fit_l41_natural_rectifier(h_real_u[tr], h_pair_u, h_off_all, h_ver_tp, h_edit_all, u_blend, w_down41, k=16)
        oof_real.append(float(h_real_u[i] @ row_tr))

    u_rect, row_rect16, sel16 = fit_l41_natural_rectifier(h_real_u, h_pair_u, h_off_all, h_ver_tp, h_edit_all, u_blend, w_down41, k=16)
    sig_real = h_real_u @ row_rect16
    sig_ed = h_edit_all @ row_rect16
    sig_ver = h_ver_tp @ row_rect16
    sig_off = h_off_all @ row_rect16

    diag = {
        "k": int(sel16.numel()),
        "oof_real_unch": [round(v, 4) for v in oof_real],
        "oof_real_pos_rate": round(sum(1 for v in oof_real if v > 0) / len(oof_real), 3),
        "real_unch_mean": round(float(sig_real.mean()), 4),
        "edit_rms": round(float(sig_ed.pow(2).mean().sqrt()), 5),
        "edit_max_abs": round(float(sig_ed.abs().max()), 5),
        "verified_tp_max_abs": round(float(sig_ver.abs().max()), 5),
        "off_min": round(float(sig_off.min()), 5),
        "unch_vs_edit_rms_ratio": round(float(sig_real.mean() / sig_ed.pow(2).mean().sqrt().clamp_min(1e-6)), 2),
        "unch_vs_verified_max_ratio": round(float(sig_real.mean() / sig_ver.abs().max().clamp_min(1e-6)), 2),
    }
    print("L41 Natural Rectifier Diagnostics:", json.dumps(diag, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []
    mean_on = float(sig_real.mean().clamp_min(1e-5))

    # Zero control on [29, 31, 34, 37, 38, 41]
    tmp_fac = out_dir / "fac_zero.safetensors"
    save_file({"41.A": row_rect16.bfloat16(), "41.B": (-1.0 * u_rect).bfloat16()}, str(tmp_fac))
    z_info = compile_explicit(ROOT / "champion.zip", tmp_fac, ad_root / "V6_zero_41", "zero", rank=1)
    compiled.append({"adapter": "V6_zero_41", "sha256": z_info["adapter_sha256"], "control": "zero"})

    # Family A: L41 Natural Rectifier down_proj ONLY across 4 finely-spaced gain brackets (rho = 0.14, 0.22, 0.30, 0.40 on L41 alone)
    for tag, rho in (("r14", 0.14), ("r22", 0.22), ("r30", 0.30), ("r40", 0.40)):
        gain = rho * y41_norm / mean_on
        fac_p = out_dir / f"fac_{tag}.safetensors"
        save_file({"41.A": row_rect16.bfloat16(), "41.B": (-gain * u_rect).bfloat16()}, str(fac_p))
        for ctrl in ("candidate", "random"):
            lbl = f"V6_L41_rect_{tag}_{ctrl}"
            info = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / lbl, ctrl, rank=1)
            compiled.append({"adapter": lbl, "sha256": info["adapter_sha256"], "control": ctrl, "rho_41": rho})

    # Family B: Exact ML2's L41 slice ONLY (dropping upstream L35 completely!) at 1.25x and 1.75x
    for tag, mult in (("m125", 1.25), ("m175", 1.75)):
        fac_p = out_dir / f"fac_ml1_{tag}.safetensors"
        save_file({"41.A": ml2["41.A"], "41.B": (ml2["41.B"].float() * mult).bfloat16()}, str(fac_p))
        for ctrl in ("candidate", "random"):
            lbl = f"V6_ML1_only41_{tag}_{ctrl}"
            info = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / lbl, ctrl, rank=1)
            compiled.append({"adapter": lbl, "sha256": info["adapter_sha256"], "control": ctrl, "mult": mult})

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diag, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V6 strictly-downstream L41 adapters!", flush=True)


if __name__ == "__main__":
    main()
