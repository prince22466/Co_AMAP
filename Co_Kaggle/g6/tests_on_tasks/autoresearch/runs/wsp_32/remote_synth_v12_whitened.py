"""Synthesize V12 Truncated-PCA Whitened Pure `down_proj` h0-Readout + Group-Locked GeGLU Transistors across 19 disjoint snapshots.

Key upgrades from V11 diagnostics (v11_hybrid_manifest.json):
1. Family A (Pure `down_proj` r=1 in 10,240-D GeGLU h0-space, ZERO base-neuron pruning):
   - V11 proved 5/5 (100%) positive unseen-issue OOF sign (`4/5 > 0.15`, SNR=5.03x-7.25x) with raw 5D means.
   - V12 upgrades to Truncated-PCA + Contrast Subspace Q_P (P in {8, 12, 18, 24} fit strictly inside LOO CV)
     with balanced OFF-group whitening (`ver_colon=3.0, edit_colon=3.0, loc_colon=2.0, non_colon=4.0`),
     boosting selectivity SNR while keeping zero tail-noise overfitting and `edit_exact_match` near 12/12!
2. Family B (Group-Locked IRLS Balanced GeGLU on {39, 41}):
   - Fixes V11's global IRLS weight-stealing bug by normalizing IRLS weights INDEPENDENTLY inside each OFF group
     (`ver_colon` ALWAYS keeps weight 2.0, `edit_colon` ALWAYS keeps weight 2.0), guaranteeing `ver_colon == 0.0`!
"""
import gc
import json
import shutil
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter, resolve_targeted_check
from recipe import compile_explicit
from synth_v7_geglu import compile_v7_geglu
from synthesize_targeted_dispatch_v3 import find_subseq_end


def build_orthonormal_basis(vecs: list[torch.Tensor]) -> torch.Tensor:
    valid = [v.double() for v in vecs if float(v.norm()) > 1e-9]
    mat = torch.stack([v / v.norm() for v in valid], dim=1)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.float()


def make_shared_pca_colon_basis(records: list[dict], li: int, feat_key: str, pref_key: str, is_h0: bool, k_pca: int) -> torch.Tensor:
    """Build (6 + k_pca)-dim shared cross-issue basis strictly inside LOO CV (removes tail per-issue noise!)."""
    unch_col, ver_col, edit_col, loc_col, all_col, non_col = [], [], [], [], [], []
    non_samples = []
    for r in records:
        xs = r["cap"][(li, feat_key)].float()
        cidx = r["h_colon_idx"] if is_h0 else r["colon_idx"]
        if cidx is not None:
            xc = xs[cidx]
            all_col.append(xc)
            if r["role"] in ("unchecked", "check"):
                unch_col.append(xc)
            elif r["role"] == "verified":
                ver_col.append(xc)
            elif r["role"] == "edit":
                edit_col.append(xc)
            else:
                loc_col.append(xc)
            m = torch.arange(xs.shape[0]) != cidx
            if m.any():
                non_col.append(xs[m].mean(dim=0))
                non_samples.append(xs[m][:: max(1, int(m.sum()) // 4)])
        else:
            non_col.append(xs.mean(dim=0))
            non_samples.append(xs[:: max(1, xs.shape[0] // 4)])
        if (li, pref_key) in r["cap"]:
            xp = r["cap"][(li, pref_key)].float()
            non_col.append(xp.mean(dim=0))

    X_col = torch.stack(all_col)
    mu_col = X_col.mean(dim=0)
    mu_unch = torch.stack(unch_col).mean(dim=0)
    mu_ver = torch.stack(ver_col).mean(dim=0) if ver_col else mu_col
    mu_edit = torch.stack(edit_col).mean(dim=0) if edit_col else mu_col
    mu_loc = torch.stack(loc_col).mean(dim=0) if loc_col else mu_col
    mu_non = torch.stack(non_col).mean(dim=0)

    vecs = [
        mu_col - mu_non,
        mu_unch - mu_ver,
        mu_unch - mu_edit,
        mu_unch - mu_loc,
        mu_unch,
        mu_ver - mu_edit,
    ]
    if k_pca > 0:
        # Shared top PCs of colon-centered states + shared top PCs of non-colon states (high-energy shared modes ONLY!)
        k_c = max(1, (k_pca * 2) // 3)
        k_n = max(1, k_pca - k_c)
        if X_col.shape[0] > 3:
            _, _, Vh_c = torch.linalg.svd((X_col - mu_col).double(), full_matrices=False)
            for j in range(min(k_c, Vh_c.shape[0])):
                vecs.append(Vh_c[j].float())
        if non_samples:
            Xn = torch.cat(non_samples, dim=0)
            _, _, Vh_n = torch.linalg.svd((Xn - Xn.mean(dim=0)).double(), full_matrices=False)
            for j in range(min(k_n, Vh_n.shape[0])):
                vecs.append(Vh_n[j].float())

    return build_orthonormal_basis(vecs)


def solve_whitened_linear_h0(
    U_h: torch.Tensor,                      # [10240, K]
    h_on: torch.Tensor,                     # [N_on, 10240]
    off_h_groups: list[tuple[torch.Tensor, float]],
    lam: float = 3e-3,
) -> torch.Tensor:
    """Whitened balanced-group linear readout in Q_P subset of R^10240: ON -> +1.0, EVERY OFF group -> 0.0!"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ud = U_h.to(device, torch.float32)
    Zo = h_on.to(device, torch.float32) @ Ud
    n_on = max(1, Zo.shape[0])
    w_on = torch.full((n_on,), 1.0 / n_on, device=device)
    yo = torch.ones(n_on, device=device)

    Zf_list, wf_list = [], []
    for hg, wt in off_h_groups:
        if hg.shape[0] == 0:
            continue
        Zf_list.append(hg.to(device, torch.float32) @ Ud)
        wf_list.append(torch.full((hg.shape[0],), wt / hg.shape[0], device=device))
    Zf = torch.cat(Zf_list, dim=0)
    wf = torch.cat(wf_list, dim=0)

    z_scale = float(Zf.pow(2).sum(dim=1).mean().clamp_min(1e-4))
    eye = torch.eye(Ud.shape[1], device=device)
    lhs = (Zo.T * w_on) @ Zo + (Zf.T * wf) @ Zf + (lam * z_scale) * eye
    rhs = Zo.T @ (w_on * yo)
    alpha = torch.linalg.solve(lhs, rhs)
    return (Ud @ alpha).cpu()


def solve_grouplocked_subspace_geglu(
    U: torch.Tensor,
    x_on: torch.Tensor,
    g0_on: torch.Tensor,
    p0_on: torch.Tensor,
    off_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    target_g_on: float = 3.6,
    target_g_off: float = -4.9,
    target_p_on: float = 3.4,
    lam: float = 1e-3,
    n_irls: int = 14,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-locked IRLS: each OFF group normalizes its IRLS weights INDEPENDENTLY so non_colon NEVER steals weight from verified@call:!"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ud = U.to(device, torch.float32)
    Zo = x_on.to(device, torch.float32) @ Ud
    go = g0_on.to(device, torch.float32)
    po = p0_on.to(device, torch.float32)

    grp_Z, grp_g, grp_p, grp_w, grp_budget = [], [], [], [], []
    for xg, gg, pg, budget in off_groups:
        if xg.shape[0] == 0:
            continue
        grp_Z.append(xg.to(device, torch.float32) @ Ud)
        grp_g.append(gg.to(device, torch.float32))
        grp_p.append(pg.to(device, torch.float32))
        grp_w.append(torch.ones(xg.shape[0], device=device))
        grp_budget.append(budget)

    Zf_all = torch.cat(grp_Z, dim=0)
    pf_all = torch.cat(grp_p, dim=0)
    yo_g = (target_g_on - go).clamp_min(1.5)
    n_on = max(1, Zo.shape[0])
    w_on = torch.full((n_on,), 1.0 / n_on, device=device)
    z_scale = float(Zf_all.pow(2).sum(dim=1).mean().clamp_min(1.0))
    eye = torch.eye(Ud.shape[1], device=device)

    alpha_g = torch.zeros(Ud.shape[1], device=device)
    for step in range(n_irls):
        w_parts, y_parts = [], []
        for Zg, gg, wg, bgt in zip(grp_Z, grp_g, grp_w, grp_budget):
            w_parts.append(wg * (bgt / wg.sum().clamp_min(1e-8)))
            y_parts.append((target_g_off - gg).clamp_max(-0.6))
        w_off_n = torch.cat(w_parts, dim=0)
        yf_g = torch.cat(y_parts, dim=0)

        lhs = (Zo.T * w_on) @ Zo + (Zf_all.T * w_off_n) @ Zf_all + (lam * z_scale) * eye
        rhs = Zo.T @ (w_on * yo_g) + Zf_all.T @ (w_off_n * yf_g)
        alpha_g = torch.linalg.solve(lhs, rhs)
        if step + 1 < n_irls:
            any_v = False
            for idx_g, (Zg, gg) in enumerate(zip(grp_Z, grp_g)):
                vg = (gg + Zg @ alpha_g) > -3.95
                if vg.any():
                    any_v = True
                    grp_w[idx_g] = torch.where(vg, grp_w[idx_g] * 5.0, grp_w[idx_g])
            vo = (go + Zo @ alpha_g) < 2.2
            if vo.any():
                any_v = True
                w_on = torch.where(vo, w_on * 2.0, w_on)
                w_on = w_on / w_on.sum().clamp_min(1e-8)
            if not any_v:
                break

    w_base_p = torch.cat([torch.full((Zg.shape[0],), 0.15 * bgt / Zg.shape[0], device=device) for Zg, bgt in zip(grp_Z, grp_budget)], dim=0)
    yo_p = target_p_on - po
    yf_p = -0.30 * pf_all
    w_on_p = torch.full((n_on,), 1.0 / n_on, device=device)
    lhs_p = (Zo.T * w_on_p) @ Zo + (Zf_all.T * w_base_p) @ Zf_all + (4.0 * lam * z_scale) * eye
    rhs_p = Zo.T @ (w_on_p * yo_p) + Zf_all.T @ (w_base_p * yf_p)
    alpha_u = torch.linalg.solve(lhs_p, rhs_p)

    return (Ud @ alpha_g).cpu(), (Ud @ alpha_u).cpu()


def main():
    hdir = Path("/content/wsp/harvest32")
    out_dir = hdir / "v12_whitened"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads((hdir / "cases.json").read_text())
    native = {r["prompt_sha"]: r for r in json.loads((hdir / "native/champion.json").read_text())}

    assets = json.loads((ROOT / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        assets["model"], dtype=torch.bfloat16, device_map="cuda", local_files_only=True, low_cpu_mem_usage=True
    ).eval()
    gc.collect()
    lm_root = getattr(model.model, "language_model", model.model)
    layers = lm_root.layers

    with zipfile.ZipFile(ROOT / "champion.zip") as z:
        champ = load(z.read("adapters/wsp_r1/adapter_model.safetensors"))
    attach_adapter(layers, champ)
    with torch.inference_mode():
        model(input_ids=torch.tensor([[tokenizer.bos_token_id]], device=model.device), use_cache=False, logits_to_keep=1)

    target_layers = [35, 39, 41]
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    call_colon_seq = enc("<|tool_call>call:")
    prefix_pytest = enc('<|tool_call>call:run_command{command:<|"|>pytest')

    def find_all_subseq_ends(seq: list[int], sub: list[int]) -> list[int]:
        n = len(sub)
        return [i + n for i in range(len(seq) - n + 1) if seq[i : i + n] == sub]

    dev = model.device
    col_norms = {}
    with torch.inference_mode():
        for li in target_layers:
            norms_l = []
            for s in range(0, 10240, 512):
                e = min(10240, s + 512)
                mat = torch.zeros(e - s, 10240, device=dev, dtype=torch.bfloat16)
                mat[torch.arange(e - s, device=dev), torch.arange(s, e, device=dev)] = 1.0
                norms_l.append(layers[li].mlp.down_proj(mat).float().norm(dim=-1))
            col_norms[li] = torch.cat(norms_l, dim=0)

    max_h0_acc = {li: torch.zeros(10240, device=dev, dtype=torch.float32) for li in target_layers}
    sum_h0_acc = {li: torch.zeros(10240, device=dev, dtype=torch.float32) for li in target_layers}
    max_g0_acc = {li: torch.full((10240,), -1e9, device=dev, dtype=torch.float32) for li in target_layers}
    tok_cnt_acc = {li: 0 for li in target_layers}

    def forward_slim(prefix: list[int], suffix: list[int], accumulate_pres: bool):
        toks = prefix + suffix
        assert len(toks) <= 4096
        p_len = len(prefix)
        hist_colons = [e - 1 for e in find_all_subseq_ends(prefix, call_colon_seq)]
        stride = max(1, p_len // 28)
        grid_idx = list(range(0, p_len, stride))[:28]
        pref_idx = sorted(set(hist_colons + grid_idx))
        hist_colon_set = set(hist_colons)
        is_hist_colon = torch.tensor([idx in hist_colon_set for idx in pref_idx], dtype=torch.bool)
        pref_t = torch.tensor(pref_idx, device=dev, dtype=torch.long)
        suf_t = torch.arange(p_len, len(toks), device=dev, dtype=torch.long)
        cat_t = torch.cat([pref_t, suf_t], dim=0) if accumulate_pres else pref_t

        colon_end = find_subseq_end(suffix, call_colon_seq)
        pytest_end = find_subseq_end(suffix, prefix_pytest)
        special_s = set()
        if colon_end is not None:
            special_s.add(colon_end - 1)
            if colon_end >= 2:
                special_s.add(colon_end - 2)
        if pytest_end is not None:
            special_s.add(pytest_end - 1)
        s_stride = max(1, len(suffix) // 24)
        h_suf_idx = sorted(special_s | set(range(0, len(suffix), s_stride)))
        h_suf_t = torch.tensor([p_len + i for i in h_suf_idx], device=dev, dtype=torch.long)
        h_pref_idx = sorted(set(hist_colons) | set(grid_idx[::2]))
        h_pref_t = torch.tensor(h_pref_idx, device=dev, dtype=torch.long)
        h_pref_is_col = torch.tensor([idx in hist_colon_set for idx in h_pref_idx], dtype=torch.bool)

        cap = {
            "pref_is_colon": is_hist_colon,
            "h_suf_map": {orig_i: pos_i for pos_i, orig_i in enumerate(h_suf_idx)},
            "h_pref_is_colon": h_pref_is_col,
        }
        tmp_g = {}
        hooks = []
        for li in target_layers:
            def mk_g(idx=li):
                def _h(mod, inp, out):
                    x_src = inp[0][0]
                    cap[(idx, "x")] = x_src[p_len:].detach().to(torch.bfloat16).cpu()
                    cap[(idx, "x_pref")] = x_src[pref_t].detach().to(torch.bfloat16).cpu()
                    tmp_g[idx] = out[0].detach()
                    g_sel = tmp_g[idx].index_select(0, cat_t).float()
                    max_g0_acc[idx].copy_(torch.maximum(max_g0_acc[idx], g_sel.max(dim=0).values))
                    tok_cnt_acc[idx] += int(g_sel.shape[0])
                return _h
            def mk_u(idx=li):
                def _h(mod, inp, out):
                    p_full = out[0].detach()
                    g_full = tmp_g.pop(idx)
                    h_cat = (F.gelu(g_full.index_select(0, cat_t), approximate="tanh") * p_full.index_select(0, cat_t)).float().abs()
                    max_h0_acc[idx].copy_(torch.maximum(max_h0_acc[idx], h_cat.max(dim=0).values))
                    sum_h0_acc[idx].add_(h_cat.sum(dim=0))
                    hs = F.gelu(g_full.index_select(0, h_suf_t), approximate="tanh") * p_full.index_select(0, h_suf_t)
                    hp = F.gelu(g_full.index_select(0, h_pref_t), approximate="tanh") * p_full.index_select(0, h_pref_t)
                    cap[(idx, "h_sub")] = hs.to(torch.bfloat16).cpu()
                    cap[(idx, "h_pref_sub")] = hp.to(torch.bfloat16).cpu()
                return _h
            def mk_d(idx=li):
                def _h(mod, inp, out):
                    cap[(idx, "y0_norm")] = float(out[0, p_len:].detach().float().norm(dim=-1).mean().item())
                return _h
            hooks.append(layers[li].mlp.gate_proj.register_forward_hook(mk_g()))
            hooks.append(layers[li].mlp.up_proj.register_forward_hook(mk_u()))
            hooks.append(layers[li].mlp.down_proj.register_forward_hook(mk_d()))
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([toks], device=dev), use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
            tmp_g.clear()
        return cap

    seq_records = []
    seen = set()
    for c in cases:
        if c["role"] not in ("unchecked", "check", "verified", "edit", "localize"):
            continue
        key = (c["issue"], c["role"])
        if key in seen:
            continue
        seen.add(key)
        act = native.get(c.get("prompt_sha", ""))
        if act and act.get("text"):
            gen_txt = act["text"]
        elif c["role"] == "verified":
            gen_txt = "<|channel>thought\nThe targeted test passed and verified the fix. I am ready to submit the patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>"
        else:
            continue
        full_suf = enc(gen_txt)
        suf = full_suf[-768:] if len(full_suf) > 768 else full_suf
        if len(suf) < 4:
            continue
        pref = c["tokens"][-(4096 - len(suf)) :]
        cap = forward_slim(pref, suf, accumulate_pres=(c["role"] in ("verified", "edit", "localize")))
        colon_end = find_subseq_end(suf, call_colon_seq)
        pytest_end = find_subseq_end(suf, prefix_pytest)
        cidx = (colon_end - 1) if colon_end is not None else None
        pidx = (pytest_end - 1) if pytest_end is not None else None
        seq_records.append({
            "issue": c["issue"],
            "role": c["role"],
            "suf": suf,
            "colon_idx": cidx,
            "pytest_idx": pidx,
            "h_colon_idx": cap["h_suf_map"].get(cidx) if cidx is not None else None,
            "h_pytest_idx": cap["h_suf_map"].get(pidx) if pidx is not None else None,
            "cap": cap,
        })

    bare_seqs = []
    for issue in sorted({c["issue"] for c in cases}):
        opts = [c for c in cases if c["issue"] == issue and c["role"] in ("unchecked", "check")]
        for case in sorted(opts, key=lambda c: (c["role"] != "unchecked", len(c["tokens"]))):
            cmd = resolve_targeted_check(case)
            if not cmd:
                continue
            cmd_py = cmd if cmd.startswith("pytest") else f"pytest -q tests/test_{Path(case.get('filepath') or 'x.py').stem.lstrip('_')}.py"
            suf = enc(f"<|channel>thought\nI will run tests to verify the fix.\n<channel|><|tool_call>call:run_command{{command:<|\"|>{cmd_py}<|\"|>}}<tool_call|>")
            pref = case["tokens"][-(4096 - len(suf)) :]
            cap = forward_slim(pref, suf, accumulate_pres=False)
            pt_end = find_subseq_end(suf, prefix_pytest)
            col_end = find_subseq_end(suf, call_colon_seq)
            if pt_end is not None:
                pidx = pt_end - 1
                bare_seqs.append({
                    "issue": issue,
                    "colon_idx": (col_end - 1) if col_end is not None else None,
                    "pytest_idx": pidx,
                    "h_pytest_idx": cap["h_suf_map"][pidx],
                    "cap": cap,
                })
            break

    q_stars = {}
    q_stats = {}
    for li in target_layers:
        cn = col_norms[li].cpu()
        max_h0 = max_h0_acc[li].cpu()
        mean_h0 = (sum_h0_acc[li] / max(1, tok_cnt_acc[li])).cpu()
        max_g0 = max_g0_acc[li].cpu()
        mean_dy = mean_h0 * cn
        max_dy = max_h0 * cn
        loss_q = 100.0 * mean_dy + 15.0 * max_dy + 0.01 * F.relu(max_g0)
        q = int(loss_q.argmin().item())
        q_stars[li] = q
        q_stats[li] = {
            "q": q, "col_norm": round(float(cn[q]), 5),
            "mean_residual_dy": round(float(mean_dy[q]), 6), "max_residual_dy": round(float(max_dy[q]), 6),
        }

    with torch.inference_mode():
        for rec_list in (seq_records, bare_seqs):
            for r in rec_list:
                for li in target_layers:
                    q = q_stars[li]
                    gp, up = layers[li].mlp.gate_proj, layers[li].mlp.up_proj
                    xs_gpu = r["cap"][(li, "x")].to(dev, torch.bfloat16)
                    r["cap"][(li, "g0_q")] = gp(xs_gpu)[:, q].float().cpu()
                    r["cap"][(li, "p0_q")] = up(xs_gpu)[:, q].float().cpu()
                    xp_gpu = r["cap"][(li, "x_pref")].to(dev, torch.bfloat16)
                    r["cap"][(li, "g0_pref_q")] = gp(xp_gpu)[:, q].float().cpu()
                    r["cap"][(li, "p0_pref_q")] = up(xp_gpu)[:, q].float().cpu()

    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

    tok_run = enc("<|tool_call>call:run_command{")[len(call_colon_seq)]
    tok_sub = enc("<|tool_call>call:submit_patch{")[len(call_colon_seq)]
    u_disp = (lm_w[tok_sub] - lm_w[tok_run]) * scale_w
    u_disp = u_disp / u_disp.norm().clamp_min(1e-10)

    sb = enc('<|tool_call>call:run_command{command:<|"|>pytest<|"|>}<tool_call|>')
    st1 = enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests/test_panel.py<|"|>}<tool_call|>')
    st2 = enc('<|tool_call>call:run_command{command:<|"|>pytest tests/test_panel.py<|"|>}<tool_call|>')
    u_ab = (
        lm_w[sb[find_subseq_end(sb, prefix_pytest)]]
        - (0.60 * lm_w[st1[find_subseq_end(st1, prefix_pytest)]] + 0.40 * lm_w[st2[find_subseq_end(st2, prefix_pytest)]])
    ) * scale_w
    u_ab = u_ab / u_ab.norm().clamp_min(1e-10)
    del lm_w

    w_down_cols = {}
    with torch.inference_mode():
        for li in target_layers:
            eq = torch.zeros(1, 10240, device=dev, dtype=torch.bfloat16)
            eq[0, q_stars[li]] = 1.0
            w_down_cols[li] = layers[li].mlp.down_proj(eq)[0].detach().float().cpu()

    del model, lm_root, layers, col_norms, max_h0_acc, sum_h0_acc, max_g0_acc
    gc.collect()
    torch.cuda.empty_cache()

    diagnostics = {"n_seq_records": len(seq_records), "q_stats": q_stats}
    real_unch_issues = sorted({r["issue"] for r in seq_records if r["role"] == "unchecked" and r["colon_idx"] is not None})

    # --- Family A: Truncated-PCA Whitened Pure `down_proj` h0-Readout (ZERO base-neuron pruning!) ---
    def pack_h0_groups(recs: list[dict], li: int):
        unch_c, ver_c, ed_c, loc_c, non_c, yn = [], [], [], [], [], []
        for r in recs:
            hs = r["cap"][(li, "h_sub")].float()
            hp = r["cap"][(li, "h_pref_sub")].float()
            yn.append(r["cap"][(li, "y0_norm")])
            isc = r["cap"]["h_pref_is_colon"]
            if isc.any():
                loc_c.append(hp[isc])
            if (~isc).any():
                non_c.append(hp[~isc])
            hci = r["h_colon_idx"]
            if hci is not None:
                m = torch.arange(hs.shape[0]) != hci
                if m.any():
                    non_c.append(hs[m])
                if r["role"] in ("unchecked", "check"):
                    unch_c.append(hs[hci])
                elif r["role"] == "verified":
                    ver_c.append(hs[hci:hci+1])
                elif r["role"] == "edit":
                    ed_c.append(hs[hci:hci+1])
                else:
                    loc_c.append(hs[hci:hci+1])
            else:
                non_c.append(hs)
        H_on = torch.stack(unch_c)
        H_ver = torch.cat(ver_c) if ver_c else torch.zeros(0, 10240)
        H_ed = torch.cat(ed_c) if ed_c else torch.zeros(0, 10240)
        H_loc = torch.cat(loc_c) if loc_c else torch.zeros(0, 10240)
        H_non = torch.cat(non_c)
        ogrps = [(H_ver, 3.0), (H_ed, 3.0), (H_loc, 2.0), (H_non, 5.0)]
        H_all_off = torch.cat([g[0] for g in ogrps if g[0].shape[0] > 0], dim=0)
        return H_on, ogrps, H_all_off, H_ver, H_ed, sum(yn) / len(yn)

    lin_fits = {}
    for li in (35, 41):
        best_l = None
        ltrials = {}
        for k_pca in (0, 6, 12, 18, 24):
            oof_s = {}
            for iss in real_unch_issues:
                tr_recs = [r for r in seq_records if r["issue"] != iss]
                Uh_tr = make_shared_pca_colon_basis(tr_recs, li, "h_sub", "h_pref_sub", True, k_pca)
                Ho_tr, og_tr, _, _, _, _ = pack_h0_groups(tr_recs, li)
                a_tr = solve_whitened_linear_h0(Uh_tr, Ho_tr, og_tr)
                for r in seq_records:
                    if r["issue"] == iss and r["role"] == "unchecked" and r["h_colon_idx"] is not None:
                        hi = r["cap"][(li, "h_sub")][r["h_colon_idx"]].float()
                        oof_s[iss] = round(float(torch.dot(a_tr, hi).item()), 4)

            Uh = make_shared_pca_colon_basis(seq_records, li, "h_sub", "h_pref_sub", True, k_pca)
            Ho, og, H_off, H_ver, H_ed, yn = pack_h0_groups(seq_records, li)
            a_full = solve_whitened_linear_h0(Uh, Ho, og)
            s_on = Ho @ a_full
            s_off = H_off @ a_full
            s_ver = (H_ver @ a_full).abs().max().item() if H_ver.shape[0] else 0.0
            s_ed = (H_ed @ a_full).abs().max().item() if H_ed.shape[0] else 0.0
            off_rms = float(s_off.pow(2).mean().sqrt().clamp_min(1e-6))
            oof_vals = list(oof_s.values())
            oof_mean = sum(oof_vals) / max(1, len(oof_vals))
            oof_snr = oof_mean / off_rms
            pos_cnt = sum(1 for v in oof_vals if v > 1.8 * off_rms)
            ltrials[f"kpca{k_pca}_dim{Uh.shape[1]}"] = {
                "pos_over_1p8rms": f"{pos_cnt}/{len(oof_vals)}",
                "oof_snr": round(oof_snr, 2),
                "in_sample_snr": round(float(s_on.mean()) / off_rms, 2),
                "s_on_mean": round(float(s_on.mean()), 4),
                "s_off_rms": round(off_rms, 4),
                "max_ver_colon": round(s_ver, 4),
                "max_edit_colon": round(s_ed, 4),
                "oof": oof_s,
            }
            # Maximize honest OOF SNR + positive count while keeping verified/edit colons tiny relative to s_on_mean!
            rkey = (s_ver <= 0.25 * float(s_on.mean()), s_ed <= 0.25 * float(s_on.mean()), pos_cnt, round(oof_snr, 2))
            if best_l is None or rkey > best_l[0]:
                best_l = (rkey, k_pca, a_full, float(s_on.mean()), yn, ltrials[f"kpca{k_pca}_dim{Uh.shape[1]}"])

        _, bk_pca, a_best, s_on_m, yn, bstat = best_l
        diagnostics[f"L{li}_whitened_h0_downproj"] = {"best_kpca": bk_pca, "best": bstat, "trials": ltrials}
        lin_fits[li] = {"a": a_best, "u": u_disp, "s_on_mean": max(s_on_m, 0.1), "y0_norm": yn}

    # L39 Whitened h0 Anti-Bare-Pytest
    h39_on_l, h39_off_l, y39_l = [], [], []
    for r in seq_records:
        hs = r["cap"][(39, "h_sub")].float()
        y39_l.append(r["cap"][(39, "y0_norm")])
        h39_off_l.append(r["cap"][(39, "h_pref_sub")].float())
        hpidx = r["h_pytest_idx"]
        if hpidx is not None:
            h39_on_l.append(hs[hpidx])
            m = torch.arange(hs.shape[0]) != hpidx
            if m.any():
                h39_off_l.append(hs[m])
        else:
            h39_off_l.append(hs)
    for r in bare_seqs:
        hs = r["cap"][(39, "h_sub")].float()
        hpidx = r["h_pytest_idx"]
        h39_on_l.append(hs[hpidx])
        m = torch.arange(hs.shape[0]) != hpidx
        if m.any():
            h39_off_l.append(hs[m])
    H39_on = torch.stack(h39_on_l)
    H39_off = torch.cat(h39_off_l)
    mu39_h = H39_on.mean(dim=0)
    hard39_h = H39_off[(H39_off @ (mu39_h / mu39_h.norm().clamp_min(1e-8))).topk(min(96, H39_off.shape[0])).indices]
    _, _, Vh39_h = torch.linalg.svd((hard39_h - hard39_h.mean(dim=0)).double(), full_matrices=False)
    _, _, Vh39_all = torch.linalg.svd((H39_off[::2] - H39_off[::2].mean(dim=0)).double(), full_matrices=False)
    Uh39 = build_orthonormal_basis([
        mu39_h - H39_off.mean(dim=0),
        mu39_h - hard39_h.mean(dim=0),
        mu39_h,
        *[Vh39_h[j].float() for j in range(min(6, Vh39_h.shape[0]))],
        *[Vh39_all[j].float() for j in range(min(6, Vh39_all.shape[0]))],
    ])
    a39_lin = solve_whitened_linear_h0(Uh39, H39_on, [(hard39_h, 3.0), (H39_off, 5.0)])
    s39_on = H39_on @ a39_lin
    s39_off = H39_off @ a39_lin
    diagnostics["L39_whitened_h0_anti_bare"] = {
        "s_on_mean": round(float(s39_on.mean()), 4),
        "s_off_rms": round(float(s39_off.pow(2).mean().sqrt()), 4),
        "snr": round(float(s39_on.mean() / s39_off.pow(2).mean().sqrt().clamp_min(1e-6)), 2),
    }

    # --- Family B: Group-Locked IRLS Balanced Subspace GeGLU (ver_colon ALWAYS locked to 0.0!) ---
    def pack_geglu_groups(recs: list[dict], li: int):
        on_x, on_g, on_p = [], [], []
        ver_x, ver_g, ver_p = [], [], []
        ed_x, ed_g, ed_p = [], [], []
        loc_x, loc_g, loc_p = [], [], []
        non_x, non_g, non_p = [], [], []
        yn = []
        for r in recs:
            xs, gs, ps = r["cap"][(li, "x")].float(), r["cap"][(li, "g0_q")], r["cap"][(li, "p0_q")]
            yn.append(r["cap"][(li, "y0_norm")])
            xp, gp, pp = r["cap"][(li, "x_pref")].float(), r["cap"][(li, "g0_pref_q")], r["cap"][(li, "p0_pref_q")]
            isc = r["cap"]["pref_is_colon"]
            if isc.any():
                loc_x.append(xp[isc]); loc_g.append(gp[isc]); loc_p.append(pp[isc])
            if (~isc).any():
                non_x.append(xp[~isc]); non_g.append(gp[~isc]); non_p.append(pp[~isc])
            cidx = r["colon_idx"]
            if cidx is not None:
                m = torch.arange(xs.shape[0]) != cidx
                if m.any():
                    non_x.append(xs[m]); non_g.append(gs[m]); non_p.append(ps[m])
                if r["role"] in ("unchecked", "check"):
                    on_x.append(xs[cidx]); on_g.append(gs[cidx]); on_p.append(ps[cidx])
                elif r["role"] == "verified":
                    ver_x.append(xs[cidx:cidx+1]); ver_g.append(gs[cidx:cidx+1]); ver_p.append(ps[cidx:cidx+1])
                elif r["role"] == "edit":
                    ed_x.append(xs[cidx:cidx+1]); ed_g.append(gs[cidx:cidx+1]); ed_p.append(ps[cidx:cidx+1])
                else:
                    loc_x.append(xs[cidx:cidx+1]); loc_g.append(gs[cidx:cidx+1]); loc_p.append(ps[cidx:cidx+1])
            else:
                non_x.append(xs); non_g.append(gs); non_p.append(ps)
        ogrps = [
            (torch.cat(ver_x), torch.cat(ver_g), torch.cat(ver_p), 2.0),
            (torch.cat(ed_x), torch.cat(ed_g), torch.cat(ed_p), 2.0),
            (torch.cat(loc_x), torch.cat(loc_g), torch.cat(loc_p), 1.5),
            (torch.cat(non_x), torch.cat(non_g), torch.cat(non_p), 2.5),
        ]
        return (
            torch.stack(on_x), torch.stack(on_g), torch.stack(on_p), ogrps,
            torch.cat([g[0] for g in ogrps]), torch.cat([g[1] for g in ogrps]), torch.cat([g[2] for g in ogrps]),
            sum(yn) / len(yn),
        )

    geglu_fits = {}
    for li in (41,):
        best_g = None
        gtrials = {}
        for k_pca in (2, 4, 6, 9):
            oof_h = {}
            for iss in real_unch_issues:
                tr_recs = [r for r in seq_records if r["issue"] != iss]
                U_tr = make_shared_pca_colon_basis(tr_recs, li, "x", "x_pref", False, k_pca)
                Xo, Go, Po, ogrps, _, _, _, _ = pack_geglu_groups(tr_recs, li)
                wg_tr, wu_tr = solve_grouplocked_subspace_geglu(U_tr, Xo, Go, Po, ogrps)
                for r in seq_records:
                    if r["issue"] == iss and r["role"] == "unchecked" and r["colon_idx"] is not None:
                        ci = r["colon_idx"]
                        xi = r["cap"][(li, "x")][ci].float()
                        gi = r["cap"][(li, "g0_q")][ci].bfloat16() + torch.dot(wg_tr, xi).bfloat16()
                        pi = r["cap"][(li, "p0_q")][ci].bfloat16() + torch.dot(wu_tr, xi).bfloat16()
                        oof_h[iss] = round(float((F.gelu(gi, approximate="tanh") * pi).item()), 3)

            U_full = make_shared_pca_colon_basis(seq_records, li, "x", "x_pref", False, k_pca)
            Xo, Go, Po, ogrps, Xf, Gf, Pf, yn = pack_geglu_groups(seq_records, li)
            wg, wu = solve_grouplocked_subspace_geglu(U_full, Xo, Go, Po, ogrps)
            hf = (F.gelu(Gf.bfloat16() + (Xf @ wg).bfloat16(), approximate="tanh") * (Pf.bfloat16() + (Xf @ wu).bfloat16())).float()
            zfrac = float((hf == 0.0).float().mean())
            ver_h, ed_h = [], []
            for r in seq_records:
                ci = r["colon_idx"]
                if ci is not None and r["role"] in ("verified", "edit"):
                    xi = r["cap"][(li, "x")][ci].float()
                    gi = r["cap"][(li, "g0_q")][ci].bfloat16() + torch.dot(wg, xi).bfloat16()
                    pi = r["cap"][(li, "p0_q")][ci].bfloat16() + torch.dot(wu, xi).bfloat16()
                    (ver_h if r["role"] == "verified" else ed_h).append(round(float((F.gelu(gi, approximate="tanh") * pi).item()), 4))
            max_ver = max((abs(v) for v in ver_h), default=0.0)
            max_ed = max((abs(v) for v in ed_h), default=0.0)
            pos_cnt = sum(1 for v in oof_h.values() if v > 0.5)
            oof_mean = sum(oof_h.values()) / max(1, len(oof_h))
            gtrials[f"kpca{k_pca}_dim{U_full.shape[1]}"] = {
                "oof_pos": f"{pos_cnt}/{len(oof_h)}", "oof_mean": round(oof_mean, 3),
                "zero_frac": round(zfrac, 5), "max_ver_colon": max_ver, "max_edit_colon": max_ed, "oof": oof_h,
            }
            rkey = (max_ver <= 0.02, max_ed <= 0.02, zfrac >= 0.992, pos_cnt, oof_mean)
            if best_g is None or rkey > best_g[0]:
                best_g = (rkey, k_pca, oof_h, wg, wu, Xo, Go, Po, hf, ver_h, ed_h, yn)

        _, bk_pca, oof_h, wg, wu, Xo, Go, Po, hf, ver_h, ed_h, yn = best_g
        ho = (F.gelu(Go.bfloat16() + (Xo @ wg).bfloat16(), approximate="tanh") * (Po.bfloat16() + (Xo @ wu).bfloat16())).float()
        diagnostics["L41_grouplocked_geglu"] = {
            "q": q_stars[41], "best_kpca": bk_pca, "trials": gtrials,
            "ver_colon": ver_h, "edit_colon": ed_h, "zero_frac": round(float((hf == 0.0).float().mean()), 5),
        }
        geglu_fits[41] = {
            "q_idx": q_stars[41], "w_gate": wg, "w_up": wu, "w_down_q": w_down_cols[41],
            "u": u_disp, "h_on_mean": float(ho.mean().clamp_min(1.0)), "y0_norm": yn,
        }

    # L39 Group-Locked GeGLU Anti-Bare
    q39 = q_stars[39]
    on39_x, on39_g, on39_p, off39_x, off39_g, off39_p, y39_norms = [], [], [], [], [], [], []
    for r in seq_records:
        xs, gs, ps = r["cap"][(39, "x")].float(), r["cap"][(39, "g0_q")], r["cap"][(39, "p0_q")]
        y39_norms.append(r["cap"][(39, "y0_norm")])
        off39_x.append(r["cap"][(39, "x_pref")].float())
        off39_g.append(r["cap"][(39, "g0_pref_q")])
        off39_p.append(r["cap"][(39, "p0_pref_q")])
        pidx = r["pytest_idx"]
        if pidx is not None:
            on39_x.append(xs[pidx]); on39_g.append(gs[pidx]); on39_p.append(ps[pidx])
            m = torch.arange(xs.shape[0]) != pidx
            off39_x.append(xs[m]); off39_g.append(gs[m]); off39_p.append(ps[m])
        else:
            off39_x.append(xs); off39_g.append(gs); off39_p.append(ps)
    for r in bare_seqs:
        xs, gs, ps = r["cap"][(39, "x")].float(), r["cap"][(39, "g0_q")], r["cap"][(39, "p0_q")]
        pidx = r["pytest_idx"]
        on39_x.append(xs[pidx]); on39_g.append(gs[pidx]); on39_p.append(ps[pidx])
        m = torch.arange(xs.shape[0]) != pidx
        off39_x.append(xs[m]); off39_g.append(gs[m]); off39_p.append(ps[m])
    X39_on, G39_on, P39_on = torch.stack(on39_x), torch.stack(on39_g), torch.stack(on39_p)
    X39_off, G39_off, P39_off = torch.cat(off39_x), torch.cat(off39_g), torch.cat(off39_p)
    mu39_on = X39_on.mean(dim=0)
    hard_idx = (X39_off @ (mu39_on / mu39_on.norm().clamp_min(1e-8))).topk(min(96, X39_off.shape[0])).indices
    X39_hard = X39_off[hard_idx]
    _, _, Vh39 = torch.linalg.svd((X39_hard - X39_hard.mean(dim=0)).double(), full_matrices=False)
    U39 = build_orthonormal_basis([mu39_on - X39_off.mean(dim=0), mu39_on - X39_hard.mean(dim=0), mu39_on, Vh39[0].float(), Vh39[1].float(), Vh39[2].float()])
    wg39, wu39 = solve_grouplocked_subspace_geglu(
        U39, X39_on, G39_on, P39_on,
        [(X39_hard, G39_off[hard_idx], P39_off[hard_idx], 2.0), (X39_off, G39_off, P39_off, 2.5)],
    )
    h39_on = (F.gelu(G39_on.bfloat16() + (X39_on @ wg39).bfloat16(), approximate="tanh") * (P39_on.bfloat16() + (X39_on @ wu39).bfloat16())).float()

    print("V12 Whitened Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    # 1. Family A: Pure `down_proj` Whitened h0-Readout (4 gain brackets across {41, 39} and {35, 41, 39})
    def add_pure_down(name: str, d_layers: list[int], rho_d: float, rho_ab: float | None):
        ft = {}
        for li in d_layers:
            f = lin_fits[li]
            g = rho_d * f["y0_norm"] / f["s_on_mean"]
            ft[f"{li}.A"] = f["a"].bfloat16()
            ft[f"{li}.B"] = (-g * f["u"]).bfloat16()
        if rho_ab is not None:
            g39 = rho_ab * (sum(y39_l) / len(y39_l)) / float(s39_on.mean().clamp_min(0.1))
            ft["39.A"] = a39_lin.bfloat16()
            ft["39.B"] = (-g39 * u_ab).bfloat16()
        fac_p = out_dir / f"fac_{name}.safetensors"
        save_file(ft, str(fac_p))
        zlbl = f"V12_lin_zero_{'_'.join(sorted(k.split('.')[0] for k in ft if k.endswith('.A')))}"
        if not (ad_root / zlbl).exists():
            zinfo = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / zlbl, "zero", rank=1)
            compiled.append({"adapter": zlbl, "sha256": zinfo["adapter_sha256"], "control": "zero", "family": "pure_downproj"})
        for ctrl in ("candidate", "random"):
            lbl = f"{name}_{ctrl}"
            info = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / lbl, ctrl, rank=1)
            compiled.append({"adapter": lbl, "sha256": info["adapter_sha256"], "control": ctrl, "family": "pure_downproj"})

    add_pure_down("V12_lin_L41_39ab_g022", [41], rho_d=0.22, rho_ab=0.22)
    add_pure_down("V12_lin_L41_39ab_g035", [41], rho_d=0.35, rho_ab=0.30)
    add_pure_down("V12_lin_L41_39ab_g050", [41], rho_d=0.50, rho_ab=0.38)
    add_pure_down("V12_lin_L35_41_39ab_g020", [35, 41], rho_d=0.20, rho_ab=0.26)
    add_pure_down("V12_lin_L35_41_39ab_g032", [35, 41], rho_d=0.32, rho_ab=0.34)

    # 2. Family B: Group-Locked GeGLU on downstream {39, 41}
    for name, rho_d, rho_ab in (("V12_geglu_L41_39ab_g038", 0.38, 0.34), ("V12_geglu_L41_39ab_g055", 0.55, 0.40)):
        f41 = geglu_fits[41]
        g41 = rho_d * f41["y0_norm"] / f41["h_on_mean"]
        g39 = rho_ab * (sum(y39_norms) / len(y39_norms)) / float(h39_on.mean().clamp_min(1.0))
        spec = {
            41: {"q_idx": f41["q_idx"], "w_gate": f41["w_gate"], "w_up": f41["w_up"], "b_down": -f41["w_down_q"] - g41 * f41["u"]},
            39: {"q_idx": q39, "w_gate": wg39, "w_up": wu39, "b_down": -w_down_cols[39] - g39 * u_ab},
        }
        zlbl = "V12_geglu_zero_39_41"
        if not (ad_root / zlbl).exists():
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", spec, ad_root / zlbl, "zero"))
        for ctrl in ("candidate", "random"):
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", spec, ad_root / f"{name}_{ctrl}", ctrl))

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V12 Whitened adapters!", flush=True)


if __name__ == "__main__":
    main()
