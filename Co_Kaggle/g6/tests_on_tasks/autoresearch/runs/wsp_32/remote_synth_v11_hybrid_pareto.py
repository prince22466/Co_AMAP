"""Synthesize V11 Hybrid Pareto Candidates:
- Family B (GeGLU dim=5..7, k_svd in {0,1}): Combines V9's 4/5 honest unseen-issue OOF (`dim=5..6`)
  with V10's balanced group weights (`verified_colon=0.0`, `edit_colon=0.0`, `3/3` verified preserved!)
  AND selects neuron q* by actual residual output norm `mean_|h0[:, q]| * ||W_down[:, q]||`.
- Family A (Pure `down_proj` Decoupled h0-Subspace Readout, r=1): Zero `gate_proj`/`up_proj` edits ->
  ZERO base-neuron pruning on edit/thought tokens, fitting `a_down` in 10,240-D GeGLU h0-space!
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


def make_lowdim_colon_basis(records: list[dict], li: int, feat_key: str, pref_key: str, mode: str) -> torch.Tensor:
    """Build ultra-low-dim (K = 5, 6, or 7) task-invariant basis strictly inside LOO CV!"""
    unch_col, ver_col, edit_col, loc_col, all_col, non_col, prev_tok = [], [], [], [], [], [], []
    for r in records:
        xs = r["cap"][(li, feat_key)].float()
        cidx = r["colon_idx"]
        if cidx is not None:
            xc = xs[cidx]
            all_col.append(xc)
            if r["role"] in ("unchecked", "check"):
                unch_col.append(xc)
            elif r["role"] == "verified":
                ver_col.append(xc)
            elif r["role"] == "edit":
                edit_col.append(xc)
            elif r["role"] == "localize":
                loc_col.append(xc)
            m = torch.arange(xs.shape[0]) != cidx
            if m.any():
                non_col.append(xs[m].mean(dim=0))
                if cidx >= 1:
                    prev_tok.append(xs[cidx - 1])
        else:
            non_col.append(xs.mean(dim=0))
        if (li, pref_key) in r["cap"]:
            non_col.append(r["cap"][(li, pref_key)].float().mean(dim=0))

    X_col = torch.stack(all_col)
    mu_col = X_col.mean(dim=0)
    mu_unch = torch.stack(unch_col).mean(dim=0)
    mu_ver = torch.stack(ver_col).mean(dim=0)
    mu_edit = torch.stack(edit_col).mean(dim=0)
    mu_loc = torch.stack(loc_col).mean(dim=0)
    mu_non = torch.stack(non_col).mean(dim=0)
    mu_prev = torch.stack(prev_tok).mean(dim=0) if prev_tok else mu_non

    # Core 5D basis (achieved 4/5 honest unseen-issue OOF in V9!)
    vecs = [
        mu_col - mu_non,       # 1. `<|tool_call>call:` colon vs thought/arg/prefill
        mu_unch - mu_ver,      # 2. Unchecked vs verified (`test passed`) at `:`
        mu_unch - mu_edit,     # 3. Post-edit vs pre-edit (`edit_file`) at `:`
        mu_unch - mu_loc,      # 4. Post-edit vs initial localization at `:`
        mu_unch,               # 5. Radial centering direction
    ]
    if mode in ("d6", "d7_svd1"):
        vecs.append(mu_col - mu_prev)  # 6. `:` vs immediately preceding `call` token
    if mode == "d7_svd1" and X_col.shape[0] > 4:
        _, _, Vh = torch.linalg.svd((X_col - mu_col).double(), full_matrices=False)
        vecs.append(Vh[0].float())     # 7. Single dominant colon-state PC (no high-dim overfitting!)

    return build_orthonormal_basis(vecs)


def solve_balanced_subspace_geglu(
    U: torch.Tensor,
    x_on: torch.Tensor,
    g0_on: torch.Tensor,
    p0_on: torch.Tensor,
    off_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    target_g_on: float = 3.6,
    target_g_off: float = -4.7,
    target_p_on: float = 3.4,
    lam: float = 1e-3,
    n_irls: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ud = U.to(device, torch.float32)
    Zo = x_on.to(device, torch.float32) @ Ud
    go = g0_on.to(device, torch.float32)
    po = p0_on.to(device, torch.float32)

    Zf_list, gf_list, pf_list, wf_init_list = [], [], [], []
    for xg, gg, pg, grp_weight in off_groups:
        if xg.shape[0] == 0:
            continue
        Zf_list.append(xg.to(device, torch.float32) @ Ud)
        gf_list.append(gg.to(device, torch.float32))
        pf_list.append(pg.to(device, torch.float32))
        wf_init_list.append(torch.full((xg.shape[0],), grp_weight / xg.shape[0], device=device))

    Zf = torch.cat(Zf_list, dim=0)
    gf = torch.cat(gf_list, dim=0)
    pf = torch.cat(pf_list, dim=0)
    w_off_base = torch.cat(wf_init_list, dim=0)
    total_off_weight = float(w_off_base.sum().item())

    yo_g = (target_g_on - go).clamp_min(1.5)
    yf_g = (target_g_off - gf).clamp_max(-0.5)

    n_on = max(1, Zo.shape[0])
    w_on = torch.full((n_on,), 1.0 / n_on, device=device)
    w_off = w_off_base.clone()
    z_scale = float(Zf.pow(2).sum(dim=1).mean().clamp_min(1.0))
    eye = torch.eye(Ud.shape[1], device=device)

    alpha_g = torch.zeros(Ud.shape[1], device=device)
    for step in range(n_irls):
        w_off_n = w_off * (total_off_weight / w_off.sum().clamp_min(1e-8))
        lhs = (Zo.T * w_on) @ Zo + (Zf.T * w_off_n) @ Zf + (lam * z_scale) * eye
        rhs = Zo.T @ (w_on * yo_g) + Zf.T @ (w_off_n * yf_g)
        alpha_g = torch.linalg.solve(lhs, rhs)
        if step + 1 < n_irls:
            pred_gf = gf + (Zf @ alpha_g)
            pred_go = go + (Zo @ alpha_g)
            viol_off = pred_gf > -3.90
            viol_on = pred_go < 2.2
            if not viol_off.any() and not viol_on.any():
                break
            w_off = torch.where(viol_off, w_off * 5.0, w_off)
            w_on = torch.where(viol_on, w_on * 2.2, w_on)

    yo_p = target_p_on - po
    yf_p = -0.30 * pf
    w_on_p = torch.full((n_on,), 1.0 / n_on, device=device)
    w_off_p = w_off_base * (0.55 / total_off_weight)
    lhs_p = (Zo.T * w_on_p) @ Zo + (Zf.T * w_off_p) @ Zf + (4.0 * lam * z_scale) * eye
    rhs_p = Zo.T @ (w_on_p * yo_p) + Zf.T @ (w_off_p * yf_p)
    alpha_u = torch.linalg.solve(lhs_p, rhs_p)

    return (Ud @ alpha_g).cpu(), (Ud @ alpha_u).cpu()


def solve_balanced_linear_h0(
    U_h: torch.Tensor,         # [10240, K]
    h_on: torch.Tensor,        # [N_on, 10240]
    off_h_groups: list[tuple[torch.Tensor, float]],
    lam: float = 2e-3,
) -> torch.Tensor:
    """Solve pure down_proj linear readout in 10,240-D GeGLU h0-space: ON -> +1.0, ALL OFF groups -> 0.0 (ZERO base-neuron pruning!)."""
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


def main():
    hdir = Path("/content/wsp/harvest32")
    out_dir = hdir / "v11_hybrid"
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
    # Compute exact column norms ||W_down[:, q]|| on GPU for all 10,240 columns so q* minimizes ACTUAL output perturbation!
    col_norms = {}
    with torch.inference_mode():
        eye_chunk = 512
        for li in target_layers:
            norms_l = []
            for s in range(0, 10240, eye_chunk):
                e = min(10240, s + eye_chunk)
                mat = torch.zeros(e - s, 10240, device=dev, dtype=torch.bfloat16)
                mat[torch.arange(e - s, device=dev), torch.arange(s, e, device=dev)] = 1.0
                norms_l.append(layers[li].mlp.down_proj(mat).float().norm(dim=-1))
            col_norms[li] = torch.cat(norms_l, dim=0)  # [10240] on GPU!

    max_h0_acc = {li: torch.zeros(10240, device=dev, dtype=torch.float32) for li in target_layers}
    sum_h0_acc = {li: torch.zeros(10240, device=dev, dtype=torch.float32) for li in target_layers}
    max_g0_acc = {li: torch.full((10240,), -1e9, device=dev, dtype=torch.float32) for li in target_layers}
    tok_cnt_acc = {li: 0 for li in target_layers}

    def forward_slim(prefix: list[int], suffix: list[int], accumulate_pres: bool):
        toks = prefix + suffix
        assert len(toks) <= 4096
        p_len = len(prefix)
        hist_colons = [e - 1 for e in find_all_subseq_ends(prefix, call_colon_seq)]
        stride = max(1, p_len // 32)
        grid_idx = list(range(0, p_len, stride))[:32]
        pref_idx = sorted(set(hist_colons + grid_idx))
        hist_colon_set = set(hist_colons)
        is_hist_colon = torch.tensor([idx in hist_colon_set for idx in pref_idx], dtype=torch.bool)
        pref_t = torch.tensor(pref_idx, device=dev, dtype=torch.long)
        suf_t = torch.arange(p_len, len(toks), device=dev, dtype=torch.long)
        cat_t = torch.cat([pref_t, suf_t], dim=0) if accumulate_pres else pref_t

        # Subsample suffix tokens for 10240-D h0 storage (always keep colon, colon-1, pytest, + up to 24 evenly spaced tokens -> <1.5 MB/seq!)
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
                    # Save compact subsampled 10240-D h0 slices (~1 MB/seq!) for Family A pure down_proj readout!
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

    # Pick neuron q* minimizing ACTUAL RESIDUAL OUTPUT IMPACT: (mean_h0 * ||W_down[:, q]||) + 0.15 * (max_h0 * ||W_down[:, q]||)!
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
            "q": q,
            "col_norm": round(float(cn[q]), 5),
            "mean_h0": round(float(mean_h0[q]), 6),
            "max_h0": round(float(max_h0[q]), 6),
            "mean_residual_dy": round(float(mean_dy[q]), 6),
            "max_residual_dy": round(float(max_dy[q]), 6),
        }
        print(f"Layer {li} min-residual-impact neuron: {q_stats[li]}", flush=True)

    with torch.inference_mode():
        for rec_list in (seq_records, bare_seqs):
            for r in rec_list:
                for li in target_layers:
                    q = q_stars[li]
                    gp = layers[li].mlp.gate_proj
                    up = layers[li].mlp.up_proj
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

    # --- Helpers for Family B (Low-dim Balanced GeGLU, mode in {d5, d6, d7_svd1}) ---
    def pack_geglu_groups(recs: list[dict], li: int):
        on_x, on_g, on_p = [], [], []
        ver_x, ver_g, ver_p = [], [], []
        ed_x, ed_g, ed_p = [], [], []
        loc_x, loc_g, loc_p = [], [], []
        non_x, non_g, non_p = [], [], []
        y_norms = []
        for r in recs:
            xs = r["cap"][(li, "x")].float()
            gs = r["cap"][(li, "g0_q")]
            ps = r["cap"][(li, "p0_q")]
            y_norms.append(r["cap"][(li, "y0_norm")])
            xp = r["cap"][(li, "x_pref")].float()
            gp = r["cap"][(li, "g0_pref_q")]
            pp = r["cap"][(li, "p0_pref_q")]
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
            (torch.cat(ver_x), torch.cat(ver_g), torch.cat(ver_p), 1.6),
            (torch.cat(ed_x), torch.cat(ed_g), torch.cat(ed_p), 1.6),
            (torch.cat(loc_x), torch.cat(loc_g), torch.cat(loc_p), 1.2),
            (torch.cat(non_x), torch.cat(non_g), torch.cat(non_p), 2.5),
        ]
        return (
            torch.stack(on_x), torch.stack(on_g), torch.stack(on_p), ogrps,
            torch.cat([g[0] for g in ogrps]), torch.cat([g[1] for g in ogrps]), torch.cat([g[2] for g in ogrps]),
            sum(y_norms) / len(y_norms),
        )

    diagnostics = {"q_stats": q_stats}
    geglu_fits = {}
    real_unch_issues = sorted({r["issue"] for r in seq_records if r["role"] == "unchecked" and r["colon_idx"] is not None})

    for li in (35, 41):
        best_cfg = None
        trials = {}
        for mode in ("d5", "d6", "d7_svd1"):
            oof_h = {}
            for iss in real_unch_issues:
                tr_recs = [r for r in seq_records if r["issue"] != iss]
                U_tr = make_lowdim_colon_basis(tr_recs, li, "x", "x_pref", mode)
                Xo, Go, Po, ogrps, _, _, _, _ = pack_geglu_groups(tr_recs, li)
                wg_tr, wu_tr = solve_balanced_subspace_geglu(U_tr, Xo, Go, Po, ogrps)
                for r in seq_records:
                    if r["issue"] == iss and r["role"] == "unchecked" and r["colon_idx"] is not None:
                        ci = r["colon_idx"]
                        xi = r["cap"][(li, "x")][ci].float()
                        gi = r["cap"][(li, "g0_q")][ci].bfloat16() + torch.dot(wg_tr, xi).bfloat16()
                        pi = r["cap"][(li, "p0_q")][ci].bfloat16() + torch.dot(wu_tr, xi).bfloat16()
                        oof_h[iss] = round(float((F.gelu(gi, approximate="tanh") * pi).item()), 3)

            U_full = make_lowdim_colon_basis(seq_records, li, "x", "x_pref", mode)
            Xo, Go, Po, ogrps, Xf, Gf, Pf, y_norm = pack_geglu_groups(seq_records, li)
            wg, wu = solve_balanced_subspace_geglu(U_full, Xo, Go, Po, ogrps)
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
            trials[mode] = {
                "dim": int(U_full.shape[1]), "oof_pos": pos_cnt, "oof_mean": round(oof_mean, 3),
                "zero_frac": round(zfrac, 5), "max_ver_colon": max_ver, "max_edit_colon": max_ed, "oof": oof_h,
            }
            # Prioritize BOTH clean verified/edit preservation (<=0.05) AND high unseen OOF pos_cnt!
            rkey = (max_ver <= 0.05, max_ed <= 0.05, zfrac >= 0.985, pos_cnt, oof_mean, zfrac)
            if best_cfg is None or rkey > best_cfg[0]:
                best_cfg = (rkey, mode, oof_h, wg, wu, Xo, Go, Po, hf, ver_h, ed_h, y_norm)

        _, best_mode, oof_h, wg, wu, Xo, Go, Po, hf, ver_h, ed_h, y_norm = best_cfg
        ho = (F.gelu(Go.bfloat16() + (Xo @ wg).bfloat16(), approximate="tanh") * (Po.bfloat16() + (Xo @ wu).bfloat16())).float()
        diagnostics[f"L{li}_geglu_lowdim"] = {
            "q": q_stars[li], "best_mode": best_mode, "trials": trials,
            "oof": oof_h, "h_on_mean": round(float(ho.mean()), 3),
            "zero_frac": round(float((hf == 0.0).float().mean()), 5),
            "ver_colon": ver_h, "edit_colon": ed_h,
        }
        geglu_fits[li] = {
            "q_idx": q_stars[li], "w_gate": wg, "w_up": wu, "w_down_q": w_down_cols[li],
            "u": u_disp, "h_on_mean": float(ho.mean().clamp_min(1.0)), "y0_norm": y_norm,
        }

    # L39 anti-bare GeGLU
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
    U39 = build_orthonormal_basis([mu39_on - X39_off.mean(dim=0), mu39_on - X39_hard.mean(dim=0), mu39_on, Vh39[0].float(), Vh39[1].float()])
    wg39, wu39 = solve_balanced_subspace_geglu(
        U39, X39_on, G39_on, P39_on,
        [(X39_hard, G39_off[hard_idx], P39_off[hard_idx], 2.0), (X39_off, G39_off, P39_off, 2.5)],
    )
    h39_on = (F.gelu(G39_on.bfloat16() + (X39_on @ wg39).bfloat16(), approximate="tanh") * (P39_on.bfloat16() + (X39_on @ wu39).bfloat16())).float()

    # --- Family A: Pure `down_proj` Decoupled 10,240-D h0-Subspace Readout (ZERO base-neuron pruning!) ---
    def make_h0_basis_and_groups(recs: list[dict], li: int):
        unch_c, ver_c, ed_c, loc_c, non_c = [], [], [], [], []
        y_norms = []
        for r in recs:
            hs = r["cap"][(li, "h_sub")].float()
            hp = r["cap"][(li, "h_pref_sub")].float()
            y_norms.append(r["cap"][(li, "y0_norm")])
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
        H_ver = torch.cat(ver_c)
        H_ed = torch.cat(ed_c)
        H_loc = torch.cat(loc_c)
        H_non = torch.cat(non_c)
        mu_on = H_on.mean(dim=0)
        mu_all_c = torch.cat([H_on, H_ver, H_ed, H_loc], dim=0).mean(dim=0)
        U_h = build_orthonormal_basis([
            mu_all_c - H_non.mean(dim=0),
            mu_on - H_ver.mean(dim=0),
            mu_on - H_ed.mean(dim=0),
            mu_on - H_loc.mean(dim=0),
            mu_on,
        ])
        ogrps = [(H_ver, 2.0), (H_ed, 2.0), (H_loc, 1.5), (H_non, 4.0)]
        return U_h, H_on, ogrps, torch.cat([H_ver, H_ed, H_loc, H_non], dim=0), sum(y_norms) / len(y_norms)

    lin_fits = {}
    for li in (35, 41):
        oof_lin = {}
        for iss in real_unch_issues:
            tr_recs = [r for r in seq_records if r["issue"] != iss]
            Uh_tr, Ho_tr, og_tr, _, _ = make_h0_basis_and_groups(tr_recs, li)
            a_tr = solve_balanced_linear_h0(Uh_tr, Ho_tr, og_tr)
            for r in seq_records:
                if r["issue"] == iss and r["role"] == "unchecked" and r["h_colon_idx"] is not None:
                    hi = r["cap"][(li, "h_sub")][r["h_colon_idx"]].float()
                    oof_lin[iss] = round(float(torch.dot(a_tr, hi).item()), 4)
        Uh, Ho, og, H_all_off, yn = make_h0_basis_and_groups(seq_records, li)
        a_full = solve_balanced_linear_h0(Uh, Ho, og)
        s_on = Ho @ a_full
        s_off = H_all_off @ a_full
        snr = float(s_on.mean() / s_off.pow(2).mean().sqrt().clamp_min(1e-6))
        diagnostics[f"L{li}_pure_downproj_h0"] = {
            "oof_s": oof_lin,
            "oof_pos_count": f"{sum(1 for v in oof_lin.values() if v > 0.15)}/{len(oof_lin)}",
            "s_on_mean": round(float(s_on.mean()), 4),
            "s_off_rms": round(float(s_off.pow(2).mean().sqrt()), 4),
            "selectivity_snr": round(snr, 2),
        }
        lin_fits[li] = {"a": a_full, "u": u_disp, "s_on_mean": float(s_on.mean().clamp_min(0.1)), "y0_norm": yn}

    # Also fit L39 pure down_proj anti-bare-pytest in h0-space!
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
    hard39_h = H39_off[(H39_off @ (mu39_h / mu39_h.norm().clamp_min(1e-8))).topk(min(64, H39_off.shape[0])).indices]
    Uh39 = build_orthonormal_basis([mu39_h - H39_off.mean(dim=0), mu39_h - hard39_h.mean(dim=0), mu39_h])
    a39_lin = solve_balanced_linear_h0(Uh39, H39_on, [(hard39_h, 2.0), (H39_off, 4.0)])
    s39_on = H39_on @ a39_lin
    s39_off = H39_off @ a39_lin
    diagnostics["L39_pure_downproj_h0"] = {
        "s_on_mean": round(float(s39_on.mean()), 4),
        "s_off_rms": round(float(s39_off.pow(2).mean().sqrt()), 4),
        "selectivity_snr": round(float(s39_on.mean() / s39_off.pow(2).mean().sqrt().clamp_min(1e-6)), 2),
    }

    print("V11 Hybrid Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    # 1. Compile Family A (Pure down_proj r=1 — ZERO base-neuron pruning!)
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
        zlbl = f"V11_lin_zero_{'_'.join(sorted(k.split('.')[0] for k in ft if k.endswith('.A')))}"
        if not (ad_root / zlbl).exists():
            zinfo = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / zlbl, "zero", rank=1)
            compiled.append({"adapter": zlbl, "sha256": zinfo["adapter_sha256"], "control": "zero", "family": "pure_downproj"})
        for ctrl in ("candidate", "random"):
            lbl = f"{name}_{ctrl}"
            info = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / lbl, ctrl, rank=1)
            compiled.append({"adapter": lbl, "sha256": info["adapter_sha256"], "control": ctrl, "family": "pure_downproj"})

    add_pure_down("V11_lin_L41_39ab_mid", [41], rho_d=0.32, rho_ab=0.28)
    add_pure_down("V11_lin_L41_39ab_hi", [41], rho_d=0.50, rho_ab=0.38)
    add_pure_down("V11_lin_L35_41_39ab_mid", [35, 41], rho_d=0.22, rho_ab=0.28)

    # 2. Compile Family B (Low-dim Balanced GeGLU with min-residual-impact q*)
    def add_geglu_fam(name: str, d_layers: list[int], rho_d: float, rho_ab: float | None):
        spec = {}
        for li in d_layers:
            f = geglu_fits[li]
            g = rho_d * f["y0_norm"] / f["h_on_mean"]
            spec[li] = {"q_idx": f["q_idx"], "w_gate": f["w_gate"], "w_up": f["w_up"], "b_down": -f["w_down_q"] - g * f["u"]}
        if rho_ab is not None:
            g39 = rho_ab * (sum(y39_norms) / len(y39_norms)) / float(h39_on.mean().clamp_min(1.0))
            spec[39] = {"q_idx": q39, "w_gate": wg39, "w_up": wu39, "b_down": -w_down_cols[39] - g39 * u_ab}
        zlbl = f"V11_geglu_zero_{'_'.join(map(str, sorted(spec)))}"
        if not (ad_root / zlbl).exists():
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", spec, ad_root / zlbl, "zero"))
        for ctrl in ("candidate", "random"):
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", spec, ad_root / f"{name}_{ctrl}", ctrl))

    add_geglu_fam("V11_geglu_L41_39ab_mid", [41], rho_d=0.40, rho_ab=0.35)
    add_geglu_fam("V11_geglu_L41_39ab_hi", [41], rho_d=0.60, rho_ab=0.42)
    add_geglu_fam("V11_geglu_L35_41_39ab_mid", [35, 41], rho_d=0.28, rho_ab=0.35)

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V11 Hybrid Pareto adapters!", flush=True)


if __name__ == "__main__":
    main()
