"""Synthesize V10 Balanced-Group Colon-State Subspace GeGLU Transistors.

Fixes V9's two exact bottlenecks identified in v9_subspace_manifest.json:
1. Balanced group weights (verified_colon=1.5, edit_colon=1.5, loc_pref_colon=1.0, non_colon=2.0 vs ON=1.0)
   so verified@call: and edit@call: are NEVER drowned out by 16,800 thought tokens -> verified_colon_h == 0.0, edit_colon_h == 0.0!
2. Task-general colon-centered state SVD subspace (K in {7, 9, 11, 13} fit strictly inside LOO CV)
   so honest unseen-issue OOF stays high (>=3-4/5) while off_bf16_zero_frac >= 0.998 (`edit_exact_match >= 10/12`!).
3. Calibrated moderate gains (rho_disp in 0.28..0.52, rho_ab in 0.30..0.42) matching V9_L41_39ab_mid's clean 5/5 pytest commands.
"""
import gc
import json
import shutil
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter, resolve_targeted_check
from synth_v7_geglu import compile_v7_geglu
from synthesize_targeted_dispatch_v3 import find_subseq_end


def build_orthonormal_basis(vecs: list[torch.Tensor]) -> torch.Tensor:
    valid = [v.double() for v in vecs if float(v.norm()) > 1e-9]
    mat = torch.stack([v / v.norm() for v in valid], dim=1)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.float()


def make_colon_state_basis(records: list[dict], li: int, k_svd: int = 4) -> torch.Tensor:
    """Build (6 + k_svd)-dim subspace from training records ONLY (100% leak-free inside LOO CV)."""
    unch_col, ver_col, edit_col, loc_col, all_col, non_col, prev_tok = [], [], [], [], [], [], []
    for r in records:
        xs = r["cap"][(li, "x")].float()
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
        if (li, "x_pref") in r["cap"]:
            non_col.append(r["cap"][(li, "x_pref")].float().mean(dim=0))

    X_col = torch.stack(all_col)  # [N_col, 2560]
    mu_col = X_col.mean(dim=0)
    mu_unch = torch.stack(unch_col).mean(dim=0)
    mu_ver = torch.stack(ver_col).mean(dim=0)
    mu_edit = torch.stack(edit_col).mean(dim=0)
    mu_loc = torch.stack(loc_col).mean(dim=0)
    mu_non = torch.stack(non_col).mean(dim=0)
    mu_prev = torch.stack(prev_tok).mean(dim=0) if prev_tok else mu_non

    vecs = [
        mu_col - mu_non,       # 1. `<|tool_call>call:` colon vs thought/arg/prefill
        mu_col - mu_prev,      # 2. `:` vs immediately preceding `call` token
        mu_unch,               # 3. Radial RMSNorm sphere direction
        mu_unch - mu_ver,      # 4. Unchecked vs verified (`test passed`) at `:`
        mu_unch - mu_edit,     # 5. Post-edit vs pre-edit (`edit_file`) at `:`
        mu_unch - mu_loc,      # 6. Post-edit vs initial localization at `:`
    ]
    if k_svd > 0 and X_col.shape[0] > 4:
        # Subtract local `:` mean to isolate pure conversation-history state axes across all colons!
        Xc = (X_col - mu_col).double()
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        for j in range(min(k_svd, Vh.shape[0])):
            vecs.append(Vh[j].float())

    return build_orthonormal_basis(vecs)


def solve_balanced_subspace_geglu(
    U: torch.Tensor,              # [D, K]
    x_on: torch.Tensor,           # [N_on, D]
    g0_on: torch.Tensor,          # [N_on]
    p0_on: torch.Tensor,          # [N_on]
    off_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    target_g_on: float = 3.6,
    target_g_off: float = -4.9,
    target_p_on: float = 3.4,
    lam: float = 8e-4,
    n_irls: int = 14,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced-group IRLS solver so verified@call:, edit@call:, loc@call:, AND non_colon are ALL strictly <= -3.95 (BF16 0.0!)."""
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
    yf_g = (target_g_off - gf).clamp_max(-0.6)

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
            viol_off = pred_gf > -3.95
            viol_on = pred_go < 2.2
            if not viol_off.any() and not viol_on.any():
                break
            w_off = torch.where(viol_off, w_off * 6.5, w_off)
            w_on = torch.where(viol_on, w_on * 2.0, w_on)

    # Fit alpha_up with bounded OFF magnitude so 0.0 * p_new == 0.0 in BF16 and ON has p_new ~ +3.4
    yo_p = target_p_on - po
    yf_p = -0.35 * pf
    w_on_p = torch.full((n_on,), 1.0 / n_on, device=device)
    w_off_p = w_off_base * (0.60 / total_off_weight)
    lhs_p = (Zo.T * w_on_p) @ Zo + (Zf.T * w_off_p) @ Zf + (4.0 * lam * z_scale) * eye
    rhs_p = Zo.T @ (w_on_p * yo_p) + Zf.T @ (w_off_p * yf_p)
    alpha_u = torch.linalg.solve(lhs_p, rhs_p)

    return (Ud @ alpha_g).cpu(), (Ud @ alpha_u).cpu()


def main():
    hdir = Path("/content/wsp/harvest32")
    out_dir = hdir / "v10_subspace"
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
    max_h0_acc = {li: torch.zeros(10240, device=dev, dtype=torch.float32) for li in target_layers}
    sum_h0_acc = {li: torch.zeros(10240, device=dev, dtype=torch.float32) for li in target_layers}
    max_g0_acc = {li: torch.full((10240,), -1e9, device=dev, dtype=torch.float32) for li in target_layers}
    sum_g0_acc = {li: torch.zeros(10240, device=dev, dtype=torch.float32) for li in target_layers}
    tok_cnt_acc = {li: 0 for li in target_layers}

    def forward_slim(prefix: list[int], suffix: list[int], accumulate_pres: bool):
        toks = prefix + suffix
        assert len(toks) <= 4096
        p_len = len(prefix)
        hist_colons = [e - 1 for e in find_all_subseq_ends(prefix, call_colon_seq)]
        stride = max(1, p_len // 40)
        grid_idx = list(range(0, p_len, stride))[:40]
        pref_idx = sorted(set(hist_colons + grid_idx))
        hist_colon_set = set(hist_colons)
        is_hist_colon = torch.tensor([idx in hist_colon_set for idx in pref_idx], dtype=torch.bool)
        pref_t = torch.tensor(pref_idx, device=dev, dtype=torch.long)
        suf_t = torch.arange(p_len, len(toks), device=dev, dtype=torch.long)
        cat_t = torch.cat([pref_t, suf_t], dim=0) if accumulate_pres else pref_t

        cap = {"pref_is_colon": is_hist_colon}
        tmp_g = {}
        hooks = []
        for li in target_layers:
            def mk_g(idx=li):
                def _h(mod, inp, out):
                    x_src = inp[0][0]
                    cap[(idx, "x")] = x_src[p_len:].detach().to(torch.bfloat16).cpu()
                    cap[(idx, "x_pref")] = x_src[pref_t].detach().to(torch.bfloat16).cpu()
                    g_sel = out[0].index_select(0, cat_t).detach()
                    tmp_g[idx] = g_sel
                    gf = g_sel.float()
                    max_g0_acc[idx].copy_(torch.maximum(max_g0_acc[idx], gf.max(dim=0).values))
                    sum_g0_acc[idx].add_(gf.sum(dim=0))
                    tok_cnt_acc[idx] += int(gf.shape[0])
                return _h
            def mk_u(idx=li):
                def _h(mod, inp, out):
                    p_sel = out[0].index_select(0, cat_t).detach()
                    g_sel = tmp_g.pop(idx)
                    h_abs = (F.gelu(g_sel, approximate="tanh") * p_sel).float().abs()
                    max_h0_acc[idx].copy_(torch.maximum(max_h0_acc[idx], h_abs.max(dim=0).values))
                    sum_h0_acc[idx].add_(h_abs.sum(dim=0))
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
        seq_records.append({
            "issue": c["issue"],
            "role": c["role"],
            "suf": suf,
            "colon_idx": (colon_end - 1) if colon_end is not None else None,
            "pytest_idx": (pytest_end - 1) if pytest_end is not None else None,
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
                bare_seqs.append({
                    "issue": issue,
                    "colon_idx": (col_end - 1) if col_end is not None else None,
                    "pytest_idx": pt_end - 1,
                    "cap": cap,
                })
            break

    q_stars = {}
    q_stats = {}
    for li in target_layers:
        max_h0 = max_h0_acc[li].cpu()
        mean_h0 = (sum_h0_acc[li] / max(1, tok_cnt_acc[li])).cpu()
        max_g0 = max_g0_acc[li].cpu()
        mean_g0 = (sum_g0_acc[li] / max(1, tok_cnt_acc[li])).cpu()
        # Minimize BOTH max_h0 AND mean_h0 so base perturbation -w_down_q * h0_q is < 0.002 everywhere!
        loss_q = 100.0 * mean_h0 + 10.0 * max_h0 + 0.05 * F.relu(max_g0)
        q = int(loss_q.argmin().item())
        q_stars[li] = q
        q_stats[li] = {
            "q": q,
            "max_h0": round(float(max_h0[q]), 6),
            "mean_h0": round(float(mean_h0[q]), 6),
            "max_g0": round(float(max_g0[q]), 3),
            "mean_g0": round(float(mean_g0[q]), 3),
        }
        print(f"Layer {li} quietest neuron: {q_stats[li]}", flush=True)

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

    del model, lm_root, layers, max_h0_acc, sum_h0_acc, max_g0_acc, sum_g0_acc
    gc.collect()
    torch.cuda.empty_cache()

    def pack_balanced_groups(recs: list[dict], li: int):
        on_x, on_g, on_p = [], [], []
        ver_c_x, ver_c_g, ver_c_p = [], [], []
        edit_c_x, edit_c_g, edit_c_p = [], [], []
        loc_c_x, loc_c_g, loc_c_p = [], [], []
        non_c_x, non_c_g, non_c_p = [], [], []
        y_norms = []

        for r in recs:
            xs = r["cap"][(li, "x")].float()
            gs = r["cap"][(li, "g0_q")]
            ps = r["cap"][(li, "p0_q")]
            y_norms.append(r["cap"][(li, "y0_norm")])

            # Split prefill into historical `call:` colons vs regular prefill tokens!
            xp = r["cap"][(li, "x_pref")].float()
            gp = r["cap"][(li, "g0_pref_q")]
            pp = r["cap"][(li, "p0_pref_q")]
            isc = r["cap"]["pref_is_colon"]
            if isc.any():
                loc_c_x.append(xp[isc])
                loc_c_g.append(gp[isc])
                loc_c_p.append(pp[isc])
            if (~isc).any():
                non_c_x.append(xp[~isc])
                non_c_g.append(gp[~isc])
                non_c_p.append(pp[~isc])

            cidx = r["colon_idx"]
            if cidx is not None:
                m = torch.arange(xs.shape[0]) != cidx
                if m.any():
                    non_c_x.append(xs[m])
                    non_c_g.append(gs[m])
                    non_c_p.append(ps[m])
                if r["role"] in ("unchecked", "check"):
                    on_x.append(xs[cidx])
                    on_g.append(gs[cidx])
                    on_p.append(ps[cidx])
                elif r["role"] == "verified":
                    ver_c_x.append(xs[cidx : cidx + 1])
                    ver_c_g.append(gs[cidx : cidx + 1])
                    ver_c_p.append(ps[cidx : cidx + 1])
                elif r["role"] == "edit":
                    edit_c_x.append(xs[cidx : cidx + 1])
                    edit_c_g.append(gs[cidx : cidx + 1])
                    edit_c_p.append(ps[cidx : cidx + 1])
                else:
                    loc_c_x.append(xs[cidx : cidx + 1])
                    loc_c_g.append(gs[cidx : cidx + 1])
                    loc_c_p.append(ps[cidx : cidx + 1])
            else:
                non_c_x.append(xs)
                non_c_g.append(gs)
                non_c_p.append(ps)

        off_groups = [
            (torch.cat(ver_c_x, dim=0), torch.cat(ver_c_g, dim=0), torch.cat(ver_c_p, dim=0), 1.6),   # 1.6x weight on verified@call:!
            (torch.cat(edit_c_x, dim=0), torch.cat(edit_c_g, dim=0), torch.cat(edit_c_p, dim=0), 1.6), # 1.6x weight on edit@call:!
            (torch.cat(loc_c_x, dim=0), torch.cat(loc_c_g, dim=0), torch.cat(loc_c_p, dim=0), 1.2),   # 1.2x weight on localize+prefill@call:!
            (torch.cat(non_c_x, dim=0), torch.cat(non_c_g, dim=0), torch.cat(non_c_p, dim=0), 2.5),   # 2.5x weight on all non-colon tokens!
        ]
        all_off_x = torch.cat([g[0] for g in off_groups], dim=0)
        all_off_g = torch.cat([g[1] for g in off_groups], dim=0)
        all_off_p = torch.cat([g[2] for g in off_groups], dim=0)
        return (
            torch.stack(on_x), torch.stack(on_g), torch.stack(on_p),
            off_groups, all_off_x, all_off_g, all_off_p, sum(y_norms) / len(y_norms),
        )

    disp_fits = {}
    diagnostics = {"q_stats": q_stats}

    for li in (35, 41):
        q = q_stars[li]
        real_unch_issues = sorted({r["issue"] for r in seq_records if r["role"] == "unchecked" and r["colon_idx"] is not None})

        best_cfg = None
        k_trials = {}
        for k_svd in (2, 4, 6, 8, 10):
            oof_honest = {}
            for iss in real_unch_issues:
                tr_recs = [r for r in seq_records if r["issue"] != iss]
                U_tr = make_colon_state_basis(tr_recs, li, k_svd=k_svd)
                Xo, Go, Po, ogrps, _, _, _, _ = pack_balanced_groups(tr_recs, li)
                wg_tr, wu_tr = solve_balanced_subspace_geglu(U_tr, Xo, Go, Po, ogrps)
                for r in seq_records:
                    if r["issue"] == iss and r["role"] == "unchecked" and r["colon_idx"] is not None:
                        ci = r["colon_idx"]
                        xi = r["cap"][(li, "x")][ci].float()
                        gi = r["cap"][(li, "g0_q")][ci].bfloat16() + torch.dot(wg_tr, xi).bfloat16()
                        pi = r["cap"][(li, "p0_q")][ci].bfloat16() + torch.dot(wu_tr, xi).bfloat16()
                        hi = F.gelu(gi, approximate="tanh") * pi
                        oof_honest[iss] = round(float(hi.item()), 3)

            U_full = make_colon_state_basis(seq_records, li, k_svd=k_svd)
            Xo, Go, Po, ogrps, Xf, Gf, Pf, y_norm = pack_balanced_groups(seq_records, li)
            wg, wu = solve_balanced_subspace_geglu(U_full, Xo, Go, Po, ogrps)
            gf_new = Gf.bfloat16() + (Xf @ wg).bfloat16()
            pf_new = Pf.bfloat16() + (Xf @ wu).bfloat16()
            hf_new = (F.gelu(gf_new, approximate="tanh") * pf_new).float()
            zero_frac = float((hf_new == 0.0).float().mean())
            max_off = float(hf_new.abs().max())

            ver_h, edit_h = [], []
            for r in seq_records:
                ci = r["colon_idx"]
                if ci is not None and r["role"] in ("verified", "edit"):
                    xi = r["cap"][(li, "x")][ci].float()
                    gi = r["cap"][(li, "g0_q")][ci].bfloat16() + torch.dot(wg, xi).bfloat16()
                    pi = r["cap"][(li, "p0_q")][ci].bfloat16() + torch.dot(wu, xi).bfloat16()
                    hv = float((F.gelu(gi, approximate="tanh") * pi).item())
                    (ver_h if r["role"] == "verified" else edit_h).append(round(hv, 5))

            max_ver = max((abs(v) for v in ver_h), default=0.0)
            max_edit = max((abs(v) for v in edit_h), default=0.0)
            pos_cnt = sum(1 for v in oof_honest.values() if v > 1.0)
            oof_mean = sum(oof_honest.values()) / max(1, len(oof_honest))
            k_trials[f"svd{k_svd}_dim{U_full.shape[1]}"] = {
                "oof_pos": pos_cnt,
                "oof_mean": round(oof_mean, 3),
                "zero_frac": round(zero_frac, 5),
                "max_off": round(max_off, 5),
                "max_ver_colon": round(max_ver, 5),
                "max_edit_colon": round(max_edit, 5),
                "oof": oof_honest,
            }
            # Smooth hierarchical rank key: NEVER pick a low zero_frac or positive verified/edit colon!
            rank_key = (
                max_ver == 0.0,
                max_edit <= 0.02,
                zero_frac >= 0.997,
                round(zero_frac, 3),
                pos_cnt,
                oof_mean,
            )
            if best_cfg is None or rank_key > best_cfg[0]:
                best_cfg = (rank_key, k_svd, U_full.shape[1], oof_honest, wg, wu, Xo, Go, Po, hf_new, ver_h, edit_h, y_norm)

        _, best_k_svd, best_dim, oof_honest, wg, wu, Xo, Go, Po, hf_new, ver_h, edit_h, y_norm = best_cfg
        ho_new = (F.gelu(Go.bfloat16() + (Xo @ wg).bfloat16(), approximate="tanh") * (Po.bfloat16() + (Xo @ wu).bfloat16())).float()

        diagnostics[f"L{li}_balanced_dispatch"] = {
            "q": q,
            "best_k_svd": best_k_svd,
            "best_subspace_dim": best_dim,
            "k_trials": k_trials,
            "honest_unseen_issue_oof_h": oof_honest,
            "honest_oof_pos_count": f"{sum(1 for v in oof_honest.values() if v > 1.0)}/{len(oof_honest)}",
            "h_on_mean": round(float(ho_new.mean()), 3),
            "h_on_min": round(float(ho_new.min()), 3),
            "off_bf16_zero_frac": round(float((hf_new == 0.0).float().mean()), 5),
            "off_h_max_abs": round(float(hf_new.abs().max()), 6),
            "verified_colon_h": ver_h,
            "edit_colon_h": edit_h,
        }
        disp_fits[li] = {
            "q_idx": q,
            "w_gate": wg,
            "w_up": wu,
            "w_down_q": w_down_cols[li],
            "u": u_disp,
            "h_on_mean": float(ho_new.mean().clamp_min(1.0)),
            "y0_norm": y_norm,
        }

    # Fit L39 6D-Subspace Anti-Bare-Pytest Transistor (`off_bf16_zero_frac >= 0.9999`!)
    q39 = q_stars[39]
    on39_x, on39_g, on39_p = [], [], []
    off39_x, off39_g, off39_p = [], [], []
    y39_norms = []
    for r in seq_records:
        xs = r["cap"][(39, "x")].float()
        gs = r["cap"][(39, "g0_q")]
        ps = r["cap"][(39, "p0_q")]
        y39_norms.append(r["cap"][(39, "y0_norm")])
        off39_x.append(r["cap"][(39, "x_pref")].float())
        off39_g.append(r["cap"][(39, "g0_pref_q")])
        off39_p.append(r["cap"][(39, "p0_pref_q")])
        pidx = r["pytest_idx"]
        if pidx is not None:
            on39_x.append(xs[pidx])
            on39_g.append(gs[pidx])
            on39_p.append(ps[pidx])
            m = torch.arange(xs.shape[0]) != pidx
            off39_x.append(xs[m])
            off39_g.append(gs[m])
            off39_p.append(ps[m])
        else:
            off39_x.append(xs)
            off39_g.append(gs)
            off39_p.append(ps)
    for r in bare_seqs:
        xs = r["cap"][(39, "x")].float()
        gs = r["cap"][(39, "g0_q")]
        ps = r["cap"][(39, "p0_q")]
        pidx = r["pytest_idx"]
        on39_x.append(xs[pidx])
        on39_g.append(gs[pidx])
        on39_p.append(ps[pidx])
        m = torch.arange(xs.shape[0]) != pidx
        off39_x.append(xs[m])
        off39_g.append(gs[m])
        off39_p.append(ps[m])

    X39_on = torch.stack(on39_x)
    G39_on = torch.stack(on39_g)
    P39_on = torch.stack(on39_p)
    X39_off = torch.cat(off39_x, dim=0)
    G39_off = torch.cat(off39_g, dim=0)
    P39_off = torch.cat(off39_p, dim=0)

    mu39_on = X39_on.mean(dim=0)
    mu39_off = X39_off.mean(dim=0)
    dots39 = X39_off @ (mu39_on / mu39_on.norm().clamp_min(1e-8))
    hard_idx = dots39.topk(min(96, X39_off.shape[0])).indices
    X39_hard = X39_off[hard_idx]
    _, _, Vh39 = torch.linalg.svd((X39_hard - X39_hard.mean(dim=0)).double(), full_matrices=False)
    U39 = build_orthonormal_basis([
        mu39_on - mu39_off,
        mu39_on - X39_hard.mean(dim=0),
        mu39_on,
        Vh39[0].float(),
        Vh39[1].float(),
        Vh39[2].float(),
    ])
    wg39, wu39 = solve_balanced_subspace_geglu(
        U39, X39_on, G39_on, P39_on,
        [(X39_hard, G39_off[hard_idx], P39_off[hard_idx], 2.0), (X39_off, G39_off, P39_off, 2.5)],
    )
    h39_on = (F.gelu(G39_on.bfloat16() + (X39_on @ wg39).bfloat16(), approximate="tanh") * (P39_on.bfloat16() + (X39_on @ wu39).bfloat16())).float()
    h39_off = (F.gelu(G39_off.bfloat16() + (X39_off @ wg39).bfloat16(), approximate="tanh") * (P39_off.bfloat16() + (X39_off @ wu39).bfloat16())).float()
    w_down_q39 = w_down_cols[39]

    diagnostics["L39_subspace6_anti_bare"] = {
        "q": q39,
        "h_on_mean": round(float(h39_on.mean()), 3),
        "off_bf16_zero_frac": round(float((h39_off == 0.0).float().mean()), 5),
        "off_h_max_abs": round(float(h39_off.abs().max()), 6),
    }
    print("V10 Balanced Subspace Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    def add_family(name: str, disp_layers: list[int], rho_disp: float, rho_ab: float | None):
        spec = {}
        for li in disp_layers:
            f = disp_fits[li]
            g = rho_disp * f["y0_norm"] / f["h_on_mean"]
            spec[li] = {"q_idx": f["q_idx"], "w_gate": f["w_gate"], "w_up": f["w_up"], "b_down": -f["w_down_q"] - g * f["u"]}
        if rho_ab is not None:
            g39 = rho_ab * (sum(y39_norms) / len(y39_norms)) / float(h39_on.mean().clamp_min(1.0))
            spec[39] = {"q_idx": q39, "w_gate": wg39, "w_up": wu39, "b_down": -w_down_q39 - g39 * u_ab}
        zlbl = f"V10_zero_{'_'.join(map(str, sorted(spec)))}"
        if not (ad_root / zlbl).exists():
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", spec, ad_root / zlbl, "zero"))
        for ctrl in ("candidate", "random"):
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", spec, ad_root / f"{name}_{ctrl}", ctrl))

    # Calibrated gains (matching V9_L41_39ab_mid's clean 5/5 pytest commands without over-steering!)
    add_family("V10_L41_39ab_g035", [41], rho_disp=0.35, rho_ab=0.32)
    add_family("V10_L41_39ab_g050", [41], rho_disp=0.50, rho_ab=0.38)
    add_family("V10_L41_only_g045", [41], rho_disp=0.45, rho_ab=None)
    add_family("V10_L35_41_39ab_g025", [35, 41], rho_disp=0.25, rho_ab=0.32)
    add_family("V10_L35_41_39ab_g038", [35, 41], rho_disp=0.38, rho_ab=0.38)

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V10 Balanced Subspace GeGLU adapters!", flush=True)


if __name__ == "__main__":
    main()
