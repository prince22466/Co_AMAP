"""Synthesize V9 Task-Invariant Low-Dim Subspace GeGLU Transistors (80 MB RAM streaming dead-neuron scan + honest LOO OOF)."""
import gc
import hashlib
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
    """Return [D, K] orthonormal basis spanning vecs via float64 QR."""
    mat = torch.stack([v.double() / v.double().norm().clamp_min(1e-12) for v in vecs], dim=1)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.float()


def solve_subspace_geglu_pair(
    U: torch.Tensor,         # [D, K] low-dim orthonormal basis (K = 4..6 << D=2560!)
    x_on: torch.Tensor,      # [N_on, D]
    g0_on: torch.Tensor,     # [N_on]
    p0_on: torch.Tensor,     # [N_on]
    x_off: torch.Tensor,     # [N_off, D] (includes prefill + edit + localize + verified + thought/arg!)
    g0_off: torch.Tensor,    # [N_off]
    p0_off: torch.Tensor,    # [N_off]
    target_g_on: float = 3.8,
    target_g_off: float = -4.8,
    target_p_on: float = 3.5,
    lam: float = 5e-4,
    n_irls: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit w_gate, w_up in span(U) so unseen ON has g_new~+3.6, p_new~+3.5 (h~+12) and OFF has g_new<=-3.95 (BF16 exact 0.0!)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ud = U.to(device, torch.float32)
    Zo = x_on.to(device, torch.float32) @ Ud    # [N_on, K]
    Zf = x_off.to(device, torch.float32) @ Ud   # [N_off, K]

    go = g0_on.to(device, torch.float32)
    gf = g0_off.to(device, torch.float32)
    po = p0_on.to(device, torch.float32)
    pf = p0_off.to(device, torch.float32)

    yo_g = (target_g_on - go).clamp_min(1.5)
    yf_g = (target_g_off - gf).clamp_max(-0.5)

    n_on = max(1, Zo.shape[0])
    n_off = max(1, Zf.shape[0])
    w_on = torch.full((n_on,), 1.0 / n_on, device=device)
    w_off = torch.full((n_off,), 3.0 / n_off, device=device)
    z_scale = float(Zf.pow(2).sum(dim=1).mean().clamp_min(1.0))
    eye = torch.eye(Ud.shape[1], device=device)

    alpha_g = torch.zeros(Ud.shape[1], device=device)
    for step in range(n_irls):
        w_off_n = w_off * (3.0 / w_off.sum().clamp_min(1e-8))
        lhs = (Zo.T * w_on) @ Zo + (Zf.T * w_off_n) @ Zf + (lam * z_scale) * eye
        rhs = Zo.T @ (w_on * yo_g) + Zf.T @ (w_off_n * yf_g)
        alpha_g = torch.linalg.solve(lhs, rhs)
        if step + 1 < n_irls:
            pred_gf = gf + (Zf @ alpha_g)
            pred_go = go + (Zo @ alpha_g)
            viol_off = pred_gf > -3.92
            viol_on = pred_go < 2.0
            if not viol_off.any() and not viol_on.any():
                break
            w_off = torch.where(viol_off, w_off * 7.0, w_off)
            w_on = torch.where(viol_on, w_on * 2.2, w_on)

    yo_p = target_p_on - po
    yf_p = -0.20 * pf
    w_on_p = torch.full((n_on,), 1.0 / n_on, device=device)
    w_off_p = torch.full((n_off,), 0.40 / n_off, device=device)
    lhs_p = (Zo.T * w_on_p) @ Zo + (Zf.T * w_off_p) @ Zf + (5.0 * lam * z_scale) * eye
    rhs_p = Zo.T @ (w_on_p * yo_p) + Zf.T @ (w_off_p * yf_p)
    alpha_u = torch.linalg.solve(lhs_p, rhs_p)

    return (Ud @ alpha_g).cpu(), (Ud @ alpha_u).cpu()


def make_dispatch_basis(records: list[dict], li: int, k_dim: int = 6) -> torch.Tensor:
    """Build K-dim task-invariant basis from training records ONLY (safe inside LOO CV!)."""
    unch_col, ver_col, edit_col, loc_col, all_col, non_col, hard_non_col = [], [], [], [], [], [], []
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
                    hard_non_col.append(xs[cidx - 1])
        else:
            non_col.append(xs.mean(dim=0))
        if (li, "x_pref") in r["cap"]:
            non_col.append(r["cap"][(li, "x_pref")].float().mean(dim=0))

    mu_unch = torch.stack(unch_col).mean(dim=0)
    mu_ver = torch.stack(ver_col).mean(dim=0)
    mu_edit = torch.stack(edit_col).mean(dim=0)
    mu_loc = torch.stack(loc_col).mean(dim=0)
    mu_col = torch.stack(all_col).mean(dim=0)
    mu_non = torch.stack(non_col).mean(dim=0)
    mu_prev = torch.stack(hard_non_col).mean(dim=0) if hard_non_col else mu_non

    vecs = [
        mu_col - mu_non,       # 1. `<|tool_call>call:` colon vs all thought/arg/prefill tokens
        mu_unch - mu_ver,      # 2. Global-attention state: NO check run after edit vs verified pass!
        mu_unch - mu_edit,     # 3. Global-attention state: post-edit vs pre-edit!
        mu_unch,               # 4. Radial centering on RMSNorm sphere
        mu_unch - mu_loc,      # 5. Post-edit vs initial localization `call:`
        mu_col - mu_prev,     # 6. Sharp single-token contrast between `:` and preceding `call` token!
    ]
    return build_orthonormal_basis(vecs[:k_dim])


def main():
    hdir = Path("/content/wsp/harvest32")
    out_dir = hdir / "v9_subspace"
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

    # Streaming GPU accumulators for [10240] neuron stats -> ZERO [S, 10240] CPU allocation!
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
        pref_t = torch.tensor(pref_idx, device=dev, dtype=torch.long)
        suf_t = torch.arange(p_len, len(toks), device=dev, dtype=torch.long)
        cat_t = torch.cat([pref_t, suf_t], dim=0) if accumulate_pres else pref_t

        cap = {}
        tmp_g = {}
        hooks = []
        for li in target_layers:
            def mk_g(idx=li):
                def _h(mod, inp, out):
                    # Slice ON GPU in BF16 BEFORE .cpu() -> < 2 MB per sequence instead of 500 MB!
                    x_src = inp[0][0]
                    cap[(idx, "x")] = x_src[p_len:].detach().to(torch.bfloat16).cpu()
                    cap[(idx, "x_pref")] = x_src[pref_t].detach().to(torch.bfloat16).cpu()
                    g_sel = out[0].index_select(0, cat_t).detach()  # [M, 10240] BF16 on GPU
                    tmp_g[idx] = g_sel
                    gf = g_sel.float()
                    max_g0_acc[idx].copy_(torch.maximum(max_g0_acc[idx], gf.max(dim=0).values))
                    sum_g0_acc[idx].add_(gf.sum(dim=0))
                    tok_cnt_acc[idx] += int(gf.shape[0])
                return _h
            def mk_u(idx=li):
                def _h(mod, inp, out):
                    p_sel = out[0].index_select(0, cat_t).detach()  # [M, 10240] BF16 on GPU
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

    # Pick 100% BF16-DEAD neuron q* per layer from streaming GPU accumulators!
    q_stars = {}
    q_stats = {}
    for li in target_layers:
        max_h0 = max_h0_acc[li].cpu()
        mean_h0 = (sum_h0_acc[li] / max(1, tok_cnt_acc[li])).cpu()
        max_g0 = max_g0_acc[li].cpu()
        mean_g0 = (sum_g0_acc[li] / max(1, tok_cnt_acc[li])).cpu()
        exact_dead = (max_h0 == 0.0) & (max_g0 <= -3.9)
        dead_count = int(exact_dead.sum().item())
        if exact_dead.any():
            loss_q = torch.where(exact_dead, (max_g0 + 4.2).abs() + 0.2 * (mean_g0 + 5.5).abs(), torch.full_like(max_g0, 1e9))
        else:
            loss_q = 1000.0 * max_h0 + 100.0 * mean_h0 + 0.05 * F.relu(max_g0 + 3.5)
        q = int(loss_q.argmin().item())
        q_stars[li] = q
        q_stats[li] = {
            "q": q,
            "exact_dead_pool": dead_count,
            "max_h0": round(float(max_h0[q]), 6),
            "mean_h0": round(float(mean_h0[q]), 6),
            "max_g0": round(float(max_g0[q]), 3),
            "mean_g0": round(float(mean_g0[q]), 3),
        }
        print(f"Layer {li} quiet neuron: {q_stats[li]}", flush=True)

    # Now evaluate ONLY the 1D scalar g0[:, q*] and p0[:, q*] through the exact W4A16 modules on GPU (takes 0.05s, < 1 MB RAM!)
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

    # Extract unembedding rows + 3 down_proj columns, then FREE THE ENTIRE MODEL before solving!
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

    # NOTE on w_down[:, q]: In W4A16 compressed-tensors, check if `.weight` is packed int4 or float/bf16!
    # Evaluate exact W_down[:, q] by passing one-hot e_q through `layers[li].mlp.down_proj(e_q)`!
    w_down_cols = {}
    with torch.inference_mode():
        for li in target_layers:
            eq = torch.zeros(1, 10240, device=dev, dtype=torch.bfloat16)
            eq[0, q_stars[li]] = 1.0
            w_down_cols[li] = layers[li].mlp.down_proj(eq)[0].detach().float().cpu()

    del model, lm_root, layers, max_h0_acc, sum_h0_acc, max_g0_acc, sum_g0_acc
    gc.collect()
    torch.cuda.empty_cache()
    print("Model freed cleanly; solving low-dim subspace GeGLU gates...", flush=True)

    def pack_dispatch_tensors(recs: list[dict], li: int, include_prefill_off: bool):
        on_x, on_g, on_p, on_meta = [], [], [], []
        off_x, off_g, off_p = [], [], []
        y_norms = []
        for r in recs:
            xs = r["cap"][(li, "x")].float()
            gs = r["cap"][(li, "g0_q")]
            ps = r["cap"][(li, "p0_q")]
            y_norms.append(r["cap"][(li, "y0_norm")])
            cidx = r["colon_idx"]
            if include_prefill_off:
                off_x.append(r["cap"][(li, "x_pref")].float())
                off_g.append(r["cap"][(li, "g0_pref_q")])
                off_p.append(r["cap"][(li, "p0_pref_q")])

            if r["role"] in ("unchecked", "check") and cidx is not None:
                on_x.append(xs[cidx])
                on_g.append(gs[cidx])
                on_p.append(ps[cidx])
                on_meta.append((r["issue"], r["role"]))
                m = torch.arange(xs.shape[0]) != cidx
                off_x.append(xs[m])
                off_g.append(gs[m])
                off_p.append(ps[m])
            else:
                off_x.append(xs)
                off_g.append(gs)
                off_p.append(ps)
                if cidx is not None:
                    rep = 16 if r["role"] == "verified" else 10
                    off_x.append(xs[cidx : cidx + 1].repeat(rep, 1))
                    off_g.append(gs[cidx : cidx + 1].repeat(rep))
                    off_p.append(ps[cidx : cidx + 1].repeat(rep))
        return (
            torch.stack(on_x), torch.stack(on_g), torch.stack(on_p), on_meta,
            torch.cat(off_x, dim=0), torch.cat(off_g, dim=0), torch.cat(off_p, dim=0),
            sum(y_norms) / len(y_norms),
        )

    disp_fits = {}
    diagnostics = {"q_stats": q_stats}

    for li in (35, 41):
        q = q_stars[li]
        inc_pref = True
        real_unch_issues = sorted({r["issue"] for r in seq_records if r["role"] == "unchecked" and r["colon_idx"] is not None})

        best_cfg = None
        k_trials = {}
        for k_dim in (4, 5, 6):
            oof_honest = {}
            for iss in real_unch_issues:
                tr_recs = [r for r in seq_records if r["issue"] != iss]
                U_tr = make_dispatch_basis(tr_recs, li, k_dim=k_dim)
                Xo, Go, Po, _, Xf, Gf, Pf, _ = pack_dispatch_tensors(tr_recs, li, inc_pref)
                wg_tr, wu_tr = solve_subspace_geglu_pair(U_tr, Xo, Go, Po, Xf, Gf, Pf)
                for r in seq_records:
                    if r["issue"] == iss and r["role"] == "unchecked" and r["colon_idx"] is not None:
                        ci = r["colon_idx"]
                        xi = r["cap"][(li, "x")][ci].float()
                        gi = r["cap"][(li, "g0_q")][ci].bfloat16() + torch.dot(wg_tr, xi).bfloat16()
                        pi = r["cap"][(li, "p0_q")][ci].bfloat16() + torch.dot(wu_tr, xi).bfloat16()
                        hi = F.gelu(gi, approximate="tanh") * pi
                        oof_honest[iss] = round(float(hi.item()), 3)

            U_full = make_dispatch_basis(seq_records, li, k_dim=k_dim)
            Xo, Go, Po, meta, Xf, Gf, Pf, y_norm = pack_dispatch_tensors(seq_records, li, inc_pref)
            wg, wu = solve_subspace_geglu_pair(U_full, Xo, Go, Po, Xf, Gf, Pf)
            gf_new = Gf.bfloat16() + (Xf @ wg).bfloat16()
            pf_new = Pf.bfloat16() + (Xf @ wu).bfloat16()
            hf_new = (F.gelu(gf_new, approximate="tanh") * pf_new).float()
            zero_frac = float((hf_new == 0.0).float().mean())
            max_off = float(hf_new.abs().max())
            pos_cnt = sum(1 for v in oof_honest.values() if v > 1.0)
            oof_mean = sum(oof_honest.values()) / max(1, len(oof_honest))
            k_trials[f"K{k_dim}"] = {
                "oof_pos": pos_cnt, "oof_mean": round(oof_mean, 3),
                "zero_frac": round(zero_frac, 5), "max_off": round(max_off, 6),
                "oof": oof_honest,
            }
            rank_key = (zero_frac >= 0.995, max_off <= 0.02, pos_cnt, oof_mean, zero_frac)
            if best_cfg is None or rank_key > best_cfg[0]:
                best_cfg = (rank_key, k_dim, oof_honest, wg, wu, Xo, Go, Po, Xf, Gf, Pf, y_norm, hf_new)

        _, best_k, oof_honest, wg, wu, Xo, Go, Po, Xf, Gf, Pf, y_norm, hf_new = best_cfg
        go_new = Go.bfloat16() + (Xo @ wg).bfloat16()
        po_new = Po.bfloat16() + (Xo @ wu).bfloat16()
        ho_new = (F.gelu(go_new, approximate="tanh") * po_new).float()

        ver_h, edit_h = [], []
        for r in seq_records:
            ci = r["colon_idx"]
            if ci is not None and r["role"] in ("verified", "edit"):
                xi = r["cap"][(li, "x")][ci].float()
                gi = r["cap"][(li, "g0_q")][ci].bfloat16() + torch.dot(wg, xi).bfloat16()
                pi = r["cap"][(li, "p0_q")][ci].bfloat16() + torch.dot(wu, xi).bfloat16()
                hv = float((F.gelu(gi, approximate="tanh") * pi).item())
                (ver_h if r["role"] == "verified" else edit_h).append(round(hv, 5))

        diagnostics[f"L{li}_subspace_dispatch"] = {
            "q": q,
            "best_k": best_k,
            "k_trials": k_trials,
            "n_on": int(Xo.shape[0]),
            "n_off": int(Xf.shape[0]),
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

    # Fit L39 3D-Subspace Anti-Bare-Pytest Transistor
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
    hard39 = X39_off[dots39.topk(min(64, X39_off.shape[0])).indices].mean(dim=0)
    U39 = build_orthonormal_basis([mu39_on - mu39_off, mu39_on - hard39, mu39_on])
    wg39, wu39 = solve_subspace_geglu_pair(U39, X39_on, G39_on, P39_on, X39_off, G39_off, P39_off)

    h39_on = (F.gelu(G39_on.bfloat16() + (X39_on @ wg39).bfloat16(), approximate="tanh") * (P39_on.bfloat16() + (X39_on @ wu39).bfloat16())).float()
    h39_off = (F.gelu(G39_off.bfloat16() + (X39_off @ wg39).bfloat16(), approximate="tanh") * (P39_off.bfloat16() + (X39_off @ wu39).bfloat16())).float()
    w_down_q39 = w_down_cols[39]

    diagnostics["L39_subspace3_anti_bare"] = {
        "q": q39,
        "n_on": int(X39_on.shape[0]),
        "n_off": int(X39_off.shape[0]),
        "h_on_mean": round(float(h39_on.mean()), 3),
        "off_bf16_zero_frac": round(float((h39_off == 0.0).float().mean()), 5),
        "off_h_max_abs": round(float(h39_off.abs().max()), 6),
    }
    print("V9 Subspace GeGLU Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

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
        zlbl = f"V9_zero_{'_'.join(map(str, sorted(spec)))}"
        if not (ad_root / zlbl).exists():
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", spec, ad_root / zlbl, "zero"))
        for ctrl in ("candidate", "random"):
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", spec, ad_root / f"{name}_{ctrl}", ctrl))

    # 1. Pure L41 subspace causal-readout GeGLU
    add_family("V9_L41_sub_mid", [41], rho_disp=0.45, rho_ab=None)
    add_family("V9_L41_sub_hi", [41], rho_disp=0.85, rho_ab=None)
    # 2. Downstream {39, 41}: L41 subspace dispatch + L39 subspace anti-bare-pytest (both > 38!)
    add_family("V9_L41_39ab_mid", [41], rho_disp=0.50, rho_ab=0.35)
    add_family("V9_L41_39ab_hi", [41], rho_disp=0.85, rho_ab=0.45)
    # 3. Prefill-safe Global {35, 41} + L39 anti-bare-pytest
    add_family("V9_L35_41_39ab_mid", [35, 41], rho_disp=0.30, rho_ab=0.35)
    add_family("V9_L35_41_39ab_hi", [35, 41], rho_disp=0.50, rho_ab=0.45)

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V9 Subspace GeGLU adapters!", flush=True)


if __name__ == "__main__":
    main()
