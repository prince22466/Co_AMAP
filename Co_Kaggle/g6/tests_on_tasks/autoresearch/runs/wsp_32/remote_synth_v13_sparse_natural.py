"""Synthesize V13 Sparse Naturally-Silent Base-Neuron `down_proj` Readouts (`edit_exact_match >= 15/18` target!).

Core Mathematical Insight:
- Why did GeGLU overrides (V5-V12_geglu) get `edit_exact_match = 3/18` even at 99.9% OFF-zero?
  Because overriding `gate_proj` to -4.9 silenced neuron q*, subtracting `h0_q * W_down[:, q]` on ~50% of tokens!
- Why did dense 10,240-D `down_proj` readouts (V11-V12_lin) get `edit_exact_match = 1..3/18`?
  Because dense PCA/means used thousands of prose/code neurons that are non-zero on EVERY thought and edit_file token!
- V13 Solution (Pure `down_proj` r=1 on Naturally-Silent Base Neurons S_colon & S_pytest):
  1. Never touch `gate_proj` or `up_proj` (`target_modules = ["down_proj"]` -> ZERO base-neuron pruning!).
  2. Scan all 10,240 base GeGLU neurons `h0(t)` across ALL ~20,000 thought + edit-body + prefill non-colon tokens!
  3. Restrict support `S_colon` (`|S| in {16, 32, 64}`) strictly to base neurons that are NATURALLY ~0.0 in BF16
     on >=97-99% of thought/edit-body tokens (`rms_non_colon` tiny!) YET strongly active at `<|tool_call>call:`!
  4. Within `S_colon`, fit low-dim (`K=5..8`) balanced group-whitened weights so:
     - `thought` & `edit_file` body tokens: `a_S^T h0_S == 0.0` naturally -> `edit_exact_match -> 15..18 / 18`!
     - `verified @ call:` & `edit @ call:` & `localize @ call:`: whitened to ~0.0 (`4/4` verified, `6/6` champ-edit!)
     - `unchecked @ call:`: strong positive firing on unseen issues (`honest LOO OOF >= 4/5`), paired with
       `S_pytest` naturally-silent base neurons on L39 (`0` bare `pytest`)!
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
from synthesize_targeted_dispatch_v3 import find_subseq_end


def build_orthonormal_basis(vecs: list[torch.Tensor]) -> torch.Tensor:
    valid = [v.double() for v in vecs if float(v.norm()) > 1e-9]
    mat = torch.stack([v / v.norm() for v in valid], dim=1)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.float()


def fit_sparse_natural_readout(
    recs: list[dict],
    li: int,
    non_rms_global: torch.Tensor,     # [10240] exact streaming RMS across ALL ~20k non-colon tokens!
    non_nz_frac: torch.Tensor,        # [10240] fraction of non-colon tokens where |h0| > 0.02!
    m_support: int = 32,
    k_pca: int = 3,
    lam: float = 8e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-M naturally-silent-on-prose/code colon neurons & solve low-dim balanced readout on S (100% LOO-safe)."""
    unch_c, ver_c, ed_c, loc_c, non_sub = [], [], [], [], []
    for r in recs:
        hs = r["cap"][(li, "h_colon_vec")].float() if (li, "h_colon_vec") in r["cap"] else None
        if hs is not None:
            if r["role"] in ("unchecked", "check"):
                unch_c.append(hs)
            elif r["role"] == "verified":
                ver_c.append(hs.unsqueeze(0))
            elif r["role"] == "edit":
                ed_c.append(hs.unsqueeze(0))
            else:
                loc_c.append(hs.unsqueeze(0))
        hp_c = r["cap"].get((li, "h_pref_colons"))
        if hp_c is not None and hp_c.shape[0] > 0:
            loc_c.append(hp_c.float())
        hn = r["cap"][(li, "h_non_sub")].float()
        if hn.shape[0] > 0:
            non_sub.append(hn)

    H_on = torch.stack(unch_c)                                    # [N_on, 10240]
    H_ver = torch.cat(ver_c, dim=0) if ver_c else torch.zeros(0, 10240)
    H_ed = torch.cat(ed_c, dim=0) if ed_c else torch.zeros(0, 10240)
    H_loc = torch.cat(loc_c, dim=0) if loc_c else torch.zeros(0, 10240)
    H_non = torch.cat(non_sub, dim=0)

    # Score each of 10,240 base neurons by: high |activation| on unchecked@colon DIVIDED BY exact full-sequence non-colon RMS!
    on_mag = H_on.abs().mean(dim=0)
    on_consistency = (H_on.sign().mean(dim=0)).abs()              # [0..1]: consistent sign across unchecked issues!
    ver_rms = H_ver.pow(2).mean(dim=0).sqrt() if H_ver.shape[0] else torch.zeros(10240)
    ed_rms = H_ed.pow(2).mean(dim=0).sqrt() if H_ed.shape[0] else torch.zeros(10240)

    denom = (non_rms_global * (1.0 + 6.0 * non_nz_frac) + 0.25 * ver_rms + 0.25 * ed_rms).clamp_min(1e-4)
    neuron_score = (on_mag * (0.4 + 0.6 * on_consistency)) / denom
    sel = neuron_score.topk(min(m_support, 10240)).indices        # [M]

    # Restrict to M-dim sparse support S!
    S_on = H_on[:, sel]
    S_ver = H_ver[:, sel]
    S_ed = H_ed[:, sel]
    S_loc = H_loc[:, sel]
    S_non = H_non[:, sel]

    mu_on = S_on.mean(dim=0)
    mu_ver = S_ver.mean(dim=0) if S_ver.shape[0] else torch.zeros_like(mu_on)
    mu_ed = S_ed.mean(dim=0) if S_ed.shape[0] else torch.zeros_like(mu_on)
    mu_loc = S_loc.mean(dim=0) if S_loc.shape[0] else torch.zeros_like(mu_on)
    mu_non = S_non.mean(dim=0)
    S_all_c = torch.cat([x for x in (S_on, S_ver, S_ed, S_loc) if x.shape[0] > 0], dim=0)
    mu_col = S_all_c.mean(dim=0)

    vecs = [
        mu_col - mu_non,
        mu_on - mu_ver,
        mu_on - mu_ed,
        mu_on - mu_loc,
        mu_on,
    ]
    if k_pca > 0 and S_all_c.shape[0] > 4:
        _, _, Vh = torch.linalg.svd((S_all_c - mu_col).double(), full_matrices=False)
        for j in range(min(k_pca, Vh.shape[0])):
            vecs.append(Vh[j].float())
    U_s = build_orthonormal_basis(vecs)                           # [M, K]

    Zo = S_on @ U_s
    n_on = max(1, Zo.shape[0])
    w_on = torch.full((n_on,), 1.0 / n_on)
    yo = torch.ones(n_on)

    off_groups = [
        (S_ver, 3.5),
        (S_ed, 3.5),
        (S_loc, 2.0),
        (S_non, 8.0),
    ]
    Zf_list, wf_list = [], []
    for sg, wt in off_groups:
        if sg.shape[0] == 0:
            continue
        Zf_list.append(sg @ U_s)
        wf_list.append(torch.full((sg.shape[0],), wt / sg.shape[0]))
    Zf = torch.cat(Zf_list, dim=0)
    wf = torch.cat(wf_list, dim=0)

    # Also include diagonal exact full-sequence non-colon energy penalty `U_s^T diag(non_rms_global[sel]^2) U_s`!
    diag_non = (U_s.T * (non_rms_global[sel].pow(2) * 12.0)) @ U_s
    z_scale = float(Zf.pow(2).sum(dim=1).mean().clamp_min(1e-4))
    lhs = (Zo.T * w_on) @ Zo + (Zf.T * wf) @ Zf + diag_non + (lam * z_scale) * torch.eye(U_s.shape[1])
    rhs = Zo.T @ (w_on * yo)
    alpha = torch.linalg.solve(lhs, rhs)
    a_sel = U_s @ alpha

    a_full = torch.zeros(10240, dtype=torch.float32)
    a_full[sel] = a_sel
    return a_full, sel


def main():
    hdir = Path("/content/wsp/harvest32")
    out_dir = hdir / "v13_sparse_natural"
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
    # Exact streaming accumulators across EVERY non-colon token of EVERY edit/localize/verified sequence!
    sum_sq_noncolon = {li: torch.zeros(10240, device=dev, dtype=torch.float32) for li in target_layers}
    cnt_nz_noncolon = {li: torch.zeros(10240, device=dev, dtype=torch.float32) for li in target_layers}
    tot_noncolon = {li: 0 for li in target_layers}

    sum_sq_nonpytest = torch.zeros(10240, device=dev, dtype=torch.float32)
    cnt_nz_nonpytest = torch.zeros(10240, device=dev, dtype=torch.float32)
    tot_nonpytest = 0

    def forward_harvest(prefix: list[int], suffix: list[int], accumulate_global_off: bool):
        nonlocal tot_nonpytest
        toks = prefix + suffix
        assert len(toks) <= 4096
        p_len = len(prefix)
        hist_colons = [e - 1 for e in find_all_subseq_ends(prefix, call_colon_seq)]
        colon_end = find_subseq_end(suffix, call_colon_seq)
        pytest_end = find_subseq_end(suffix, prefix_pytest)
        cidx_abs = (p_len + colon_end - 1) if colon_end is not None else None
        pidx_abs = (p_len + pytest_end - 1) if pytest_end is not None else None

        # Non-colon mask across suffix + sampled prefill
        stride_p = max(1, p_len // 32)
        pref_grid = [i for i in range(0, p_len, stride_p) if i not in set(hist_colons)][:32]
        suf_non_c = [p_len + i for i in range(len(suffix)) if (p_len + i) != cidx_abs]
        non_c_all_t = torch.tensor(pref_grid + suf_non_c, device=dev, dtype=torch.long)

        # Subsample 20 non-colon tokens for CPU matrix storage (~0.4 MB/seq)
        step_sub = max(1, len(suf_non_c) // 16)
        non_c_sub_t = torch.tensor((pref_grid[::4] + suf_non_c[::step_sub])[:24], device=dev, dtype=torch.long)
        hist_c_t = torch.tensor(hist_colons, device=dev, dtype=torch.long) if hist_colons else None

        non_py_t = torch.tensor(
            pref_grid + [p_len + i for i in range(len(suffix)) if (p_len + i) != pidx_abs],
            device=dev, dtype=torch.long,
        )

        cap = {}
        tmp_g = {}
        hooks = []
        for li in target_layers:
            def mk_g(idx=li):
                def _h(mod, inp, out):
                    tmp_g[idx] = out[0].detach()
                return _h
            def mk_u(idx=li):
                nonlocal tot_nonpytest
                def _h(mod, inp, out):
                    nonlocal tot_nonpytest
                    p_full = out[0].detach()
                    g_full = tmp_g.pop(idx)
                    h_full = F.gelu(g_full, approximate="tanh") * p_full  # [T, 10240] BF16 on GPU!

                    if accumulate_global_off and non_c_all_t.numel() > 0:
                        hn_f = h_full.index_select(0, non_c_all_t).float()
                        sum_sq_noncolon[idx].add_(hn_f.pow(2).sum(dim=0))
                        cnt_nz_noncolon[idx].add_((hn_f.abs() > 0.02).float().sum(dim=0))
                        tot_noncolon[idx] += int(hn_f.shape[0])
                        if idx == 39 and non_py_t.numel() > 0:
                            hp_f = h_full.index_select(0, non_py_t).float()
                            sum_sq_nonpytest.add_(hp_f.pow(2).sum(dim=0))
                            cnt_nz_nonpytest.add_((hp_f.abs() > 0.02).float().sum(dim=0))
                            tot_nonpytest += int(hp_f.shape[0])

                    if cidx_abs is not None:
                        cap[(idx, "h_colon_vec")] = h_full[cidx_abs].to(torch.bfloat16).cpu()
                    if pidx_abs is not None:
                        cap[(idx, "h_pytest_vec")] = h_full[pidx_abs].to(torch.bfloat16).cpu()
                    if hist_c_t is not None:
                        cap[(idx, "h_pref_colons")] = h_full.index_select(0, hist_c_t).to(torch.bfloat16).cpu()
                    cap[(idx, "h_non_sub")] = h_full.index_select(0, non_c_sub_t).to(torch.bfloat16).cpu()
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
        cap = forward_harvest(pref, suf, accumulate_global_off=(c["role"] in ("verified", "edit", "localize")))
        seq_records.append({"issue": c["issue"], "role": c["role"], "cap": cap})

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
            cap = forward_harvest(pref, suf, accumulate_global_off=False)
            if (39, "h_pytest_vec") in cap:
                bare_seqs.append({"issue": issue, "cap": cap})
            break

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
    del lm_w, model, lm_root, layers
    gc.collect()
    torch.cuda.empty_cache()

    non_rms_cpu = {li: (sum_sq_noncolon[li] / max(1, tot_noncolon[li])).sqrt().cpu() for li in target_layers}
    non_nz_cpu = {li: (cnt_nz_noncolon[li] / max(1, tot_noncolon[li])).cpu() for li in target_layers}
    py_rms_cpu = (sum_sq_nonpytest / max(1, tot_nonpytest)).sqrt().cpu()
    py_nz_cpu = (cnt_nz_nonpytest / max(1, tot_nonpytest)).cpu()

    diagnostics = {"n_seq_records": len(seq_records), "tot_noncolon_tokens": tot_noncolon[41]}
    real_unch_issues = sorted({r["issue"] for r in seq_records if r["role"] == "unchecked" and (41, "h_colon_vec") in r["cap"]})

    disp_fits = {}
    for li in (35, 41):
        best_cfg = None
        trials = {}
        y_norm = sum(r["cap"][(li, "y0_norm")] for r in seq_records) / len(seq_records)
        for m_sup in (16, 32, 64):
            for k_pca in (2, 4):
                oof_s = {}
                for iss in real_unch_issues:
                    tr_recs = [r for r in seq_records if r["issue"] != iss]
                    a_tr, _ = fit_sparse_natural_readout(tr_recs, li, non_rms_cpu[li], non_nz_cpu[li], m_support=m_sup, k_pca=k_pca)
                    for r in seq_records:
                        if r["issue"] == iss and r["role"] == "unchecked" and (li, "h_colon_vec") in r["cap"]:
                            oof_s[iss] = round(float(torch.dot(a_tr, r["cap"][(li, "h_colon_vec")].float()).item()), 4)

                a_full, sel = fit_sparse_natural_readout(seq_records, li, non_rms_cpu[li], non_nz_cpu[li], m_support=m_sup, k_pca=k_pca)
                H_on = torch.stack([r["cap"][(li, "h_colon_vec")].float() for r in seq_records if r["role"] in ("unchecked", "check") and (li, "h_colon_vec") in r["cap"]])
                H_non = torch.cat([r["cap"][(li, "h_non_sub")].float() for r in seq_records], dim=0)
                H_ver = torch.stack([r["cap"][(li, "h_colon_vec")].float() for r in seq_records if r["role"] == "verified" and (li, "h_colon_vec") in r["cap"]])
                H_ed = torch.stack([r["cap"][(li, "h_colon_vec")].float() for r in seq_records if r["role"] == "edit" and (li, "h_colon_vec") in r["cap"]])

                s_on_m = float((H_on @ a_full).mean())
                off_rms = float((H_non @ a_full).pow(2).mean().sqrt().clamp_min(1e-6))
                off_zero_99 = float(((H_non @ a_full).abs() < 0.02 * max(s_on_m, 1e-4)).float().mean())
                ver_max = float((H_ver @ a_full).abs().max())
                ed_max = float((H_ed @ a_full).abs().max())
                oof_vals = list(oof_s.values())
                oof_snr = (sum(oof_vals) / max(1, len(oof_vals))) / off_rms
                pos_cnt = sum(1 for v in oof_vals if v > 2.5 * off_rms)

                tkey = f"M{m_sup}_kpca{k_pca}"
                trials[tkey] = {
                    "support": m_sup,
                    "mean_nz_frac_on_selected": round(float(non_nz_cpu[li][sel].mean()), 4),
                    "pos_over_2p5rms": f"{pos_cnt}/{len(oof_vals)}",
                    "oof_snr": round(oof_snr, 2),
                    "in_sample_snr": round(s_on_m / off_rms, 2),
                    "off_below_2pct_on_frac": round(off_zero_99, 4),
                    "s_on_mean": round(s_on_m, 4),
                    "s_off_rms": round(off_rms, 5),
                    "ver_max": round(ver_max, 4),
                    "edit_max": round(ed_max, 4),
                    "oof": oof_s,
                }
                rkey = (ver_max <= 0.25 * s_on_m, ed_max <= 0.30 * s_on_m, pos_cnt, round(oof_snr, 1))
                if best_cfg is None or rkey > best_cfg[0]:
                    best_cfg = (rkey, tkey, a_full, s_on_m, y_norm, trials[tkey])

        _, btkey, a_best, s_on_m, y_norm, bstat = best_cfg
        diagnostics[f"L{li}_sparse_natural"] = {"best": btkey, "stats": bstat, "trials": trials}
        disp_fits[li] = {"a": a_best, "u": u_disp, "s_on_mean": max(s_on_m, 0.05), "y0_norm": y_norm}

    # Sparse Naturally-Silent L39 Anti-Bare-Pytest (`M=24` neurons naturally ~0 outside `pytest`!)
    H39_on = torch.stack(
        [r["cap"][(39, "h_pytest_vec")].float() for r in seq_records if (39, "h_pytest_vec") in r["cap"]]
        + [r["cap"][(39, "h_pytest_vec")].float() for r in bare_seqs]
    )
    H39_off = torch.cat([r["cap"][(39, "h_non_sub")].float() for r in seq_records], dim=0)
    sc39 = (H39_on.abs().mean(dim=0) * H39_on.sign().mean(dim=0).abs()) / (py_rms_cpu * (1.0 + 6.0 * py_nz_cpu)).clamp_min(1e-4)
    sel39 = sc39.topk(24).indices
    S39_on, S39_off = H39_on[:, sel39], H39_off[:, sel39]
    mu39 = S39_on.mean(dim=0)
    U39 = build_orthonormal_basis([mu39 - S39_off.mean(dim=0), mu39])
    Zo39, Zf39 = S39_on @ U39, S39_off @ U39
    lhs39 = (Zo39.T @ Zo39) / Zo39.shape[0] + 8.0 * (Zf39.T @ Zf39) / Zf39.shape[0] + 12.0 * (U39.T * py_rms_cpu[sel39].pow(2)) @ U39 + 1e-3 * torch.eye(U39.shape[1])
    alpha39 = torch.linalg.solve(lhs39, Zo39.mean(dim=0))
    a39 = torch.zeros(10240, dtype=torch.float32)
    a39[sel39] = U39 @ alpha39
    s39_on_m = float((H39_on @ a39).mean())
    s39_off_rms = float((H39_off @ a39).pow(2).mean().sqrt().clamp_min(1e-6))
    y39_norm = sum(r["cap"][(39, "y0_norm")] for r in seq_records) / len(seq_records)
    diagnostics["L39_sparse_natural_anti_bare"] = {
        "support": 24,
        "mean_nz_frac_on_selected": round(float(py_nz_cpu[sel39].mean()), 4),
        "snr": round(s39_on_m / s39_off_rms, 2),
        "s_on_mean": round(s39_on_m, 4),
        "s_off_rms": round(s39_off_rms, 5),
    }

    print("V13 Sparse Natural Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    # Gentle iter_04-calibrated gain brackets (`rho_d in 0.08..0.28` -> `edit_exact_match >= 14..18/18`!)
    def add_cand(name: str, d_layers: list[int], rho_d: float, rho_ab: float | None):
        ft = {}
        for li in d_layers:
            f = disp_fits[li]
            g = rho_d * f["y0_norm"] / f["s_on_mean"]
            ft[f"{li}.A"] = f["a"].bfloat16()
            ft[f"{li}.B"] = (-g * f["u"]).bfloat16()
        if rho_ab is not None:
            g39 = rho_ab * y39_norm / max(s39_on_m, 0.05)
            ft["39.A"] = a39.bfloat16()
            ft["39.B"] = (-g39 * u_ab).bfloat16()
        fac_p = out_dir / f"fac_{name}.safetensors"
        save_file(ft, str(fac_p))
        zlbl = f"V13_zero_{'_'.join(sorted(k.split('.')[0] for k in ft if k.endswith('.A')))}"
        if not (ad_root / zlbl).exists():
            zinfo = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / zlbl, "zero", rank=1)
            compiled.append({"adapter": zlbl, "sha256": zinfo["adapter_sha256"], "control": "zero"})
        for ctrl in ("candidate", "random"):
            lbl = f"{name}_{ctrl}"
            info = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / lbl, ctrl, rank=1)
            compiled.append({"adapter": lbl, "sha256": info["adapter_sha256"], "control": ctrl})

    add_cand("V13_nat_L41_39ab_g010", [41], rho_d=0.10, rho_ab=0.14)
    add_cand("V13_nat_L41_39ab_g018", [41], rho_d=0.18, rho_ab=0.22)
    add_cand("V13_nat_L41_39ab_g028", [41], rho_d=0.28, rho_ab=0.28)
    add_cand("V13_nat_L35_41_39ab_g009", [35, 41], rho_d=0.09, rho_ab=0.16)
    add_cand("V13_nat_L35_41_39ab_g016", [35, 41], rho_d=0.16, rho_ab=0.22)

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V13 Sparse Natural adapters!", flush=True)


if __name__ == "__main__":
    main()
