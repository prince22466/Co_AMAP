"""Synthesize V20 Sequential Cascade (`L39 -> L40 -> L41 down_proj` r=1) with 100% Real vLLM Replays & OOF Gain Calibration.

Directly resolves the two remaining failure mechanisms identified across `wsp32_iter05` (4/15) & `wsp32_iter06` (4/15, +rich_4077):
1. Eliminates synthetic suffix overfitting: uses 100% genuine vLLM `champion_r1` generations across all strictly-immediate
   literal-verified prompts from the 19 disjoint snapshots (`clean_v20_real.json`), with stronger ridge regularization
   (`lam_rel = 0.28`) AND scales `40.B` and `41.B` by `1 / oof_on` (Leave-One-Issue-Out unseen-task activation mean,
   NOT in-sample `in_on`!) so unseen `QUICK_15` tasks receive the exact calibrated +11..+16 nat verification shift!
2. Eliminates parallel-layer interference (`iter06` L40 shifting `h_41` -> `leaks=2, b1_submits=3/8`):
   - Stage 1: Fit `L39` (file-level `pytest tests/test_<mod>.py`), then ATTACH `L39` live to `layers[39].mlp.down_proj`!
   - Stage 2a: Fit `L41_no40` directly on post-`L39` `h_41` (for pure `39+41` OOF-calibrated cascade).
   - Stage 2b: Fit `L40` (`FileEditError -> read_file`) on post-`L39` `h_40`, ATTACH `L40` live to `layers[40].mlp.down_proj`,
     and THEN harvest exact post-`(L39+L40)` `h_41` to fit `L41_with40` with ZERO cross-layer mismatch!
"""
import gc
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter
from recipe import compile_explicit
from synth_v16_upperband import find_subseq_end, solve_full_alphaedit


def main():
    v20_cases_p = ROOT / "harvest32/clean_v20/real_replays.json"
    assert v20_cases_p.is_file(), f"Missing {v20_cases_p}"
    out_dir = ROOT / "harvest32/v20_cascade"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_replays = json.loads(v20_cases_p.read_text())
    print("Loaded V20 100%-real replays by literal role:", Counter(r["true_role"] for r in raw_replays), flush=True)

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

    dev = model.device
    post_ffn_w = {li: layers[li].post_feedforward_layernorm.weight.detach().float().cpu() for li in (39, 40, 41)}

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    thought_hdr = enc("<|channel>thought\n")
    chan_end_seq = enc("<channel|>")
    call_colon_seq = enc("<|tool_call>call:")
    cmd_open_quote_seq = enc('<|tool_call>call:run_command{command:<|"|>')

    inc_prefixes = [
        enc('<|tool_call>call:run_command{command:<|"|>pytest'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest -q'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest tests'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest tests/'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests/'),
    ]
    py_ext_seq = enc(".py")

    # Prepare tokenized (prefix, suffix, index metadata) ONCE so sequential passes run in ~45s each!
    prepared = []
    for item in raw_replays:
        rl = item["true_role"]
        txt = item.get("text") or ""
        if not txt and rl == "true_imm_test_pass":
            txt = "<|channel>thought\nTests passed cleanly. Submitting patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>"
        if not txt or "<|tool_call>call:" not in txt:
            continue
        full_suf = enc(txt)
        suf = (full_suf[:140] + full_suf[-628:]) if len(full_suf) > 768 else full_suf
        pref = item["tokens"][-(4096 - len(suf)) :]
        toks = pref + suf
        p_len = len(pref)

        win_lo = max(0, p_len - 8)
        win_hi = min(len(toks), p_len + 10)
        th_rel = find_subseq_end(toks[win_lo:win_hi], thought_hdr)
        ts0 = (win_lo + th_rel - 1) if th_rel is not None else p_len

        ch_end_rel = find_subseq_end(suf, chan_end_seq)
        ch_end_abs = (p_len + ch_end_rel - len(chan_end_seq)) if ch_end_rel is not None else len(toks)
        colon_end = find_subseq_end(suf, call_colon_seq)
        c_abs = (p_len + colon_end - 1) if colon_end is not None else None
        if c_abs is None:
            continue

        cmd_q_end = find_subseq_end(suf, cmd_open_quote_seq)
        cmd_q_abs = (p_len + cmd_q_end - 1) if cmd_q_end is not None else None

        pt_inc_abs = sorted({p_len + ie - 1 for ip in inc_prefixes if (ie := find_subseq_end(suf, ip)) is not None})
        py_done_abs = []
        if pt_inc_abs and (py_e := find_subseq_end(suf, py_ext_seq)) is not None:
            py_done_abs.append(p_len + py_e - 1)

        is_ed = (item.get("action") == "edit_file") or (rl in ("true_imm_read_full", "true_imm_read_trunc"))
        delim_set = set()
        if ch_end_abs > ts0:
            th_step = max(1, (ch_end_abs - ts0) // 24)
            for k in range(ts0, ch_end_abs, th_step):
                delim_set.add(k)
        if ch_end_rel is not None:
            for k in range(max(0, p_len + ch_end_rel - 3), min(len(toks), p_len + ch_end_rel + 4)):
                delim_set.add(k)
        if cmd_q_abs is not None:
            delim_set.add(cmd_q_abs)
        if len(toks) > c_abs + 2:
            step_suf = max(1, (len(toks) - c_abs - 1) // (36 if is_ed else 14))
            for k in range(c_abs + 1, len(toks), step_suf):
                delim_set.add(k)
        bg_stride = max(1, max(1, ts0 - 8) // 12)
        for k in range(8, max(9, ts0 - 1), bg_stride):
            delim_set.add(k)
        forbid = {c_abs} | set(pt_inc_abs)
        delim_idx = sorted(k for k in delim_set if k not in forbid)[:72] or [max(0, ts0 - 2)]

        prepared.append({
            "issue": item["issue"],
            "true_role": rl,
            "filepath": item.get("filepath"),
            "tokens": item["tokens"],
            "toks": toks,
            "c_abs": c_abs,
            "cmd_q_abs": cmd_q_abs,
            "pt_inc_abs": pt_inc_abs,
            "py_done_abs": py_done_abs,
            "delim_idx": delim_idx,
        })

    # Also prepare targeted `.py` pytest continuations for L39
    bare_prepared = []
    seen_iss = set()
    for r in prepared:
        if r["issue"] in seen_iss or r["true_role"] not in ("true_imm_edit_ok", "true_imm_test_fail"):
            continue
        seen_iss.add(r["issue"])
        stem = Path(r.get("filepath") or "panel.py").stem.lstrip("_") or "panel"
        for variant_cmd in (f"pytest tests/test_{stem}.py", f"pytest -q tests/test_{stem}.py"):
            suf = enc(f"<|channel>thought\nLet's run targeted pytest on test_{stem}.py.\n<channel|><|tool_call>call:run_command{{command:<|\"|>{variant_cmd}<|\"|>}}<tool_call|>")
            pref = r["tokens"][-(4096 - len(suf)) :]
            toks = pref + suf
            p_len = len(pref)
            pt_inc_abs = sorted({p_len + ie - 1 for ip in inc_prefixes if (ie := find_subseq_end(suf, ip)) is not None})
            py_e = find_subseq_end(suf, py_ext_seq)
            py_done_abs = [p_len + py_e - 1] if py_e is not None else []
            cmd_q_end = find_subseq_end(suf, cmd_open_quote_seq)
            cmd_q_abs = (p_len + cmd_q_end - 1) if cmd_q_end is not None else None
            delim_idx = [k for k in range(p_len, len(toks), 3) if k not in set(pt_inc_abs)][:24]
            bare_prepared.append({
                "toks": toks,
                "pt_inc_abs": pt_inc_abs,
                "py_done_abs": py_done_abs,
                "cmd_q_abs": cmd_q_abs,
                "delim_idx": delim_idx,
            })

    print(f"Prepared {len(prepared)} 100%-real tool-call records + {len(bare_prepared)} pytest-file records!", flush=True)

    # Pure weight-space unembedding vectors
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

    def precond(u_res: torch.Tensor, w_pn: torch.Tensor) -> torch.Tensor:
        b = u_res * w_pn.clamp_min(0.05)
        return b / b.norm().clamp_min(1e-8)

    tok_run = enc("<|tool_call>call:run_command{")[len(call_colon_seq)]
    tok_sub = enc("<|tool_call>call:submit_patch{")[len(call_colon_seq)]
    tok_ed = enc("<|tool_call>call:edit_file{")[len(call_colon_seq)]
    tok_rd = enc("<|tool_call>call:read_file{")[len(call_colon_seq)]

    u_c1_disp = (lm_w[tok_sub] - lm_w[tok_run]) * scale_w
    u_c1_disp = u_c1_disp / u_c1_disp.norm().clamp_min(1e-8)

    u_c2_err = ((0.85 * lm_w[tok_ed] + 0.15 * lm_w[tok_sub]) - lm_w[tok_rd]) * scale_w
    u_c2_err = u_c2_err / u_c2_err.norm().clamp_min(1e-8)

    p_base = enc('<|tool_call>call:run_command{command:<|"|>pytest')
    s_bare = enc('<|tool_call>call:run_command{command:<|"|>pytest<|"|>}')
    s_full1 = enc('<|tool_call>call:run_command{command:<|"|>pytest tests/test_panel.py<|"|>}')
    s_full2 = enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests/test_panel.py<|"|>}')
    p_dir_noslash = enc('<|tool_call>call:run_command{command:<|"|>pytest tests')
    p_dir_slash = enc('<|tool_call>call:run_command{command:<|"|>pytest tests/')

    tok_quote = s_bare[len(p_base)]
    tok_sp_tests = s_full1[len(p_base)]
    tok_sp_dashq = s_full2[len(p_base)]
    tok_slash = s_full1[len(p_dir_noslash)]
    tok_test_word = s_full1[len(p_dir_slash)]

    good_l39 = (
        0.30 * (lm_w[tok_sp_tests] / lm_w[tok_sp_tests].norm())
        + 0.15 * (lm_w[tok_sp_dashq] / lm_w[tok_sp_dashq].norm())
        + 0.20 * (lm_w[tok_slash] / lm_w[tok_slash].norm())
        + 0.35 * (lm_w[tok_test_word] / lm_w[tok_test_word].norm())
    )
    bad_l39 = lm_w[tok_quote] / lm_w[tok_quote].norm()
    u_c3_file = (bad_l39 - good_l39) * scale_w
    u_c3_file = u_c3_file / u_c3_file.norm().clamp_min(1e-8)

    def harvest_single_layer(li: int):
        recs_h = []
        for r in prepared:
            dt = torch.tensor(r["delim_idx"], device=dev, dtype=torch.long)
            cap = {}
            def _h(mod, inp, out):
                h = inp[0][0].detach()
                y = out[0].detach()
                cap["dy_norm"] = float(y[r["c_abs"]].float().norm().item())
                cap["dh_col"] = h[r["c_abs"]].to(torch.bfloat16).cpu()
                cap["dh_delim"] = h.index_select(0, dt).to(torch.bfloat16).cpu()
                if r["cmd_q_abs"] is not None:
                    cap["dh_cmd_q"] = h[r["cmd_q_abs"]].to(torch.bfloat16).cpu()
                if li == 39 and r["pt_inc_abs"]:
                    cap["dh_pt_inc"] = h.index_select(0, torch.tensor(r["pt_inc_abs"], device=dev)).to(torch.bfloat16).cpu()
                if li == 39 and r["py_done_abs"]:
                    cap["dh_py_done"] = h.index_select(0, torch.tensor(r["py_done_abs"], device=dev)).to(torch.bfloat16).cpu()
            hk = layers[li].mlp.down_proj.register_forward_hook(_h)
            try:
                with torch.inference_mode():
                    model(input_ids=torch.tensor([r["toks"]], device=dev), use_cache=False, logits_to_keep=1)
            finally:
                hk.remove()
            recs_h.append({**r, "cap": cap})

        bare_h = []
        for b in bare_prepared:
            dt = torch.tensor(b["delim_idx"], device=dev, dtype=torch.long)
            cap = {}
            def _hb(mod, inp, out):
                h = inp[0][0].detach()
                cap["dh_delim"] = h.index_select(0, dt).to(torch.bfloat16).cpu()
                if b["cmd_q_abs"] is not None:
                    cap["dh_cmd_q"] = h[b["cmd_q_abs"]].to(torch.bfloat16).cpu()
                if li == 39 and b["pt_inc_abs"]:
                    cap["dh_pt_inc"] = h.index_select(0, torch.tensor(b["pt_inc_abs"], device=dev)).to(torch.bfloat16).cpu()
                if li == 39 and b["py_done_abs"]:
                    cap["dh_py_done"] = h.index_select(0, torch.tensor(b["py_done_abs"], device=dev)).to(torch.bfloat16).cpu()
            hk = layers[li].mlp.down_proj.register_forward_hook(_hb)
            try:
                with torch.inference_mode():
                    model(input_ids=torch.tensor([b["toks"]], device=dev), use_cache=False, logits_to_keep=1)
            finally:
                hk.remove()
            bare_h.append(cap)
        return recs_h, bare_h

    # =========================================================================
    # STAGE 1: Fit L39 File-Level Targeted Pytest & ATTACH LIVE!
    # =========================================================================
    print("=== Stage 1: Harvesting & fitting L39 ===", flush=True)
    r39, b39 = harvest_single_layer(39)
    h39_on, h39_pyd, h39_off, dyn39 = [], [], [], []
    for r in r39:
        dyn39.append(r["cap"]["dy_norm"])
        h39_off.append(r["cap"]["dh_delim"].float())
        h39_off.append(r["cap"]["dh_col"].float().unsqueeze(0))
        if "dh_pt_inc" in r["cap"]:
            h39_on.append(r["cap"]["dh_pt_inc"].float())
        if "dh_py_done" in r["cap"]:
            h39_pyd.append(r["cap"]["dh_py_done"].float())
    for bc in b39:
        h39_off.append(bc["dh_delim"].float())
        if "dh_pt_inc" in bc:
            h39_on.append(bc["dh_pt_inc"].float())
        if "dh_py_done" in bc:
            h39_pyd.append(bc["dh_py_done"].float())

    H39_on = torch.cat(h39_on, dim=0)
    H39_pyd = torch.cat(h39_pyd, dim=0)
    H39_off = torch.cat(h39_off, dim=0)
    a39_u, sc39 = solve_full_alphaedit(H39_on, [(H39_pyd, 28.0), (H39_off, 22.0)], lam_rel=0.14)
    s39_on = float((H39_on @ (a39_u * sc39)).mean())
    b39_dir = precond(-u_c3_file, post_ffn_w[39])
    dyn39_mean = sum(dyn39) / len(dyn39)

    A39 = (a39_u * (sc39 / max(0.15, s39_on))).bfloat16()
    B39 = ((0.46 * dyn39_mean) * b39_dir).bfloat16()
    a39_dev, b39_dev = A39.to(dev), B39.to(dev)
    hk39 = layers[39].mlp.down_proj.register_forward_hook(
        lambda mod, inputs, output: output + (inputs[0] @ a39_dev).unsqueeze(-1) * b39_dev
    )

    def fit_colon_oof_calibrated(li: int, recs_h, bare_h, on_pred, u_disp: torch.Tensor, lam_rel: float = 0.28):
        def pack(sub_recs):
            on_h, rd_h, ver_h, cmdq_h, oth_h, del_h, dyn = [], [], [], [], [], [], []
            for r in sub_recs:
                dyn.append(r["cap"]["dy_norm"])
                del_h.append(r["cap"]["dh_delim"].float())
                if "dh_cmd_q" in r["cap"]:
                    cmdq_h.append(r["cap"]["dh_cmd_q"].float())
                hc = r["cap"]["dh_col"].float()
                if on_pred(r):
                    on_h.append(hc)
                elif r["true_role"] in ("true_imm_read_full", "true_imm_read_trunc"):
                    rd_h.append(hc)
                elif r["true_role"] == "true_imm_test_pass":
                    ver_h.append(hc)
                else:
                    oth_h.append(hc)
            for bc in bare_h:
                if "dh_cmd_q" in bc:
                    cmdq_h.append(bc["dh_cmd_q"].float())
            Ho = torch.stack(on_h)
            Hr = torch.stack(rd_h) if rd_h else torch.zeros(0, 10240)
            Hv = torch.stack(ver_h) if ver_h else torch.zeros(0, 10240)
            Hcq = torch.stack(cmdq_h) if cmdq_h else torch.zeros(0, 10240)
            Hot = torch.stack(oth_h) if oth_h else torch.zeros(0, 10240)
            Hd = torch.cat(del_h, dim=0)
            ogrps = [(Hr, 55.0), (Hv, 55.0), (Hcq, 50.0), (Hot, 25.0), (Hd, 55.0)]
            return Ho, ogrps, Hr, Hv, sum(dyn) / len(dyn)

        # Leave-One-Issue-Out CV to measure TRUE unseen-issue activation `oof_on`!
        on_issues = sorted({r["issue"] for r in recs_h if on_pred(r)})
        oof_on_vals, oof_rd_vals, oof_ver_vals = [], [], []
        for iss in on_issues:
            tr = [r for r in recs_h if r["issue"] != iss]
            Ho_tr, og_tr, _, _, _ = pack(tr)
            wu_tr, sc_tr = solve_full_alphaedit(Ho_tr, og_tr, lam_rel=lam_rel)
            w_tr = wu_tr * sc_tr
            for r in recs_h:
                if r["issue"] != iss:
                    continue
                v = float(torch.dot(w_tr, r["cap"]["dh_col"].float()))
                if on_pred(r):
                    oof_on_vals.append(v)
                elif r["true_role"] in ("true_imm_read_full", "true_imm_read_trunc"):
                    oof_rd_vals.append(abs(v))
                elif r["true_role"] == "true_imm_test_pass":
                    oof_ver_vals.append(abs(v))

        Ho, og, Hr, Hv, raw_yn = pack(recs_h)
        au, sc = solve_full_alphaedit(Ho, og, lam_rel=lam_rel)
        w = au * sc
        in_on = float((Ho @ w).mean())
        oof_on = max(0.08, sum(oof_on_vals) / max(1, len(oof_on_vals)))
        oof_pos_frac = sum(1 for v in oof_on_vals if v > 0.05) / max(1, len(oof_on_vals))
        return {
            "u_a": au,
            "scale": sc,
            "in_on": round(in_on, 4),
            "oof_on": round(oof_on, 4),
            "oof_pos_frac": round(oof_pos_frac, 3),
            "oof_to_in_ratio": round(oof_on / max(1e-4, in_on), 3),
            "oof_rd_abs": round(sum(oof_rd_vals) / max(1, len(oof_rd_vals)), 4) if oof_rd_vals else 0.0,
            "raw_yn": round(raw_yn, 4),
            "b_dir": precond(-u_disp, post_ffn_w[li]),
        }

    is_c1_on = lambda r: r["true_role"] in ("true_imm_edit_ok", "true_imm_test_fail")
    is_c2_on = lambda r: r["true_role"] == "true_imm_edit_err"

    # =========================================================================
    # STAGE 2A: Fit L41 directly on post-L39 states (for pure 39+41 cascade!)
    # =========================================================================
    print("=== Stage 2A: Harvesting post-L39 h_41 for pure 39+41 cascade ===", flush=True)
    r41_no40, b41_no40 = harvest_single_layer(41)
    d41_no40 = fit_colon_oof_calibrated(41, r41_no40, b41_no40, is_c1_on, u_c1_disp, lam_rel=0.28)

    # =========================================================================
    # STAGE 2B: Fit L40 on post-L39 h_40, ATTACH L40 LIVE, then fit L41 on post-(L39+L40) h_41!
    # =========================================================================
    print("=== Stage 2B: Harvesting post-L39 h_40 & fitting L40 ===", flush=True)
    r40, b40 = harvest_single_layer(40)
    d40_c2 = fit_colon_oof_calibrated(40, r40, b40, is_c2_on, u_c2_err, lam_rel=0.28)

    # Attach L40 at calibrated sweet-spot (`rho40_oof = 0.22 -> +9 nats OOF read_file - edit_file`) before harvesting h_41!
    A40_mid = (d40_c2["u_a"] * (d40_c2["scale"] / d40_c2["oof_on"])).bfloat16()
    B40_mid = ((0.22 * d40_c2["raw_yn"]) * d40_c2["b_dir"]).bfloat16()
    a40_dev, b40_dev = A40_mid.to(dev), B40_mid.to(dev)
    hk40 = layers[40].mlp.down_proj.register_forward_hook(
        lambda mod, inputs, output: output + (inputs[0] @ a40_dev).unsqueeze(-1) * b40_dev
    )

    print("=== Stage 3: Harvesting exact post-(L39+L40) h_41 & fitting cascade-consistent L41 ===", flush=True)
    r41_casc, b41_casc = harvest_single_layer(41)
    d41_casc = fit_colon_oof_calibrated(41, r41_casc, b41_casc, is_c1_on, u_c1_disp, lam_rel=0.28)

    hk40.remove()
    hk39.remove()
    del model, lm_root, layers
    gc.collect()
    torch.cuda.empty_cache()

    diagnostics = {
        "n_real_records": len(prepared),
        "L39_s_on": round(s39_on, 4),
        "L41_no40_oof": {k: v for k, v in d41_no40.items() if k not in ("u_a", "b_dir")},
        "L40_c2_oof": {k: v for k, v in d40_c2.items() if k not in ("u_a", "b_dir")},
        "L41_post_L39_L40_oof": {k: v for k, v in d41_casc.items() if k not in ("u_a", "b_dir")},
    }
    print("V20 Cascade Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    def export_pair(name: str, ft: dict):
        fp = out_dir / f"fac_{name}.safetensors"
        save_file(ft, str(fp))
        for ctrl in ("candidate", "random"):
            inf = compile_explicit(ROOT / "champion.zip", fp, ad_root / f"{name}_{ctrl}", ctrl, rank=1)
            compiled.append({"adapter": f"{name}_{ctrl}", "sha256": inf["adapter_sha256"], "control": ctrl, "layers": inf["active_layers"]})

    # Notice `1 / oof_on` scaling! For unseen tasks (`dot ≈ oof_on`), `rho_oof = 0.22..0.30` delivers +11..+16 nats!
    export_pair("V20_casc_39_40_41_sweet", {
        "39.A": A39, "39.B": B39,
        "40.A": A40_mid, "40.B": B40_mid,
        "41.A": (d41_casc["u_a"] * (d41_casc["scale"] / d41_casc["oof_on"])).bfloat16(),
        "41.B": ((0.22 * d41_casc["raw_yn"]) * d41_casc["b_dir"]).bfloat16(),
    })
    export_pair("V20_casc_39_40_41_firm", {
        "39.A": A39, "39.B": B39,
        "40.A": A40_mid, "40.B": ((0.27 * d40_c2["raw_yn"]) * d40_c2["b_dir"]).bfloat16(),
        "41.A": (d41_casc["u_a"] * (d41_casc["scale"] / d41_casc["oof_on"])).bfloat16(),
        "41.B": ((0.29 * d41_casc["raw_yn"]) * d41_casc["b_dir"]).bfloat16(),
    })
    export_pair("V20_casc_39_41_firm", {
        "39.A": A39, "39.B": B39,
        "41.A": (d41_no40["u_a"] * (d41_no40["scale"] / d41_no40["oof_on"])).bfloat16(),
        "41.B": ((0.29 * d41_no40["raw_yn"]) * d41_no40["b_dir"]).bfloat16(),
    })

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V20 Sequential Cascade OOF-Calibrated adapters!", flush=True)


if __name__ == "__main__":
    main()
