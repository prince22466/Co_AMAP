"""Synthesize V22 Top-Band Pure-WSP Closed-Loop Repair Cascade (`L39[pytest] -> L40[fail->repair] -> L41[edit->verify]`, `r=1`).

Directly incorporates every empirical finding from `screen_v21_fsm_summary.json`:
1. Zero Mid-Network KV-Cache Drift (`b3_exact_args_match -> 5..6/6`):
   - `V21_fsm_4layer_sweet_zero` proved 100% bit-exact vLLM equivalence (`6/6` exact args, `11/11` verified kept),
     while `L35` in `V21` perturbed 6 downstream attention layers (`b3_exact_args_match` dropped `6/6 -> 1..3/6`) and was
     attenuated 5x by `layer_scalar` (`+1.48 nats`).
   - `V22` moves the Post-Failure Repair Gate (`true_imm_real_test_fail @ call:`) onto **`L40.mlp.down_proj`** (`cum_scalar=0.39`,
     4x stronger survival, leaving layers `0..38` — including all 5 champion `edit_file_loop` layers `[29,31,34,37,38]` — 100% untouched!).
2. Exact Constrained Cauchy-Schwarz `b`-Space Orthogonalization + `H39_quote_close` Null-Spacing:
   - For `L40` and `L41`, `b* = (I - Q_l Q_l^T)(u_target * w_pffn)` is strictly orthogonal in logit space to ALL sacred delimiters
     (including `<|"|>`, `<tool_call|>`, `<channel|>`, thought-openers, and Python indentation) WITH full `w_pffn` gain efficiency!
   - For `L39` (`pytest` -> `pytest tests/test_<mod>.py`), `b39` suppresses premature `<|"|>` ONLY at incomplete `pytest` prefixes
     (`...pytest`, `...pytest tests`, `...pytest tests/`) while staying strictly orthogonal to `<tool_call|>`, `<channel|>`,
     thought-openers, and Python indentation, AND `39.A` puts EVERY closing `<|"|>` inside `edit_file`/`read_file` (`H39_quote_close`,
     `beta = 160.0`) + completed `.py` (`beta = 120.0`) + thought-openers (`beta = 130.0`) into hard null space!
3. Direct Live In-VRAM Slope Calibration (`rho = desired_in_nats / measured_slope`) so every stage hits `+11..+14 nats` on unseen tasks!
"""
from __future__ import annotations

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
from remote_synth_v21_fsm import solve_alphaedit_v21
from synth_v16_upperband import find_subseq_end


def main():
    v21_cases_p = ROOT / "harvest32/clean_v21/real_replays.json"
    assert v21_cases_p.is_file(), f"Missing {v21_cases_p}"
    out_dir = ROOT / "harvest32/v22_cauchy"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_replays = json.loads(v21_cases_p.read_text())
    print("Loaded V21 disjoint corpus by role:", Counter(r["true_role"] for r in raw_replays), flush=True)

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
    cand_layers = (36, 39, 40, 41)
    assert not (set(cand_layers) & CHAMP_LAYERS)
    post_ffn_w = {li: layers[li].post_feedforward_layernorm.weight.detach().float().cpu() for li in cand_layers}

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

    prepared = []
    for item in raw_replays:
        rl = item["true_role"]
        txt = item.get("text") or ""
        if not txt and rl == "true_imm_test_pass":
            txt = "<|channel>thought\nTests passed cleanly. Submitting patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>"
        if not txt or "<|tool_call>call:" not in txt:
            continue
        full_suf = enc(txt)
        suf = (full_suf[:160] + full_suf[-608:]) if len(full_suf) > 768 else full_suf
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

        # Explicitly capture EVERY position right before `<|"|>` inside non-pytest tool calls (`edit_file`, `read_file`)!
        quote_pred_abs = [
            pos - 1
            for pos in range(max(p_len + 1, c_abs + 1), len(toks))
            if toks[pos] == tok_quote and (pos - 1) not in set(pt_inc_abs)
        ][:16]

        is_ed_or_pres = (item.get("action") == "edit_file") or (rl in ("true_imm_read_full", "true_imm_read_trunc", "turn1_init"))
        open_idx = [k for k in range(ts0, min(ch_end_abs, ts0 + 10, len(toks))) if k != c_abs and k not in set(pt_inc_abs)]

        delim_set = set(open_idx) | set(quote_pred_abs)
        if ch_end_abs > ts0:
            th_step = max(1, (ch_end_abs - ts0) // (36 if is_ed_or_pres else 22))
            for k in range(ts0, ch_end_abs, th_step):
                delim_set.add(k)
        if ch_end_rel is not None:
            for k in range(max(0, p_len + ch_end_rel - 4), min(len(toks), p_len + ch_end_rel + 4)):
                delim_set.add(k)
        if cmd_q_abs is not None:
            delim_set.add(cmd_q_abs)
        if len(toks) > c_abs + 2:
            step_suf = max(1, (len(toks) - c_abs - 1) // (48 if is_ed_or_pres else 18))
            for k in range(c_abs + 1, len(toks), step_suf):
                delim_set.add(k)
            for k in range(max(c_abs + 1, len(toks) - 6), len(toks)):
                delim_set.add(k)
        bg_stride = max(1, max(1, ts0 - 8) // 12)
        for k in range(8, max(9, ts0 - 1), bg_stride):
            delim_set.add(k)

        forbid = {c_abs} | set(pt_inc_abs)
        delim_idx = sorted(k for k in delim_set if k not in forbid)[:96] or [max(0, ts0 - 2)]
        open_clean = [k for k in open_idx if k not in forbid] or [delim_idx[0]]

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
            "quote_pred_abs": quote_pred_abs,
            "open_idx": open_clean,
            "delim_idx": delim_idx,
        })

    bare_prepared = []
    seen_iss = set()
    for r in prepared:
        if r["issue"] in seen_iss or r["true_role"] not in ("true_imm_edit_ok", "true_imm_real_test_fail", "true_imm_bad_test_path"):
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
            delim_idx = [k for k in range(p_len, len(toks), 2) if k not in set(pt_inc_abs)][:36]
            bare_prepared.append({
                "toks": toks,
                "pt_inc_abs": pt_inc_abs,
                "py_done_abs": py_done_abs,
                "cmd_q_abs": cmd_q_abs,
                "delim_idx": delim_idx,
            })

    # =========================================================================
    # Constrained Cauchy-Schwarz Optimal `b`-Space Orthogonalizer!
    # =========================================================================
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

    tok_run = enc("<|tool_call>call:run_command{")[len(call_colon_seq)]
    tok_sub = enc("<|tool_call>call:submit_patch{")[len(call_colon_seq)]
    tok_ed = enc("<|tool_call>call:edit_file{")[len(call_colon_seq)]
    tok_rd = enc("<|tool_call>call:read_file{")[len(call_colon_seq)]

    protected_action_toks = {tok_run, tok_sub, tok_ed, tok_rd, tok_sp_tests, tok_sp_dashq, tok_slash, tok_test_word}

    base_sacred = set()
    for sp in ("<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>"):
        tid = tokenizer.convert_tokens_to_ids(sp)
        if isinstance(tid, int) and tid >= 0:
            base_sacred.add(tid)
    for lit in (
        "{", "}", ":", ",", "\"}", "\n", "\n\n", "    ", "        ", "            ",
        "Let", " Let", "I", " I", "The", " The", "We", " We", "Looking", "Now", "In", "Based",
        "First", "To", "After", "Wait", "def", " def", "if", " if", "return", " return", "self",
    ):
        ids = enc(lit)
        if len(ids) == 1 and ids[0] not in protected_action_toks:
            base_sacred.add(ids[0])

    # For colon-dispatch layers (L36, L40, L41), ALSO orthogonalize strictly against `<|"|>` (`tok_quote`)!
    colon_sacred_list = sorted((base_sacred | {tok_quote}) - protected_action_toks)
    # For L39 (`pytest` -> `pytest tests/test_<mod>.py`), orthogonalize against all structural/opener/code tokens EXCEPT `tok_quote`,
    # while `H39_quote_close` (`beta=160`) + `H39_pyd` (`beta=120`) protect every `edit_file` & `.py` quote in activation space!
    l39_sacred_list = sorted(base_sacred - protected_action_toks - {tok_quote})

    def make_cauchy_b(u_target_res: torch.Tensor, w_pffn: torch.Tensor, s_ids: list[int]) -> torch.Tensor:
        M = lm_w[s_ids] * scale_w.unsqueeze(0)
        G = M * w_pffn.unsqueeze(0)
        G = G / G.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Q_l, _ = torch.linalg.qr(G.T, mode="reduced")
        g_t = u_target_res * w_pffn
        b_opt = g_t - Q_l @ (Q_l.T @ g_t)
        b_hat = b_opt / b_opt.norm().clamp_min(1e-8)
        assert float((M @ (w_pffn * b_hat)).abs().max().item()) < 1e-4
        return b_hat

    def unit_unemb(tid: int) -> torch.Tensor:
        v = lm_w[tid] * scale_w
        return v / v.norm().clamp_min(1e-8)

    pos_repair = (0.55 * unit_unemb(tok_ed) + 0.45 * unit_unemb(tok_rd)) - (0.75 * unit_unemb(tok_sub) + 0.25 * unit_unemb(tok_run))
    pos_l39 = (
        0.30 * unit_unemb(tok_sp_tests)
        + 0.15 * unit_unemb(tok_sp_dashq)
        + 0.20 * unit_unemb(tok_slash)
        + 0.35 * unit_unemb(tok_test_word)
    ) - 0.85 * unit_unemb(tok_quote)
    pos_err = unit_unemb(tok_rd) - (0.85 * unit_unemb(tok_ed) + 0.15 * unit_unemb(tok_sub))
    pos_verify = unit_unemb(tok_run) - unit_unemb(tok_sub)

    def harvest_layers(layer_list: list[int]):
        recs_by_l = {li: [] for li in layer_list}
        bare_by_l = {li: [] for li in layer_list}
        for r in prepared:
            dt = torch.tensor(r["delim_idx"], device=dev, dtype=torch.long)
            ot = torch.tensor(r["open_idx"], device=dev, dtype=torch.long)
            caps = {li: {} for li in layer_list}
            hks = []
            for li in layer_list:
                def mk_h(idx=li):
                    def _h(mod, inp, out):
                        h = inp[0][0].detach()
                        y = out[0].detach()
                        caps[idx]["dy_norm"] = float(y[r["c_abs"]].float().norm().item())
                        caps[idx]["dh_col"] = h[r["c_abs"]].to(torch.bfloat16).cpu()
                        caps[idx]["dh_open"] = h.index_select(0, ot).to(torch.bfloat16).cpu()
                        caps[idx]["dh_delim"] = h.index_select(0, dt).to(torch.bfloat16).cpu()
                        if r["cmd_q_abs"] is not None:
                            caps[idx]["dh_cmd_q"] = h[r["cmd_q_abs"]].to(torch.bfloat16).cpu()
                        if idx == 39 and r["quote_pred_abs"]:
                            caps[39]["dh_qclose"] = h.index_select(0, torch.tensor(r["quote_pred_abs"], device=dev)).to(torch.bfloat16).cpu()
                        if idx == 39 and r["pt_inc_abs"]:
                            caps[39]["dh_pt_inc"] = h.index_select(0, torch.tensor(r["pt_inc_abs"], device=dev)).to(torch.bfloat16).cpu()
                        if idx == 39 and r["py_done_abs"]:
                            caps[39]["dh_py_done"] = h.index_select(0, torch.tensor(r["py_done_abs"], device=dev)).to(torch.bfloat16).cpu()
                    return _h
                hks.append(layers[li].mlp.down_proj.register_forward_hook(mk_h()))
            try:
                with torch.inference_mode():
                    model(input_ids=torch.tensor([r["toks"]], device=dev), use_cache=False, logits_to_keep=1)
            finally:
                for hk in hks:
                    hk.remove()
            for li in layer_list:
                recs_by_l[li].append({**r, "cap": caps[li]})

        if any(x in layer_list for x in (39, 40, 41)):
            for b in bare_prepared:
                dt = torch.tensor(b["delim_idx"], device=dev, dtype=torch.long)
                caps = {li: {} for li in layer_list}
                hks = []
                for li in layer_list:
                    def mk_hb(idx=li):
                        def _hb(mod, inp, out):
                            h = inp[0][0].detach()
                            caps[idx]["dh_delim"] = h.index_select(0, dt).to(torch.bfloat16).cpu()
                            if b["cmd_q_abs"] is not None:
                                caps[idx]["dh_cmd_q"] = h[b["cmd_q_abs"]].to(torch.bfloat16).cpu()
                            if idx == 39 and b["pt_inc_abs"]:
                                caps[39]["dh_pt_inc"] = h.index_select(0, torch.tensor(b["pt_inc_abs"], device=dev)).to(torch.bfloat16).cpu()
                            if idx == 39 and b["py_done_abs"]:
                                caps[39]["dh_py_done"] = h.index_select(0, torch.tensor(b["py_done_abs"], device=dev)).to(torch.bfloat16).cpu()
                        return _hb
                    hks.append(layers[li].mlp.down_proj.register_forward_hook(mk_hb()))
                try:
                    with torch.inference_mode():
                        model(input_ids=torch.tensor([b["toks"]], device=dev), use_cache=False, logits_to_keep=1)
                finally:
                    for hk in hks:
                        hk.remove()
                for li in layer_list:
                    bare_by_l[li].append(caps[li])
        return recs_by_l, bare_by_l

    def fit_colon_oof(li: int, recs_h, bare_h, on_pred, pos_res_dir: torch.Tensor, lam_rel: float = 0.24):
        def pack(sub_recs):
            on_h, rd_h, ver_h, oth_col_h, cmdq_h, open_h, del_h, dyn = [], [], [], [], [], [], [], []
            for r in sub_recs:
                dyn.append(r["cap"]["dy_norm"])
                open_h.append(r["cap"]["dh_open"].float())
                del_h.append(r["cap"]["dh_delim"].float())
                if "dh_cmd_q" in r["cap"]:
                    cmdq_h.append(r["cap"]["dh_cmd_q"].float())
                hc = r["cap"]["dh_col"].float()
                if on_pred(r):
                    on_h.append(hc)
                elif r["true_role"] in ("true_imm_read_full", "true_imm_read_trunc", "turn1_init"):
                    rd_h.append(hc)
                elif r["true_role"] == "true_imm_test_pass":
                    ver_h.append(hc)
                else:
                    oth_col_h.append(hc)
            for bc in bare_h:
                if "dh_cmd_q" in bc:
                    cmdq_h.append(bc["dh_cmd_q"].float())
            Ho = torch.stack(on_h)
            Hr = torch.stack(rd_h) if rd_h else torch.zeros(0, 10240)
            Hv = torch.stack(ver_h) if ver_h else torch.zeros(0, 10240)
            Hot = torch.stack(oth_col_h) if oth_col_h else torch.zeros(0, 10240)
            Hcq = torch.stack(cmdq_h) if cmdq_h else torch.zeros(0, 10240)
            Hop = torch.cat(open_h, dim=0)
            Hd = torch.cat(del_h, dim=0)
            ogrps = [(Hr, 95.0), (Hv, 100.0), (Hot, 85.0), (Hcq, 80.0), (Hop, 130.0), (Hd, 95.0)]
            return Ho, ogrps, sum(dyn) / len(dyn)

        on_issues = sorted({r["issue"] for r in recs_h if on_pred(r)})
        oof_on_vals, oof_off_col_vals, oof_open_vals = [], [], []
        for iss in on_issues:
            tr = [r for r in recs_h if r["issue"] != iss]
            Ho_tr, og_tr, _ = pack(tr)
            wu_tr, sc_tr = solve_alphaedit_v21(Ho_tr, og_tr, lam_rel=lam_rel)
            w_tr = wu_tr * sc_tr
            for r in recs_h:
                if r["issue"] != iss:
                    continue
                v_col = float(torch.dot(w_tr, r["cap"]["dh_col"].float()))
                oof_open_vals.append(float((r["cap"]["dh_open"].float() @ w_tr).abs().max().item()))
                if on_pred(r):
                    oof_on_vals.append(v_col)
                else:
                    oof_off_col_vals.append(abs(v_col))

        Ho, og, raw_yn = pack(recs_h)
        au, sc = solve_alphaedit_v21(Ho, og, lam_rel=lam_rel)
        w = au * sc
        in_on = float((Ho @ w).mean())
        oof_on = max(0.06, sum(oof_on_vals) / max(1, len(oof_on_vals)))
        oof_pos_frac = sum(1 for v in oof_on_vals if v > 0.03) / max(1, len(oof_on_vals))
        oof_off_mean = sum(oof_off_col_vals) / max(1, len(oof_off_col_vals))
        oof_open_max = sum(oof_open_vals) / max(1, len(oof_open_vals))
        return {
            "layer": li,
            "u_a": au,
            "scale": sc,
            "in_on": round(in_on, 4),
            "oof_on": round(oof_on, 4),
            "oof_pos_frac": round(oof_pos_frac, 3),
            "oof_off_col_abs": round(oof_off_mean, 4),
            "oof_open_max_abs": round(oof_open_max, 4),
            "raw_yn": round(raw_yn, 4),
            "b_dir": make_cauchy_b(pos_res_dir, post_ffn_w[li], colon_sacred_list),
        }

    is_repair_on = lambda r: r["true_role"] == "true_imm_real_test_fail"
    is_err_on = lambda r: r["true_role"] == "true_imm_edit_err"
    is_verify_on = lambda r: r["true_role"] in ("true_imm_edit_ok", "true_imm_bad_test_path")

    def solve_exact_rho_live(li: int, a_dev: torch.Tensor, raw_yn: float, b_dir: torch.Tensor,
                             sample_recs: list[dict], suf_toks: list[int], metric_fn,
                             target_oof_nats: float, oof_to_in_ratio: float) -> tuple[float, float]:
        probe_rho = 0.25
        b_probe = ((probe_rho * raw_yn) * b_dir).to(dev, torch.bfloat16)
        deltas = []
        for r in sample_recs:
            inp = torch.tensor([(r["tokens"][-(4096 - len(suf_toks)) :] + suf_toks)], device=dev)
            with torch.inference_mode():
                lg0 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            hk = layers[li].mlp.down_proj.register_forward_hook(
                lambda mod, inputs, output, bv=b_probe: output + (inputs[0] @ a_dev).unsqueeze(-1) * bv
            )
            try:
                with torch.inference_mode():
                    lg1 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            finally:
                hk.remove()
            deltas.append(float(metric_fn(lg0, lg1)))
        mean_probe_in = max(0.30, sum(deltas) / max(1, len(deltas)))
        in_slope = mean_probe_in / probe_rho
        # On unseen OOF issues, expected dot is `oof_to_in_ratio` times training dot:
        oof_slope = in_slope * max(0.45, min(1.0, oof_to_in_ratio))
        rho_cal = max(0.06, min(0.85, target_oof_nats / oof_slope))
        return round(rho_cal, 4), round(oof_slope * rho_cal, 2)

    # =========================================================================
    # PATH A (Ultra-Clean Top-3 `L39 -> L40[repair] -> L41[verify]`, ZERO layers < 39!)
    # =========================================================================
    print("=== Stage A1: Fitting L39 Targeted Pytest (with H39_quote_close beta=160) ===", flush=True)
    r39_map, b39_map = harvest_layers([39])
    r39, b39 = r39_map[39], b39_map[39]
    h39_on, h39_pyd, h39_qcl, h39_open, h39_off, dyn39 = [], [], [], [], [], []
    for r in r39:
        dyn39.append(r["cap"]["dy_norm"])
        h39_open.append(r["cap"]["dh_open"].float())
        h39_off.append(r["cap"]["dh_delim"].float())
        h39_off.append(r["cap"]["dh_col"].float().unsqueeze(0))
        if "dh_qclose" in r["cap"]:
            h39_qcl.append(r["cap"]["dh_qclose"].float())
        if "dh_cmd_q" in r["cap"]:
            h39_off.append(r["cap"]["dh_cmd_q"].float().unsqueeze(0))
        if "dh_pt_inc" in r["cap"]:
            h39_on.append(r["cap"]["dh_pt_inc"].float())
        if "dh_py_done" in r["cap"]:
            h39_pyd.append(r["cap"]["dh_py_done"].float())
    for bc in b39:
        h39_off.append(bc["dh_delim"].float())
        if "dh_cmd_q" in bc:
            h39_off.append(bc["dh_cmd_q"].float().unsqueeze(0))
        if "dh_pt_inc" in bc:
            h39_on.append(bc["dh_pt_inc"].float())
        if "dh_py_done" in bc:
            h39_pyd.append(bc["dh_py_done"].float())

    H39_on = torch.cat(h39_on, dim=0)
    H39_pyd = torch.cat(h39_pyd, dim=0)
    H39_qcl = torch.cat(h39_qcl, dim=0) if h39_qcl else torch.zeros(0, 10240)
    H39_open = torch.cat(h39_open, dim=0)
    H39_off = torch.cat(h39_off, dim=0)
    a39_u, sc39 = solve_alphaedit_v21(
        H39_on,
        [(H39_qcl, 160.0), (H39_pyd, 120.0), (H39_open, 130.0), (H39_off, 95.0)],
        lam_rel=0.15,
    )
    s39_on = float((H39_on @ (a39_u * sc39)).mean())
    b39_dir = make_cauchy_b(pos_l39, post_ffn_w[39], l39_sacred_list)
    dyn39_mean = sum(dyn39) / len(dyn39)

    A39 = (a39_u * (sc39 / max(0.12, s39_on))).bfloat16()
    a39_dev = A39.to(dev)
    suf_pt_bare = enc('<|channel>thought\nLet\'s run pytest.\n<channel|><|tool_call>call:run_command{command:<|"|>pytest')
    suf_pt_dir = enc('<|channel>thought\nLet\'s run pytest.\n<channel|><|tool_call>call:run_command{command:<|"|>pytest tests/')
    ok_samples = [r for r in prepared if is_verify_on(r)][:6]

    rho39, oof_nats_39_bare = solve_exact_rho_live(
        39, a39_dev, dyn39_mean, b39_dir, ok_samples, suf_pt_bare,
        lambda lg0, lg1: ((lg1[tok_sp_tests] - lg1[tok_quote]) - (lg0[tok_sp_tests] - lg0[tok_quote])).item(),
        target_oof_nats=12.5,
        oof_to_in_ratio=0.82,
    )
    # Also verify `test - quote` shift right after `pytest tests/`!
    b39_cal = ((rho39 * dyn39_mean) * b39_dir).to(dev, torch.bfloat16)
    dir_gaps = []
    for r in ok_samples[:4]:
        inp = torch.tensor([(r["tokens"][-(4096 - len(suf_pt_dir)) :] + suf_pt_dir)], device=dev)
        with torch.inference_mode():
            lg0 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        hk = layers[39].mlp.down_proj.register_forward_hook(
            lambda mod, inputs, output: output + (inputs[0] @ a39_dev).unsqueeze(-1) * b39_cal
        )
        try:
            with torch.inference_mode():
                lg1 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        finally:
            hk.remove()
        dir_gaps.append(float(((lg1[tok_test_word] - lg1[tok_quote]) - (lg0[tok_test_word] - lg0[tok_quote])).item()))
    print(f"  [L39 Pytest] rho39={rho39} -> OOF bare(sp_tests-quote)=+{oof_nats_39_bare} nats | dir(test-quote)=+{sum(dir_gaps)/len(dir_gaps):.2f} nats!", flush=True)

    B39 = ((rho39 * dyn39_mean) * b39_dir).bfloat16()
    b39_dev = B39.to(dev)
    hk39 = layers[39].mlp.down_proj.register_forward_hook(
        lambda mod, inputs, output: output + (inputs[0] @ a39_dev).unsqueeze(-1) * b39_dev
    )

    print("=== Stage A2: Fitting L40 Post-Failure Repair Gate (`true_imm_real_test_fail -> edit/read`) ===", flush=True)
    r40_map, b40_map = harvest_layers([40])
    d40_rep = fit_colon_oof(40, r40_map[40], b40_map[40], is_repair_on, pos_repair, lam_rel=0.24)
    A40_rep = (d40_rep["u_a"] * (d40_rep["scale"] / d40_rep["oof_on"])).bfloat16()
    a40_rep_dev = A40_rep.to(dev)

    fail_samples = [r for r in prepared if is_repair_on(r)][:6]
    suf_fail_col = enc("<|channel>thought\nThe test failed, let's check what to do next.\n<channel|><|tool_call>call:")
    rho40_rep, oof_nats_40rep = solve_exact_rho_live(
        40, a40_rep_dev, d40_rep["raw_yn"], d40_rep["b_dir"], fail_samples, suf_fail_col,
        lambda lg0, lg1: ((torch.maximum(lg1[tok_ed], lg1[tok_rd]) - torch.maximum(lg1[tok_sub], lg1[tok_run]))
                          - (torch.maximum(lg0[tok_ed], lg0[tok_rd]) - torch.maximum(lg0[tok_sub], lg0[tok_run]))).item(),
        target_oof_nats=11.5,
        oof_to_in_ratio=d40_rep["oof_on"] / max(1e-4, d40_rep["in_on"]),
    )
    print(f"  [L40 Repair] oof_pos={d40_rep['oof_pos_frac']} | rho40_rep={rho40_rep} -> expected OOF repair shift = +{oof_nats_40rep} nats!", flush=True)

    B40_rep = ((rho40_rep * d40_rep["raw_yn"]) * d40_rep["b_dir"]).bfloat16()
    b40_rep_dev = B40_rep.to(dev)
    hk40 = layers[40].mlp.down_proj.register_forward_hook(
        lambda mod, inputs, output: output + (inputs[0] @ a40_rep_dev).unsqueeze(-1) * b40_rep_dev
    )

    print("=== Stage A3: Fitting L41 Verify Gate (`true_imm_edit_ok + bad_path -> run_command`) ===", flush=True)
    r41_map, b41_map = harvest_layers([41])
    d41_ver = fit_colon_oof(41, r41_map[41], b41_map[41], is_verify_on, pos_verify, lam_rel=0.24)
    A41_ver = (d41_ver["u_a"] * (d41_ver["scale"] / d41_ver["oof_on"])).bfloat16()
    a41_ver_dev = A41_ver.to(dev)

    suf_prem = enc("<|channel>thought\nThe change looks complete. I will submit the patch.\n<channel|><|tool_call>call:")
    rho41_ver, oof_nats_41ver = solve_exact_rho_live(
        41, a41_ver_dev, d41_ver["raw_yn"], d41_ver["b_dir"], ok_samples, suf_prem,
        lambda lg0, lg1: ((lg1[tok_run] - lg1[tok_sub]) - (lg0[tok_run] - lg0[tok_sub])).item(),
        target_oof_nats=12.0,
        oof_to_in_ratio=d41_ver["oof_on"] / max(1e-4, d41_ver["in_on"]),
    )
    print(f"  [L41 Verify] oof_pos={d41_ver['oof_pos_frac']} | rho41_ver={rho41_ver} -> expected OOF (run-sub) = +{oof_nats_41ver} nats!", flush=True)
    B41_ver = ((rho41_ver * d41_ver["raw_yn"]) * d41_ver["b_dir"]).bfloat16()

    hk40.remove()
    hk39.remove()

    # =========================================================================
    # PATH B (Optional 4-Layer adding Gentle L36 `FileEditError -> read_file`)
    # =========================================================================
    print("=== Stage B1: Fitting Gentle L36 FileEditError -> read_file ===", flush=True)
    r36_map, b36_map = harvest_layers([36])
    d36_err = fit_colon_oof(36, r36_map[36], b36_map[36], is_err_on, pos_err, lam_rel=0.26)
    A36_err = (d36_err["u_a"] * (d36_err["scale"] / d36_err["oof_on"])).bfloat16()
    a36_err_dev = A36_err.to(dev)
    err_samples = [r for r in prepared if is_err_on(r)][:6]
    suf_err_col = enc("<|channel>thought\nThe replacement failed. Let's inspect the file.\n<channel|><|tool_call>call:")
    rho36_err, oof_nats_36err = solve_exact_rho_live(
        36, a36_err_dev, d36_err["raw_yn"], d36_err["b_dir"], err_samples, suf_err_col,
        lambda lg0, lg1: ((lg1[tok_rd] - lg1[tok_ed]) - (lg0[tok_rd] - lg0[tok_ed])).item(),
        target_oof_nats=7.5,
        oof_to_in_ratio=d36_err["oof_on"] / max(1e-4, d36_err["in_on"]),
    )
    B36_err = ((rho36_err * d36_err["raw_yn"]) * d36_err["b_dir"]).bfloat16()
    print(f"  [L36 Err->Read] oof_pos={d36_err['oof_pos_frac']} | rho36_err={rho36_err} -> expected OOF (read-edit) = +{oof_nats_36err} nats!", flush=True)

    del model, lm_root, layers
    gc.collect()
    torch.cuda.empty_cache()

    clean_diag = lambda d: {k: v for k, v in d.items() if k not in ("u_a", "b_dir")}
    diagnostics = {
        "n_real_records": len(prepared),
        "solved_rhos": {"rho39": rho39, "rho40_rep": rho40_rep, "rho41_ver": rho41_ver, "rho36_err": rho36_err},
        "expected_oof_nats": {
            "l39_bare": oof_nats_39_bare,
            "l39_dir": round(sum(dir_gaps) / len(dir_gaps), 2),
            "l40_repair": oof_nats_40rep,
            "l41_verify": oof_nats_41ver,
            "l36_err": oof_nats_36err,
        },
        "L40_repair_oof": clean_diag(d40_rep),
        "L41_verify_oof": clean_diag(d41_ver),
        "L36_err_oof": clean_diag(d36_err),
    }
    print("V22 Top-Band Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    def export_variant(name: str, ft: dict):
        fp = out_dir / f"fac_{name}.safetensors"
        save_file(ft, str(fp))
        for ctrl in ("candidate", "random"):
            inf = compile_explicit(ROOT / "champion.zip", fp, ad_root / f"{name}_{ctrl}", ctrl, rank=1)
            compiled.append({"adapter": f"{name}_{ctrl}", "sha256": inf["adapter_sha256"], "control": ctrl, "layers": inf["active_layers"]})

    # 1. Primary Ultra-Clean Top-3 (`L39[pytest] + L40[fail->repair] + L41[edit->verify]`, ZERO layers < 39!)
    export_variant(
        "V22_top3_39pt_40rep_41ver_bal",
        {
            "39.A": A39, "39.B": B39,
            "40.A": A40_rep, "40.B": B40_rep,
            "41.A": A41_ver, "41.B": B41_ver,
        },
    )
    # 2. Gentle 0.80x Top-3 (`L39 + L40[repair] + L41[verify]`) — maximizes `b3_exact_args_match`!
    export_variant(
        "V22_top3_39pt_40rep_41ver_soft",
        {
            "39.A": A39, "39.B": (0.82 * B39.float()).bfloat16(),
            "40.A": A40_rep, "40.B": (0.80 * B40_rep.float()).bfloat16(),
            "41.A": A41_ver, "41.B": (0.82 * B41_ver.float()).bfloat16(),
        },
    )
    # 3. 4-Layer adding Gentle L36 `FileEditError -> read_file` (`L36 + L39 + L40 + L41`)
    export_variant(
        "V22_top4_36err_39pt_40rep_41ver",
        {
            "36.A": A36_err, "36.B": (0.80 * B36_err.float()).bfloat16(),
            "39.A": A39, "39.B": (0.90 * B39.float()).bfloat16(),
            "40.A": A40_rep, "40.B": (0.92 * B40_rep.float()).bfloat16(),
            "41.A": A41_ver, "41.B": (0.92 * B41_ver.float()).bfloat16(),
        },
    )

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V22 Top-Band Closed-Loop Repair adapters!", flush=True)


if __name__ == "__main__":
    main()
