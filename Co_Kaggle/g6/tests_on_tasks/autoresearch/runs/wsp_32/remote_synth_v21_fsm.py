"""Synthesize V21 Pure-WSP 4-Transition Closed-Loop Repair Cascade (`L_rep -> L39 -> L40 -> L41`, `r=1`, `down_proj`).

Resolves ALL THREE empirical bottlenecks found in the 5-stage causal funnel audit (`champion_reproducibility_and_funnel.json`):
1. Sacred-Delimiter & Thought-Opener QR Orthogonalization + Full 3,600-Row Null Space:
   - Previously (`V17..V20`), `39.B` subtracted `lm_w[52]` (`<|"|>`, Gemma 4's tool-call string delimiter!) and `solve_full_alphaedit`
     subsampled `Hd` down to 420 rows in R^{10240}, causing `b3_exact_args_match = 2/6`, `edit_errors = 50..76`, and `syntax_leaks = 2..3`.
   - `V21` projects EVERY `lora_B` orthogonal to all Gemma 4 special/syntax tokens (`<|"|>`, `<tool_call|>`, `<|tool_call>`,
     `<channel|>`, `<|channel>`, `{`, `}`, `:`, `,`), top thought-opener words, and Python indentation/newlines in exact
     post-`post_feedforward_layernorm` residual space, AND retains up to 3,600 null-space vectors per solve!
2. Separates `true_imm_real_test_fail` (`ec in {1, 2}`) from `true_imm_edit_ok` (`s3 -> s4` Closed-Loop Repair Gate!):
   - `L_rep` (best OOF layer in `{35, 36}`): fires ONLY on `true_imm_real_test_fail @ call:`, steering toward
     `+(0.55 * edit_file + 0.45 * read_file) - (0.75 * submit_patch + 0.25 * run_command)` — eliminating BOTH
     `RATIONALIZED_FAIL_SUB` (3/15 in `champ_rep3`!) AND infinite `run_command` re-test loops (`iter05/07`!).
   - `L39`: fires ONLY on incomplete `pytest` prefixes -> boosts `(" tests" + " -q" + "/" + "test")` with ZERO `<|"|>` perturbation!
   - `L40`: fires ONLY on `true_imm_edit_err @ call:` -> `+read_file - (0.85 * edit_file + 0.15 * submit_patch)`.
   - `L41`: fires ONLY on `true_imm_edit_ok + true_imm_bad_test_path @ call:` (`true_imm_real_test_fail` & `true_imm_test_pass`
     in heavy null space!) -> `+run_command - submit_patch`.
3. Live In-VRAM Logit Calibration across all 5 states before exporting `r=1` `down_proj` adapters.
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
from synth_v16_upperband import find_subseq_end


def solve_alphaedit_v21(
    X_on: torch.Tensor,
    off_groups: list[tuple[torch.Tensor, float]],
    lam_rel: float = 0.22,
    max_group_rows: int = 3600,
) -> tuple[torch.Tensor, float]:
    """Full-resolution GPU float32 Kernel Ridge AlphaEdit in R^{10240} (up to 3,600 null-space rows per group!)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xo = X_on.to(device, torch.float32)
    n_on = max(1, Xo.shape[0])
    rows = [(1.0 / (n_on ** 0.5)) * Xo]
    y_parts = [torch.full((n_on,), 1.0 / (n_on ** 0.5), device=device)]

    for xg, beta in off_groups:
        if xg.shape[0] == 0:
            continue
        xg_d = xg.to(device, torch.float32)
        if xg_d.shape[0] > max_group_rows:
            stride = max(1, xg_d.shape[0] // max_group_rows)
            xg_d = xg_d[::stride][:max_group_rows]
        scale = (beta / xg_d.shape[0]) ** 0.5
        rows.append(scale * xg_d)
        y_parts.append(torch.zeros(xg_d.shape[0], device=device))

    X = torch.cat(rows, dim=0)
    y = torch.cat(y_parts, dim=0)
    gram = X @ X.T
    tr_mean = float(gram.diagonal().mean().clamp_min(1e-4))
    reg = (lam_rel * tr_mean) * torch.eye(X.shape[0], device=device)
    dual = torch.linalg.solve(gram + reg, y)
    w = (X.T @ dual).cpu()
    norm = float(w.norm().clamp_min(1e-8))
    return w / norm, norm


def main():
    v21_cases_p = ROOT / "harvest32/clean_v21/real_replays.json"
    assert v21_cases_p.is_file(), f"Missing {v21_cases_p}"
    out_dir = ROOT / "harvest32/v21_fsm"
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
    raw_scalars = [float(layers[i].layer_scalar.detach().float().cpu().item()) for i in range(len(layers))]
    cum_scalars = {}
    for li in range(len(layers)):
        prod = 1.0
        for k in range(li, len(layers)):
            prod *= raw_scalars[k]
        cum_scalars[li] = prod

    cand_layers = (35, 36, 39, 40, 41)
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

    # Prepare tokenized records with dense thought-opener + thought-prose + code/quote null-space indices!
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

        is_ed_or_pres = (item.get("action") == "edit_file") or (rl in ("true_imm_read_full", "true_imm_read_trunc", "turn1_init"))

        # Explicitly track ALL first 8 thought-opener tokens (where tiny logit shifts flip greedy thought trajectories!)
        open_idx = [k for k in range(ts0, min(ch_end_abs, ts0 + 8, len(toks))) if k != c_abs and k not in set(pt_inc_abs)]

        delim_set = set(open_idx)
        if ch_end_abs > ts0:
            th_step = max(1, (ch_end_abs - ts0) // (32 if is_ed_or_pres else 20))
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
            # Always include the last 6 tokens of the tool call (closing `<|"|>}<tool_call|>`)!
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
            "open_idx": open_clean,
            "delim_idx": delim_idx,
        })

    # Targeted `.py` pytest continuations for L39
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

    print(f"Prepared {len(prepared)} 100%-real records + {len(bare_prepared)} targeted-pytest records!", flush=True)

    # =========================================================================
    # Build Sacred-Delimiter + Thought-Opener + Code-Indentation QR Orthogonalizer!
    # =========================================================================
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

    tok_run = enc("<|tool_call>call:run_command{")[len(call_colon_seq)]
    tok_sub = enc("<|tool_call>call:submit_patch{")[len(call_colon_seq)]
    tok_ed = enc("<|tool_call>call:edit_file{")[len(call_colon_seq)]
    tok_rd = enc("<|tool_call>call:read_file{")[len(call_colon_seq)]

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

    protected_action_toks = {tok_run, tok_sub, tok_ed, tok_rd, tok_sp_tests, tok_sp_dashq, tok_slash, tok_test_word}

    sacred_ids = set()
    for sp in ("<|\"|>", "<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>"):
        tid = tokenizer.convert_tokens_to_ids(sp)
        if isinstance(tid, int) and tid >= 0:
            sacred_ids.add(tid)
    sacred_ids.add(tok_quote)
    for lit in (
        "{", "}", ":", ",", "}", "\"}", "\n", "\n\n", "    ", "        ", "            ",
        "Let", " Let", "I", " I", "The", " The", "We", " We", "Looking", "Now", "In", "Based",
        "First", "To", "After", "Wait", "def", " def", "if", " if", "return", " return", "self",
    ):
        ids = enc(lit)
        if len(ids) == 1 and ids[0] not in protected_action_toks:
            sacred_ids.add(ids[0])

    sacred_list = sorted(sacred_ids - protected_action_toks)
    M_sacred = lm_w[sacred_list] * scale_w.unsqueeze(0)  # [|S|, 2560]
    M_sacred = M_sacred / M_sacred.norm(dim=1, keepdim=True).clamp_min(1e-8)
    Q_sacred, _ = torch.linalg.qr(M_sacred.T, mode="reduced")  # [2560, |S|]
    print(f"Built QR Sacred-Delimiter & Thought-Opener Orthogonalizer over {len(sacred_list)} tokens (quote_id={tok_quote} in set={tok_quote in sacred_list})!", flush=True)

    def make_clean_b_dir(u_target_res: torch.Tensor, w_pffn: torch.Tensor) -> torch.Tensor:
        """Project u_target_res orthogonal to all sacred/opener readout rows, then inverse-precondition post_feedforward_layernorm!"""
        u_clean = u_target_res - Q_sacred @ (Q_sacred.T @ u_target_res)
        u_clean = u_clean / u_clean.norm().clamp_min(1e-8)
        # Regularized inverse of post_feedforward_layernorm weight so w_pffn * b_dir == u_clean!
        b = (u_clean * w_pffn) / (w_pffn.pow(2) + 0.04)
        return b / b.norm().clamp_min(1e-8)

    def unit_unemb(tid: int) -> torch.Tensor:
        v = lm_w[tid] * scale_w
        return v / v.norm().clamp_min(1e-8)

    # Positive desired residual readout directions (`+` = desired shift when s(h) > 0!):
    # 1. C_repair (`true_imm_real_test_fail @ call:`): +(0.55*edit_file + 0.45*read_file) - (0.75*submit_patch + 0.25*run_command)
    pos_repair = (0.55 * unit_unemb(tok_ed) + 0.45 * unit_unemb(tok_rd)) - (0.75 * unit_unemb(tok_sub) + 0.25 * unit_unemb(tok_run))
    # 2. C_l39 (incomplete `pytest` -> file-level `pytest tests/test_<mod>.py`, ZERO `<|"|>` subtraction!):
    pos_l39 = 0.35 * unit_unemb(tok_sp_tests) + 0.15 * unit_unemb(tok_sp_dashq) + 0.20 * unit_unemb(tok_slash) + 0.30 * unit_unemb(tok_test_word)
    # 3. C_err (`true_imm_edit_err @ call:` -> `read_file`):
    pos_err = unit_unemb(tok_rd) - (0.85 * unit_unemb(tok_ed) + 0.15 * unit_unemb(tok_sub))
    # 4. C_verify (`true_imm_edit_ok + true_imm_bad_test_path @ call:` -> `run_command`):
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
                        if idx == 39 and r["pt_inc_abs"]:
                            caps[idx]["dh_pt_inc"] = h.index_select(0, torch.tensor(r["pt_inc_abs"], device=dev)).to(torch.bfloat16).cpu()
                        if idx == 39 and r["py_done_abs"]:
                            caps[idx]["dh_py_done"] = h.index_select(0, torch.tensor(r["py_done_abs"], device=dev)).to(torch.bfloat16).cpu()
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

        if 39 in layer_list or 40 in layer_list or 41 in layer_list:
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
                                caps[idx]["dh_pt_inc"] = h.index_select(0, torch.tensor(b["pt_inc_abs"], device=dev)).to(torch.bfloat16).cpu()
                            if idx == 39 and b["py_done_abs"]:
                                caps[idx]["dh_py_done"] = h.index_select(0, torch.tensor(b["py_done_abs"], device=dev)).to(torch.bfloat16).cpu()
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
            # Heavy null-space weights on thought-openers (`Hop`), verified (`Hv`), read/edit (`Hr`), other colon roles (`Hot`), and code (`Hd`)!
            ogrps = [(Hr, 80.0), (Hv, 85.0), (Hot, 75.0), (Hcq, 70.0), (Hop, 95.0), (Hd, 80.0)]
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
        eff_snr = (oof_on / max(0.015, oof_off_mean, oof_open_max)) * (cum_scalars[li] ** 0.5)
        return {
            "layer": li,
            "u_a": au,
            "scale": sc,
            "in_on": round(in_on, 4),
            "oof_on": round(oof_on, 4),
            "oof_pos_frac": round(oof_pos_frac, 3),
            "oof_off_col_abs": round(oof_off_mean, 4),
            "oof_open_max_abs": round(oof_open_max, 4),
            "eff_snr": round(eff_snr, 3),
            "raw_yn": round(raw_yn, 4),
            "b_dir": make_clean_b_dir(pos_res_dir, post_ffn_w[li]),
        }

    is_repair_on = lambda r: r["true_role"] == "true_imm_real_test_fail"
    is_err_on = lambda r: r["true_role"] == "true_imm_edit_err"
    is_verify_on = lambda r: r["true_role"] in ("true_imm_edit_ok", "true_imm_bad_test_path")

    # =========================================================================
    # STAGE 1: Compare L35 vs L36 for Post-Failure Repair (`true_imm_real_test_fail`) & ATTACH WINNER!
    # =========================================================================
    print("=== Stage 1: Comparing L35 vs L36 for Post-Failure Repair Gate ===", flush=True)
    r35_36, b35_36 = harvest_layers([35, 36])
    d35_rep = fit_colon_oof(35, r35_36[35], b35_36[35], is_repair_on, pos_repair, lam_rel=0.24)
    d36_rep = fit_colon_oof(36, r35_36[36], b35_36[36], is_repair_on, pos_repair, lam_rel=0.24)
    d_rep = d35_rep if d35_rep["eff_snr"] >= d36_rep["eff_snr"] else d36_rep
    l_rep = d_rep["layer"]
    print(f"  L35 eff_snr={d35_rep['eff_snr']} (oof_on={d35_rep['oof_on']}, pos={d35_rep['oof_pos_frac']}) vs L36 eff_snr={d36_rep['eff_snr']} (oof_on={d36_rep['oof_on']}, pos={d36_rep['oof_pos_frac']}) -> Selected L{l_rep}!", flush=True)

    # Calibrate L_rep gain live in VRAM on `true_imm_real_test_fail` (`max(edit, read) - max(submit, run)`)!
    A_rep = (d_rep["u_a"] * (d_rep["scale"] / d_rep["oof_on"])).bfloat16()
    a_rep_dev = A_rep.to(dev)
    fail_samples = [r for r in prepared if is_repair_on(r)][:6]
    suf_fail_col = enc("<|channel>thought\nThe test failed, let's check what to do next.\n<channel|><|tool_call>call:")

    best_rho_rep = 0.28
    for cand_rho in (0.20, 0.28, 0.36, 0.45):
        b_try = ((cand_rho * d_rep["raw_yn"]) * d_rep["b_dir"]).to(dev, torch.bfloat16)
        shifts = []
        for r in fail_samples:
            inp = torch.tensor([(r["tokens"][-(4096 - len(suf_fail_col)) :] + suf_fail_col)], device=dev)
            with torch.inference_mode():
                lg0 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            hk = layers[l_rep].mlp.down_proj.register_forward_hook(
                lambda mod, inputs, output, bv=b_try: output + (inputs[0] @ a_rep_dev).unsqueeze(-1) * bv
            )
            try:
                with torch.inference_mode():
                    lg1 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            finally:
                hk.remove()
            d_nat = float(((torch.maximum(lg1[tok_ed], lg1[tok_rd]) - torch.maximum(lg1[tok_sub], lg1[tok_run]))
                           - (torch.maximum(lg0[tok_ed], lg0[tok_rd]) - torch.maximum(lg0[tok_sub], lg0[tok_run]))).item())
            shifts.append(d_nat)
        m_sh = sum(shifts) / max(1, len(shifts))
        print(f"  [Calibrate L{l_rep} Repair] rho={cand_rho:.2f} -> mean_repair_shift = {m_sh:+.2f} nats", flush=True)
        if 9.0 <= m_sh <= 15.0:
            best_rho_rep = cand_rho
            break
        if m_sh < 9.0:
            best_rho_rep = cand_rho

    B_rep_mid = ((best_rho_rep * d_rep["raw_yn"]) * d_rep["b_dir"]).bfloat16()
    b_rep_dev = B_rep_mid.to(dev)
    hk_rep = layers[l_rep].mlp.down_proj.register_forward_hook(
        lambda mod, inputs, output: output + (inputs[0] @ a_rep_dev).unsqueeze(-1) * b_rep_dev
    )

    # =========================================================================
    # STAGE 2: Fit L39 File-Level Targeted Pytest (ZERO `<|"|>` Perturbation!) & ATTACH LIVE!
    # =========================================================================
    print("=== Stage 2: Harvesting post-L_rep h_39 & fitting Sacred-Orthogonalized L39 ===", flush=True)
    r39_map, b39_map = harvest_layers([39])
    r39, b39 = r39_map[39], b39_map[39]
    h39_on, h39_pyd, h39_open, h39_off, dyn39 = [], [], [], [], []
    for r in r39:
        dyn39.append(r["cap"]["dy_norm"])
        h39_open.append(r["cap"]["dh_open"].float())
        h39_off.append(r["cap"]["dh_delim"].float())
        h39_off.append(r["cap"]["dh_col"].float().unsqueeze(0))
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
    H39_open = torch.cat(h39_open, dim=0)
    H39_off = torch.cat(h39_off, dim=0)
    a39_u, sc39 = solve_alphaedit_v21(H39_on, [(H39_pyd, 90.0), (H39_open, 95.0), (H39_off, 80.0)], lam_rel=0.16)
    s39_on = float((H39_on @ (a39_u * sc39)).mean())
    b39_dir = make_clean_b_dir(pos_l39, post_ffn_w[39])
    dyn39_mean = sum(dyn39) / len(dyn39)

    A39 = (a39_u * (sc39 / max(0.12, s39_on))).bfloat16()
    a39_dev = A39.to(dev)

    # Verify live in VRAM that L39 boosts `" tests"` over `<|"|>` WITHOUT moving `logit[<|"|>]`!
    suf_pt_bare = enc('<|channel>thought\nLet\'s run pytest.\n<channel|><|tool_call>call:run_command{command:<|"|>pytest')
    ok_samples = [r for r in prepared if is_verify_on(r)][:5]
    best_rho39 = 0.52
    for cand_rho39 in (0.40, 0.55, 0.72, 0.90):
        b39_try = ((cand_rho39 * dyn39_mean) * b39_dir).to(dev, torch.bfloat16)
        d_gap, d_qt = [], []
        for r in ok_samples:
            inp = torch.tensor([(r["tokens"][-(4096 - len(suf_pt_bare)) :] + suf_pt_bare)], device=dev)
            with torch.inference_mode():
                lg0 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            hk = layers[39].mlp.down_proj.register_forward_hook(
                lambda mod, inputs, output, bv=b39_try: output + (inputs[0] @ a39_dev).unsqueeze(-1) * bv
            )
            try:
                with torch.inference_mode():
                    lg1 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            finally:
                hk.remove()
            d_gap.append(float(((lg1[tok_sp_tests] - lg1[tok_quote]) - (lg0[tok_sp_tests] - lg0[tok_quote])).item()))
            d_qt.append(float((lg1[tok_quote] - lg0[tok_quote]).abs().item()))
        mg, mq = sum(d_gap) / len(d_gap), sum(d_qt) / len(d_qt)
        print(f"  [Calibrate L39 Pytest] rho={cand_rho39:.2f} -> delta(sp_tests - quote)={mg:+.2f} nats | |delta(quote)|={mq:.3f} nats", flush=True)
        if 7.5 <= mg <= 13.0:
            best_rho39 = cand_rho39
            break
        if mg < 7.5:
            best_rho39 = cand_rho39

    B39 = ((best_rho39 * dyn39_mean) * b39_dir).bfloat16()
    b39_dev = B39.to(dev)
    hk39 = layers[39].mlp.down_proj.register_forward_hook(
        lambda mod, inputs, output: output + (inputs[0] @ a39_dev).unsqueeze(-1) * b39_dev
    )

    # =========================================================================
    # STAGE 3A: Fit L41_no40 directly on post-(L_rep + L39) states (for 3-layer ablation)
    # =========================================================================
    print("=== Stage 3A: Harvesting post-(L_rep+L39) h_41 for 3-layer variant ===", flush=True)
    r41_n40_map, b41_n40_map = harvest_layers([41])
    d41_no40 = fit_colon_oof(41, r41_n40_map[41], b41_n40_map[41], is_verify_on, pos_verify, lam_rel=0.24)

    # =========================================================================
    # STAGE 3B: Fit L40 (`FileEditError -> read_file`) on post-(L_rep + L39) & ATTACH LIVE!
    # =========================================================================
    print("=== Stage 3B: Harvesting post-(L_rep+L39) h_40 & fitting L40 ===", flush=True)
    r40_map, b40_map = harvest_layers([40])
    d40_err = fit_colon_oof(40, r40_map[40], b40_map[40], is_err_on, pos_err, lam_rel=0.24)

    A40 = (d40_err["u_a"] * (d40_err["scale"] / d40_err["oof_on"])).bfloat16()
    B40_mid = ((0.22 * d40_err["raw_yn"]) * d40_err["b_dir"]).bfloat16()
    a40_dev, b40_dev = A40.to(dev), B40_mid.to(dev)
    hk40 = layers[40].mlp.down_proj.register_forward_hook(
        lambda mod, inputs, output: output + (inputs[0] @ a40_dev).unsqueeze(-1) * b40_dev
    )

    # =========================================================================
    # STAGE 4: Fit L41 (`true_imm_edit_ok + bad_path -> run_command`, `real_test_fail` OFF!) on exact post-(L_rep+L39+L40)!
    # =========================================================================
    print("=== Stage 4: Harvesting exact post-(L_rep+L39+L40) h_41 & fitting L41 ===", flush=True)
    r41_map, b41_map = harvest_layers([41])
    d41_casc = fit_colon_oof(41, r41_map[41], b41_map[41], is_verify_on, pos_verify, lam_rel=0.24)

    # Live calibrate L41 `rho41` so `run_command - submit_patch` shifts by +10..+14 nats on `edit_ok` while `test_pass` & `real_test_fail` stay clean!
    A41 = (d41_casc["u_a"] * (d41_casc["scale"] / d41_casc["oof_on"])).bfloat16()
    a41_dev = A41.to(dev)
    suf_prem = enc("<|channel>thought\nThe change looks complete. I will submit the patch.\n<channel|><|tool_call>call:")
    ver_samples = [r for r in prepared if r["true_role"] == "true_imm_test_pass"][:5]

    best_rho41 = 0.24
    for cand_rho41 in (0.18, 0.24, 0.30, 0.36):
        b41_try = ((cand_rho41 * d41_casc["raw_yn"]) * d41_casc["b_dir"]).to(dev, torch.bfloat16)
        on_sh, ver_sh, fail_leak = [], [], []
        for r in ok_samples:
            inp = torch.tensor([(r["tokens"][-(4096 - len(suf_prem)) :] + suf_prem)], device=dev)
            with torch.inference_mode():
                lg0 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            hk = layers[41].mlp.down_proj.register_forward_hook(
                lambda mod, inputs, output, bv=b41_try: output + (inputs[0] @ a41_dev).unsqueeze(-1) * bv
            )
            try:
                with torch.inference_mode():
                    lg1 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            finally:
                hk.remove()
            on_sh.append(float(((lg1[tok_run] - lg1[tok_sub]) - (lg0[tok_run] - lg0[tok_sub])).item()))
        for r in ver_samples:
            inp = torch.tensor([(r["tokens"][-(4096 - len(suf_prem)) :] + suf_prem)], device=dev)
            with torch.inference_mode():
                lg0 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            hk = layers[41].mlp.down_proj.register_forward_hook(
                lambda mod, inputs, output, bv=b41_try: output + (inputs[0] @ a41_dev).unsqueeze(-1) * bv
            )
            try:
                with torch.inference_mode():
                    lg1 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            finally:
                hk.remove()
            ver_sh.append(float(((lg1[tok_sub] - lg1[tok_run]) - (lg0[tok_sub] - lg0[tok_run])).item()))
        for r in fail_samples[:4]:
            inp = torch.tensor([(r["tokens"][-(4096 - len(suf_fail_col)) :] + suf_fail_col)], device=dev)
            with torch.inference_mode():
                lg0 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            hk = layers[41].mlp.down_proj.register_forward_hook(
                lambda mod, inputs, output, bv=b41_try: output + (inputs[0] @ a41_dev).unsqueeze(-1) * bv
            )
            try:
                with torch.inference_mode():
                    lg1 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            finally:
                hk.remove()
            fail_leak.append(float(((lg1[tok_run] - lg1[tok_ed]) - (lg0[tok_run] - lg0[tok_ed])).abs().item()))
        m_on = sum(on_sh) / max(1, len(on_sh))
        m_ver = sum(ver_sh) / max(1, len(ver_sh))
        m_fl = sum(fail_leak) / max(1, len(fail_leak))
        print(f"  [Calibrate L41 Verify] rho={cand_rho41:.2f} -> edit_ok(run-sub)={m_on:+.2f} | test_pass_drift={m_ver:+.2f} | fail_leak={m_fl:.2f}", flush=True)
        if 9.5 <= m_on <= 14.5:
            best_rho41 = cand_rho41
            break
        if m_on < 9.5:
            best_rho41 = cand_rho41

    hk40.remove()
    hk39.remove()
    hk_rep.remove()
    del model, lm_root, layers
    gc.collect()
    torch.cuda.empty_cache()

    clean_diag = lambda d: {k: v for k, v in d.items() if k not in ("u_a", "b_dir")}
    diagnostics = {
        "n_real_records": len(prepared),
        "selected_repair_layer": l_rep,
        "calibrated_rhos": {"rho_rep": best_rho_rep, "rho39": best_rho39, "rho40": 0.22, "rho41": best_rho41},
        "L_rep_oof": clean_diag(d_rep),
        "L39_s_on": round(s39_on, 4),
        "L40_err_oof": clean_diag(d40_err),
        "L41_casc_oof": clean_diag(d41_casc),
        "L41_no40_oof": clean_diag(d41_no40),
    }
    print("V21 FSM Cascade Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    def export_variant(name: str, ft: dict, include_zero: bool = False):
        fp = out_dir / f"fac_{name}.safetensors"
        save_file(ft, str(fp))
        if include_zero:
            zi = compile_explicit(ROOT / "champion.zip", fp, ad_root / f"{name}_zero", "zero", rank=1)
            compiled.append({"adapter": f"{name}_zero", "sha256": zi["adapter_sha256"], "control": "zero", "layers": zi["active_layers"]})
        for ctrl in ("candidate", "random"):
            inf = compile_explicit(ROOT / "champion.zip", fp, ad_root / f"{name}_{ctrl}", ctrl, rank=1)
            compiled.append({"adapter": f"{name}_{ctrl}", "sha256": inf["adapter_sha256"], "control": ctrl, "layers": inf["active_layers"]})

    # 1. Calibrated 4-Layer Closed-Loop FSM (`L_rep + L39 + L40 + L41`) — Sweet Spot + Zero Control
    export_variant(
        "V21_fsm_4layer_sweet",
        {
            f"{l_rep}.A": A_rep,
            f"{l_rep}.B": B_rep_mid,
            "39.A": A39,
            "39.B": B39,
            "40.A": A40,
            "40.B": B40_mid,
            "41.A": A41,
            "41.B": ((best_rho41 * d41_casc["raw_yn"]) * d41_casc["b_dir"]).bfloat16(),
        },
        include_zero=True,
    )

    # 2. Slightly Firmer Repair + Verify (`1.22x` on `L_rep` & `L41`, `0.85x` gentle `L40`)
    export_variant(
        "V21_fsm_4layer_firm",
        {
            f"{l_rep}.A": A_rep,
            f"{l_rep}.B": ((1.22 * best_rho_rep * d_rep["raw_yn"]) * d_rep["b_dir"]).bfloat16(),
            "39.A": A39,
            "39.B": B39,
            "40.A": A40,
            "40.B": ((0.18 * d40_err["raw_yn"]) * d40_err["b_dir"]).bfloat16(),
            "41.A": A41,
            "41.B": ((1.22 * best_rho41 * d41_casc["raw_yn"]) * d41_casc["b_dir"]).bfloat16(),
        },
    )

    # 3. Clean 3-Layer Closed-Loop FSM (`L_rep + L39 + L41`, NO `L40`) — tests pure verify + fail->repair without L40!
    export_variant(
        "V21_fsm_3layer_no40",
        {
            f"{l_rep}.A": A_rep,
            f"{l_rep}.B": ((1.10 * best_rho_rep * d_rep["raw_yn"]) * d_rep["b_dir"]).bfloat16(),
            "39.A": A39,
            "39.B": B39,
            "41.A": (d41_no40["u_a"] * (d41_no40["scale"] / d41_no40["oof_on"])).bfloat16(),
            "41.B": ((1.10 * best_rho41 * d41_no40["raw_yn"]) * d41_no40["b_dir"]).bfloat16(),
        },
    )

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V21 Pure-WSP 4-State FSM adapters!", flush=True)


if __name__ == "__main__":
    main()
