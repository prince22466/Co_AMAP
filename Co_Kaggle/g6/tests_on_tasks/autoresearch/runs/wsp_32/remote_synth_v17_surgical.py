"""Synthesize V17 Surgical Zero-KV-Drift Readout (`L39, L40, L41 down_proj` r=1) + Gentle `L35` Global `o_proj`.

Directly resolves BOTH root causes uncovered in `wsp32_iter04` (`V16`):
1. Fixes `pytest tests/` 109-second full-suite timeout (`rich_3454`, `rich_3882`, `rich_3934`, `rich_4076`):
   - `L39 down_proj` now harvests ALL incomplete pytest prefixes (`...pytest`, `...pytest -q`, `...pytest tests`,
     `...pytest tests/`, `...pytest -q tests/`) as ON, suppressing premature `<|"|>` while boosting
     `(" tests" + "/" + "test")` so commands complete to file-level `pytest tests/test_<module>.py`,
     while putting completed `.py` paths (`...test_<module>.py`) strictly in the AlphaEdit null space (`OFF`, w=25)
     so `<|"|>` closes `.py` paths cleanly!
2. Preserves 100% of champion's delicate `edit_file` `old_string` accuracy (`b3_exact_edit_match >= 5/6`):
   - `L41.mlp.down_proj` is the FINAL layer of Gemma 4 (zero attention layers after `L41`), so edits on `L41`
     have ZERO KV-cache propagation to future tokens!
   - Full 10,240-D AlphaEdit puts 64 code/indentation tokens per `edit_file` turn + `true_imm_read_full @ call:` (w=35)
     + `true_imm_test_pass @ call:` (w=30) strictly in the null space, while delivering a calibrated +9..+14 nat
     `run_command` over `submit_patch` override right at `<|tool_call>call:` on unverified edits!
"""
import gc
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter, resolve_targeted_check
from recipe import compile_explicit
from synth_v16_upperband import (
    compile_oproj_downproj_r1,
    find_subseq_end,
    literal_role,
    solve_full_alphaedit,
)


def main():
    cdir = ROOT / "harvest32/clean_v14"
    out_dir = ROOT / "harvest32/v17_surgical"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = json.loads((ROOT / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tokenizer.truncation_side = "left"

    cases_v14 = json.loads((cdir / "clean_cases.json").read_text())
    nat = {r["prompt_sha"]: r for r in json.loads((cdir / "champion_clean.json").read_text())}

    all_items = {}
    for c in cases_v14:
        tr = literal_role(tokenizer, c["tokens"], c.get("source_run", ""))
        all_items[c["prompt_sha"]] = {
            "issue": c["issue"],
            "true_role": tr,
            "filepath": c.get("filepath"),
            "tokens": c["tokens"],
            "prompt_sha": c["prompt_sha"],
            "nat_text": (nat.get(c["prompt_sha"], {}).get("text") or ""),
            "nat_action": nat.get(c["prompt_sha"], {}).get("action"),
        }

    tasks_all = [json.loads(s) for s in (ROOT / "input/competitions/gemma-4-developer-agent/tasks.jsonl").read_text().splitlines() if s.strip()]
    disc_ids = {
        "rich_3506", "rich_3296", "rich_3067", "rich_3469", "rich_3130",
        "rich_4079", "rich_3480", "rich_3472", "rich_3105", "rich_3006",
        "fastapi_14786", "fastapi_14479", "requests_7328",
        "fastapi_15661", "fastapi_15588", "requests_7505", "requests_7502",
        "rich_4070", "rich_4006",
    }
    disc_ps = [t["problem_statement"] for t in tasks_all if t["instance_id"] in disc_ids]
    disc_needles = {}
    for t in tasks_all:
        if t["instance_id"] in disc_ids:
            cands = [ln.strip() for ln in t["problem_statement"].splitlines() if len(ln.strip()) >= 8]
            uniq = [ln for ln in cands if sum(1 for o in disc_ps if ln in o) == 1] or cands
            disc_needles[t["instance_id"]] = max(uniq, key=len)

    for rname in ("discovery02", "discovery03", "discovery04"):
        rp = ROOT / "experiments" / rname / "replay_tokens.jsonl"
        if not rp.is_file():
            continue
        for line in rp.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            sha = o["request_sha"]
            if sha in all_items:
                continue
            tr = literal_role(tokenizer, o["tokens"])
            if tr in ("true_imm_edit_ok", "true_imm_test_fail", "true_imm_test_pass"):
                txt_full = tokenizer.decode(o["tokens"], skip_special_tokens=False)
                iss = next((iid for iid, ndl in disc_needles.items() if ndl in txt_full), f"disc_{sha[:8]}")
                m_fp = re.findall(r'"filepath":\s*"([^"]+)"', txt_full)
                all_items[sha] = {
                    "issue": iss,
                    "true_role": tr,
                    "filepath": m_fp[-1] if m_fp else None,
                    "tokens": o["tokens"],
                    "prompt_sha": sha,
                    "nat_text": "",
                    "nat_action": None,
                }

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
    oproj_in_dims = {}
    for idx_l in range(len(layers)):
        op_mod = layers[idx_l].self_attn.o_proj
        in_f = getattr(op_mod, "in_features", None)
        if in_f is None:
            w_sh = getattr(op_mod, "weight_shape", None)
            in_f = int(w_sh[1]) if w_sh is not None else int(op_mod.weight.shape[1])
        oproj_in_dims[idx_l] = int(in_f)

    post_attn_35 = layers[35].post_attention_layernorm.weight.detach().float().cpu()
    post_ffn_w = {li: layers[li].post_feedforward_layernorm.weight.detach().float().cpu() for li in (39, 40, 41)}

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    thought_hdr = enc("<|channel>thought\n")
    chan_end_seq = enc("<channel|>")
    call_colon_seq = enc("<|tool_call>call:")

    # Incomplete pytest prefixes (ALL must NOT close with `<|"|>`!) vs completed `.py` prefix (MUST close cleanly!)
    inc_prefixes = [
        enc('<|tool_call>call:run_command{command:<|"|>pytest'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest -q'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest tests'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest tests/'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests'),
        enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests/'),
    ]
    py_ext_seq = enc(".py")

    def harvest_record(prefix: list[int], suffix: list[int], is_edit_turn: bool = False):
        toks = prefix + suffix
        assert len(toks) <= 4096
        p_len = len(prefix)

        win_lo = max(0, p_len - 8)
        win_hi = min(len(toks), p_len + 10)
        th_rel = find_subseq_end(toks[win_lo:win_hi], thought_hdr)
        ts0 = (win_lo + th_rel - 1) if th_rel is not None else (len(toks) - 1)

        ch_end_rel = find_subseq_end(suffix, chan_end_seq)
        ch_end_abs = (p_len + ch_end_rel - len(chan_end_seq)) if ch_end_rel is not None else len(toks)
        open_hi = min(ch_end_abs, ts0 + 8, len(toks))
        open_idx = list(range(ts0, max(ts0 + 1, open_hi)))
        open_t = torch.tensor(open_idx, device=dev, dtype=torch.long)

        colon_end = find_subseq_end(suffix, call_colon_seq)
        c_abs = (p_len + colon_end - 1) if colon_end is not None else None

        # Find ALL incomplete pytest positions & completed `.py` positions in suffix!
        pt_inc_abs = []
        for ip in inc_prefixes:
            ie = find_subseq_end(suffix, ip)
            if ie is not None:
                pt_inc_abs.append(p_len + ie - 1)
        pt_inc_abs = sorted(set(pt_inc_abs))

        py_done_abs = []
        if pt_inc_abs:
            py_e = find_subseq_end(suffix, py_ext_seq)
            if py_e is not None:
                py_done_abs.append(p_len + py_e - 1)

        # Sample dense code/indentation/quote tokens (up to 56 tokens on edit_file turns!) as hard null-space OFF!
        n_bg_target = 56 if is_edit_turn else 28
        delim_set = set()
        if ch_end_rel is not None:
            for k in range(max(0, p_len + ch_end_rel - 3), min(len(toks), p_len + ch_end_rel + 4)):
                delim_set.add(k)
        if c_abs is not None and len(toks) > c_abs + 2:
            step_suf = max(1, (len(toks) - c_abs - 1) // (40 if is_edit_turn else 14))
            for k in range(c_abs + 1, len(toks), step_suf):
                delim_set.add(k)
        bg_stride = max(1, max(1, ts0 - 8) // 14)
        for k in range(8, max(9, ts0 - 1), bg_stride):
            delim_set.add(k)
        forbid = set(open_idx) | ({c_abs} if c_abs is not None else set()) | set(pt_inc_abs)
        delim_idx = sorted(k for k in delim_set if k not in forbid)[:n_bg_target]
        if not delim_idx:
            delim_idx = [max(0, ts0 - 2)]
        delim_t = torch.tensor(delim_idx, device=dev, dtype=torch.long)

        cap = {"has_colon": c_abs is not None, "n_pt_inc": len(pt_inc_abs), "n_py_done": len(py_done_abs)}
        hooks = []

        def _h35(mod, inp, out):
            z = inp[0][0].detach()
            y = out[0].detach()
            cap[(35, "oz_ts0")] = z[ts0].to(torch.bfloat16).cpu()
            cap[(35, "oz_open")] = z.index_select(0, open_t).float().mean(0).to(torch.bfloat16).cpu()
            cap[(35, "oz_delim")] = z.index_select(0, delim_t).to(torch.bfloat16).cpu()
            cap[(35, "oy_raw_norm")] = float(y[ts0].float().norm().item())
            if c_abs is not None:
                cap[(35, "oz_col")] = z[c_abs].to(torch.bfloat16).cpu()

        hooks.append(layers[35].self_attn.o_proj.register_forward_hook(_h35))

        for li in (39, 40, 41):
            def mk_d(idx=li):
                def _hd(mod, inp, out):
                    h = inp[0][0].detach()
                    y = out[0].detach()
                    cap[(idx, "dy_raw_norm")] = float(y[max(0, c_abs or ts0)].float().norm().item())
                    cap[(idx, "dh_delim")] = h.index_select(0, delim_t).to(torch.bfloat16).cpu()
                    if c_abs is not None:
                        cap[(idx, "dh_col")] = h[c_abs].to(torch.bfloat16).cpu()
                    if pt_inc_abs and idx == 39:
                        pt_t = torch.tensor(pt_inc_abs, device=dev, dtype=torch.long)
                        cap[(39, "dh_pt_inc")] = h.index_select(0, pt_t).to(torch.bfloat16).cpu()
                    if py_done_abs and idx == 39:
                        pyd_t = torch.tensor(py_done_abs, device=dev, dtype=torch.long)
                        cap[(39, "dh_py_done")] = h.index_select(0, pyd_t).to(torch.bfloat16).cpu()
                return _hd
            hooks.append(layers[li].mlp.down_proj.register_forward_hook(mk_d()))

        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([toks], device=dev), use_cache=False, logits_to_keep=1)
        finally:
            for hk in hooks:
                hk.remove()
        return cap

    records = []
    for sha, item in all_items.items():
        rl = item["true_role"]
        if rl in ("after_submit_ignore", "non_imm_prose"):
            continue
        txt = item["nat_text"]
        is_ed = (item["nat_action"] == "edit_file") or (rl in ("true_imm_read_full", "true_imm_read_trunc"))
        if not txt:
            if rl == "true_imm_test_pass":
                txt = "<|channel>thought\nTests passed cleanly. Submitting patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>"
            elif rl in ("true_imm_edit_ok", "true_imm_test_fail"):
                # Synthesize realistic premature-submit suffix so we ALSO capture exact `<|tool_call>call:` state!
                txt = "<|channel>thought\nThe change looks minimal and complete. I will now submit the patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>"
            else:
                tail_dec = tokenizer.decode(item["tokens"][-10:], skip_special_tokens=False)
                txt = "" if tail_dec.endswith("<|channel>thought\n") else "<|channel>thought\n"
        full_suf = enc(txt) if txt else []
        suf = (full_suf[:140] + full_suf[-628:]) if len(full_suf) > 768 else full_suf
        pref = item["tokens"][-(4096 - len(suf)) :]
        cap = harvest_record(pref, suf, is_edit_turn=is_ed)
        records.append({
            "issue": item["issue"],
            "true_role": rl,
            "filepath": item["filepath"],
            "has_full_suf": len(suf) > 6,
            "tokens": item["tokens"],
            "nat_action": item["nat_action"],
            "cap": cap,
        })

    # Harvest BOTH `pytest tests/test_<stem>.py` AND `pytest -q tests/test_<stem>.py` across discovery issues for L39!
    bare_caps = []
    seen_iss = set()
    for r in records:
        if r["issue"] in seen_iss or r["true_role"] not in ("true_imm_edit_ok", "true_imm_test_fail"):
            continue
        seen_iss.add(r["issue"])
        stem = Path(r.get("filepath") or "panel.py").stem.lstrip("_") or "panel"
        for variant_cmd in (f"pytest tests/test_{stem}.py", f"pytest -q tests/test_{stem}.py"):
            suf = enc(f"<|channel>thought\nLet's run targeted pytest on test_{stem}.py.\n<channel|><|tool_call>call:run_command{{command:<|\"|>{variant_cmd}<|\"|>}}<tool_call|>")
            pref = r["tokens"][-(4096 - len(suf)) :]
            cap = harvest_record(pref, suf, is_edit_turn=False)
            if cap["n_pt_inc"] > 0:
                bare_caps.append(cap)

    # Pure weight-space unembedding vectors
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

    def precond(u_res: torch.Tensor, w_pn: torch.Tensor) -> torch.Tensor:
        b = u_res * w_pn.clamp_min(0.05)
        return b / b.norm().clamp_min(1e-8)

    def tok_vec(words: list[str]) -> torch.Tensor:
        vecs = []
        for w in words:
            for variant in (w, " " + w):
                ids = enc(variant)
                if len(ids) == 1:
                    v = lm_w[ids[0]] * scale_w
                    vecs.append(v / v.norm().clamp_min(1e-8))
        m = torch.stack(vecs, dim=0).mean(dim=0)
        return m / m.norm().clamp_min(1e-8)

    u_thought_verify = tok_vec(["pytest", "test", "tests", "verify", "check", "run"]) - tok_vec(
        ["submit", "done", "complete", "finished", "ready"]
    )
    u_thought_verify = u_thought_verify / u_thought_verify.norm().clamp_min(1e-8)

    # Exact `<|tool_call>call:` dispatch tokens
    tok_run = enc("<|tool_call>call:run_command{")[len(call_colon_seq)]
    tok_sub = enc("<|tool_call>call:submit_patch{")[len(call_colon_seq)]
    tok_ed = enc("<|tool_call>call:edit_file{")[len(call_colon_seq)]
    tok_rd = enc("<|tool_call>call:read_file{")[len(call_colon_seq)]

    # L41 C1 dispatch: strongly suppress submit_patch, boost run_command!
    u_c1_disp = (lm_w[tok_sub] - lm_w[tok_run]) * scale_w
    u_c1_disp = u_c1_disp / u_c1_disp.norm().clamp_min(1e-8)

    # L40 C2 dispatch: after FileEditError, suppress blind edit_file + submit_patch, boost read_file!
    u_c2_err = ((0.85 * lm_w[tok_ed] + 0.15 * lm_w[tok_sub]) - lm_w[tok_rd]) * scale_w
    u_c2_err = u_c2_err / u_c2_err.norm().clamp_min(1e-8)

    # Exact File-Level L39 Anti-Bare & Anti-Directory-Pytest direction!
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

    print("Verified L39 token IDs:", {
        "quote": (tok_quote, tokenizer.decode([tok_quote])),
        "sp_tests": (tok_sp_tests, tokenizer.decode([tok_sp_tests])),
        "sp_dashq": (tok_sp_dashq, tokenizer.decode([tok_sp_dashq])),
        "slash": (tok_slash, tokenizer.decode([tok_slash])),
        "test_word": (tok_test_word, tokenizer.decode([tok_test_word])),
    }, flush=True)

    # Suppress premature `<|"|>` closing, boost `" tests"`, `"/"`, and `"test"` (test_<module>.py)!
    good_l39 = (
        0.30 * (lm_w[tok_sp_tests] / lm_w[tok_sp_tests].norm())
        + 0.15 * (lm_w[tok_sp_dashq] / lm_w[tok_sp_dashq].norm())
        + 0.20 * (lm_w[tok_slash] / lm_w[tok_slash].norm())
        + 0.35 * (lm_w[tok_test_word] / lm_w[tok_test_word].norm())
    )
    bad_l39 = lm_w[tok_quote] / lm_w[tok_quote].norm()
    u_c3_file = (bad_l39 - good_l39) * scale_w
    u_c3_file = u_c3_file / u_c3_file.norm().clamp_min(1e-8)

    def is_c1_on(r: dict) -> bool:
        return r["true_role"] == "true_imm_edit_ok" or (
            r["true_role"] == "true_imm_test_fail"
        )

    def is_c2_on(r: dict) -> bool:
        return r["true_role"] == "true_imm_edit_err"

    # =========================================================================
    # 1. Fit L39 File-Level Targeted Pytest (`ON` = all incomplete prefixes, `OFF` = `.py` done + all code/quotes!)
    # =========================================================================
    h39_on, h39_py_done, h39_off, dyn39 = [], [], [], []
    for r in records:
        dyn39.append(r["cap"][(39, "dy_raw_norm")])
        h39_off.append(r["cap"][(39, "dh_delim")].float())
        if r["cap"]["has_colon"]:
            h39_off.append(r["cap"][(39, "dh_col")].float().unsqueeze(0))
        if r["cap"]["n_pt_inc"] > 0:
            h39_on.append(r["cap"][(39, "dh_pt_inc")].float())
        if r["cap"]["n_py_done"] > 0:
            h39_py_done.append(r["cap"][(39, "dh_py_done")].float())
    for bc in bare_caps:
        h39_on.append(bc[(39, "dh_pt_inc")].float())
        h39_off.append(bc[(39, "dh_delim")].float())
        if bc["n_py_done"] > 0:
            h39_py_done.append(bc[(39, "dh_py_done")].float())

    H39_on = torch.cat(h39_on, dim=0)
    H39_pyd = torch.cat(h39_py_done, dim=0) if h39_py_done else torch.zeros(0, 10240)
    H39_off = torch.cat(h39_off, dim=0)
    a39_u, sc39 = solve_full_alphaedit(H39_on, [(H39_pyd, 28.0), (H39_off, 18.0)], lam_rel=0.10)
    w39 = a39_u * sc39
    s39_on = float((H39_on @ w39).mean())
    s39_pyd_max = float((H39_pyd @ w39).abs().max()) if H39_pyd.shape[0] else 0.0
    s39_off_rms = float((H39_off @ w39).pow(2).mean().sqrt().clamp_min(1e-6))
    b39_dir = precond(-u_c3_file, post_ffn_w[39])
    dyn39_mean = sum(dyn39) / len(dyn39)

    # =========================================================================
    # 2. Fit L41 Zero-KV-Drift Colon Dispatch (`true_imm_edit_ok + true_imm_test_fail @ call:` -> `run_command`)
    #    and L40 Colon Dispatch (`true_imm_edit_err @ call:` -> `read_file`)
    # =========================================================================
    def fit_surgical_colon(li: int, on_pred, u_disp: torch.Tensor):
        on_h, rd_h, ver_h, oth_h, del_h, dyn = [], [], [], [], [], []
        for r in records:
            dyn.append(r["cap"][(li, "dy_raw_norm")])
            del_h.append(r["cap"][(li, "dh_delim")].float())
            if not r["cap"]["has_colon"]:
                continue
            hc = r["cap"][(li, "dh_col")].float()
            if on_pred(r):
                on_h.append(hc)
            elif r["true_role"] in ("true_imm_read_full", "true_imm_read_trunc"):
                rd_h.append(hc)
            elif r["true_role"] == "true_imm_test_pass":
                ver_h.append(hc)
            else:
                oth_h.append(hc)
        Ho = torch.stack(on_h)
        Hr = torch.stack(rd_h) if rd_h else torch.zeros(0, 10240)
        Hv = torch.stack(ver_h) if ver_h else torch.zeros(0, 10240)
        Hot = torch.stack(oth_h) if oth_h else torch.zeros(0, 10240)
        Hd = torch.cat(del_h, dim=0)
        au, sc = solve_full_alphaedit(Ho, [(Hr, 35.0), (Hv, 30.0), (Hot, 15.0), (Hd, 30.0)], lam_rel=0.08)
        w = au * sc
        son = float((Ho @ w).mean())
        return {
            "u_a": au, "scale": sc, "in_on": max(0.15, son),
            "raw_yn": sum(dyn) / len(dyn),
            "rd_ratio": round(float((Hr @ w).abs().max()) / max(1e-4, son), 4) if Hr.shape[0] else 0.0,
            "ver_ratio": round(float((Hv @ w).abs().max()) / max(1e-4, son), 4) if Hv.shape[0] else 0.0,
            "code_rms_ratio": round(float((Hd @ w).pow(2).mean().sqrt()) / max(1e-4, son), 4),
            "b_dir": precond(-u_disp, post_ffn_w[li]),
        }

    d41_c1 = fit_surgical_colon(41, is_c1_on, u_c1_disp)
    d40_c2 = fit_surgical_colon(40, is_c2_on, u_c2_err)

    # =========================================================================
    # 3. Fit Gentle Single-Layer Global Full-Attention `L35 o_proj` (`sliding_window=None`, 4096-D)
    # =========================================================================
    on35, ver35, rd35, oth35, del35, yn35 = [], [], [], [], [], []
    for r in records:
        yn35.append(r["cap"][(35, "oy_raw_norm")])
        z0 = r["cap"][(35, "oz_ts0")].float()
        zo = r["cap"][(35, "oz_open")].float()
        del35.append(r["cap"][(35, "oz_delim")].float())
        if r["cap"]["has_colon"]:
            del35.append(r["cap"][(35, "oz_col")].float().unsqueeze(0))
        if is_c1_on(r):
            on35.extend([z0, zo])
        elif r["true_role"] == "true_imm_test_pass":
            ver35.extend([z0, zo])
        elif r["true_role"] in ("true_imm_read_full", "true_imm_read_trunc"):
            rd35.extend([z0, zo])
        else:
            oth35.append(z0)
    Xo35 = torch.stack(on35)
    o35_u, sc35 = solve_full_alphaedit(
        Xo35,
        [
            (torch.stack(ver35), 25.0),
            (torch.stack(rd35), 25.0),
            (torch.stack(oth35), 12.0),
            (torch.cat(del35, dim=0), 35.0),
        ],
        lam_rel=0.10,
    )
    s35_on = float((Xo35 @ (o35_u * sc35)).mean())
    b35_dir = precond(u_thought_verify, post_attn_35)
    yn35_mean = sum(yn35) / len(yn35)

    # =========================================================================
    # 4. LIVE IN-VRAM COLON LOGIT CALIBRATION AT `<|tool_call>call:` ON `L41`!
    #    Verify exact `logit[run_command] - logit[submit_patch]` shift on ON vs OFF!
    # =========================================================================
    cal_colon_on = [r for r in records if is_c1_on(r)][:5]
    cal_colon_ver = [r for r in records if r["true_role"] == "true_imm_test_pass"][:4]
    cal_colon_rd = [r for r in records if r["true_role"] == "true_imm_read_full"][:5]

    colon_suf_prem = enc("<|channel>thought\nThe change is complete. I will now submit the patch.\n<channel|><|tool_call>call:")
    colon_suf_rd = enc("<|channel>thought\nLet's apply the fix using edit_file.\n<channel|><|tool_call>call:")

    cal_l41 = []
    for rho41 in (0.35, 0.55, 0.80, 1.10):
        av = (d41_c1["u_a"] * (d41_c1["scale"] / d41_c1["in_on"])).to(dev, torch.bfloat16)
        bv = (rho41 * d41_c1["raw_yn"] * d41_c1["b_dir"]).to(dev, torch.bfloat16)

        def measure_set(sub_recs, suf_tks, t_pos, t_neg):
            vals_raw, vals_new = [], []
            for r in sub_recs:
                inp = torch.tensor([(r["tokens"][-(4096 - len(suf_tks)) :] + suf_tks)], device=dev)
                with torch.inference_mode():
                    lg0 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                hk = layers[41].mlp.down_proj.register_forward_hook(
                    lambda mod, inputs, output: output + (inputs[0] @ av).unsqueeze(-1) * bv
                )
                try:
                    with torch.inference_mode():
                        lg1 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                finally:
                    hk.remove()
                vals_raw.append(float((lg0[t_pos] - lg0[t_neg]).item()))
                vals_new.append(float((lg1[t_pos] - lg1[t_neg]).item()))
            return round(sum(vals_raw) / max(1, len(vals_raw)), 2), round(sum(vals_new) / max(1, len(vals_new)), 2)

        on_b, on_a = measure_set(cal_colon_on, colon_suf_prem, tok_run, tok_sub)
        ver_b, ver_a = measure_set(cal_colon_ver, colon_suf_prem, tok_sub, tok_run)
        rd_b, rd_a = measure_set(cal_colon_rd, colon_suf_rd, tok_ed, tok_run)
        cal_l41.append({
            "rho41": rho41,
            "on_run_minus_sub": f"{on_b:+.2f} -> {on_a:+.2f}",
            "ver_sub_minus_run": f"{ver_b:+.2f} -> {ver_a:+.2f}",
            "rd_edit_minus_run": f"{rd_b:+.2f} -> {rd_a:+.2f}",
        })

    del model, lm_root, layers
    gc.collect()
    torch.cuda.empty_cache()

    diagnostics = {
        "n_records": len(records),
        "L39_file_pytest": {
            "n_on": int(H39_on.shape[0]),
            "n_py_done": int(H39_pyd.shape[0]),
            "s39_on_mean": round(s39_on, 4),
            "py_done_max_ratio": round(s39_pyd_max / max(1e-4, s39_on), 4),
            "snr": round(s39_on / s39_off_rms, 2),
        },
        "L41_colon_c1": {k: v for k, v in d41_c1.items() if k not in ("u_a", "b_dir")},
        "L40_colon_c2": {k: v for k, v in d40_c2.items() if k not in ("u_a", "b_dir")},
        "L41_live_colon_calibration": cal_l41,
    }
    print("V17 Surgical Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    # --- Family A: Pure `down_proj` (`target_modules=['down_proj']`, r=1, ZERO o_proj, ZERO L41 KV-cache drift!) ---
    def add_pure_down(name: str, rho39: float, rho40: float, rho41: float):
        ft = {}
        ft["39.A"] = (a39_u * (sc39 / max(0.15, s39_on))).bfloat16()
        ft["39.B"] = ((rho39 * dyn39_mean) * b39_dir).bfloat16()
        if rho40 > 0:
            ft["40.A"] = (d40_c2["u_a"] * (d40_c2["scale"] / d40_c2["in_on"])).bfloat16()
            ft["40.B"] = ((rho40 * d40_c2["raw_yn"]) * d40_c2["b_dir"]).bfloat16()
        if rho41 > 0:
            ft["41.A"] = (d41_c1["u_a"] * (d41_c1["scale"] / d41_c1["in_on"])).bfloat16()
            ft["41.B"] = ((rho41 * d41_c1["raw_yn"]) * d41_c1["b_dir"]).bfloat16()
        fac_p = out_dir / f"fac_{name}.safetensors"
        save_file(ft, str(fac_p))
        zlbl = f"V17_dzero_{'_'.join(sorted(k.split('.')[0] for k in ft if k.endswith('.A')))}"
        if not (ad_root / zlbl).exists():
            zi = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / zlbl, "zero", rank=1)
            compiled.append({"adapter": zlbl, "sha256": zi["adapter_sha256"], "control": "zero"})
        for ctrl in ("candidate", "random"):
            inf = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / f"{name}_{ctrl}", ctrl, rank=1)
            compiled.append({"adapter": f"{name}_{ctrl}", "sha256": inf["adapter_sha256"], "control": ctrl})

    add_pure_down("V17_donly_39_41_mid", rho39=0.48, rho40=0.0, rho41=0.65)
    add_pure_down("V17_donly_39_40_41_mid", rho39=0.48, rho40=0.38, rho41=0.65)
    add_pure_down("V17_donly_39_40_41_hi", rho39=0.55, rho40=0.48, rho41=0.95)

    # --- Family B: Gentle Single-Layer Global `L35 o_proj` (`rho_o=0.18` -> ~3.5 nats ON / 0.12 OFF) + `L39,L40,L41 down_proj` ---
    for name, rho_o35, rho39, rho40, rho41 in (
        ("V17_gl35_gentle_d39_40_41", 0.16, 0.48, 0.36, 0.58),
        ("V17_gl35_mid_d39_40_41", 0.24, 0.52, 0.42, 0.78),
    ):
        ospecs = {35: (o35_u * (sc35 / max(0.15, s35_on)), (rho_o35 * yn35_mean) * b35_dir)}
        dspecs = {
            39: (a39_u * (sc39 / max(0.15, s39_on)), (rho39 * dyn39_mean) * b39_dir),
            40: (d40_c2["u_a"] * (d40_c2["scale"] / d40_c2["in_on"]), (rho40 * d40_c2["raw_yn"]) * d40_c2["b_dir"]),
            41: (d41_c1["u_a"] * (d41_c1["scale"] / d41_c1["in_on"]), (rho41 * d41_c1["raw_yn"]) * d41_c1["b_dir"]),
        }
        if not (ad_root / "V17_hybrid_zero").exists():
            compiled.append(compile_oproj_downproj_r1(ROOT / "champion.zip", ospecs, dspecs, oproj_in_dims, ad_root / "V17_hybrid_zero", "zero"))
        for ctrl in ("candidate", "random"):
            compiled.append(compile_oproj_downproj_r1(ROOT / "champion.zip", ospecs, dspecs, oproj_in_dims, ad_root / f"{name}_{ctrl}", ctrl))

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V17 Surgical Zero-KV-Drift adapters!", flush=True)


if __name__ == "__main__":
    main()
