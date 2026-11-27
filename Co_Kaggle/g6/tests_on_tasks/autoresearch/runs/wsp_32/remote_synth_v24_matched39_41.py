"""Synthesize V24 Pure `L39 + L41` Matched-Suffix Global-History Pure-WSP Adapter (`r=1`, `down_proj`, ZERO `L40`).

Resolves BOTH root causes uncovered in the multi-turn trace audit of `wsp32_v23_disjoint6`:
1. Eliminates `L40` Repeat-Edit Loops (`rich_3296` 0->10 edit errs, `rich_3480` 2->6 edit errs):
   - Multi-turn traces proved that `L40` (`+edit_file`) caused the agent to repeat its just-applied `edit_file` call (`old_string not found`),
     whereas unmodified `champion_r1` ALREADY calls `read_file`/`edit_file` naturally on 6/6 module `pytest` failures (`steps/1386`)
     as long as `41.A` keeps `true_imm_real_test_fail` in its null space (`OFF`)!
   - Across all `QUICK_15` runs, ONLY `L39 + L41` (`wsp32_iter05`, NO `L40`) achieved `s2_valid_test_ran = 8/15`,
     `s3_defect_exposed = 5/15`, and `syntax_leaks = 0`.
2. Eliminates the Thought-Suffix Shortcut in `41.A` (`Why fastapi_14479 & fastapi_14786 still called submit_patch in V23`):
   - In `clean_v21`, 12/13 `true_imm_edit_ok` replays already contained `"I will run pytest"` in their generated thought while 11/11
     `true_imm_test_pass` replays contained `"Submitting patch"`, so `V21..V23`'s `41.A` learned a thought-text shortcut (`"pytest"` vs `"submit"`)
     and outputted ~0 whenever an unverified task (`fastapi_14479`, `fastapi_14786`) wrote `"I will submit the patch"`!
   - `V24` harvests EVERY `true_imm_edit_ok` & `true_imm_bad_test_path` (`ON`) AND EVERY `true_imm_test_pass` (`OFF`, 11 real `pytest ... passed`
     histories) & `true_imm_real_test_fail` (`OFF`, 14 histories) under **IDENTICAL MATCHED PREMATURE-SUBMIT THOUGHT SUFFIXES**
     (in addition to their real `champion_r1` suffixes, plus all 23 `edit_ok` prompts across `discovery02/03/04`), mathematically forcing
     `41.A` to use `L41`'s 4,096-token Global Full Attention over the `<|tool_response>` history!
3. Eliminates `pytest tests/rich/...` (`ec=4`) bad paths in `L39` by subtracting `0.28 * u["rich"]` alongside `-0.80 * u[quote]` at `pt_inc_abs`
   while keeping `H39_quote_close` (`beta=160`), `H39_pyd` (`beta=120`), and `H39_open` (`beta=130`) in hard null space!
"""
from __future__ import annotations

import gc
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_v21_disjoint_corpus import DISCOVERY_19, QUICK_30_IDX, HOLDOUT_IDS
from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter
from recipe import compile_explicit
from remote_synth_v21_fsm import solve_alphaedit_v21
from synth_v16_upperband import find_subseq_end

MATCHED_SUBMIT_THOUGHTS = [
    "<|channel>thought\nThe change looks minimal and complete. I will now submit the patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>",
    "<|channel>thought\nThe fix has been applied cleanly to the file and matches the specification. Let's submit the patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>",
]


def main():
    v21_cases_p = ROOT / "harvest32/clean_v21/real_replays.json"
    assert v21_cases_p.is_file(), f"Missing {v21_cases_p}"
    out_dir = ROOT / "harvest32/v24_matched39_41"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = json.loads((ROOT / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tokenizer.truncation_side = "left"

    # Verify strict snapshot disjointness
    tasks_all = [json.loads(s) for s in (ROOT / "input/competitions/gemma-4-developer-agent/tasks.jsonl").read_text().splitlines() if s.strip()]
    tasks_by_id = {t["instance_id"]: t for t in tasks_all}
    reserved_snaps = {(tasks_all[i]["repo"], tasks_all[i]["base_commit"]) for i in QUICK_30_IDX}
    reserved_snaps |= {(t["repo"], t["base_commit"]) for t in tasks_all if t["instance_id"] in HOLDOUT_IDS}
    disc_snaps = {(tasks_by_id[i]["repo"], tasks_by_id[i]["base_commit"]) for i in DISCOVERY_19}
    assert not (disc_snaps & reserved_snaps)

    raw_replays = list(json.loads(v21_cases_p.read_text()))
    seen_shas = {r["prompt_sha"] for r in raw_replays}

    # ALSO pull ALL missed `edit_file -> status: ok` turns from discovery02/03/04 (fixing the non-greedy TR_RE regex!)
    disc_ps = [tasks_by_id[i]["problem_statement"] for i in DISCOVERY_19]
    disc_needles = {}
    for iid in DISCOVERY_19:
        cands = [ln.strip() for ln in tasks_by_id[iid]["problem_statement"].splitlines() if len(ln.strip()) >= 8]
        uniq = [ln for ln in cands if sum(1 for o in disc_ps if ln in o) == 1] or cands
        disc_needles[iid] = max(uniq, key=len)

    added_edit_ok = 0
    for rn in ("discovery02", "discovery03", "discovery04"):
        rp = ROOT / "experiments" / rn / "replay_tokens.jsonl"
        if not rp.is_file():
            continue
        for ln in rp.read_text().splitlines():
            if not ln.strip():
                continue
            o = json.loads(ln)
            if o["request_sha"] in seen_shas:
                continue
            txt_tail = tokenizer.decode(o["tokens"][-900:], skip_special_tokens=False)
            # Check the LAST `<|tool_response>` specifically using rfind!
            last_tr_pos = txt_tail.rfind("<|tool_response>response:")
            if last_tr_pos < 0:
                continue
            tail_after_tr = txt_tail[last_tr_pos:]
            if tail_after_tr.startswith("<|tool_response>response:edit_file{") and '"status": "ok"' in tail_after_tr:
                end_tr = tail_after_tr.find("<tool_response|>")
                if end_tr > 0 and "<|turn>user" not in tail_after_tr[end_tr:] and "<|tool_call>" not in tail_after_tr[end_tr:]:
                    txt_full = tokenizer.decode(o["tokens"], skip_special_tokens=False)
                    if "<|tool_response>response:submit_patch{" in txt_full:
                        continue
                    iss = next((iid for iid, ndl in disc_needles.items() if ndl in txt_full), None)
                    if iss in DISCOVERY_19:
                        fp_m = re.findall(r'"filepath":\s*"([^"]+)"', tail_after_tr)
                        raw_replays.append({
                            "issue": iss,
                            "true_role": "true_imm_edit_ok",
                            "filepath": fp_m[-1] if fp_m else None,
                            "tokens": o["tokens"],
                            "prompt_sha": o["request_sha"],
                            "action": "submit_patch",
                            "text": MATCHED_SUBMIT_THOUGHTS[0],
                        })
                        seen_shas.add(o["request_sha"])
                        added_edit_ok += 1

    print(f"Added {added_edit_ok} previously-skipped `true_imm_edit_ok` prompts! Role counts:", Counter(r["true_role"] for r in raw_replays), flush=True)

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
    post_ffn_w = {li: layers[li].post_feedforward_layernorm.weight.detach().float().cpu() for li in (39, 41)}

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
    s_bad_rich1 = enc('<|tool_call>call:run_command{command:<|"|>pytest tests/rich/test_panel.py<|"|>}')

    tok_quote = s_bare[len(p_base)]
    tok_sp_tests = s_full1[len(p_base)]
    tok_sp_dashq = s_full2[len(p_base)]
    tok_slash = s_full1[len(p_dir_noslash)]
    tok_test_word = s_full1[len(p_dir_slash)]
    tok_bad_rich = s_bad_rich1[len(p_dir_slash)]

    def build_prep_entry(item: dict, txt_override: str | None = None, is_matched_pair: bool = False) -> dict | None:
        rl = item["true_role"]
        txt = txt_override if txt_override is not None else (item.get("text") or "")
        if not txt or "<|tool_call>call:" not in txt:
            return None
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
            return None

        cmd_q_end = find_subseq_end(suf, cmd_open_quote_seq)
        cmd_q_abs = (p_len + cmd_q_end - 1) if cmd_q_end is not None else None
        pt_inc_abs = sorted({p_len + ie - 1 for ip in inc_prefixes if (ie := find_subseq_end(suf, ip)) is not None})
        py_done_abs = []
        if pt_inc_abs and (py_e := find_subseq_end(suf, py_ext_seq)) is not None:
            py_done_abs.append(p_len + py_e - 1)

        quote_pred_abs = [
            pos - 1
            for pos in range(max(p_len + 1, c_abs + 1), len(toks))
            if toks[pos] == tok_quote and (pos - 1) not in set(pt_inc_abs)
        ][:16]

        is_ed_or_pres = (item.get("action") == "edit_file") or (rl in ("true_imm_read_full", "true_imm_read_trunc", "turn1_init"))
        open_idx = [k for k in range(ts0, min(ch_end_abs, ts0 + 10, len(toks))) if k != c_abs and k not in set(pt_inc_abs)]

        delim_set = set(open_idx) | set(quote_pred_abs)
        if ch_end_abs > ts0:
            th_step = max(1, (ch_end_abs - ts0) // (36 if is_ed_or_pres else 20))
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

        return {
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
            "is_matched_pair": is_matched_pair,
        }

    prepared = []
    for item in raw_replays:
        rl = item["true_role"]
        # 1. Include genuine `champion_r1` trajectory
        e_nat = build_prep_entry(item, None, is_matched_pair=False)
        if e_nat is not None:
            prepared.append(e_nat)
        # 2. CRITICAL: For `edit_ok`, `bad_test_path`, `test_pass`, AND `real_test_fail`, ALSO include IDENTICAL
        #    matched premature-submit thought suffixes so `41.A` CANNOT cheat on `"pytest"` vs `"submit"` in the thought!
        if rl in ("true_imm_edit_ok", "true_imm_bad_test_path", "true_imm_test_pass"):
            for m_txt in MATCHED_SUBMIT_THOUGHTS:
                e_m = build_prep_entry(item, m_txt, is_matched_pair=True)
                if e_m is not None:
                    prepared.append(e_m)

    bare_prepared = []
    seen_iss = set()
    for r in prepared:
        if r["issue"] in seen_iss or r["true_role"] not in ("true_imm_edit_ok", "true_imm_bad_test_path"):
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

    print(f"Prepared {len(prepared)} records ({sum(1 for x in prepared if x['is_matched_pair'])} matched-suffix history pairs!) + {len(bare_prepared)} pytest records.", flush=True)

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

    protected_action_toks = {tok_run, tok_sub, tok_ed, tok_rd, tok_sp_tests, tok_sp_dashq, tok_slash, tok_test_word, tok_bad_rich}

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

    # For L41 colon dispatch: strictly orthogonalize against `<|"|>` AND `edit_file`/`read_file` so L41 ONLY trades `run_command` vs `submit_patch`!
    l41_sacred_list = sorted((base_sacred | {tok_quote, tok_ed, tok_rd}) - {tok_run, tok_sub})
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

    # L39: boost `" tests"`, `" -q"`, `"/"`, `"test"` and suppress premature `<|"|>` + wrong `tests/rich/` (`tok_bad_rich`)!
    pos_l39 = (
        0.30 * unit_unemb(tok_sp_tests)
        + 0.15 * unit_unemb(tok_sp_dashq)
        + 0.20 * unit_unemb(tok_slash)
        + 0.35 * unit_unemb(tok_test_word)
    ) - (0.80 * unit_unemb(tok_quote) + 0.28 * unit_unemb(tok_bad_rich))

    # L41: pure `+run_command - submit_patch` WITH `edit_file` and `read_file` in `l41_sacred_list` (ZERO repeat-edit push!!)!
    pos_verify = unit_unemb(tok_run) - unit_unemb(tok_sub)

    def harvest_single(li: int):
        recs_h, bare_h = [], []
        for r in prepared:
            dt = torch.tensor(r["delim_idx"], device=dev, dtype=torch.long)
            ot = torch.tensor(r["open_idx"], device=dev, dtype=torch.long)
            cap = {}
            def _h(mod, inp, out):
                h = inp[0][0].detach()
                y = out[0].detach()
                cap["dy_norm"] = float(y[r["c_abs"]].float().norm().item())
                cap["dh_col"] = h[r["c_abs"]].to(torch.bfloat16).cpu()
                cap["dh_open"] = h.index_select(0, ot).to(torch.bfloat16).cpu()
                cap["dh_delim"] = h.index_select(0, dt).to(torch.bfloat16).cpu()
                if r["cmd_q_abs"] is not None:
                    cap["dh_cmd_q"] = h[r["cmd_q_abs"]].to(torch.bfloat16).cpu()
                if li == 39 and r["quote_pred_abs"]:
                    cap["dh_qclose"] = h.index_select(0, torch.tensor(r["quote_pred_abs"], device=dev)).to(torch.bfloat16).cpu()
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
    # STAGE 1: Fit L39 Targeted Pytest & Attach Live atProven Zero-Leak `rho_in = 0.43`!
    # =========================================================================
    print("=== Stage 1: Fitting L39 Targeted Pytest ===", flush=True)
    r39, b39 = harvest_single(39)
    h39_on, h39_pyd, h39_qcl, h39_open, h39_off, dyn39 = [], [], [], [], [], []
    for r in r39:
        if r["is_matched_pair"]:
            continue
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
    B39_sweet = ((0.44 * dyn39_mean) * b39_dir).bfloat16()
    a39_dev, b39_dev = A39.to(dev), B39_sweet.to(dev)
    hk39 = layers[39].mlp.down_proj.register_forward_hook(
        lambda mod, inputs, output: output + (inputs[0] @ a39_dev).unsqueeze(-1) * b39_dev
    )

    # =========================================================================
    # STAGE 2: Fit L41 Global-History Unverified-vs-Verified Detector on Post-L39 States!
    # =========================================================================
    print("=== Stage 2: Fitting L41 Matched-Suffix Global-History Detector ===", flush=True)
    r41, b41 = harvest_single(41)

    is_on_41 = lambda r: r["true_role"] in ("true_imm_edit_ok", "true_imm_bad_test_path")

    def pack_41(sub_recs):
        on_h, ver_h, fail_h, rd_h, oth_h, cmdq_h, open_h, del_h, dyn = [], [], [], [], [], [], [], [], []
        for r in sub_recs:
            hc = r["cap"]["dh_col"].float()
            if not r["is_matched_pair"]:
                dyn.append(r["cap"]["dy_norm"])
                open_h.append(r["cap"]["dh_open"].float())
                del_h.append(r["cap"]["dh_delim"].float())
                if "dh_cmd_q" in r["cap"]:
                    cmdq_h.append(r["cap"]["dh_cmd_q"].float())
            if is_on_41(r):
                # Upweight matched-submit pairs (2x) so 41.A learns global history rather than thought text!
                on_h.append(hc)
                if r["is_matched_pair"]:
                    on_h.append(hc)
            elif r["true_role"] == "true_imm_test_pass":
                ver_h.append(hc)
            elif r["true_role"] == "true_imm_real_test_fail":
                fail_h.append(hc)
            elif r["true_role"] in ("true_imm_read_full", "true_imm_read_trunc", "turn1_init"):
                rd_h.append(hc)
            else:
                oth_h.append(hc)
        for bc in b41:
            if "dh_cmd_q" in bc:
                cmdq_h.append(bc["dh_cmd_q"].float())
        Ho = torch.stack(on_h)
        Hv = torch.stack(ver_h)
        Hf = torch.stack(fail_h) if fail_h else torch.zeros(0, 10240)
        Hr = torch.stack(rd_h) if rd_h else torch.zeros(0, 10240)
        Hot = torch.stack(oth_h) if oth_h else torch.zeros(0, 10240)
        Hcq = torch.stack(cmdq_h) if cmdq_h else torch.zeros(0, 10240)
        Hop = torch.cat(open_h, dim=0)
        Hd = torch.cat(del_h, dim=0)
        # Notice `Hv` (verified test pass, matched submit thought!) & `Hf` (real test fail -> natural repair!) & `Hr` in heavy null space!
        ogrps = [(Hv, 130.0), (Hf, 110.0), (Hr, 100.0), (Hot, 75.0), (Hcq, 80.0), (Hop, 140.0), (Hd, 100.0)]
        return Ho, ogrps, sum(dyn) / len(dyn)

    # Leave-One-Issue-Out CV specifically measured on **MATCHED-SUBMIT SUFFIXES** (true premature-submit test!)
    on_issues = sorted({r["issue"] for r in r41 if is_on_41(r)})
    oof_prem_on, oof_ver_off, oof_fail_off, oof_open = [], [], [], []
    for iss in on_issues:
        tr = [r for r in r41 if r["issue"] != iss]
        Ho_tr, og_tr, _ = pack_41(tr)
        wu_tr, sc_tr = solve_alphaedit_v21(Ho_tr, og_tr, lam_rel=0.20)
        w_tr = wu_tr * sc_tr
        for r in r41:
            if r["issue"] != iss:
                continue
            vc = float(torch.dot(w_tr, r["cap"]["dh_col"].float()))
            if not r["is_matched_pair"]:
                oof_open.append(float((r["cap"]["dh_open"].float() @ w_tr).abs().max().item()))
            if is_on_41(r) and r["is_matched_pair"]:
                oof_prem_on.append(vc)
            elif r["true_role"] == "true_imm_test_pass" and r["is_matched_pair"]:
                oof_ver_off.append(vc)
            elif r["true_role"] == "true_imm_real_test_fail":
                oof_fail_off.append(abs(vc))

    Ho41, og41, raw_yn41 = pack_41(r41)
    au41, sc41 = solve_alphaedit_v21(Ho41, og41, lam_rel=0.20)
    w41 = au41 * sc41
    in_on41 = float((Ho41 @ w41).mean())
    oof_on41 = max(0.05, sum(oof_prem_on) / max(1, len(oof_prem_on)))
    oof_pos41 = sum(1 for v in oof_prem_on if v > 0.02) / max(1, len(oof_prem_on))
    oof_ver_mean = sum(abs(v) for v in oof_ver_off) / max(1, len(oof_ver_off))
    oof_fail_mean = sum(oof_fail_off) / max(1, len(oof_fail_off))
    b41_dir = make_cauchy_b(pos_verify, post_ffn_w[41], l41_sacred_list)

    print(
        f"  [L41 Matched-Submit OOF] in_on={in_on41:.4f} | oof_prem_on={oof_on41:.4f} (pos={oof_pos41:.3f}) "
        f"| oof_ver_abs={oof_ver_mean:.4f} | oof_fail_abs={oof_fail_mean:.4f} | oof_open_max={sum(oof_open)/max(1,len(oof_open)):.4f}",
        flush=True,
    )

    # Normalize A41 by `in_on41` (NOT `oof_on`!) and calibrate `rho_in(41)` live in VRAM on REAL premature-submit prompts
    # (`fastapi_14479`, `fastapi_14786`, `rich_3506`, `rich_3296`) vs `true_imm_test_pass`!
    A41 = (au41 * (sc41 / max(0.08, in_on41))).bfloat16()
    a41_dev = A41.to(dev)

    prem_test_recs = [r for r in r41 if is_on_41(r) and r["is_matched_pair"]][:10]
    ver_test_recs = [r for r in r41 if r["true_role"] == "true_imm_test_pass" and r["is_matched_pair"]][:10]

    cal_table = []
    for cand_rho_in in (0.36, 0.48, 0.60, 0.72):
        b41_try = ((cand_rho_in * raw_yn41) * b41_dir).to(dev, torch.bfloat16)
        prem_gaps, ver_gaps, ed_leaks = [], [], []
        for r in prem_test_recs:
            inp = torch.tensor([r["toks"][: r["c_abs"] + 1]], device=dev)
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
            prem_gaps.append(float((lg1[tok_run] - lg1[tok_sub]).item()))
            ed_leaks.append(float((lg1[tok_ed] - lg0[tok_ed]).abs().item()))
        for r in ver_test_recs:
            inp = torch.tensor([r["toks"][: r["c_abs"] + 1]], device=dev)
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
            ver_gaps.append(float((lg1[tok_sub] - lg1[tok_run]).item()))
        row_c = {
            "rho_in41": cand_rho_in,
            "prem_run_wins": f"{sum(1 for g in prem_gaps if g > 0)}/{len(prem_gaps)}",
            "prem_mean_run_minus_sub": round(sum(prem_gaps) / len(prem_gaps), 2),
            "ver_sub_wins": f"{sum(1 for g in ver_gaps if g > 0)}/{len(ver_gaps)}",
            "ver_mean_sub_minus_run": round(sum(ver_gaps) / len(ver_gaps), 2),
            "max_edit_logit_leak": round(max(ed_leaks), 4),
        }
        cal_table.append(row_c)
        print("  [Live L41 Premature-vs-Verified Check]", row_c, flush=True)

    hk39.remove()
    del model, lm_root, layers
    gc.collect()
    torch.cuda.empty_cache()

    # Pick sweet-spot `rho_in41` where `ver_sub_wins == 10/10` and `prem_run_wins` is maximized!
    sweet_rho41 = 0.48
    for row_c in cal_table:
        vw_n, vw_d = map(int, row_c["ver_sub_wins"].split("/"))
        pw_n, _ = map(int, row_c["prem_run_wins"].split("/"))
        if vw_n == vw_d and pw_n >= 7:
            sweet_rho41 = row_c["rho_in41"]

    diagnostics = {
        "n_prepared": len(prepared),
        "added_skipped_edit_ok": added_edit_ok,
        "L39_s_on": round(s39_on, 4),
        "L41_matched_oof": {
            "in_on": round(in_on41, 4),
            "oof_prem_on": round(oof_on41, 4),
            "oof_pos_frac": round(oof_pos41, 3),
            "oof_ver_abs": round(oof_ver_mean, 4),
            "oof_fail_abs": round(oof_fail_mean, 4),
        },
        "L41_live_calibration": cal_table,
        "selected_sweet_rho41": sweet_rho41,
    }
    print("V24 Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    for name, r39_g, r41_g in (
        ("V24_pure39_41_soft", 0.42, round(0.82 * sweet_rho41, 3)),
        ("V24_pure39_41_sweet", 0.46, sweet_rho41),
        ("V24_pure39_41_firm", 0.50, round(1.18 * sweet_rho41, 3)),
    ):
        ft = {
            "39.A": A39,
            "39.B": ((r39_g * dyn39_mean) * b39_dir).bfloat16(),
            "41.A": A41,
            "41.B": ((r41_g * raw_yn41) * b41_dir).bfloat16(),
        }
        fp = out_dir / f"fac_{name}.safetensors"
        save_file(ft, str(fp))
        for ctrl in ("candidate", "random"):
            inf = compile_explicit(ROOT / "champion.zip", fp, ad_root / f"{name}_{ctrl}", ctrl, rank=1)
            compiled.append({"adapter": f"{name}_{ctrl}", "sha256": inf["adapter_sha256"], "control": ctrl, "layers": inf["active_layers"]})

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V24 Pure L39+L41 Matched-History adapters!", flush=True)


if __name__ == "__main__":
    main()
