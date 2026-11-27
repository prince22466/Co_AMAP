"""Synthesize V15 Full-Space AlphaEdit Null-Space Multi-Circuit Adapters (`o_proj + down_proj` r=1).

Key V15 Upgrades over V14 (100% derived from 19 snapshot-disjoint clean_v14 cases + base weights):
1. Exact `<|channel>thought\\n` boundary alignment across both prefilled-thought and generated-thought turns.
2. Full-Space Kernel Ridge AlphaEdit (`w = X^T (X X^T + lam I)^{-1} y`) in full D_attn (2048/4096) & D_ffn (10240)
   instead of cramped 11-D PCA subspace — putting `imm_test_pass`, `imm_read_full`, `turn1_init`, AND all
   syntax/delimiter/code tokens (`<channel|>`, `<|tool_call>call:`, `<|"|>`, `}<tool_call|>`) strictly in the null space!
3. Direct Unembedding Vocabulary Contrast at Thought-Start (`u_thought_verify` & `u_thought_read`) paired with
   proven `L39` anti-bare-pytest readout (`b1_bare_pytest -> 0`), keeping strict `r=1` (`lora_alpha=1`) everywhere.
"""
import gc
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter, resolve_targeted_check


def find_subseq_end(seq: list[int], sub: list[int]) -> int | None:
    """Return index `i + len(sub)` right after the last occurrence of `sub` in `seq`, or None."""
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return None
    for i in range(n - m, -1, -1):
        if seq[i : i + m] == sub:
            return i + m
    return None


def solve_full_alphaedit(
    X_on: torch.Tensor,                      # [N_on, D]
    off_groups: list[tuple[torch.Tensor, float]],
    lam_rel: float = 0.15,
) -> tuple[torch.Tensor, float]:
    """Full-space minimum-norm Kernel Ridge AlphaEdit: X_on -> +1, every OFF group -> 0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xo = X_on.to(device, torch.float32)
    n_on = max(1, Xo.shape[0])
    rows = [(1.0 / (n_on ** 0.5)) * Xo]
    y_parts = [torch.full((n_on,), 1.0 / (n_on ** 0.5), device=device)]

    for xg, beta in off_groups:
        if xg.shape[0] == 0:
            continue
        xg_d = xg.to(device, torch.float32)
        # Subsample huge token pools evenly to keep Gram matrix <= 600x600 while spanning all turns
        if xg_d.shape[0] > 384:
            stride = max(1, xg_d.shape[0] // 384)
            xg_d = xg_d[::stride][:384]
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


def compile_oproj_downproj_r1(
    champion_zip: Path,
    oproj_specs: dict[int, tuple[torch.Tensor, torch.Tensor]],
    down_specs: dict[int, tuple[torch.Tensor, torch.Tensor]],
    oproj_in_dims: dict[int, int],
    out_dir: Path,
    control: str = "candidate",
    seed: int = 43,
) -> dict:
    assert not (set(down_specs) & CHAMP_LAYERS), "down_proj edits must be disjoint from CHAMP_LAYERS"
    with zipfile.ZipFile(champion_zip) as z:
        champ = load(z.read("adapters/wsp_r1/adapter_model.safetensors"))
        cfg = json.loads(z.read("adapters/wsp_r1/adapter_config.json"))

    all_layers = sorted(CHAMP_LAYERS | set(oproj_specs) | set(down_specs))
    gen = torch.Generator().manual_seed(seed)
    tensors = {}

    for li in all_layers:
        d_attn = oproj_in_dims[li]
        op = f"base_model.model.model.language_model.layers.{li}.self_attn.o_proj"
        dp = f"base_model.model.model.language_model.layers.{li}.mlp.down_proj"

        oa = torch.zeros(1, d_attn, dtype=torch.bfloat16)
        ob = torch.zeros(2560, 1, dtype=torch.bfloat16)
        da = torch.zeros(1, 10240, dtype=torch.bfloat16)
        db = torch.zeros(2560, 1, dtype=torch.bfloat16)

        if li in CHAMP_LAYERS:
            da[:] = champ[f"{dp}.lora_A.weight"]
            db[:] = champ[f"{dp}.lora_B.weight"]

        if li in oproj_specs and control != "zero":
            ra, rb = oproj_specs[li]
            ra, rb = ra.to(torch.bfloat16), rb.to(torch.bfloat16)
            if control == "random":
                ra = ra[torch.randperm(d_attn, generator=gen)]
                rb = rb[torch.randperm(2560, generator=gen)] * (2 * torch.randint(2, (2560,), generator=gen) - 1)
            oa[0] = ra
            ob[:, 0] = rb

        if li in down_specs and control != "zero":
            ra, rb = down_specs[li]
            ra, rb = ra.to(torch.bfloat16), rb.to(torch.bfloat16)
            if control == "random":
                ra = ra[torch.randperm(10240, generator=gen)]
                rb = rb[torch.randperm(2560, generator=gen)] * (2 * torch.randint(2, (2560,), generator=gen) - 1)
            da[0] = ra
            db[:, 0] = rb

        tensors[f"{op}.lora_A.weight"] = oa
        tensors[f"{op}.lora_B.weight"] = ob
        tensors[f"{dp}.lora_A.weight"] = da
        tensors[f"{dp}.lora_B.weight"] = db

    out_dir.mkdir(parents=True, exist_ok=True)
    w_path = out_dir / "adapter_model.safetensors"
    save_file(tensors, str(w_path))
    cfg.update(
        r=1,
        lora_alpha=1,
        layers_to_transform=all_layers,
        target_modules=["o_proj", "down_proj"],
    )
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return {
        "adapter": out_dir.name,
        "layers": all_layers,
        "control": control,
        "sha256": hashlib.sha256(w_path.read_bytes()).hexdigest(),
    }


def main():
    cdir = ROOT / "harvest32/clean_v14"
    out_dir = ROOT / "harvest32/v15_alphaedit"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads((cdir / "clean_cases.json").read_text())
    nat = {r["prompt_sha"]: r for r in json.loads((cdir / "champion_clean.json").read_text())}

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
    oproj_in_dims = {}
    for idx_l in range(len(layers)):
        op_mod = layers[idx_l].self_attn.o_proj
        in_f = getattr(op_mod, "in_features", None)
        if in_f is None:
            w_sh = getattr(op_mod, "weight_shape", None)
            in_f = int(w_sh[1]) if w_sh is not None else int(op_mod.weight.shape[1])
        oproj_in_dims[idx_l] = int(in_f)

    # Candidate mid-network o_proj layers [15..31] (avoiding late L32+ o_proj to prevent special-token leaks)
    o_layers = list(range(15, 32))
    d_layers = [39, 40, 41]
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    thought_hdr = enc("<|channel>thought\n")
    chan_end_seq = enc("<channel|>")
    call_colon_seq = enc("<|tool_call>call:")
    quote_seq = enc('<|"|>')
    prefix_pytest = enc('<|tool_call>call:run_command{command:<|"|>pytest')

    def harvest_record(prefix: list[int], suffix: list[int]):
        toks = prefix + suffix
        assert len(toks) <= 4096
        p_len = len(prefix)

        # Find EXACT `<|channel>thought\n` end across boundary (handles both prefilled & generated thought!)
        win_lo = max(0, p_len - 8)
        win_hi = min(len(toks), p_len + 10)
        th_rel = find_subseq_end(toks[win_lo:win_hi], thought_hdr)
        ts0 = (win_lo + th_rel - 1) if th_rel is not None else max(0, p_len - 1)

        # Find `<channel|>` end in suffix so thought-opening window NEVER crosses into `<|tool_call>`!
        ch_end_rel = find_subseq_end(suffix, chan_end_seq)
        ch_end_abs = (p_len + ch_end_rel - len(chan_end_seq)) if ch_end_rel is not None else len(toks)
        open_hi = min(ch_end_abs, ts0 + 7, len(toks))
        open_idx = list(range(ts0, max(ts0 + 1, open_hi)))
        open_t = torch.tensor(open_idx, device=dev, dtype=torch.long)

        colon_end = find_subseq_end(suffix, call_colon_seq)
        pytest_end = find_subseq_end(suffix, prefix_pytest)
        c_abs = (p_len + colon_end - 1) if colon_end is not None else None
        pt_abs = (p_len + pytest_end - 1) if pytest_end is not None else None

        # Collect explicit syntax/delimiter/code-argument positions in suffix + prefill as hard OFF targets!
        delim_set = set()
        if ch_end_rel is not None:
            for k in range(max(0, p_len + ch_end_rel - 2), min(len(toks), p_len + ch_end_rel + 4)):
                delim_set.add(k)
        if c_abs is not None:
            for k in range(max(p_len, c_abs - 2), min(len(toks), len(toks))):
                if (k - p_len) % max(1, max(1, len(suffix) - (colon_end or 0)) // 10) == 0:
                    delim_set.add(k)
            delim_set.add(c_abs)
        # Add 12 evenly spaced context tokens before ts0
        bg_stride = max(1, max(1, ts0 - 8) // 12)
        for k in range(8, max(9, ts0 - 1), bg_stride):
            delim_set.add(k)
        delim_idx = sorted(k for k in delim_set if k not in set(open_idx) and k != pt_abs)[:24]
        if not delim_idx:
            delim_idx = [max(0, ts0 - 2)]
        delim_t = torch.tensor(delim_idx, device=dev, dtype=torch.long)

        cap = {"has_colon": c_abs is not None, "has_pytest": pt_abs is not None, "has_thought": th_rel is not None}
        hooks = []
        for li in o_layers:
            def mk_o(idx=li):
                def _h(mod, inp, out):
                    z = inp[0][0].detach()
                    cap[(idx, "oz_ts0")] = z[ts0].to(torch.bfloat16).cpu()
                    cap[(idx, "oz_open")] = z.index_select(0, open_t).float().mean(0).to(torch.bfloat16).cpu()
                    cap[(idx, "oz_delim")] = z.index_select(0, delim_t).to(torch.bfloat16).cpu()
                return _h
            hooks.append(layers[li].self_attn.o_proj.register_forward_hook(mk_o()))

        for li in d_layers:
            def mk_d(idx=li):
                def _h(mod, inp, out):
                    h = inp[0][0].detach()
                    y = out[0].detach()
                    cap[(idx, "dy_norm")] = float(y[p_len:].float().norm(dim=-1).mean().item())
                    cap[(idx, "dh_delim")] = h.index_select(0, delim_t).to(torch.bfloat16).cpu()
                    if c_abs is not None:
                        cap[(idx, "dh_col")] = h[c_abs].to(torch.bfloat16).cpu()
                    if pt_abs is not None:
                        cap[(idx, "dh_pt")] = h[pt_abs].to(torch.bfloat16).cpu()
                return _h
            hooks.append(layers[li].mlp.down_proj.register_forward_hook(mk_d()))

        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([toks], device=dev), use_cache=False, logits_to_keep=1)
        finally:
            for hk in hooks:
                hk.remove()
        return cap

    records = []
    for c in cases:
        act = nat.get(c["prompt_sha"], {})
        txt = act.get("text") or ""
        if not txt and c["clean_role"] == "imm_test_pass":
            txt = "<|channel>thought\nTests passed cleanly. Submitting patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>"
        if not txt:
            continue
        full_suf = enc(txt)
        suf = (full_suf[:140] + full_suf[-628:]) if len(full_suf) > 768 else full_suf
        if len(suf) < 4:
            continue
        pref = c["tokens"][-(4096 - len(suf)) :]
        cap = harvest_record(pref, suf)
        records.append({
            "issue": c["issue"],
            "clean_role": c["clean_role"],
            "imm_state": c.get("imm_state", ""),
            "nat_action": act.get("action"),
            "filepath": c.get("filepath"),
            "cap": cap,
        })

    # Harvest targeted pytest suffixes for Circuit 3 (L39 anti-bare-pytest)
    bare_caps = []
    seen_iss = set()
    for c in cases:
        if c["issue"] in seen_iss or c["clean_role"] not in ("imm_edit_ok", "post_edit_prem_submit"):
            continue
        cmd = resolve_targeted_check(c)
        if not cmd:
            continue
        seen_iss.add(c["issue"])
        cmd_py = cmd if cmd.startswith("pytest") else f"pytest -q tests/test_{Path(c.get('filepath') or 'x.py').stem.lstrip('_')}.py"
        suf = enc(f"<|channel>thought\nLet's run targeted pytest to verify the fix.\n<channel|><|tool_call>call:run_command{{command:<|\"|>{cmd_py}<|\"|>}}<tool_call|>")
        pref = c["tokens"][-(4096 - len(suf)) :]
        cap = harvest_record(pref, suf)
        if cap["has_pytest"]:
            bare_caps.append(cap)

    # Pure weight-space unembedding vocabulary directions
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

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

    # Thought-Start vocabulary steering vectors (push first thought sentence directly!)
    u_thought_verify = tok_vec(["pytest", "test", "tests", "verify", "check", "run"]) - tok_vec(
        ["submit", "done", "complete", "finished", "ready", "fixed"]
    )
    u_thought_verify = u_thought_verify / u_thought_verify.norm().clamp_min(1e-8)

    u_thought_read = tok_vec(["read", "inspect", "lines", "view", "exact", "content"]) - tok_vec(
        ["edit", "replace", "submit", "again", "retry"]
    )
    u_thought_read = u_thought_read / u_thought_read.norm().clamp_min(1e-8)

    # Readout tool-call directions
    tok_run = enc("<|tool_call>call:run_command{")[len(call_colon_seq)]
    tok_sub = enc("<|tool_call>call:submit_patch{")[len(call_colon_seq)]
    tok_ed = enc("<|tool_call>call:edit_file{")[len(call_colon_seq)]
    tok_rd = enc("<|tool_call>call:read_file{")[len(call_colon_seq)]

    u_c1_disp = ((0.80 * lm_w[tok_sub] + 0.20 * lm_w[tok_ed]) - lm_w[tok_run]) * scale_w
    u_c1_disp = u_c1_disp / u_c1_disp.norm().clamp_min(1e-8)

    u_c2_err = ((0.85 * lm_w[tok_ed] + 0.15 * lm_w[tok_sub]) - lm_w[tok_rd]) * scale_w
    u_c2_err = u_c2_err / u_c2_err.norm().clamp_min(1e-8)

    sb = enc('<|tool_call>call:run_command{command:<|"|>pytest<|"|>}<tool_call|>')
    st1 = enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests/test_panel.py<|"|>}<tool_call|>')
    st2 = enc('<|tool_call>call:run_command{command:<|"|>pytest tests/test_panel.py<|"|>}<tool_call|>')
    u_c3_ab = (
        lm_w[sb[find_subseq_end(sb, prefix_pytest)]]
        - (0.60 * lm_w[st1[find_subseq_end(st1, prefix_pytest)]] + 0.40 * lm_w[st2[find_subseq_end(st2, prefix_pytest)]])
    ) * scale_w
    u_c3_ab = u_c3_ab / u_c3_ab.norm().clamp_min(1e-8)

    del lm_w, model, lm_root, layers
    gc.collect()
    torch.cuda.empty_cache()

    # Pure role predicates (never mix `old_string not found` into post-edit verification!)
    def is_c1_on(r: dict) -> bool:
        return r["clean_role"] == "imm_edit_ok" or (
            r["clean_role"] == "post_edit_prem_submit" and r["imm_state"] != "IMMEDIATE_EDIT_ERR"
        )

    def is_c2_on(r: dict) -> bool:
        return r["clean_role"] == "imm_edit_err"

    # =========================================================================
    # 1. Rank Mid-Network `o_proj` Layers [15..31] for Circuit 1 & Circuit 2 via LOO-CV!
    # =========================================================================
    c1_issues = sorted({r["issue"] for r in records if is_c1_on(r)})
    c2_issues = sorted({r["issue"] for r in records if is_c2_on(r)})
    c1_cv, c2_cv = [], []
    o_c1_fits, o_c2_fits = {}, {}

    for li in o_layers:
        def pack_c1(recs):
            on_l, ver_l, rd_l, err_l, init_l, del_l = [], [], [], [], [], []
            for r in recs:
                z0 = r["cap"][(li, "oz_ts0")].float()
                zo = r["cap"][(li, "oz_open")].float()
                del_l.append(r["cap"][(li, "oz_delim")].float())
                if is_c1_on(r):
                    on_l.append(z0)
                    on_l.append(zo)
                elif r["clean_role"] == "imm_test_pass":
                    ver_l.extend([z0, zo])
                elif r["clean_role"] in ("imm_read_full", "imm_read_trunc"):
                    rd_l.extend([z0, zo])
                elif is_c2_on(r):
                    err_l.extend([z0, zo])
                else:
                    init_l.extend([z0, zo])
            Xo = torch.stack(on_l)
            Xv = torch.stack(ver_l) if ver_l else torch.zeros(0, Xo.shape[1])
            Xr = torch.stack(rd_l) if rd_l else torch.zeros(0, Xo.shape[1])
            Xe = torch.stack(err_l) if err_l else torch.zeros(0, Xo.shape[1])
            Xi = torch.stack(init_l) if init_l else torch.zeros(0, Xo.shape[1])
            Xd = torch.cat(del_l, dim=0)
            ogrps = [(Xv, 8.0), (Xr, 8.0), (Xe, 4.0), (Xi, 5.0), (Xd, 12.0)]
            return Xo, ogrps, Xv, Xr, Xd

        oof_on, oof_ver, oof_rd, oof_del = [], [], [], []
        for iss in c1_issues:
            tr = [r for r in records if r["issue"] != iss]
            Xo_tr, og_tr, _, _, _ = pack_c1(tr)
            wu, sc = solve_full_alphaedit(Xo_tr, og_tr, lam_rel=0.15)
            w = wu * sc
            for r in records:
                if r["issue"] != iss:
                    continue
                v0 = float(torch.dot(w, r["cap"][(li, "oz_ts0")].float()))
                vd = float((r["cap"][(li, "oz_delim")].float() @ w).abs().mean())
                oof_del.append(vd)
                if is_c1_on(r):
                    oof_on.append(v0)
                elif r["clean_role"] == "imm_test_pass":
                    oof_ver.append(abs(v0))
                elif r["clean_role"] in ("imm_read_full", "imm_read_trunc"):
                    oof_rd.append(abs(v0))

        m_on = sum(oof_on) / max(1, len(oof_on))
        m_ver = sum(oof_ver) / max(1, len(oof_ver))
        m_rd = sum(oof_rd) / max(1, len(oof_rd))
        m_del = sum(oof_del) / max(1, len(oof_del))
        gap = m_on - max(m_ver, m_rd, 2.0 * m_del)
        Xo, og, Xv, Xr, Xd = pack_c1(records)
        wu, sc = solve_full_alphaedit(Xo, og, lam_rel=0.15)
        o_c1_fits[li] = {"u_a": wu, "scale": sc, "oof_on": m_on, "oof_gap": gap}
        c1_cv.append({
            "layer": li, "oof_on": round(m_on, 4), "oof_ver_abs": round(m_ver, 4),
            "oof_rd_abs": round(m_rd, 4), "oof_delim_abs": round(m_del, 4),
            "oof_gap": round(gap, 4), "w_norm": round(sc, 5),
        })

        # Circuit 2 LOO-CV (`imm_edit_err` -> `u_thought_read`)
        def pack_c2(recs):
            on_l, rd_l, ok_l, ver_l, init_l, del_l = [], [], [], [], [], []
            for r in recs:
                z0 = r["cap"][(li, "oz_ts0")].float()
                zo = r["cap"][(li, "oz_open")].float()
                del_l.append(r["cap"][(li, "oz_delim")].float())
                if is_c2_on(r):
                    on_l.append(z0)
                    on_l.append(zo)
                elif r["clean_role"] in ("imm_read_full", "imm_read_trunc"):
                    rd_l.extend([z0, zo])
                elif is_c1_on(r):
                    ok_l.extend([z0, zo])
                elif r["clean_role"] == "imm_test_pass":
                    ver_l.extend([z0, zo])
                else:
                    init_l.extend([z0, zo])
            Xo2 = torch.stack(on_l)
            Xr2 = torch.stack(rd_l) if rd_l else torch.zeros(0, Xo2.shape[1])
            Xok2 = torch.stack(ok_l) if ok_l else torch.zeros(0, Xo2.shape[1])
            Xv2 = torch.stack(ver_l) if ver_l else torch.zeros(0, Xo2.shape[1])
            Xi2 = torch.stack(init_l) if init_l else torch.zeros(0, Xo2.shape[1])
            Xd2 = torch.cat(del_l, dim=0)
            ogrps2 = [(Xr2, 10.0), (Xok2, 6.0), (Xv2, 6.0), (Xi2, 5.0), (Xd2, 12.0)]
            return Xo2, ogrps2

        oof2_on, oof2_rd, oof2_del = [], [], []
        for iss in c2_issues:
            tr = [r for r in records if r["issue"] != iss]
            Xo2_tr, og2_tr = pack_c2(tr)
            wu2, sc2 = solve_full_alphaedit(Xo2_tr, og2_tr, lam_rel=0.15)
            w2 = wu2 * sc2
            for r in records:
                if r["issue"] != iss:
                    continue
                v0 = float(torch.dot(w2, r["cap"][(li, "oz_ts0")].float()))
                oof2_del.append(float((r["cap"][(li, "oz_delim")].float() @ w2).abs().mean()))
                if is_c2_on(r):
                    oof2_on.append(v0)
                elif r["clean_role"] in ("imm_read_full", "imm_read_trunc"):
                    oof2_rd.append(abs(v0))
        m2_on = sum(oof2_on) / max(1, len(oof2_on))
        m2_rd = sum(oof2_rd) / max(1, len(oof2_rd))
        m2_del = sum(oof2_del) / max(1, len(oof2_del))
        gap2 = m2_on - max(m2_rd, 2.0 * m2_del)
        Xo2, og2 = pack_c2(records)
        wu2, sc2 = solve_full_alphaedit(Xo2, og2, lam_rel=0.15)
        o_c2_fits[li] = {"u_a": wu2, "scale": sc2, "oof_on": m2_on, "oof_gap": gap2}
        c2_cv.append({
            "layer": li, "oof_on": round(m2_on, 4), "oof_rd_abs": round(m2_rd, 4),
            "oof_delim_abs": round(m2_del, 4), "oof_gap": round(gap2, 4), "w_norm": round(sc2, 5),
        })

    c1_cv.sort(key=lambda x: x["oof_gap"], reverse=True)
    c2_cv.sort(key=lambda x: x["oof_gap"], reverse=True)
    top2_c1 = [r["layer"] for r in c1_cv[:2]]
    top3_c1 = [r["layer"] for r in c1_cv[:3]]
    top2_c2 = [r["layer"] for r in c2_cv if r["layer"] not in set(top3_c1)][:2]

    # =========================================================================
    # 2. Full-Space AlphaEdit Readout `down_proj` Detectors:
    #    - L39: Anti-Bare-Pytest (`command:<|"|>pytest` -> `pytest -q tests/...`)
    #    - L41: Optional gentle C1 readout at `<|tool_call>call:` (with Hr null-spaced!)
    #    - L40: Optional gentle C2 readout at `<|tool_call>call:` (with Hr null-spaced!)
    # =========================================================================
    h39_on, h39_off, yn39 = [], [], []
    for r in records:
        yn39.append(r["cap"][(39, "dy_norm")])
        h39_off.append(r["cap"][(39, "dh_delim")].float())
        if r["cap"]["has_colon"] and not r["cap"]["has_pytest"]:
            h39_off.append(r["cap"][(39, "dh_col")].float().unsqueeze(0))
        if r["cap"]["has_pytest"]:
            h39_on.append(r["cap"][(39, "dh_pt")].float())
    for bc in bare_caps:
        h39_on.append(bc[(39, "dh_pt")].float())
        h39_off.append(bc[(39, "dh_delim")].float())
    H39_on = torch.stack(h39_on)
    H39_off = torch.cat(h39_off, dim=0)
    a39_u, sc39 = solve_full_alphaedit(H39_on, [(H39_off, 10.0)], lam_rel=0.12)
    s39_on = float((H39_on @ (a39_u * sc39)).mean())
    s39_off_rms = float((H39_off @ (a39_u * sc39)).pow(2).mean().sqrt().clamp_min(1e-6))

    # Full-space AlphaEdit on L41 (C1 colon) & L40 (C2 colon)
    def fit_colon_alphaedit(li: int, on_pred):
        on_h, rd_h, ver_h, oth_h, del_h, yn = [], [], [], [], [], []
        for r in records:
            yn.append(r["cap"][(li, "dy_norm")])
            del_h.append(r["cap"][(li, "dh_delim")].float())
            if not r["cap"]["has_colon"]:
                continue
            hc = r["cap"][(li, "dh_col")].float()
            if on_pred(r):
                on_h.append(hc)
            elif r["clean_role"] in ("imm_read_full", "imm_read_trunc"):
                rd_h.append(hc)
            elif r["clean_role"] == "imm_test_pass":
                ver_h.append(hc)
            else:
                oth_h.append(hc)
        Ho = torch.stack(on_h)
        Hr = torch.stack(rd_h) if rd_h else torch.zeros(0, 10240)
        Hv = torch.stack(ver_h) if ver_h else torch.zeros(0, 10240)
        Hot = torch.stack(oth_h) if oth_h else torch.zeros(0, 10240)
        Hd = torch.cat(del_h, dim=0)
        au, sc = solve_full_alphaedit(Ho, [(Hr, 14.0), (Hv, 12.0), (Hot, 8.0), (Hd, 12.0)], lam_rel=0.12)
        w = au * sc
        son = float((Ho @ w).mean())
        rd_max = float((Hr @ w).abs().max()) if Hr.shape[0] else 0.0
        ver_max = float((Hv @ w).abs().max()) if Hv.shape[0] else 0.0
        return {
            "u_a": au, "scale": sc, "s_on_mean": son,
            "rd_ratio": round(rd_max / max(1e-4, son), 4),
            "ver_ratio": round(ver_max / max(1e-4, son), 4),
            "y_norm": sum(yn) / len(yn),
        }

    d41_c1 = fit_colon_alphaedit(41, is_c1_on)
    d40_c2 = fit_colon_alphaedit(40, is_c2_on)

    diagnostics = {
        "n_clean_records": len(records),
        "top_c1_oproj_cv": c1_cv[:5],
        "top_c2_oproj_cv": [r for r in c2_cv if r["layer"] not in set(top3_c1)][:4],
        "selected_c1_layers": top3_c1,
        "selected_c2_layers": top2_c2,
        "C3_L39_full_alphaedit": {
            "n_on": int(H39_on.shape[0]),
            "s_on_mean": round(s39_on, 4),
            "s_off_rms": round(s39_off_rms, 4),
            "snr": round(s39_on / s39_off_rms, 2),
        },
        "C1_L41_colon_rd_ratio": d41_c1["rd_ratio"],
        "C1_L41_colon_ver_ratio": d41_c1["ver_ratio"],
        "C2_L40_colon_rd_ratio": d40_c2["rd_ratio"],
    }
    print("V15 Full-Space AlphaEdit Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []
    yn39_mean = sum(yn39) / len(yn39)

    def build_variant(
        name: str,
        c1_layers: list[int],
        gain_c1: float,
        c2_layers: list[int],
        gain_c2: float,
        rho_c3: float = 0.28,
        rho_d41: float = 0.0,
        rho_d40: float = 0.0,
    ):
        ospecs = {}
        for li in c1_layers:
            f = o_c1_fits[li]
            # Exact AlphaEdit scaling: (u_a * scale) @ z_on ≈ +1.0, so B = gain_c1 * u_thought_verify!
            ospecs[li] = (f["u_a"] * f["scale"], gain_c1 * u_thought_verify)
        for li in c2_layers:
            f = o_c2_fits[li]
            ospecs[li] = (f["u_a"] * f["scale"], gain_c2 * u_thought_read)

        dspecs = {}
        if rho_c3 > 0:
            g39 = rho_c3 * yn39_mean / max(0.1, s39_on)
            dspecs[39] = (a39_u * sc39, -g39 * u_c3_ab)
        if rho_d41 > 0:
            g41 = rho_d41 * d41_c1["y_norm"] / max(0.1, d41_c1["s_on_mean"])
            dspecs[41] = (d41_c1["u_a"] * d41_c1["scale"], -g41 * u_c1_disp)
        if rho_d40 > 0:
            g40 = rho_d40 * d40_c2["y_norm"] / max(0.1, d40_c2["s_on_mean"])
            dspecs[40] = (d40_c2["u_a"] * d40_c2["scale"], -g40 * u_c2_err)

        if not (ad_root / "V15_hybrid_zero").exists():
            compiled.append(compile_oproj_downproj_r1(ROOT / "champion.zip", ospecs, dspecs, oproj_in_dims, ad_root / "V15_hybrid_zero", "zero"))
        for ctrl in ("candidate", "random"):
            compiled.append(compile_oproj_downproj_r1(ROOT / "champion.zip", ospecs, dspecs, oproj_in_dims, ad_root / f"{name}_{ctrl}", ctrl))

    # Bracketed gain sweep matching iter_25/26 Thought-Start sweet spot (0.8 .. 2.2) + L39 anti-bare-pytest!
    build_variant("V15_ae_c1c3_gentle", top2_c1, gain_c1=0.95, c2_layers=[], gain_c2=0.0, rho_c3=0.28)
    build_variant("V15_ae_c1c3_mid", top3_c1, gain_c1=1.35, c2_layers=[], gain_c2=0.0, rho_c3=0.30)
    build_variant("V15_ae_c1c2c3_mid", top2_c1, gain_c1=1.35, c2_layers=top2_c2, gain_c2=1.20, rho_c3=0.30)
    build_variant("V15_ae_c1c2c3_hi", top3_c1, gain_c1=1.85, c2_layers=top2_c2, gain_c2=1.55, rho_c3=0.32)
    build_variant("V15_ae_full_readout", top2_c1, gain_c1=1.35, c2_layers=top2_c2, gain_c2=1.25, rho_c3=0.30, rho_d41=0.20, rho_d40=0.20)

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V15 Full-Space AlphaEdit adapters!", flush=True)


if __name__ == "__main__":
    main()
