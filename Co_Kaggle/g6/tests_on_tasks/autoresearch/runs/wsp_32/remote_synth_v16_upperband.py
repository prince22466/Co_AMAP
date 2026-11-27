"""Synthesize V16 Upper-Band (`L30..41`) Post-RMSNorm Calibrated AlphaEdit Multi-Circuit (`o_proj + down_proj` r=1).

Breakthrough Architectural & Data Fixes (from `Gemma4TextDecoderLayer` & literal token audit):
1. Gemma 4 `layer_scalar` bottleneck discovery: `layer_scalar[23] = 0.0659` attenuates L15..L23 perturbations by ~5,000x
   before final norm, whereas upper band `L30..L41` (`layer_scalar = 0.71..0.89`, including Global Full-Attention `L35` & `L41`)
   has 1,000x-2,500x higher residual survival!
2. Exact `post_attention_layernorm` & `post_feedforward_layernorm` RMSNorm calibration: scales `lora_B` by raw pre-norm
   `||o_proj(z)||_2` (and `||down_proj(h)||_2`) pre-conditioned by `post_*_layernorm.weight` so perturbations produce exact,
   calibrated logit shifts without saturating RMSNorm.
3. 100% Literal-Token Prompt Classification directly from `tok.decode(tokens)` across all 276 prompts in our 19 disjoint
   public snapshots (`discovery02+03+04` + `preservation/exact`), eliminating all `events.jsonl` index-shift contamination
   AND explicitly targeting:
   - C1A: `true_imm_edit_ok` -> `u_thought_verify` (`pytest` verification before `submit_patch`)
   - C1B: `true_imm_test_fail` -> `u_thought_fix` (never rationalize `exit_code != 0` into `submit_patch`!)
   - C2 : `true_imm_edit_err` -> `u_thought_read` (`FileEditError` -> `read_file` recovery)
   - C3 : `L39 down_proj` -> anti-bare-pytest (`pytest` -> `pytest -q tests/...`)
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


def find_subseq_end(seq: list[int], sub: list[int]) -> int | None:
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return None
    for i in range(n - m, -1, -1):
        if seq[i : i + m] == sub:
            return i + m
    return None


TEST_CMD = re.compile(r"(?:^|\s|&&|;)(?:pytest|python(?:3)?\s+-m\s+pytest|python(?:3)?\s+-c\s+.*?\bassert\b)")
TR_RE = re.compile(r"<\|tool_response>response:(\w+)\{(.*?)\}<tool_response\|>", re.S)
TC_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.S)


def literal_role(tok, tks: list[int], source_run: str = "") -> str:
    if source_run == "exact_verified":
        return "true_imm_test_pass"
    txt = tok.decode(tks, skip_special_tokens=False)
    trs = list(TR_RE.finditer(txt))
    tcs = list(TC_RE.finditer(txt))
    if not trs:
        return "turn1_init" if "<|turn>user" in txt and txt.count("<|turn>model") <= 1 else "non_imm_prose"
    if any(m.group(1) == "submit_patch" for m in trs):
        return "after_submit_ignore"
    after = txt[trs[-1].end() :]
    if "<|turn>user" in after or "<|tool_call>" in after or len(after) > 80:
        return "non_imm_followup"
    has_ok_edit = any(m.group(1) in ("edit_file", "write_file") and '"status": "ok"' in m.group(2) for m in trs)
    fn, body = trs[-1].group(1), trs[-1].group(2)
    if fn in ("edit_file", "write_file"):
        return "true_imm_edit_ok" if '"status": "ok"' in body else "true_imm_edit_err"
    if fn == "read_file":
        return "true_imm_read_trunc" if '"is_truncated": true' in body else "true_imm_read_full"
    if fn == "run_command":
        last_cmd = tcs[-1].group(2) if tcs and tcs[-1].group(1) == "run_command" else ""
        is_test = bool(TEST_CMD.search(last_cmd)) or "pytest" in body or "passed" in body or "FAILED" in body
        if has_ok_edit and is_test:
            if ('"exit_code": 0' in body or '"returncode": 0' in body) and ("passed" in body or "python" in last_cmd):
                return "true_imm_test_pass"
            if '"exit_code": 0' not in body and '"returncode": 0' not in body:
                return "true_imm_test_fail"
        return "true_imm_cmd_explore"
    return f"other_{fn}"


def solve_full_alphaedit(
    X_on: torch.Tensor,
    off_groups: list[tuple[torch.Tensor, float]],
    lam_rel: float = 0.12,
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
        if xg_d.shape[0] > 420:
            stride = max(1, xg_d.shape[0] // 420)
            xg_d = xg_d[::stride][:420]
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
    seed: int = 47,
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
    out_dir = ROOT / "harvest32/v16_upperband"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = json.loads((ROOT / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tokenizer.truncation_side = "left"

    # 1. Build 100% literal-token-verified corpus across clean_v14 + all 3 disjoint discovery replay_tokens!
    cases_v14 = json.loads((cdir / "clean_cases.json").read_text())
    nat = {r["prompt_sha"]: r for r in json.loads((cdir / "champion_clean.json").read_text())}

    # Map prompt_sha -> issue from clean_v14 and discovery tasks
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

    # Pull any additional `true_imm_edit_ok` / `true_imm_test_fail` / `true_imm_test_pass` prompts from discovery02/03/04!
    tasks_all = [json.loads(s) for s in (ROOT / "input/competitions/gemma-4-developer-agent/tasks.jsonl").read_text().splitlines() if s.strip()]
    disc_ids = {
        "rich_3506", "rich_3296", "rich_3067", "rich_3469", "rich_3130",
        "rich_4079", "rich_3480", "rich_3472", "rich_3105", "rich_3006",
        "fastapi_14786", "fastapi_14479", "requests_7328",
        "fastapi_15661", "fastapi_15588", "requests_7505", "requests_7502",
        "rich_4070", "rich_4006",
    }
    disc_needles = {}
    disc_ps = [t["problem_statement"] for t in tasks_all if t["instance_id"] in disc_ids]
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

    print("V16 Literal-Verified Corpus Counts:", Counter(x["true_role"] for x in all_items.values()), flush=True)

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
    post_attn_w = {}
    post_ffn_w = {}
    cum_scalars = {}
    raw_scalars = [float(layers[i].layer_scalar.detach().float().cpu().item()) for i in range(len(layers))]
    for idx_l in range(len(layers)):
        op_mod = layers[idx_l].self_attn.o_proj
        in_f = getattr(op_mod, "in_features", None)
        if in_f is None:
            w_sh = getattr(op_mod, "weight_shape", None)
            in_f = int(w_sh[1]) if w_sh is not None else int(op_mod.weight.shape[1])
        oproj_in_dims[idx_l] = int(in_f)
        post_attn_w[idx_l] = layers[idx_l].post_attention_layernorm.weight.detach().float().cpu()
        post_ffn_w[idx_l] = layers[idx_l].post_feedforward_layernorm.weight.detach().float().cpu()
        prod = 1.0
        for k in range(idx_l, len(layers)):
            prod *= raw_scalars[k]
        cum_scalars[idx_l] = prod

    # Upper high-survival band [30..41] (cumulative layer_scalar = 0.088 .. 0.4805, 500x-2500x above L23!)
    o_layers = list(range(30, 42))
    d_layers = [39, 40, 41]
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    thought_hdr = enc("<|channel>thought\n")
    chan_end_seq = enc("<channel|>")
    call_colon_seq = enc("<|tool_call>call:")
    prefix_pytest = enc('<|tool_call>call:run_command{command:<|"|>pytest')

    def harvest_record(prefix: list[int], suffix: list[int]):
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
        pytest_end = find_subseq_end(suffix, prefix_pytest)
        c_abs = (p_len + colon_end - 1) if colon_end is not None else None
        pt_abs = (p_len + pytest_end - 1) if pytest_end is not None else None

        # Hard null-space positions: <channel|>, <|tool_call>call:, tool arguments, and prefill context!
        delim_set = set()
        if ch_end_rel is not None:
            for k in range(max(0, p_len + ch_end_rel - 3), min(len(toks), p_len + ch_end_rel + 4)):
                delim_set.add(k)
        if c_abs is not None:
            for k in range(c_abs, len(toks), max(1, max(1, len(toks) - c_abs) // 10)):
                delim_set.add(k)
            delim_set.add(c_abs)
        bg_stride = max(1, max(1, ts0 - 8) // 12)
        for k in range(8, max(9, ts0 - 1), bg_stride):
            delim_set.add(k)
        delim_idx = sorted(k for k in delim_set if k not in set(open_idx) and k != pt_abs)[:26]
        if not delim_idx:
            delim_idx = [max(0, ts0 - 2)]
        delim_t = torch.tensor(delim_idx, device=dev, dtype=torch.long)

        cap = {"has_colon": c_abs is not None, "has_pytest": pt_abs is not None}
        hooks = []
        for li in o_layers:
            def mk_o(idx=li):
                def _h(mod, inp, out):
                    z = inp[0][0].detach()
                    y = out[0].detach()
                    cap[(idx, "oz_ts0")] = z[ts0].to(torch.bfloat16).cpu()
                    cap[(idx, "oz_open")] = z.index_select(0, open_t).float().mean(0).to(torch.bfloat16).cpu()
                    cap[(idx, "oz_delim")] = z.index_select(0, delim_t).to(torch.bfloat16).cpu()
                    cap[(idx, "oy_raw_norm")] = float(y[ts0].float().norm().item())
                    if c_abs is not None:
                        cap[(idx, "oz_col")] = z[c_abs].to(torch.bfloat16).cpu()
                        cap[(idx, "oy_col_norm")] = float(y[c_abs].float().norm().item())
                return _h
            hooks.append(layers[li].self_attn.o_proj.register_forward_hook(mk_o()))

        for li in d_layers:
            def mk_d(idx=li):
                def _h(mod, inp, out):
                    h = inp[0][0].detach()
                    y = out[0].detach()
                    cap[(idx, "dy_raw_norm")] = float(y[max(0, c_abs or ts0)].float().norm().item())
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
    for sha, item in all_items.items():
        rl = item["true_role"]
        if rl in ("after_submit_ignore", "non_imm_prose"):
            continue
        txt = item["nat_text"]
        if not txt:
            if rl == "true_imm_test_pass":
                txt = "<|channel>thought\nTests passed cleanly. Submitting patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>"
            else:
                # Prefill-only Thought-Start suffix (if prompt doesn't already end with <|channel>thought\n)
                tail_dec = tokenizer.decode(item["tokens"][-10:], skip_special_tokens=False)
                txt = "" if tail_dec.endswith("<|channel>thought\n") else "<|channel>thought\n"
        full_suf = enc(txt) if txt else []
        suf = (full_suf[:140] + full_suf[-628:]) if len(full_suf) > 768 else full_suf
        pref = item["tokens"][-(4096 - len(suf)) :]
        cap = harvest_record(pref, suf)
        records.append({
            "issue": item["issue"],
            "true_role": rl,
            "filepath": item["filepath"],
            "has_full_suf": len(suf) > 6,
            "tokens": item["tokens"],
            "nat_text": item["nat_text"],
            "nat_action": item["nat_action"],
            "cap": cap,
        })

    # Harvest targeted pytest suffixes on true_imm_edit_ok for Circuit 3 (L39 anti-bare-pytest)
    bare_caps = []
    seen_iss = set()
    for r in records:
        if r["issue"] in seen_iss or r["true_role"] != "true_imm_edit_ok":
            continue
        cmd = resolve_targeted_check({"filepath": r["filepath"], "issue": r["issue"]})
        cmd_py = cmd if (cmd and cmd.startswith("pytest")) else f"pytest -q tests/test_{Path(r.get('filepath') or 'x.py').stem.lstrip('_')}.py"
        seen_iss.add(r["issue"])
        suf = enc(f"<|channel>thought\nLet's run targeted pytest to verify the fix.\n<channel|><|tool_call>call:run_command{{command:<|\"|>{cmd_py}<|\"|>}}<tool_call|>")
        pref = r["tokens"][-(4096 - len(suf)) :]
        cap = harvest_record(pref, suf)
        if cap["has_pytest"]:
            bare_caps.append(cap)

    # Pure weight-space unembedding directions pre-conditioned for Gemma 4 RMSNorm!
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

    def tok_vec(words: list[str]) -> torch.Tensor:
        vecs = []
        for w in words:
            for variant in (w, " " + w, "\n" + w):
                ids = enc(variant)
                if len(ids) == 1:
                    v = lm_w[ids[0]] * scale_w
                    vecs.append(v / v.norm().clamp_min(1e-8))
        m = torch.stack(vecs, dim=0).mean(dim=0)
        return m / m.norm().clamp_min(1e-8)

    # C1A + C1B Thought-Start steering: push verification & repair reasoning, suppress premature/rationalized submit!
    u_thought_verify = tok_vec(["pytest", "test", "tests", "verify", "check", "run", "fix"]) - tok_vec(
        ["submit", "done", "complete", "finished", "ready", "expected"]
    )
    u_thought_verify = u_thought_verify / u_thought_verify.norm().clamp_min(1e-8)

    # C2 Thought-Start steering (`FileEditError` -> `read_file` exact line inspection):
    u_thought_read = tok_vec(["read", "inspect", "lines", "view", "exact", "content"]) - tok_vec(
        ["edit", "replace", "submit", "again", "retry"]
    )
    u_thought_read = u_thought_read / u_thought_read.norm().clamp_min(1e-8)

    # Readout `<|tool_call>call:` directions:
    tok_run = enc("<|tool_call>call:run_command{")[len(call_colon_seq)]
    tok_sub = enc("<|tool_call>call:submit_patch{")[len(call_colon_seq)]
    tok_ed = enc("<|tool_call>call:edit_file{")[len(call_colon_seq)]
    tok_rd = enc("<|tool_call>call:read_file{")[len(call_colon_seq)]

    u_c1_disp = (lm_w[tok_sub] - (0.75 * lm_w[tok_run] + 0.25 * lm_w[tok_ed])) * scale_w
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

    def precond_post_norm(u_res: torch.Tensor, w_pn: torch.Tensor) -> torch.Tensor:
        """Transpose Jacobian through Gemma 4's post_*_layernorm so RMSNorm(y + dy) writes straight along u_res!"""
        b = u_res * w_pn.clamp_min(0.05)
        return b / b.norm().clamp_min(1e-8)

    # =========================================================================
    # 1. Rank Upper-Band `o_proj` Layers [30..41] for C1 (`edit_ok + test_fail`) & C2 (`edit_err`)!
    # =========================================================================
    def is_c1_on(r: dict) -> bool:
        return r["true_role"] in ("true_imm_edit_ok", "true_imm_test_fail")

    def is_c2_on(r: dict) -> bool:
        return r["true_role"] == "true_imm_edit_err"

    c1_issues = sorted({r["issue"] for r in records if is_c1_on(r)})
    c2_issues = sorted({r["issue"] for r in records if is_c2_on(r)})
    c1_cv, c2_cv = [], []
    o_c1_fits, o_c2_fits = {}, {}

    for li in o_layers:
        raw_yn = sum(r["cap"][(li, "oy_raw_norm")] for r in records) / len(records)

        def pack_c1(recs):
            on_l, ver_l, rd_l, err_l, oth_l, del_l = [], [], [], [], [], []
            for r in recs:
                z0 = r["cap"][(li, "oz_ts0")].float()
                zo = r["cap"][(li, "oz_open")].float()
                del_l.append(r["cap"][(li, "oz_delim")].float())
                rl = r["true_role"]
                if is_c1_on(r):
                    on_l.append(z0)
                    if r["has_full_suf"]:
                        on_l.append(zo)
                elif rl == "true_imm_test_pass":
                    ver_l.append(z0)
                    if r["has_full_suf"]:
                        ver_l.append(zo)
                elif rl in ("true_imm_read_full", "true_imm_read_trunc"):
                    rd_l.extend([z0, zo] if r["has_full_suf"] else [z0])
                elif is_c2_on(r):
                    err_l.extend([z0, zo] if r["has_full_suf"] else [z0])
                else:
                    oth_l.extend([z0, zo] if r["has_full_suf"] else [z0])
            Xo = torch.stack(on_l)
            Xv = torch.stack(ver_l) if ver_l else torch.zeros(0, Xo.shape[1])
            Xr = torch.stack(rd_l) if rd_l else torch.zeros(0, Xo.shape[1])
            Xe = torch.stack(err_l) if err_l else torch.zeros(0, Xo.shape[1])
            Xot = torch.stack(oth_l) if oth_l else torch.zeros(0, Xo.shape[1])
            Xd = torch.cat(del_l, dim=0)
            ogrps = [(Xv, 12.0), (Xr, 10.0), (Xe, 5.0), (Xot, 6.0), (Xd, 16.0)]
            return Xo, ogrps

        oof_on, oof_ver, oof_rd, oof_del = [], [], [], []
        for iss in c1_issues:
            tr = [r for r in records if r["issue"] != iss]
            Xo_tr, og_tr = pack_c1(tr)
            wu, sc = solve_full_alphaedit(Xo_tr, og_tr, lam_rel=0.12)
            w = wu * sc
            for r in records:
                if r["issue"] != iss:
                    continue
                v0 = float(torch.dot(w, r["cap"][(li, "oz_ts0")].float()))
                oof_del.append(float((r["cap"][(li, "oz_delim")].float() @ w).abs().mean()))
                if is_c1_on(r):
                    oof_on.append(v0)
                elif r["true_role"] == "true_imm_test_pass":
                    oof_ver.append(abs(v0))
                elif r["true_role"] in ("true_imm_read_full", "true_imm_read_trunc"):
                    oof_rd.append(abs(v0))

        m_on = sum(oof_on) / max(1, len(oof_on))
        m_ver = sum(oof_ver) / max(1, len(oof_ver))
        m_rd = sum(oof_rd) / max(1, len(oof_rd))
        m_del = sum(oof_del) / max(1, len(oof_del))
        # Weight OOF gap by sqrt(cum_scalar) so layers that actually reach final norm win!
        eff_score = (m_on - max(m_ver, m_rd, 2.2 * m_del)) * (cum_scalars[li] ** 0.5)
        Xo, og = pack_c1(records)
        wu, sc = solve_full_alphaedit(Xo, og, lam_rel=0.12)
        in_on = float((Xo @ (wu * sc)).mean())
        o_c1_fits[li] = {
            "u_a": wu, "scale": sc, "in_on": max(0.15, in_on),
            "raw_yn": raw_yn, "eff_score": eff_score,
            "b_dir": precond_post_norm(u_thought_verify, post_attn_w[li]),
        }
        c1_cv.append({
            "layer": li, "cum_scalar": round(cum_scalars[li], 4), "raw_yn": round(raw_yn, 3),
            "oof_on": round(m_on, 4), "oof_ver_abs": round(m_ver, 4),
            "oof_rd_abs": round(m_rd, 4), "oof_delim_abs": round(m_del, 4),
            "eff_score": round(eff_score, 4),
        })

        # Circuit 2 (`true_imm_edit_err` -> `u_thought_read`)
        def pack_c2(recs):
            on_l, rd_l, ok_l, ver_l, oth_l, del_l = [], [], [], [], [], []
            for r in recs:
                z0 = r["cap"][(li, "oz_ts0")].float()
                zo = r["cap"][(li, "oz_open")].float()
                del_l.append(r["cap"][(li, "oz_delim")].float())
                rl = r["true_role"]
                if is_c2_on(r):
                    on_l.extend([z0, zo] if r["has_full_suf"] else [z0])
                elif rl in ("true_imm_read_full", "true_imm_read_trunc"):
                    rd_l.extend([z0, zo] if r["has_full_suf"] else [z0])
                elif is_c1_on(r):
                    ok_l.extend([z0, zo] if r["has_full_suf"] else [z0])
                elif rl == "true_imm_test_pass":
                    ver_l.append(z0)
                else:
                    oth_l.append(z0)
            Xo2 = torch.stack(on_l)
            Xr2 = torch.stack(rd_l) if rd_l else torch.zeros(0, Xo2.shape[1])
            Xok2 = torch.stack(ok_l) if ok_l else torch.zeros(0, Xo2.shape[1])
            Xv2 = torch.stack(ver_l) if ver_l else torch.zeros(0, Xo2.shape[1])
            Xot2 = torch.stack(oth_l) if oth_l else torch.zeros(0, Xo2.shape[1])
            Xd2 = torch.cat(del_l, dim=0)
            ogrps2 = [(Xr2, 12.0), (Xok2, 7.0), (Xv2, 8.0), (Xot2, 6.0), (Xd2, 16.0)]
            return Xo2, ogrps2

        oof2_on, oof2_rd, oof2_del = [], [], []
        for iss in c2_issues:
            tr = [r for r in records if r["issue"] != iss]
            Xo2_tr, og2_tr = pack_c2(tr)
            wu2, sc2 = solve_full_alphaedit(Xo2_tr, og2_tr, lam_rel=0.12)
            w2 = wu2 * sc2
            for r in records:
                if r["issue"] != iss:
                    continue
                v0 = float(torch.dot(w2, r["cap"][(li, "oz_ts0")].float()))
                oof2_del.append(float((r["cap"][(li, "oz_delim")].float() @ w2).abs().mean()))
                if is_c2_on(r):
                    oof2_on.append(v0)
                elif r["true_role"] in ("true_imm_read_full", "true_imm_read_trunc"):
                    oof2_rd.append(abs(v0))
        m2_on = sum(oof2_on) / max(1, len(oof2_on))
        m2_rd = sum(oof2_rd) / max(1, len(oof2_rd))
        m2_del = sum(oof2_del) / max(1, len(oof2_del))
        eff2 = (m2_on - max(m2_rd, 2.2 * m2_del)) * (cum_scalars[li] ** 0.5)
        Xo2, og2 = pack_c2(records)
        wu2, sc2 = solve_full_alphaedit(Xo2, og2, lam_rel=0.12)
        in2_on = float((Xo2 @ (wu2 * sc2)).mean())
        o_c2_fits[li] = {
            "u_a": wu2, "scale": sc2, "in_on": max(0.15, in2_on),
            "raw_yn": raw_yn, "eff_score": eff2,
            "b_dir": precond_post_norm(u_thought_read, post_attn_w[li]),
        }
        c2_cv.append({
            "layer": li, "cum_scalar": round(cum_scalars[li], 4),
            "oof_on": round(m2_on, 4), "oof_rd_abs": round(m2_rd, 4),
            "oof_delim_abs": round(m2_del, 4), "eff_score": round(eff2, 4),
        })

    # Avoid L41 for Thought-Start o_proj so L41 stays clean for final readout; rank [30..40]!
    c1_ranked = sorted([r for r in c1_cv if r["layer"] <= 40], key=lambda x: x["eff_score"], reverse=True)
    c2_ranked = sorted([r for r in c2_cv if r["layer"] <= 40], key=lambda x: x["eff_score"], reverse=True)
    # Ensure Global Full-Attention L35 is included if in top-5, plus highest eff_score companions!
    top3_c1 = [r["layer"] for r in c1_ranked[:3]]
    if 35 not in top3_c1 and any(r["layer"] == 35 and r["eff_score"] > 0.05 for r in c1_ranked):
        top3_c1 = [top3_c1[0], 35, top3_c1[1]]
    top2_c1 = top3_c1[:2]
    top2_c2 = [r["layer"] for r in c2_ranked if r["layer"] not in set(top3_c1)][:2]

    # =========================================================================
    # 2. Upper-Band `down_proj` Readouts (L39 Anti-Bare, L41 C1 Colon, L40 C2 Colon)
    #    Pre-conditioned by `post_feedforward_layernorm.weight` and scaled by `dy_raw_norm`!
    # =========================================================================
    h39_on, h39_off, dyn39 = [], [], []
    for r in records:
        dyn39.append(r["cap"][(39, "dy_raw_norm")])
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
    a39_u, sc39 = solve_full_alphaedit(H39_on, [(H39_off, 12.0)], lam_rel=0.10)
    s39_on = float((H39_on @ (a39_u * sc39)).mean())
    b39_dir = precond_post_norm(-u_c3_ab, post_ffn_w[39])
    dyn39_mean = sum(dyn39) / len(dyn39)

    def fit_colon_down(li: int, on_pred, u_disp: torch.Tensor):
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
        au, sc = solve_full_alphaedit(Ho, [(Hr, 18.0), (Hv, 16.0), (Hot, 10.0), (Hd, 16.0)], lam_rel=0.10)
        w = au * sc
        son = float((Ho @ w).mean())
        rd_max = float((Hr @ w).abs().max()) if Hr.shape[0] else 0.0
        ver_max = float((Hv @ w).abs().max()) if Hv.shape[0] else 0.0
        return {
            "u_a": au, "scale": sc, "in_on": max(0.15, son),
            "raw_yn": sum(dyn) / len(dyn),
            "rd_ratio": round(rd_max / max(1e-4, son), 4),
            "ver_ratio": round(ver_max / max(1e-4, son), 4),
            "b_dir": precond_post_norm(-u_disp, post_ffn_w[li]),
        }

    d41_c1 = fit_colon_down(41, is_c1_on, u_c1_disp)
    d40_c2 = fit_colon_down(40, is_c2_on, u_c2_err)

    # =========================================================================
    # 3. LIVE IN-VRAM LOGIT-SHIFT CALIBRATION CHECK BEFORE COMPILING ADAPTERS!
    #    Verify that rho_o in {0.30, 0.50, 0.75} actually moves Thought-Start
    #    verify-minus-submit logits by +1.5 .. +4.0 nats on ON states and ~0.0 on OFF!
    # =========================================================================
    v_ids = [enc(w)[0] for w in (" pytest", " test", " verify", " check") if len(enc(w)) == 1]
    s_ids = [enc(w)[0] for w in (" submit", " done", " complete") if len(enc(w)) == 1]
    cal_on = [r for r in records if is_c1_on(r)][:4]
    cal_off = [r for r in records if r["true_role"] in ("true_imm_test_pass", "true_imm_read_full")][:4]

    def measure_logit_delta(rho_o: float, layers_use: list[int]) -> dict:
        deltas = {"on": [], "off": []}
        for tag, subset in (("on", cal_on), ("off", cal_off)):
            for r in subset:
                tail_dec = tokenizer.decode(r["tokens"][-10:], skip_special_tokens=False)
                suf = [] if tail_dec.endswith("<|channel>thought\n") else thought_hdr
                inp = torch.tensor([(r["tokens"] + suf)[-3072:]], device=dev)
                with torch.inference_mode():
                    lg0 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                hks = []
                for li in layers_use:
                    f = o_c1_fits[li]
                    a_v = (f["u_a"] * (f["scale"] / f["in_on"])).to(dev, torch.bfloat16)
                    b_v = (rho_o * f["raw_yn"] * f["b_dir"]).to(dev, torch.bfloat16)
                    def mk_hk(av=a_v, bv=b_v):
                        def _hk(mod, inputs, output):
                            return output + (inputs[0] @ av).unsqueeze(-1) * bv
                        return _hk
                    hks.append(layers[li].self_attn.o_proj.register_forward_hook(mk_hk()))
                try:
                    with torch.inference_mode():
                        lg1 = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                finally:
                    for hk in hks:
                        hk.remove()
                dv = (lg1[v_ids].mean() - lg1[s_ids].mean()) - (lg0[v_ids].mean() - lg0[s_ids].mean())
                deltas[tag].append(float(dv.item()))
        return {
            "rho_o": rho_o,
            "mean_on_delta_nats": round(sum(deltas["on"]) / max(1, len(deltas["on"])), 3),
            "mean_off_delta_nats": round(sum(deltas["off"]) / max(1, len(deltas["off"])), 3),
        }

    cal_grid = [measure_logit_delta(rho, top3_c1) for rho in (0.25, 0.45, 0.70, 1.00)]

    del model, lm_root, layers
    gc.collect()
    torch.cuda.empty_cache()

    diagnostics = {
        "n_records": len(records),
        "role_counts": dict(Counter(r["true_role"] for r in records)),
        "top_c1_upper_oproj": c1_ranked[:5],
        "top_c2_upper_oproj": [r for r in c2_ranked if r["layer"] not in set(top3_c1)][:4],
        "selected_c1_layers": top3_c1,
        "selected_c2_layers": top2_c2,
        "live_thought_logit_calibration": cal_grid,
        "C1_L41_colon_rd_ratio": d41_c1["rd_ratio"],
        "C1_L41_colon_ver_ratio": d41_c1["ver_ratio"],
        "C2_L40_colon_rd_ratio": d40_c2["rd_ratio"],
    }
    print("V16 Upper-Band Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    def build_variant(
        name: str,
        c1_ls: list[int],
        rho_c1_o: float,
        c2_ls: list[int],
        rho_c2_o: float,
        rho_c3_d39: float = 0.34,
        rho_c1_d41: float = 0.0,
        rho_c2_d40: float = 0.0,
    ):
        ospecs = {}
        for li in c1_ls:
            f = o_c1_fits[li]
            ospecs[li] = (f["u_a"] * (f["scale"] / f["in_on"]), (rho_c1_o * f["raw_yn"]) * f["b_dir"])
        for li in c2_ls:
            f = o_c2_fits[li]
            ospecs[li] = (f["u_a"] * (f["scale"] / f["in_on"]), (rho_c2_o * f["raw_yn"]) * f["b_dir"])

        dspecs = {}
        if rho_c3_d39 > 0:
            dspecs[39] = (a39_u * (sc39 / max(0.15, s39_on)), (rho_c3_d39 * dyn39_mean) * b39_dir)
        if rho_c1_d41 > 0:
            dspecs[41] = (d41_c1["u_a"] * (d41_c1["scale"] / d41_c1["in_on"]), (rho_c1_d41 * d41_c1["raw_yn"]) * d41_c1["b_dir"])
        if rho_c2_d40 > 0:
            dspecs[40] = (d40_c2["u_a"] * (d40_c2["scale"] / d40_c2["in_on"]), (rho_c2_d40 * d40_c2["raw_yn"]) * d40_c2["b_dir"])

        if not (ad_root / "V16_hybrid_zero").exists():
            compiled.append(compile_oproj_downproj_r1(ROOT / "champion.zip", ospecs, dspecs, oproj_in_dims, ad_root / "V16_hybrid_zero", "zero"))
        for ctrl in ("candidate", "random"):
            compiled.append(compile_oproj_downproj_r1(ROOT / "champion.zip", ospecs, dspecs, oproj_in_dims, ad_root / f"{name}_{ctrl}", ctrl))

    # Calibrated gain brackets across upper-band o_proj + L39/L40/L41 readouts!
    build_variant("V16_up_c1c3_mid", top3_c1, rho_c1_o=0.45, c2_ls=[], rho_c2_o=0.0, rho_c3_d39=0.35, rho_c1_d41=0.22)
    build_variant("V16_up_c1c2c3_mid", top2_c1, rho_c1_o=0.45, c2_ls=top2_c2, rho_c2_o=0.42, rho_c3_d39=0.35, rho_c1_d41=0.24, rho_c2_d40=0.24)
    build_variant("V16_up_c1c2c3_hi", top3_c1, rho_c1_o=0.72, c2_ls=top2_c2, rho_c2_o=0.65, rho_c3_d39=0.38, rho_c1_d41=0.34, rho_c2_d40=0.34)
    build_variant("V16_up_c1c2c3_pure_o", top3_c1, rho_c1_o=0.65, c2_ls=top2_c2, rho_c2_o=0.58, rho_c3_d39=0.35, rho_c1_d41=0.0, rho_c2_d40=0.0)

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V16 Upper-Band Post-RMSNorm AlphaEdit adapters!", flush=True)


if __name__ == "__main__":
    main()
