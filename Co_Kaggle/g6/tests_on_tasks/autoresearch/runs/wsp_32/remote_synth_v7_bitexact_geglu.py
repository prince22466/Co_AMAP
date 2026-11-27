"""Synthesize V7 r=1 Bit-Exact Per-Token IRLS GeGLU Transistors ({41} / {35,41} dispatch + {39} anti-bare-pytest)."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter, resolve_targeted_check
from synthesize_targeted_dispatch_v3 import find_subseq_end


def solve_primal_irls_gate(
    x_on: torch.Tensor,      # [N_on, 2560] exact single-token ON states
    g0_on: torch.Tensor,     # [N_on]
    x_off: torch.Tensor,     # [N_off, 2560] ALL ~4000 individual per-token OFF states!
    g0_off: torch.Tensor,    # [N_off]
    target_on: float = 3.2,
    target_off: float = -4.6,
    lam_rel: float = 0.06,
    n_irls: int = 5,
) -> torch.Tensor:
    """Primal 2560x2560 IRLS hard-negative-mining ridge solver so 100% of OFF tokens stay <= -3.6 (BF16 exact 0.0!)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xo = x_on.to(device, torch.float32)
    yo = (target_on - g0_on.to(device, torch.float32)).clamp_min(1.5)
    Xf = x_off.to(device, torch.float32)
    yf = (target_off - g0_off.to(device, torch.float32)).clamp_max(-0.5)

    n_on = max(1, Xo.shape[0])
    n_off = max(1, Xf.shape[0])
    w_on = torch.full((n_on,), 1.0 / n_on, device=device)
    w_off = torch.full((n_off,), 2.5 / n_off, device=device)
    mean_norm_sq = float(Xf.pow(2).sum(dim=1).mean().clamp_min(1.0))
    eye = torch.eye(Xo.shape[1], device=device)

    w = torch.zeros(Xo.shape[1], device=device)
    for step in range(n_irls):
        w_off_norm = w_off * (2.5 / w_off.sum().clamp_min(1e-8))
        lhs = (Xo.T * w_on) @ Xo + (Xf.T * w_off_norm) @ Xf + (lam_rel * mean_norm_sq) * eye
        rhs = Xo.T @ (w_on * yo) + Xf.T @ (w_off_norm * yf)
        w = torch.linalg.solve(lhs, rhs)
        if step + 1 < n_irls:
            pred_g_off = g0_off.to(device, torch.float32) + (Xf @ w)
            # Boost weight on any individual OFF token above -3.7 (BF16 zero threshold!)
            viol = pred_g_off > -3.7
            if not viol.any():
                break
            w_off = torch.where(viol, w_off * 6.0, w_off)
    return w.cpu()


def compile_v7_geglu(
    champion_zip: Path,
    layer_specs: dict[int, dict],
    out_dir: Path,
    control: str = "candidate",
    seed: int = 37,
) -> dict:
    assert not (set(layer_specs) & CHAMP_LAYERS)
    with zipfile.ZipFile(champion_zip) as z:
        blob = z.read("adapters/wsp_r1/adapter_model.safetensors")
        cfg = json.loads(z.read("adapters/wsp_r1/adapter_config.json"))
    champ = load(blob)
    layers = sorted(CHAMP_LAYERS | set(layer_specs))
    gen = torch.Generator().manual_seed(seed)
    tensors = {}

    for l in layers:
        bp = f"base_model.model.model.language_model.layers.{l}.mlp"
        ga = torch.zeros(1, 2560, dtype=torch.bfloat16)
        gb = torch.zeros(10240, 1, dtype=torch.bfloat16)
        ua = torch.zeros(1, 2560, dtype=torch.bfloat16)
        ub = torch.zeros(10240, 1, dtype=torch.bfloat16)
        da = torch.zeros(1, 10240, dtype=torch.bfloat16)
        db = torch.zeros(2560, 1, dtype=torch.bfloat16)

        if l in CHAMP_LAYERS:
            da[:] = champ[f"{bp}.down_proj.lora_A.weight"]
            db[:] = champ[f"{bp}.down_proj.lora_B.weight"]
        elif l in layer_specs and control != "zero":
            sp = layer_specs[l]
            q = int(sp["q_idx"])
            wg = sp["w_gate"].to(torch.bfloat16)
            wu = sp["w_up"].to(torch.bfloat16)
            bd = sp["b_down"].to(torch.bfloat16)
            mask = torch.zeros(10240, dtype=torch.bfloat16)
            mask[q] = 1.0
            if control == "random":
                wg = wg[torch.randperm(2560, generator=gen)]
                wu = wu[torch.randperm(2560, generator=gen)]
                rq = int(torch.randint(0, 10240, (1,), generator=gen).item())
                mask = torch.zeros(10240, dtype=torch.bfloat16)
                mask[rq] = 1.0
                bd = bd[torch.randperm(2560, generator=gen)] * (2 * torch.randint(2, (2560,), generator=gen) - 1)
            ga[0] = wg
            gb[:, 0] = mask
            ua[0] = wu
            ub[:, 0] = mask
            da[0] = mask
            db[:, 0] = bd

        tensors[f"{bp}.gate_proj.lora_A.weight"] = ga
        tensors[f"{bp}.gate_proj.lora_B.weight"] = gb
        tensors[f"{bp}.up_proj.lora_A.weight"] = ua
        tensors[f"{bp}.up_proj.lora_B.weight"] = ub
        tensors[f"{bp}.down_proj.lora_A.weight"] = da
        tensors[f"{bp}.down_proj.lora_B.weight"] = db

    out_dir.mkdir(parents=True, exist_ok=True)
    w_path = out_dir / "adapter_model.safetensors"
    save_file(tensors, str(w_path))
    cfg.update(
        r=1,
        lora_alpha=1,
        layers_to_transform=layers,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return {
        "adapter": out_dir.name,
        "layers": layers,
        "control": control,
        "sha256": hashlib.sha256(w_path.read_bytes()).hexdigest(),
    }


def main():
    hdir = Path("/content/wsp/harvest32")
    out_dir = hdir / "v7_geglu"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads((hdir / "cases.json").read_text())
    native = {r["prompt_sha"]: r for r in json.loads((hdir / "native/champion.json").read_text())}

    assets = json.loads((ROOT / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        assets["model"], dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    lm_root = getattr(model.model, "language_model", model.model)
    layers = lm_root.layers

    with zipfile.ZipFile(ROOT / "champion.zip") as z:
        champ = load(z.read("adapters/wsp_r1/adapter_model.safetensors"))
    attach_adapter(layers, champ)
    with torch.inference_mode():
        model(input_ids=torch.tensor([[tokenizer.bos_token_id]], device=model.device), use_cache=False, logits_to_keep=1)

    target_layers = [35, 39, 41]  # 35 & 41 = Global Full-Attention; 39 = Anti-Bare-Pytest!
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    call_colon_seq = enc("<|tool_call>call:")
    prefix_pytest = enc('<|tool_call>call:run_command{command:<|"|>pytest')

    # Harvest full sequence of individual generated tokens on [35, 39, 41]
    def forward_seq(prefix: list[int], suffix: list[int]):
        toks = prefix + suffix
        assert len(toks) <= 4096
        p_len = len(prefix)
        cap = {}
        hooks = []
        for li in target_layers:
            def mk_g(idx=li):
                def _h(mod, inp, out):
                    cap[(idx, "x")] = inp[0][0, p_len:].float().cpu()     # [S, 2560]
                    cap[(idx, "g0")] = out[0, p_len:].float().cpu()       # [S, 10240]
                return _h
            def mk_u(idx=li):
                def _h(mod, inp, out):
                    cap[(idx, "p0")] = out[0, p_len:].float().cpu()       # [S, 10240]
                return _h
            def mk_d(idx=li):
                def _h(mod, inp, out):
                    cap[(idx, "y0_norm")] = float(out[0, p_len:].float().norm(dim=-1).mean().item())
                return _h
            hooks.append(layers[li].mlp.gate_proj.register_forward_hook(mk_g()))
            hooks.append(layers[li].mlp.up_proj.register_forward_hook(mk_u()))
            hooks.append(layers[li].mlp.down_proj.register_forward_hook(mk_d()))
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([toks], device=model.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        return cap

    # Collect per-token datasets across REAL native sequences!
    # First pick quietest neuron q* per layer using a fast scan of 16 sequences
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
            gen_txt = "<|channel>thought\nThe tests passed and verified the fix. I will now submit the patch.\n<channel|><|tool_call>call:submit_patch{}<tool_call|>"
        else:
            continue
        suf = enc(gen_txt)[:320]
        if len(suf) < 4:
            continue
        pref = c["tokens"][-(4096 - len(suf)) :]
        cap = forward_seq(pref, suf)
        colon_end = find_subseq_end(suf, call_colon_seq)  # exclusive end -> colon_idx = colon_end - 1
        pytest_end = find_subseq_end(suf, prefix_pytest)
        seq_records.append({
            "issue": c["issue"],
            "role": c["role"],
            "suf": suf,
            "colon_idx": (colon_end - 1) if colon_end is not None else None,
            "pytest_idx": (pytest_end - 1) if pytest_end is not None else None,
            "cap": cap,
        })

    # Also harvest bare-pytest -> targeted-pytest sequences on the 8 discovery issues so L39 has 8 clean ON tokens + close-quote OFF tokens
    bare_seq_records = []
    for issue in sorted({c["issue"] for c in cases}):
        opts = [c for c in cases if c["issue"] == issue and c["role"] in ("unchecked", "check")]
        for case in sorted(opts, key=lambda c: (c["role"] != "unchecked", len(c["tokens"]))):
            cmd = resolve_targeted_check(case)
            if not cmd:
                continue
            cmd_py = cmd if cmd.startswith("pytest") else f"pytest -q tests/test_{Path(case.get('filepath') or 'x.py').stem.lstrip('_')}.py"
            suf = enc(f"<|channel>thought\nI will run tests to verify the fix.\n<channel|><|tool_call>call:run_command{{command:<|\"|>{cmd_py}<|\"|>}}<tool_call|>")
            pref = case["tokens"][-(4096 - len(suf)) :]
            cap = forward_seq(pref, suf)
            pt_end = find_subseq_end(suf, prefix_pytest)
            col_end = find_subseq_end(suf, call_colon_seq)
            if pt_end is not None:
                bare_seq_records.append({
                    "issue": issue,
                    "observed_role": case["role"],
                    "colon_idx": (col_end - 1) if col_end is not None else None,
                    "pytest_idx": pt_end - 1,
                    "cap": cap,
                })
            break

    total_toks = sum(len(r["suf"]) for r in seq_records)
    print(f"Harvested {len(seq_records)} full real sequences ({total_toks} individual tokens!) + {len(bare_seq_records)} pytest sequences.", flush=True)

    # Pick 1 deadest neuron q* per layer across ALL tokens of ALL preservation sequences (verified, edit, localize)!
    q_stars = {}
    for li in target_layers:
        g_all = torch.cat([r["cap"][(li, "g0")] for r in seq_records if r["role"] in ("verified", "edit", "localize")], dim=0)
        p_all = torch.cat([r["cap"][(li, "p0")] for r in seq_records if r["role"] in ("verified", "edit", "localize")], dim=0)
        h0_bf16 = (F.gelu(g_all.bfloat16(), approximate="tanh") * p_all.bfloat16()).float()
        max_h0 = h0_bf16.abs().max(dim=0).values
        max_g0 = g_all.max(dim=0).values
        mean_g0 = g_all.mean(dim=0)
        # Prefer neurons with max_h0 == 0.0 (or smallest max_h0 < 0.01) AND max_g0 <= -2.0 AND mean_g0 >= -5.5
        valid = (max_h0 <= 0.015) & (max_g0 <= -1.8) & (mean_g0 >= -6.0)
        if not valid.any():
            valid = max_h0 <= 0.05
        score = torch.where(valid, -100.0 * max_h0 - max_g0 - 0.1 * (mean_g0 + 3.5).abs(), torch.full_like(max_g0, -1e9))
        q = int(score.argmax().item())
        q_stars[li] = q
        print(f"Layer {li} deadest neuron q*={q}: max_|h0_bf16|={max_h0[q].item():.6f}, max_g0={max_g0[q].item():.3f}, mean_g0={mean_g0[q].item():.3f}", flush=True)

    # Pure weight-space unembedding directions:
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

    # 1. Dispatch direction at `:` (`call:run_command` vs `call:submit_patch`)
    tok_run = enc("<|tool_call>call:run_command{")[len(call_colon_seq)]
    tok_sub = enc("<|tool_call>call:submit_patch{")[len(call_colon_seq)]
    print(f"Exact first token after `call:`: run={tok_run} ({tokenizer.decode([tok_run])!r}) vs sub={tok_sub} ({tokenizer.decode([tok_sub])!r})", flush=True)
    u_disp = (lm_w[tok_sub] - lm_w[tok_run]) * scale_w
    u_disp = u_disp / u_disp.norm().clamp_min(1e-10)

    # 2. Anti-bare-pytest direction at `pytest` (`command:<|"|>pytest` -> ` -q` / ` tests` vs `<|"|>`)
    sb = enc('<|tool_call>call:run_command{command:<|"|>pytest<|"|>}<tool_call|>')
    st1 = enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests/test_panel.py<|"|>}<tool_call|>')
    st2 = enc('<|tool_call>call:run_command{command:<|"|>pytest tests/test_panel.py<|"|>}<tool_call|>')
    u_ab = (
        lm_w[sb[find_subseq_end(sb, prefix_pytest)]]
        - (0.60 * lm_w[st1[find_subseq_end(st1, prefix_pytest)]] + 0.40 * lm_w[st2[find_subseq_end(st2, prefix_pytest)]])
    ) * scale_w
    u_ab = u_ab / u_ab.norm().clamp_min(1e-10)

    # Fit L35 and L41 Single-Token `call:` Dispatch Transistors!
    disp_fits = {}
    cv_report = {}

    for li in (35, 41):
        q = q_stars[li]
        on_x, on_g0, on_p0, on_issues = [], [], [], []
        off_x, off_g0, off_p0 = [], [], []
        y_norms = []

        for r in seq_records:
            x_seq = r["cap"][(li, "x")]
            g_seq = r["cap"][(li, "g0")][:, q]
            p_seq = r["cap"][(li, "p0")][:, q]
            y_norms.append(r["cap"][(li, "y0_norm")])
            cidx = r["colon_idx"]

            if r["role"] in ("unchecked", "check") and cidx is not None:
                # ONLY the exact single `:` token is ON; ALL thought tokens before `:` and arg tokens after `:` are OFF!
                on_x.append(x_seq[cidx])
                on_g0.append(g_seq[cidx])
                on_p0.append(p_seq[cidx])
                on_issues.append((r["issue"], r["role"]))
                mask_off = torch.arange(x_seq.shape[0]) != cidx
                off_x.append(x_seq[mask_off])
                off_g0.append(g_seq[mask_off])
                off_p0.append(p_seq[mask_off])
            else:
                # For verified, edit, localize: EVERY SINGLE TOKEN (including verified@call: and edit@call:!) is OFF!
                off_x.append(x_seq)
                off_g0.append(g_seq)
                off_p0.append(p_seq)
                # Duplicate verified@call: and edit@call: 8x so `call:` on verified/edit is rock-solid <= -3.8!
                if cidx is not None and r["role"] in ("verified", "edit", "localize"):
                    off_x.append(x_seq[cidx : cidx + 1].repeat(8, 1))
                    off_g0.append(g_seq[cidx : cidx + 1].repeat(8))
                    off_p0.append(p_seq[cidx : cidx + 1].repeat(8))

        X_on = torch.stack(on_x)
        G_on = torch.stack(on_g0)
        P_on = torch.stack(on_p0)
        X_off = torch.cat(off_x, dim=0)
        G_off = torch.cat(off_g0, dim=0)
        P_off = torch.cat(off_p0, dim=0)

        # Leave-one-issue-out OOF on REAL `unchecked` issues!
        oof_real = {}
        unique_real_issues = sorted({iss for iss, rl in on_issues if rl == "unchecked"})
        for iss in unique_real_issues:
            tr = torch.tensor([i_iss != iss for i_iss, _ in on_issues])
            wg_tr = solve_primal_irls_gate(X_on[tr], G_on[tr], X_off, G_off)
            for idx_i, (i_iss, rl) in enumerate(on_issues):
                if i_iss == iss and rl == "unchecked":
                    si = torch.dot(wg_tr, X_on[idx_i]).bfloat16()
                    hi = F.gelu(G_on[idx_i].bfloat16() + si, approximate="tanh") * (P_on[idx_i].bfloat16() + si)
                    oof_real[iss] = round(float(hi.item()), 3)

        wg = solve_primal_irls_gate(X_on, G_on, X_off, G_off)
        s_on_bf = (X_on @ wg).bfloat16()
        h_on_bf = (F.gelu(G_on.bfloat16() + s_on_bf, approximate="tanh") * (P_on.bfloat16() + s_on_bf)).float()

        s_off_bf = (X_off @ wg).bfloat16()
        h_off_bf = (F.gelu(G_off.bfloat16() + s_off_bf, approximate="tanh") * (P_off.bfloat16() + s_off_bf)).float()
        bf16_exact_zero_frac = float((h_off_bf == 0.0).float().mean().item())

        w_down_q = layers[li].mlp.down_proj.weight.detach().float().cpu()[:, q]
        cv_report[f"L{li}_colon_dispatch"] = {
            "q_star": q,
            "n_on_tokens": int(X_on.shape[0]),
            "n_off_tokens": int(X_off.shape[0]),
            "oof_real_unchecked_h": oof_real,
            "oof_real_pos_rate": round(sum(1 for v in oof_real.values() if v > 0.5) / max(1, len(oof_real)), 3),
            "full_h_on_mean": round(float(h_on_bf.mean()), 3),
            "off_bf16_exact_zero_frac": round(bf16_exact_zero_frac, 5),
            "off_h_max_abs": round(float(h_off_bf.abs().max()), 6),
        }
        disp_fits[li] = {
            "q_idx": q,
            "w_gate": wg,
            "w_up": wg,
            "w_down_q": w_down_q,
            "u": u_disp,
            "h_on_mean": float(h_on_bf.mean().clamp_min(1.0)),
            "y0_norm": sum(y_norms) / len(y_norms),
        }

    # Fit L39 Bit-Exact Per-Token Anti-Bare-Pytest Transistor!
    q39 = q_stars[39]
    on39_x, on39_g, on39_p = [], [], []
    off39_x, off39_g, off39_p = [], [], []
    y39_norms = []
    for r in seq_records:
        xs = r["cap"][(39, "x")]
        gs = r["cap"][(39, "g0")][:, q39]
        ps = r["cap"][(39, "p0")][:, q39]
        y39_norms.append(r["cap"][(39, "y0_norm")])
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
    for r in bare_seq_records:
        xs = r["cap"][(39, "x")]
        gs = r["cap"][(39, "g0")][:, q39]
        ps = r["cap"][(39, "p0")][:, q39]
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

    wg39 = solve_primal_irls_gate(X39_on, G39_on, X39_off, G39_off)
    s39_on = (X39_on @ wg39).bfloat16()
    h39_on = (F.gelu(G39_on.bfloat16() + s39_on, approximate="tanh") * (P39_on.bfloat16() + s39_on)).float()
    s39_off = (X39_off @ wg39).bfloat16()
    h39_off = (F.gelu(G39_off.bfloat16() + s39_off, approximate="tanh") * (P39_off.bfloat16() + s39_off)).float()
    w_down_q39 = layers[39].mlp.down_proj.weight.detach().float().cpu()[:, q39]

    cv_report["L39_anti_bare"] = {
        "q_star": q39,
        "n_on_tokens": int(X39_on.shape[0]),
        "n_off_tokens": int(X39_off.shape[0]),
        "full_h_on_mean": round(float(h39_on.mean()), 3),
        "off_bf16_exact_zero_frac": round(float((h39_off == 0.0).float().mean().item()), 5),
        "off_h_max_abs": round(float(h39_off.abs().max()), 6),
    }
    print("V7 Bit-Exact Per-Token GeGLU Report:", json.dumps(cv_report, indent=2), flush=True)

    ad_root = out_dir / "adapters"
    shutil.rmtree(ad_root, ignore_errors=True)
    compiled = []

    def build_set(name: str, disp_layers: list[int], rho_disp: float, rho_ab: float | None):
        lspecs = {}
        for li in disp_layers:
            f = disp_fits[li]
            g = rho_disp * f["y0_norm"] / f["h_on_mean"]
            lspecs[li] = {"q_idx": f["q_idx"], "w_gate": f["w_gate"], "w_up": f["w_up"], "b_down": -f["w_down_q"] - g * f["u"]}
        if rho_ab is not None:
            g39 = rho_ab * (sum(y39_norms) / len(y39_norms)) / float(h39_on.mean().clamp_min(1.0))
            lspecs[39] = {"q_idx": q39, "w_gate": wg39, "w_up": wg39, "b_down": -w_down_q39 - g39 * u_ab}

        zlbl = f"V7_zero_{'_'.join(map(str, sorted(lspecs)))}"
        if not (ad_root / zlbl).exists():
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", lspecs, ad_root / zlbl, "zero"))
        for ctrl in ("candidate", "random"):
            lbl = f"{name}_{ctrl}"
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", lspecs, ad_root / lbl, ctrl))

    # 1. Downstream L41-only + L39 anti-bare (zero layers below 38!)
    build_set("V7_L41_L39ab_mid", [41], rho_disp=0.35, rho_ab=0.25)
    build_set("V7_L41_L39ab_hi", [41], rho_disp=0.60, rho_ab=0.35)
    # 2. Global {35, 41} + L39 anti-bare (now bit-exact 0.0 on edit/localize/thought tokens even on L35!)
    build_set("V7_L35_41_L39ab_mid", [35, 41], rho_disp=0.22, rho_ab=0.25)
    build_set("V7_L35_41_L39ab_hi", [35, 41], rho_disp=0.38, rho_ab=0.35)

    (out_dir / "manifest.json").write_text(json.dumps({"cv": cv_report, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V7 Bit-Exact Per-Token GeGLU adapters!", flush=True)


if __name__ == "__main__":
    main()
