"""Synthesize V8 Pure-L41 Causal-Readout GeGLU Transistors (zero KV-cache prefill side-effects!) + gentle champion interpolations."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load, load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter, resolve_targeted_check
from recipe import compile_explicit
from synth_v7_geglu import compile_v7_geglu, solve_primal_irls_gate
from synthesize_targeted_dispatch_v3 import find_subseq_end


def main():
    hdir = Path("/content/wsp/harvest32")
    out_dir = hdir / "v8_pure41"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads((hdir / "cases.json").read_text())
    native = {r["prompt_sha"]: r for r in json.loads((hdir / "native/champion.json").read_text())}
    ml2 = load_file(str(hdir / "multilayer_candidates/ML2_late_35_41_hi.safetensors"))

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

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    call_colon_seq = enc("<|tool_call>call:")
    prefix_pytest = enc('<|tool_call>call:run_command{command:<|"|>pytest')

    # Harvest ONLY Layer 41 (final MLP layer — zero KV-cache effect on any prompt token!)
    def forward_l41_seq(prefix: list[int], suffix: list[int]):
        toks = prefix + suffix
        assert len(toks) <= 4096
        p_len = len(prefix)
        cap = {}
        hooks = [
            layers[41].mlp.gate_proj.register_forward_hook(
                lambda m, inp, out: cap.update(x=inp[0][0, p_len:].float().cpu(), g0=out[0, p_len:].float().cpu())
            ),
            layers[41].mlp.up_proj.register_forward_hook(
                lambda m, inp, out: cap.update(p0=out[0, p_len:].float().cpu())
            ),
            layers[41].mlp.down_proj.register_forward_hook(
                lambda m, inp, out: cap.update(y0_norm=float(out[0, p_len:].float().norm(dim=-1).mean().item()))
            ),
        ]
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([toks], device=model.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
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
        # Keep the TAIL of generated text (up to 768 tokens) so `<|tool_call>call:` is NEVER truncated!
        suf = full_suf[-768:] if len(full_suf) > 768 else full_suf
        if len(suf) < 4:
            continue
        pref = c["tokens"][-(4096 - len(suf)) :]
        cap = forward_l41_seq(pref, suf)
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

    # Also harvest targeted pytest suffixes on all discovery issues for anti-bare-pytest token states on L41
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
            cap = forward_l41_seq(pref, suf)
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

    # Pick globally quietest neuron q* on L41 (NO boolean mask fallback to 0 — direct argmin of max_|h0_bf16| + 0.05*relu(max_g0 + 2.0)!)
    g_pres = torch.cat([r["cap"]["g0"] for r in seq_records if r["role"] in ("verified", "edit", "localize")], dim=0)
    p_pres = torch.cat([r["cap"]["p0"] for r in seq_records if r["role"] in ("verified", "edit", "localize")], dim=0)
    h0_bf = (F.gelu(g_pres.bfloat16(), approximate="tanh") * p_pres.bfloat16()).float()
    max_h0 = h0_bf.abs().max(dim=0).values
    mean_h0 = h0_bf.abs().mean(dim=0)
    max_g0 = g_pres.max(dim=0).values
    mean_g0 = g_pres.mean(dim=0)
    # Penalty objective (minimize!): prioritize smallest max_|h0_bf16| and negative gate (mean_g0 in [-5.5, -2.0])
    loss_q = max_h0 + 2.0 * mean_h0 + 0.20 * F.relu(max_g0 + 1.5) + 0.05 * F.relu(-5.5 - mean_g0)
    q41 = int(loss_q.argmin().item())
    print(
        f"Selected L41 quietest neuron q*={q41}: max_|h0|={max_h0[q41].item():.6f}, mean_|h0|={mean_h0[q41].item():.6f}, "
        f"max_g0={max_g0[q41].item():.3f}, mean_g0={mean_g0[q41].item():.3f}",
        flush=True,
    )

    # Unembedding directions at L41
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

    u_joint = u_disp + 0.75 * u_ab
    u_joint = u_joint / u_joint.norm().clamp_min(1e-10)

    # Assemble ON (`unchecked/check @ call:`) and OFF (all edit, localize, verified, and thought/arg tokens!)
    on_disp_x, on_disp_g, on_disp_p, on_disp_meta = [], [], [], []
    off_disp_x, off_disp_g, off_disp_p = [], [], []
    y_norms = []

    for r in seq_records:
        xs = r["cap"]["x"]
        gs = r["cap"]["g0"][:, q41]
        ps = r["cap"]["p0"][:, q41]
        y_norms.append(r["cap"]["y0_norm"])
        cidx = r["colon_idx"]
        if r["role"] in ("unchecked", "check") and cidx is not None:
            on_disp_x.append(xs[cidx])
            on_disp_g.append(gs[cidx])
            on_disp_p.append(ps[cidx])
            on_disp_meta.append((r["issue"], r["role"]))
            m = torch.arange(xs.shape[0]) != cidx
            off_disp_x.append(xs[m])
            off_disp_g.append(gs[m])
            off_disp_p.append(ps[m])
        else:
            off_disp_x.append(xs)
            off_disp_g.append(gs)
            off_disp_p.append(ps)
            if cidx is not None:
                off_disp_x.append(xs[cidx : cidx + 1].repeat(10, 1))
                off_disp_g.append(gs[cidx : cidx + 1].repeat(10))
                off_disp_p.append(ps[cidx : cidx + 1].repeat(10))

    X_on_d = torch.stack(on_disp_x)
    G_on_d = torch.stack(on_disp_g)
    P_on_d = torch.stack(on_disp_p)
    X_off_d = torch.cat(off_disp_x, dim=0)
    G_off_d = torch.cat(off_disp_g, dim=0)
    P_off_d = torch.cat(off_disp_p, dim=0)
    print(f"L41 Dispatch tokens: N_on={X_on_d.shape[0]} (meta={on_disp_meta}), N_off={X_off_d.shape[0]}", flush=True)

    # Also build Joint ON (`unchecked/check @ call:` + `at_bare_pytest`) vs OFF!
    on_j_x = [X_on_d]
    on_j_g = [G_on_d]
    on_j_p = [P_on_d]
    off_j_x = [X_off_d]
    off_j_g = [G_off_d]
    off_j_p = [P_off_d]
    for r in bare_seqs:
        xs = r["cap"]["x"]
        gs = r["cap"]["g0"][:, q41]
        ps = r["cap"]["p0"][:, q41]
        pidx = r["pytest_idx"]
        cidx = r["colon_idx"]
        on_j_x.append(xs[pidx : pidx + 1])
        on_j_g.append(gs[pidx : pidx + 1])
        on_j_p.append(ps[pidx : pidx + 1])
        keep_off = torch.ones(xs.shape[0], dtype=torch.bool)
        keep_off[pidx] = False
        if cidx is not None:
            keep_off[cidx] = False
        off_j_x.append(xs[keep_off])
        off_j_g.append(gs[keep_off])
        off_j_p.append(ps[keep_off])

    X_on_j = torch.cat(on_j_x, dim=0)
    G_on_j = torch.cat(on_j_g, dim=0)
    P_on_j = torch.cat(on_j_p, dim=0)
    X_off_j = torch.cat(off_j_x, dim=0)
    G_off_j = torch.cat(off_j_g, dim=0)
    P_off_j = torch.cat(off_j_p, dim=0)

    # Solve with gentler regularization (lam_rel=0.015) so N_on=9..17 cleanly separates from verified@call:!
    oof_real = {}
    for iss in sorted({i for i, rl in on_disp_meta if rl == "unchecked"}):
        tr = torch.tensor([i != iss for i, _ in on_disp_meta])
        wg_tr = solve_primal_irls_gate(X_on_d[tr], G_on_d[tr], X_off_d, G_off_d, lam_rel=0.015, n_irls=6)
        for idx_i, (i_iss, rl) in enumerate(on_disp_meta):
            if i_iss == iss and rl == "unchecked":
                si = torch.dot(wg_tr, X_on_d[idx_i]).bfloat16()
                hi = F.gelu(G_on_d[idx_i].bfloat16() + si, approximate="tanh") * (P_on_d[idx_i].bfloat16() + si)
                oof_real[iss] = round(float(hi.item()), 3)

    wg_disp = solve_primal_irls_gate(X_on_d, G_on_d, X_off_d, G_off_d, lam_rel=0.015, n_irls=6)
    wg_joint = solve_primal_irls_gate(X_on_j, G_on_j, X_off_j, G_off_j, lam_rel=0.015, n_irls=6)

    def eval_gate(wg, Xo, Go, Po, Xf, Gf, Pf):
        so = (Xo @ wg).bfloat16()
        ho = (F.gelu(Go.bfloat16() + so, approximate="tanh") * (Po.bfloat16() + so)).float()
        sf = (Xf @ wg).bfloat16()
        hf = (F.gelu(Gf.bfloat16() + sf, approximate="tanh") * (Pf.bfloat16() + sf)).float()
        return {
            "h_on_mean": round(float(ho.mean()), 3),
            "h_on_min": round(float(ho.min()), 3),
            "off_bf16_zero_frac": round(float((hf == 0.0).float().mean()), 5),
            "off_h_max_abs": round(float(hf.abs().max()), 6),
        }

    rep = {
        "q41": q41,
        "q41_base_max_h0": round(float(max_h0[q41]), 6),
        "oof_real_unchecked": oof_real,
        "disp_gate": eval_gate(wg_disp, X_on_d, G_on_d, P_on_d, X_off_d, G_off_d, P_off_d),
        "joint_gate": eval_gate(wg_joint, X_on_j, G_on_j, P_on_j, X_off_j, G_off_j, P_off_j),
    }
    print("V8 Pure-L41 Diagnostics:", json.dumps(rep, indent=2), flush=True)

    w_down_q41 = layers[41].mlp.down_proj.weight.detach().float().cpu()[:, q41]
    y41_norm = sum(y_norms) / len(y_norms)
    ad_root = out_dir / "adapters"
    compiled = []

    # Zero control
    compiled.append(
        compile_v7_geglu(
            ROOT / "champion.zip",
            {41: {"q_idx": q41, "w_gate": wg_disp, "w_up": wg_disp, "b_down": -w_down_q41}},
            ad_root / "V8_zero_41",
            "zero",
        )
    )

    # Family 1: Pure L41 Causal-Readout GeGLU Transistors (ZERO KV-cache prefill effects!)
    for name, wg, u_vec, h_mean, rho in (
        ("V8_L41_disp_mid", wg_disp, u_disp, max(rep["disp_gate"]["h_on_mean"], 1.0), 0.45),
        ("V8_L41_disp_hi", wg_disp, u_disp, max(rep["disp_gate"]["h_on_mean"], 1.0), 0.85),
        ("V8_L41_joint_mid", wg_joint, u_joint, max(rep["joint_gate"]["h_on_mean"], 1.0), 0.50),
        ("V8_L41_joint_hi", wg_joint, u_joint, max(rep["joint_gate"]["h_on_mean"], 1.0), 0.95),
    ):
        g = rho * y41_norm / h_mean
        spec = {41: {"q_idx": q41, "w_gate": wg, "w_up": wg, "b_down": -w_down_q41 - g * u_vec}}
        for ctrl in ("candidate", "random"):
            compiled.append(compile_v7_geglu(ROOT / "champion.zip", spec, ad_root / f"{name}_{ctrl}", ctrl))

    # Family 2: Gentle 0.30x and 0.50x interpolations between Champion (5/15) and wsp32_iter01 (+requests_7315, edit_err=20)!
    for tag, alpha in (("a030", 0.30), ("a050", 0.50)):
        fac = out_dir / f"fac_{tag}.safetensors"
        save_file(
            {
                "35.A": ml2["35.A"],
                "35.B": (ml2["35.B"].float() * alpha).bfloat16(),
                "41.A": ml2["41.A"],
                "41.B": (ml2["41.B"].float() * alpha).bfloat16(),
            },
            str(fac),
        )
        for ctrl in ("candidate", "random"):
            lbl = f"V8_interp35_41_{tag}_{ctrl}"
            info = compile_explicit(ROOT / "champion.zip", fac, ad_root / lbl, ctrl, rank=1)
            compiled.append({"adapter": lbl, "sha256": info["adapter_sha256"], "control": ctrl, "alpha": alpha})

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": rep, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V8 adapters!", flush=True)


if __name__ == "__main__":
    main()
