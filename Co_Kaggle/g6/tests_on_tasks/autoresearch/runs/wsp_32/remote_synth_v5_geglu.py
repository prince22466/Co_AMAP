"""Synthesize V5 r=1 Exact-Cancellation GeGLU Transistors strictly on downstream layers {39, 40, 41}."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, ROOT, attach_adapter, build_branches, resolve_targeted_check
from synthesize_multilayer_v2 import build_pres_windows, compute_unembed_contrast
from synthesize_targeted_dispatch_v3 import find_subseq_end


def solve_dual_ridge(
    x_on: torch.Tensor,     # [N_on, 2560]
    g0_on: torch.Tensor,    # [N_on]
    x_off: torch.Tensor,    # [N_off, 2560]
    g0_off: torch.Tensor,   # [N_off]
    target_g_on: float = 3.2,
    target_g_off: float = -3.8,
    off_weight: float = 1.8,
    lam_rel: float = 0.035,
) -> torch.Tensor:
    """Class-balanced dual ridge regression so g0 + x @ w_gate hits +3.2 on ON and -3.8 on OFF."""
    y_on = (target_g_on - g0_on).clamp_min(1.2)
    y_off = (target_g_off - g0_off).clamp_max(-0.5)
    s_on = (1.0 / max(1, x_on.shape[0])) ** 0.5
    s_off = (off_weight / max(1, x_off.shape[0])) ** 0.5
    X = torch.cat([s_on * x_on, s_off * x_off], dim=0)
    y = torch.cat([s_on * y_on, s_off * y_off], dim=0)
    gram = X @ X.T
    # Trace of gram is ~ (1 + off_weight) * mean(||x||^2)
    trace_scale = float(gram.diagonal().sum().clamp_min(1.0))
    reg = gram + (lam_rel * trace_scale) * torch.eye(X.shape[0])
    return X.T @ torch.linalg.solve(reg, y)


def compile_geglu_multi(
    champion_zip: Path,
    layer_specs: dict[int, dict],  # layer -> {q_idx: int, w_gate, w_up, b_down}
    out_dir: Path,
    control: str = "candidate",
    seed: int = 35,
) -> dict:
    assert not (set(layer_specs) & CHAMP_LAYERS), f"Overlap with champion: {set(layer_specs) & CHAMP_LAYERS}"
    assert all(l > max(CHAMP_LAYERS) for l in layer_specs), f"Expected strictly downstream layers > 38, got {sorted(layer_specs)}"
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
    out_dir = hdir / "v5_geglu"
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

    target_layers = [39, 40, 41]
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    prefix_pytest = enc('<|tool_call>call:run_command{command:<|"|>pytest')
    quote_tok_id = enc('<|"|>')[0]

    def harvest_mlp_windows(prefix: list[int], suffix: list[int], wins: dict[str, tuple[int, int]]):
        toks = prefix + suffix
        assert len(toks) <= 4096
        bins = {w: {} for w in wins}
        hooks = []
        for li in target_layers:
            def mk_g(idx=li):
                def _h(mod, inp, out):
                    for wn, (a, b) in wins.items():
                        sp = slice(len(prefix) + a, len(prefix) + b)
                        bins[wn][(idx, "x")] = inp[0][0, sp].float().mean(0).cpu()
                        bins[wn][(idx, "g0")] = out[0, sp].float().mean(0).cpu()
                return _h
            def mk_u(idx=li):
                def _h(mod, inp, out):
                    for wn, (a, b) in wins.items():
                        sp = slice(len(prefix) + a, len(prefix) + b)
                        bins[wn][(idx, "p0")] = out[0, sp].float().mean(0).cpu()
                return _h
            def mk_d(idx=li):
                def _h(mod, inp, out):
                    for wn, (a, b) in wins.items():
                        sp = slice(len(prefix) + a, len(prefix) + b)
                        bins[wn][(idx, "y0")] = out[0, sp].float().mean(0).cpu()
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
        return bins

    cache_p = out_dir / "mlp_states_39_41.pt"
    if cache_p.is_file():
        st = torch.load(cache_p, map_location="cpu", weights_only=False)
    else:
        pres_states = []
        seen_p = set()
        for c in cases:
            if c["role"] not in ("verified", "edit", "localize"):
                continue
            key = (c["issue"], c["role"])
            if key in seen_p:
                continue
            seen_p.add(key)
            act = native.get(c.get("prompt_sha", ""))
            suf, wins = build_pres_windows(tokenizer, c["role"], act["text"] if act else None, c["action"], c.get("arguments") or {})
            pref = c["tokens"][-(4096 - len(suf)) :]
            hw = harvest_mlp_windows(pref, suf, wins)
            for wn, d in hw.items():
                pres_states.append({"issue": c["issue"], "role": c["role"], "window": wn, "mlp": d})

        unch_tp_states = []
        dy_tp = {li: [] for li in target_layers}
        bare_pytest_on = []
        close_quote_off = []

        for issue in sorted({c["issue"] for c in cases}):
            opts = [c for c in cases if c["issue"] == issue and c["role"] in ("unchecked", "check")]
            for case in sorted(opts, key=lambda c: (c["role"] != "unchecked", len(c["tokens"]))):
                act = native.get(case["prompt_sha"])
                if not act or act["prompt_tokens"] != len(case["tokens"]):
                    continue
                cmd = resolve_targeted_check(case)
                if not cmd:
                    continue
                pref = case["tokens"][-3904:]
                branches = build_branches(tokenizer, act["text"], cmd)
                if any(len(pref) + len(s) > 4096 for s, _ in branches.values()):
                    continue
                su, wu_w = branches["unchecked"]
                sm, wm_w = branches["useful"]
                du = harvest_mlp_windows(pref, su, {"tool_pre": wu_w["tool_pre"]})["tool_pre"]
                dm = harvest_mlp_windows(pref, sm, {"tool_pre": wm_w["tool_pre"]})["tool_pre"]
                unch_tp_states.append({"issue": issue, "observed_role": case["role"], "mlp": du})
                for li in target_layers:
                    dy_tp[li].append(du[(li, "y0")] - dm[(li, "y0")])

                # Also harvest bare pytest token vs before_close_quote for L39
                cmd_py = cmd if cmd.startswith("pytest") else f"pytest -q tests/test_{Path(case.get('filepath') or 'x.py').stem.lstrip('_')}.py"
                suf_ab = enc(f"<|channel>thought\nI will run pytest to verify the fix.\n<channel|><|tool_call>call:run_command{{command:<|\"|>{cmd_py}<|\"|>}}<tool_call|>")
                p_pt = find_subseq_end(suf_ab, prefix_pytest)
                q_pos = [idx for idx, t in enumerate(suf_ab) if t == quote_tok_id]
                if p_pt is not None and len(q_pos) >= 2:
                    p_cl = q_pos[-1]
                    dab = harvest_mlp_windows(case["tokens"][-(4096 - len(suf_ab)) :], suf_ab, {"at_pytest": (p_pt - 1, p_pt), "close_q": (p_cl - 1, p_cl)})
                    bare_pytest_on.append({"issue": issue, "mlp": dab["at_pytest"]})
                    close_quote_off.append({"issue": issue, "mlp": dab["close_q"]})
                break

        st = {
            "pres": pres_states,
            "unch_tp": unch_tp_states,
            "dy_tp": dy_tp,
            "bare_on": bare_pytest_on,
            "close_q": close_quote_off,
        }
        torch.save(st, cache_p)

    pres_states = st["pres"]
    unch_tp = st["unch_tp"]
    dy_tp = st["dy_tp"]
    bare_on = st["bare_on"]
    close_q = st["close_q"]
    print(f"Loaded MLP states: {len(unch_tp)} unch_tp, {len(pres_states)} pres, {len(bare_on)} bare_pytest", flush=True)

    # Select 1 quietest neuron q* per layer with g0 in [-3.2, -1.8] on ALL preservation states & small |p0|!
    def pick_quiet_neuron(li: int, off_mlps: list[dict]) -> int:
        g_mat = torch.stack([m[(li, "g0")] for m in off_mlps])  # [N_off, 10240]
        p_mat = torch.stack([m[(li, "p0")] for m in off_mlps])
        max_g = g_mat.max(dim=0).values
        mean_g = g_mat.mean(dim=0)
        max_abs_p = p_mat.abs().max(dim=0).values
        # Must be strictly negative across ALL preservation states (max_g <= -1.6) and not impossibly deep (mean_g >= -4.0)
        valid = (max_g <= -1.6) & (mean_g >= -4.2) & (max_abs_p <= 1.8)
        if not valid.any():
            valid = (max_g <= -1.2) & (mean_g >= -5.5)
        score = torch.where(valid, -max_g - 0.35 * max_abs_p - 0.15 * (mean_g + 2.6).abs(), torch.full_like(mean_g, -1e9))
        return int(score.argmax().item())

    # Fit L40 & L41 Unchecked Tool-Dispatch Transistors + LOO-CV!
    u_unembed_disp = compute_unembed_contrast(model, tokenizer)
    tp_transistors = {}
    cv_summary = {}

    for li in (40, 41):
        off_mlps = [r["mlp"] for r in pres_states] + [r["mlp"] for r in close_q]
        # Give extra weight to verified@tool_pre and edit states by duplicating them in X_off!
        crit_mlps = [r["mlp"] for r in pres_states if r["role"] in ("verified", "edit")]
        off_weighted = off_mlps + crit_mlps

        q_star = pick_quiet_neuron(li, off_mlps)
        x_on = torch.stack([r["mlp"][(li, "x")] for r in unch_tp])
        g0_on = torch.stack([r["mlp"][(li, "g0")][q_star] for r in unch_tp])
        p0_on = torch.stack([r["mlp"][(li, "p0")][q_star] for r in unch_tp])
        y0_norm = float(torch.stack([r["mlp"][(li, "y0")] for r in unch_tp]).norm(dim=1).mean())

        x_off = torch.stack([m[(li, "x")] for m in off_weighted])
        g0_off = torch.stack([m[(li, "g0")][q_star] for m in off_weighted])

        oof_h_on = []
        for i in range(len(unch_tp)):
            tr = torch.arange(len(unch_tp)) != i
            wg_tr = solve_dual_ridge(x_on[tr], g0_on[tr], x_off, g0_off, lam_rel=0.12)
            wu_tr = wg_tr  # Collinear w_up = w_gate guarantees p_new > 0 whenever g_new > 0!
            s_i = torch.dot(wg_tr, x_on[i])
            h_i = F.gelu(g0_on[i] + s_i, approximate="tanh") * (p0_on[i] + s_i)
            oof_h_on.append(float(h_i.item()))

        wg = solve_dual_ridge(x_on, g0_on, x_off, g0_off, lam_rel=0.12)
        wu = wg
        s_on = x_on @ wg
        h_on_act = F.gelu(g0_on + s_on, approximate="tanh") * (p0_on + s_on)

        # Evaluate exact OFF leak across ALL pres_states (verified, edit, localize)!
        x_pres = torch.stack([r["mlp"][(li, "x")] for r in pres_states])
        g0_pres = torch.stack([r["mlp"][(li, "g0")][q_star] for r in pres_states])
        p0_pres = torch.stack([r["mlp"][(li, "p0")][q_star] for r in pres_states])
        s_pres = x_pres @ wg
        h_pres_act = F.gelu(g0_pres + s_pres, approximate="tanh") * (p0_pres + s_pres)

        u_emp = torch.stack(dy_tp[li]).mean(0)
        u_emp = u_emp / u_emp.norm().clamp_min(1e-10)
        u_b = 0.60 * u_emp + 0.40 * (u_unembed_disp if torch.dot(u_emp, u_unembed_disp) >= 0 else -u_unembed_disp)
        u_b = u_b / u_b.norm().clamp_min(1e-10)

        w_down_q = layers[li].mlp.down_proj.weight.detach().float().cpu()[:, q_star]

        cv_summary[f"L{li}_dispatch"] = {
            "q_star": q_star,
            "g0_pres_max": round(float(g0_pres.max()), 4),
            "oof_h_on": [round(v, 3) for v in oof_h_on],
            "oof_pos_rate": round(sum(1 for v in oof_h_on if v > 0.5) / len(oof_h_on), 3),
            "full_h_on_mean": round(float(h_on_act.mean()), 4),
            "pres_h_max_abs": round(float(h_pres_act.abs().max()), 5),
            "on_off_ratio": round(float(h_on_act.mean() / h_pres_act.abs().max().clamp_min(1e-5)), 2),
        }
        tp_transistors[li] = {
            "q_idx": q_star,
            "w_gate": wg,
            "w_up": wu,
            "w_down_q": w_down_q,
            "u": u_b,
            "h_on_mean": float(h_on_act.mean().clamp_min(0.5)),
            "y0_norm": y0_norm,
        }

    # Fit L39 Anti-Bare-Pytest Transistor (`at_pytest` ON vs `close_q` + `pres_states` + `unch_tp` OFF!)
    sample_bare = enc('<|tool_call>call:run_command{command:<|"|>pytest<|"|>}<tool_call|>')
    sample_targ = enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests/test_panel.py<|"|>}<tool_call|>')
    sample_targ2 = enc('<|tool_call>call:run_command{command:<|"|>pytest tests/test_panel.py<|"|>}<tool_call|>')
    eb = find_subseq_end(sample_bare, prefix_pytest)
    et1 = find_subseq_end(sample_targ, prefix_pytest)
    et2 = find_subseq_end(sample_targ2, prefix_pytest)
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)
    u_ab = (lm_w[sample_bare[eb]] - (0.60 * lm_w[sample_targ[et1]] + 0.40 * lm_w[sample_targ2[et2]])) * scale_w
    u_ab = u_ab / u_ab.norm().clamp_min(1e-10)

    li39 = 39
    off39_mlps = [r["mlp"] for r in pres_states] + [r["mlp"] for r in close_q] + [r["mlp"] for r in unch_tp]
    off39_weighted = off39_mlps + [r["mlp"] for r in close_q] * 2
    q39 = pick_quiet_neuron(li39, off39_mlps)
    x39_on = torch.stack([r["mlp"][(li39, "x")] for r in bare_on])
    g39_on = torch.stack([r["mlp"][(li39, "g0")][q39] for r in bare_on])
    p39_on = torch.stack([r["mlp"][(li39, "p0")][q39] for r in bare_on])
    y39_norm = float(torch.stack([r["mlp"][(li39, "y0")] for r in bare_on]).norm(dim=1).mean())
    x39_off = torch.stack([m[(li39, "x")] for m in off39_weighted])
    g39_off = torch.stack([m[(li39, "g0")][q39] for m in off39_weighted])

    oof39 = []
    for i in range(len(bare_on)):
        tr = torch.arange(len(bare_on)) != i
        wg_tr = solve_dual_ridge(x39_on[tr], g39_on[tr], x39_off, g39_off, lam_rel=0.12)
        si = torch.dot(wg_tr, x39_on[i])
        oof39.append(float((F.gelu(g39_on[i] + si, approximate="tanh") * (p39_on[i] + si)).item()))

    wg39 = solve_dual_ridge(x39_on, g39_on, x39_off, g39_off, lam_rel=0.12)
    s39_on = x39_on @ wg39
    h39_on = F.gelu(g39_on + s39_on, approximate="tanh") * (p39_on + s39_on)
    x39_all_off = torch.stack([m[(li39, "x")] for m in off39_mlps])
    g39_all_off = torch.stack([m[(li39, "g0")][q39] for m in off39_mlps])
    p39_all_off = torch.stack([m[(li39, "p0")][q39] for m in off39_mlps])
    s39_off = x39_all_off @ wg39
    h39_off = F.gelu(g39_all_off + s39_off, approximate="tanh") * (p39_all_off + s39_off)
    w_down_q39 = layers[li39].mlp.down_proj.weight.detach().float().cpu()[:, q39]

    cv_summary["L39_anti_bare"] = {
        "q_star": q39,
        "oof_h_on": [round(v, 3) for v in oof39],
        "oof_pos_rate": round(sum(1 for v in oof39 if v > 0.5) / len(oof39), 3),
        "full_h_on_mean": round(float(h39_on.mean()), 4),
        "off_h_max_abs": round(float(h39_off.abs().max()), 5),
        "on_off_ratio": round(float(h39_on.mean() / h39_off.abs().max().clamp_min(1e-5)), 2),
    }
    print("V5 GeGLU Transistor CV Summary:", json.dumps(cv_summary, indent=2), flush=True)

    # Build 4 candidate adapters + zero + random controls!
    ad_root = out_dir / "adapters"
    shutil.rmtree(ad_root, ignore_errors=True)
    compiled = []

    def build_pair(name: str, rho_tp: float, rho_ab: float | None):
        lspecs = {}
        for li in (40, 41):
            t = tp_transistors[li]
            gain = rho_tp * t["y0_norm"] / t["h_on_mean"]
            # EXACT ALGEBRAIC CANCELLATION of -W_down[:, q*]!
            b_down = -t["w_down_q"] - gain * t["u"]
            lspecs[li] = {"q_idx": t["q_idx"], "w_gate": t["w_gate"], "w_up": t["w_up"], "b_down": b_down}
        if rho_ab is not None:
            gain39 = rho_ab * y39_norm / float(h39_on.mean().clamp_min(0.5))
            b39 = -w_down_q39 - gain39 * u_ab
            lspecs[39] = {"q_idx": q39, "w_gate": wg39, "w_up": wg39, "b_down": b39}

        z_lbl = f"GeGLU_zero_{'_'.join(map(str, sorted(lspecs)))}"
        if not (ad_root / z_lbl).exists():
            compiled.append(compile_geglu_multi(ROOT / "champion.zip", lspecs, ad_root / z_lbl, "zero"))
        for ctrl in ("candidate", "random"):
            lbl = f"{name}_{ctrl}"
            if not (ad_root / lbl).exists():
                compiled.append(compile_geglu_multi(ROOT / "champion.zip", lspecs, ad_root / lbl, ctrl))

    build_pair("GeGLU_40_41_mid", rho_tp=0.14, rho_ab=None)
    build_pair("GeGLU_40_41_hi", rho_tp=0.22, rho_ab=None)
    build_pair("GeGLU_39_40_41_mid", rho_tp=0.14, rho_ab=0.18)
    build_pair("GeGLU_39_40_41_hi", rho_tp=0.22, rho_ab=0.26)

    (out_dir / "manifest.json").write_text(json.dumps({"cv": cv_summary, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V5 GeGLU adapters!", flush=True)


if __name__ == "__main__":
    main()
