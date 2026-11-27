"""Experiment C (WSP_CONTINUATION.md): 4096-token Champion-Active GeGLU Pre-Gate Switch on Disjoint Layers."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import CHAMP_LAYERS, DATA, ROOT, attach_adapter, build_branches, resolve_targeted_check

CAND_SWITCH_LAYERS = [23, 24, 28, 30, 32, 33, 35, 36]  # Disjoint from iter_04 [29, 31, 34, 37, 38]


def solve_ridge_gate(
    x_unch: torch.Tensor,   # [Nu, 2560]
    g_unch_q: torch.Tensor, # [Nu] mean base gate over Q
    x_pres: torch.Tensor,   # [Np, 2560]
    g_pres_q: torch.Tensor, # [Np] mean base gate over Q
    target_on: float = 2.5,
    target_off: float = -3.0,
    pres_weight: float = 4.0,
    lam: float = 40.0,
) -> torch.Tensor:
    """Closed-form dual ridge regression so base_gate + x @ w_gate hits +2.5 on unchecked and -3.0 on preservation."""
    y_on = (target_on - g_unch_q).clamp_min(0.5)
    y_off = (target_off - g_pres_q).clamp_max(0.0)
    w_p = pres_weight ** 0.5
    X = torch.cat([x_unch, w_p * x_pres], dim=0)
    y = torch.cat([y_on, w_p * y_off], dim=0)
    # Normalize regularization by mean squared norm of X
    scale = float(X.pow(2).sum(dim=1).mean().clamp_min(1.0))
    G = (X @ X.T) + (lam * scale / max(1, X.shape[0])) * torch.eye(X.shape[0])
    w = X.T @ torch.linalg.solve(G, y)
    return w


def compile_geglu_adapter(
    champion_zip: Path,
    layer_idx: int,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    q_idx: torch.Tensor,
    b_down: torch.Tensor,
    out_dir: Path,
    control: str = "candidate",
    seed: int = 32,
) -> dict:
    assert layer_idx not in CHAMP_LAYERS, f"Expected disjoint switch layer, got {layer_idx}"
    with zipfile.ZipFile(champion_zip) as z:
        blob = z.read("adapters/wsp_r1/adapter_model.safetensors")
        cfg = json.loads(z.read("adapters/wsp_r1/adapter_config.json"))
    champ = load(blob)
    layers = sorted(CHAMP_LAYERS | {layer_idx})
    gen = torch.Generator().manual_seed(seed)
    tensors = {}

    for l in layers:
        base_p = f"base_model.model.model.language_model.layers.{l}.mlp"
        ga = torch.zeros(1, 2560, dtype=torch.bfloat16)
        gb = torch.zeros(10240, 1, dtype=torch.bfloat16)
        ua = torch.zeros(1, 2560, dtype=torch.bfloat16)
        ub = torch.zeros(10240, 1, dtype=torch.bfloat16)
        da = torch.zeros(1, 10240, dtype=torch.bfloat16)
        db = torch.zeros(2560, 1, dtype=torch.bfloat16)

        if l in CHAMP_LAYERS:
            da[:] = champ[f"{base_p}.down_proj.lora_A.weight"]
            db[:] = champ[f"{base_p}.down_proj.lora_B.weight"]
        elif l == layer_idx and control != "zero":
            wg = w_gate.to(torch.bfloat16)
            wu = w_up.to(torch.bfloat16)
            bd = b_down.to(torch.bfloat16)
            mask_q = torch.zeros(10240, dtype=torch.bfloat16)
            mask_q[q_idx] = 1.0
            read_q = mask_q / float(q_idx.numel())
            if control == "random":
                wg = wg[torch.randperm(2560, generator=gen)]
                wu = wu[torch.randperm(2560, generator=gen)]
                perm = torch.randperm(10240, generator=gen)
                mask_q = mask_q[perm]
                read_q = read_q[perm]
                bd = bd[torch.randperm(2560, generator=gen)] * (2 * torch.randint(2, (2560,), generator=gen) - 1)
            ga[0] = wg
            gb[:, 0] = mask_q
            ua[0] = wu
            ub[:, 0] = mask_q
            da[0] = read_q
            db[:, 0] = bd

        tensors[f"{base_p}.gate_proj.lora_A.weight"] = ga
        tensors[f"{base_p}.gate_proj.lora_B.weight"] = gb
        tensors[f"{base_p}.up_proj.lora_A.weight"] = ua
        tensors[f"{base_p}.up_proj.lora_B.weight"] = ub
        tensors[f"{base_p}.down_proj.lora_A.weight"] = da
        tensors[f"{base_p}.down_proj.lora_B.weight"] = db

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
    return {"adapter": out_dir.name, "layer": layer_idx, "control": control, "sha256": hashlib.sha256(w_path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harvest-dir", type=Path, required=True)
    p.add_argument("--native-replays", type=Path, required=True)
    args = p.parse_args()

    hdir = args.harvest_dir
    cases = json.loads((hdir / "cases.json").read_text())
    native = {r["prompt_sha"]: r for r in json.loads(args.native_replays.read_text())}

    assets = json.loads((ROOT / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        assets["model"], dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    layers = getattr(model.model, "language_model", model.model).layers

    with zipfile.ZipFile(ROOT / "champion.zip") as z:
        champ = load(z.read("adapters/wsp_r1/adapter_model.safetensors"))
    attach_adapter(layers, champ)

    header_toks = tokenizer.encode("<|channel>thought\n", add_special_tokens=False)

    def collect_mlp(tokens: list[int], span: slice):
        rec = {}
        hooks = []
        for li in CAND_SWITCH_LAYERS:
            def mk_g(idx=li):
                def _h(mod, inp, out):
                    rec[(idx, "x")] = inp[0][0, span].float().mean(0).cpu()
                    rec[(idx, "g")] = out[0, span].float().mean(0).cpu()
                return _h
            def mk_u(idx=li):
                def _h(mod, inp, out):
                    rec[(idx, "up")] = out[0, span].float().mean(0).cpu()
                return _h
            def mk_d(idx=li):
                def _h(mod, inp, out):
                    rec[(idx, "y")] = out[0, span].float().mean(0).cpu()
                return _h
            hooks.append(layers[li].mlp.gate_proj.register_forward_hook(mk_g()))
            hooks.append(layers[li].mlp.up_proj.register_forward_hook(mk_u()))
            hooks.append(layers[li].mlp.down_proj.register_forward_hook(mk_d()))
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([tokens], device=model.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        return rec

    unch_items = []
    pres_items = []
    dy_by_layer = {li: [] for li in CAND_SWITCH_LAYERS}

    # Collect preservation states (verified + edit + localize, including mid-edit token spans!)
    seen_p = set()
    for c in cases:
        if c["role"] not in ("verified", "edit", "localize"):
            continue
        key = (c["issue"], c["role"])
        if key in seen_p:
            continue
        seen_p.add(key)
        pref = c["tokens"][-3904:]
        toks = pref + header_toks
        sp = slice(len(pref), len(toks))
        pres_items.append({"issue": c["issue"], "role": c["role"], "mlp": collect_mlp(toks, sp)})
        # For edit cases, also include a mid-argument token state if native replay exists!
        act = native.get(c["prompt_sha"])
        if c["role"] == "edit" and act and act.get("tokens"):
            gen_toks = act["tokens"][:120]
            if len(pref) + len(gen_toks) <= 4096 and len(gen_toks) > 16:
                sp_mid = slice(len(pref) + len(gen_toks) - 12, len(pref) + len(gen_toks))
                pres_items.append({"issue": c["issue"], "role": "edit_arg_tokens", "mlp": collect_mlp(pref + gen_toks, sp_mid)})

    # Collect paired & genuine unchecked states
    for issue in sorted({c["issue"] for c in cases}):
        opts = [c for c in cases if c["issue"] == issue and c["role"] in ("unchecked", "check")]
        for case in sorted(opts, key=lambda c: (c["role"] != "unchecked", len(c["tokens"]))):
            act = native.get(case["prompt_sha"])
            if not act or act["prompt_tokens"] != len(case["tokens"]) or act["action"] != case["action"]:
                continue
            cmd = resolve_targeted_check(case)
            if not cmd:
                continue
            pref = case["tokens"][-3904:]
            branches = build_branches(tokenizer, act["text"], cmd)
            if any(len(pref) + len(s) > 4096 for s, _ in branches.values()):
                continue
            # Harvest thought-start prefix_end for switch detector AND thought window for dy steering direction
            toks_p0 = pref + header_toks
            mlp_p0 = collect_mlp(toks_p0, slice(len(pref), len(toks_p0)))
            if case["role"] == "unchecked":
                unch_items.append({"issue": issue, "mlp": mlp_p0})
            suf_u, win_u = branches["unchecked"]
            suf_m, win_m = branches["useful"]
            a_u, b_u = win_u["thought"]
            a_m, b_m = win_m["thought"]
            mlp_u = collect_mlp(pref + suf_u, slice(len(pref) + a_u, len(pref) + b_u))
            mlp_m = collect_mlp(pref + suf_m, slice(len(pref) + a_m, len(pref) + b_m))
            for li in CAND_SWITCH_LAYERS:
                dy_by_layer[li].append(mlp_u[(li, "y")] - mlp_m[(li, "y")])
            break

    print(f"GeGLU switch dataset: {len(unch_items)} genuine unchecked, {len(pres_items)} preservation, {len(dy_by_layer[CAND_SWITCH_LAYERS[0]])} dy pairs", flush=True)
    if len(unch_items) < 2:
        print("Need >= 2 genuine unchecked cases for leave-one-out GeGLU switch CV.", flush=True)
        return

    out_dir = hdir / "geglu_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_report = []

    for li in CAND_SWITCH_LAYERS:
        xu = torch.stack([it["mlp"][(li, "x")] for it in unch_items])
        gu = torch.stack([it["mlp"][(li, "g")] for it in unch_items])
        upu = torch.stack([it["mlp"][(li, "up")] for it in unch_items])
        xp = torch.stack([it["mlp"][(li, "x")] for it in pres_items])
        gp = torch.stack([it["mlp"][(li, "g")] for it in pres_items])

        # Select 16 quietest (consistently negative gate on preservation) neurons
        max_gp = gp.max(dim=0).values
        mean_gp = gp.mean(dim=0)
        quiet_score = -mean_gp - 0.5 * max_gp.clamp_min(-1.0)
        q_idx = quiet_score.topk(16).indices

        # Leave-one-unchecked-out CV!
        oof_unch_gelu = []
        for i in range(len(unch_items)):
            tr = torch.arange(len(unch_items)) != i
            wg = solve_ridge_gate(xu[tr], gu[tr][:, q_idx].mean(1), xp, gp[:, q_idx].mean(1))
            pred_g = gu[i, q_idx] + torch.dot(wg, xu[i])
            oof_unch_gelu.append(float(F.gelu(pred_g).mean()))

        wg_full = solve_ridge_gate(xu, gu[:, q_idx].mean(1), xp, gp[:, q_idx].mean(1))
        pres_gelu = F.gelu(gp[:, q_idx] + (xp @ wg_full).unsqueeze(1)).mean(1)
        oof_pos = sum(1 for v in oof_unch_gelu if v > 0.3) / len(oof_unch_gelu)
        ratio = (sum(oof_unch_gelu) / len(oof_unch_gelu)) / max(float(pres_gelu.abs().max()), 1e-4)
        row_cv = {
            "layer": li,
            "oof_unch_gelu": [round(v, 4) for v in oof_unch_gelu],
            "oof_pos_rate": round(oof_pos, 3),
            "pres_max_abs_gelu": round(float(pres_gelu.abs().max()), 5),
            "on_off_ratio": round(ratio, 2),
        }
        cv_report.append(row_cv)
        print("GeGLU CV:", row_cv, flush=True)

    cv_report.sort(key=lambda r: (r["oof_pos_rate"], r["on_off_ratio"]), reverse=True)
    compiled = []
    for top in cv_report[:2]:
        if top["oof_pos_rate"] < 0.65 or top["on_off_ratio"] < 3.0:
            continue
        li = top["layer"]
        xu = torch.stack([it["mlp"][(li, "x")] for it in unch_items])
        gu = torch.stack([it["mlp"][(li, "g")] for it in unch_items])
        upu = torch.stack([it["mlp"][(li, "up")] for it in unch_items])
        xp = torch.stack([it["mlp"][(li, "x")] for it in pres_items])
        gp = torch.stack([it["mlp"][(li, "g")] for it in pres_items])
        yu_norm = float(torch.stack([it["mlp"][(li, "y")] for it in unch_items]).norm(dim=1).mean())

        q_idx = (-gp.mean(0) - 0.5 * gp.max(0).values.clamp_min(-1.0)).topk(16).indices
        wg = solve_ridge_gate(xu, gu[:, q_idx].mean(1), xp, gp[:, q_idx].mean(1))

        # Ensure up_proj on Q is strictly positive (~ +2.0) when x in xu
        mu_x = xu.mean(0)
        mu_x = mu_x / mu_x.norm().clamp_min(1e-8)
        cur_up = upu[:, q_idx].mean()
        needed_up = max(0.5, float(2.0 - cur_up)) / float((xu @ mu_x).mean().clamp_min(1.0))
        wu = needed_up * mu_x

        h_active = (F.gelu(gu[:, q_idx] + (xu @ wg).unsqueeze(1)) * (upu[:, q_idx] + (xu @ wu).unsqueeze(1))).mean(1)
        u_dir = torch.stack(dy_by_layer[li]).mean(0)
        u_dir = u_dir / u_dir.norm().clamp_min(1e-8)

        mean_h = float(h_active.mean().clamp_min(0.1))
        for target_rho in (0.08, 0.14):
            gain = round(target_rho * yu_norm / mean_h, 4)
            base_name = f"GeGLU_L{li}_g{gain:.4f}"
            for ctrl in ("zero", "candidate", "random"):
                lbl = f"GeGLU_L{li}_zero" if ctrl == "zero" else f"{base_name}_{ctrl}"
                dest = out_dir / "adapters" / lbl
                if not dest.exists():
                    info = compile_geglu_adapter(ROOT / "champion.zip", li, wg, wu, q_idx, -gain * u_dir, dest, ctrl)
                    compiled.append(info)

    (out_dir / "geglu_report.json").write_text(json.dumps({"cv": cv_report, "compiled": compiled}, indent=2))
    print("Compiled GeGLU adapters:", len(compiled), flush=True)


if __name__ == "__main__":
    main()
