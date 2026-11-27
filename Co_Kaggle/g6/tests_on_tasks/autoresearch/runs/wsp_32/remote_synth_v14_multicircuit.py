"""Synthesize V14 Clean Immediate-State Multi-Circuit Adapters (`o_proj + down_proj` r=1 AND pure `down_proj` r=1).

Trained strictly on the 100% clean, zero-off-by-one immediate-tool-response states (`clean_v14`):
- Circuit 1 (Post-Edit Verification): Fires ONLY on `imm_edit_ok` + `post_edit_prem_submit` (never on `imm_read_full` or `imm_test_pass`!)
  - Thought-Start `self_attn.o_proj` on Global Full-Attention Layers [17, 23, 35] steers reasoning from premature submit / blind second edit -> targeted test plan!
  - Readout `L41` `down_proj` at `<|tool_call>call:` gently reinforces `run_command` over `submit_patch`.
- Circuit 2 (Experiment D — `FileEditError` -> `read_file` Recovery): Fires ONLY on `imm_edit_err` (`status:"error", error_type:"FileEditError"`)
  - Thought-Start `o_proj` on [23/35] + `L40` `down_proj` at `<|tool_call>call:` steers broken `edit_file` retry loops -> `read_file`!
- Circuit 3 (`L39` Anti-Bare-Pytest): Proven 17.76x SNR sparse natural readout at `command:<|"|>pytest` -> `pytest -q tests/...`.
"""
import gc
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
from recipe import compile_explicit
from synthesize_targeted_dispatch_v3 import find_subseq_end


def build_orthonormal_basis(vecs: list[torch.Tensor]) -> torch.Tensor:
    valid = [v.double() for v in vecs if float(v.norm()) > 1e-9]
    mat = torch.stack([v / v.norm() for v in valid], dim=1)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.float()


def solve_whitened_subspace_detector(
    X_on: torch.Tensor,                      # [N_on, D]
    off_groups: list[tuple[torch.Tensor, float]],
    contrast_vecs: list[torch.Tensor],
    k_pca: int = 8,
    lam: float = 1e-2,
) -> torch.Tensor:
    """Fit unit-norm linear detector `a` in shared contrast+PCA subspace: ON -> +1, EVERY OFF group -> 0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vecs = list(contrast_vecs)
    if k_pca > 0:
        all_off = torch.cat([g[0] for g in off_groups if g[0].shape[0] > 0], dim=0)
        sub_off = all_off[:: max(1, all_off.shape[0] // 256)]
        pool = torch.cat([X_on, sub_off], dim=0).double()
        _, _, Vh = torch.linalg.svd(pool - pool.mean(dim=0), full_matrices=False)
        for j in range(min(k_pca, Vh.shape[0])):
            vecs.append(Vh[j].float())
    U = build_orthonormal_basis(vecs).to(device, torch.float32)

    Zo = X_on.to(device, torch.float32) @ U
    n_on = max(1, Zo.shape[0])
    w_on = torch.full((n_on,), 1.0 / n_on, device=device)
    yo = torch.ones(n_on, device=device)

    Zf_l, wf_l = [], []
    for xg, wt in off_groups:
        if xg.shape[0] == 0:
            continue
        Zf_l.append(xg.to(device, torch.float32) @ U)
        wf_l.append(torch.full((xg.shape[0],), wt / xg.shape[0], device=device))
    Zf = torch.cat(Zf_l, dim=0)
    wf = torch.cat(wf_l, dim=0)

    z_scale = float(Zf.pow(2).sum(dim=1).mean().clamp_min(1e-4))
    eye = torch.eye(U.shape[1], device=device)
    lhs = (Zo.T * w_on) @ Zo + (Zf.T * wf) @ Zf + (lam * z_scale) * eye
    rhs = Zo.T @ (w_on * yo)
    alpha = torch.linalg.solve(lhs, rhs)
    a = (U @ alpha).cpu()
    return a / a.norm().clamp_min(1e-8)


def compile_oproj_downproj_r1(
    champion_zip: Path,
    oproj_specs: dict[int, tuple[torch.Tensor, torch.Tensor]],   # layer -> (a [D_attn], b [2560])
    down_specs: dict[int, tuple[torch.Tensor, torch.Tensor]],    # layer -> (a [10240], b [2560])
    oproj_in_dims: dict[int, int],
    out_dir: Path,
    control: str = "candidate",
    seed: int = 41,
) -> dict:
    """Compile r=1 PEFT adapter with target_modules=['o_proj', 'down_proj'], preserving champion down_proj verbatim."""
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
    out_dir = ROOT / "harvest32/v14_multicircuit"
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
    # Record exact in_features of o_proj across all 42 layers!
    oproj_in_dims = {}
    for idx_l in range(len(layers)):
        op_mod = layers[idx_l].self_attn.o_proj
        in_f = getattr(op_mod, "in_features", None)
        if in_f is None:
            w_sh = getattr(op_mod, "weight_shape", None)
            in_f = int(w_sh[1]) if w_sh is not None else int(op_mod.weight.shape[1])
        oproj_in_dims[idx_l] = int(in_f)

    o_layers = [17, 23, 35, 41]       # Global full-attention o_proj layers!
    d_layers = [35, 39, 40, 41]       # Disjoint late down_proj layers!
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    call_colon_seq = enc("<|tool_call>call:")
    prefix_pytest = enc('<|tool_call>call:run_command{command:<|"|>pytest')
    thought_hdr = enc("<|channel>thought\n")

    def harvest_record(prefix: list[int], suffix: list[int]):
        toks = prefix + suffix
        assert len(toks) <= 4096
        p_len = len(prefix)
        # Critical token positions:
        # - Turn-start / Thought-start window: [p_len - 1, p_len, ..., p_len + min(len(thought_hdr), len(suffix)-1)]
        ts_idx = sorted({max(0, p_len - 1), min(len(toks) - 1, p_len + len(thought_hdr) - 1)})
        ts_t = torch.tensor(ts_idx, device=dev, dtype=torch.long)

        colon_end = find_subseq_end(suffix, call_colon_seq)
        pytest_end = find_subseq_end(suffix, prefix_pytest)
        c_abs = (p_len + colon_end - 1) if colon_end is not None else None
        pt_abs = (p_len + pytest_end - 1) if pytest_end is not None else None

        # Sample 24 evenly spaced non-special tokens across prefill + suffix as OFF background!
        special_abs = set(ts_idx) | ({c_abs} if c_abs else set()) | ({pt_abs} if pt_abs else set())
        bg_stride = max(1, len(toks) // 24)
        bg_idx = sorted(i for i in range(8, len(toks), bg_stride) if i not in special_abs)[:24]
        bg_t = torch.tensor(bg_idx, device=dev, dtype=torch.long)

        cap = {"has_colon": c_abs is not None, "has_pytest": pt_abs is not None}
        hooks = []
        for li in o_layers:
            def mk_o(idx=li):
                def _h(mod, inp, out):
                    z = inp[0][0].detach()
                    y = out[0].detach()
                    cap[(idx, "oz_ts")] = z.index_select(0, ts_t).float().mean(0).to(torch.bfloat16).cpu()
                    cap[(idx, "oy_ts")] = y.index_select(0, ts_t).float().mean(0).to(torch.bfloat16).cpu()
                    cap[(idx, "oy_norm")] = float(y[p_len:].float().norm(dim=-1).mean().item())
                    cap[(idx, "oz_bg")] = z.index_select(0, bg_t).to(torch.bfloat16).cpu()
                    if c_abs is not None:
                        cap[(idx, "oz_col")] = z[c_abs].to(torch.bfloat16).cpu()
                        cap[(idx, "oy_col")] = y[c_abs].to(torch.bfloat16).cpu()
                return _h
            hooks.append(layers[li].self_attn.o_proj.register_forward_hook(mk_o()))

        for li in d_layers:
            def mk_d(idx=li):
                def _h(mod, inp, out):
                    h = inp[0][0].detach()
                    y = out[0].detach()
                    cap[(idx, "dy_norm")] = float(y[p_len:].float().norm(dim=-1).mean().item())
                    cap[(idx, "dh_bg")] = h.index_select(0, bg_t).to(torch.bfloat16).cpu()
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
        # Keep BOTH the thought-start head AND the tool-call tail if > 768 tokens!
        if len(full_suf) > 768:
            suf = full_suf[:128] + full_suf[-640:]
        else:
            suf = full_suf
        if len(suf) < 4:
            continue
        pref = c["tokens"][-(4096 - len(suf)) :]
        cap = harvest_record(pref, suf)
        records.append({
            "issue": c["issue"],
            "clean_role": c["clean_role"],
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

    # Pure weight-space unembedding directions
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)

    tok_run = enc("<|tool_call>call:run_command{")[len(call_colon_seq)]
    tok_sub = enc("<|tool_call>call:submit_patch{")[len(call_colon_seq)]
    tok_ed = enc("<|tool_call>call:edit_file{")[len(call_colon_seq)]
    tok_rd = enc("<|tool_call>call:read_file{")[len(call_colon_seq)]

    # Circuit 1 readout direction: suppress submit_patch & blind second edit_file, boost run_command!
    u_c1_disp = ((0.75 * lm_w[tok_sub] + 0.25 * lm_w[tok_ed]) - lm_w[tok_run]) * scale_w
    u_c1_disp = u_c1_disp / u_c1_disp.norm().clamp_min(1e-10)

    # Circuit 2 (Experiment D) readout direction: after FileEditError, suppress blind edit_file + submit_patch, boost read_file!
    u_c2_err = ((0.80 * lm_w[tok_ed] + 0.20 * lm_w[tok_sub]) - lm_w[tok_rd]) * scale_w
    u_c2_err = u_c2_err / u_c2_err.norm().clamp_min(1e-10)

    # Circuit 3 (L39 anti-bare-pytest) direction:
    sb = enc('<|tool_call>call:run_command{command:<|"|>pytest<|"|>}<tool_call|>')
    st1 = enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests/test_panel.py<|"|>}<tool_call|>')
    st2 = enc('<|tool_call>call:run_command{command:<|"|>pytest tests/test_panel.py<|"|>}<tool_call|>')
    u_c3_ab = (
        lm_w[sb[find_subseq_end(sb, prefix_pytest)]]
        - (0.60 * lm_w[st1[find_subseq_end(st1, prefix_pytest)]] + 0.40 * lm_w[st2[find_subseq_end(st2, prefix_pytest)]])
    ) * scale_w
    u_c3_ab = u_c3_ab / u_c3_ab.norm().clamp_min(1e-10)
    del lm_w, model, lm_root, layers
    gc.collect()
    torch.cuda.empty_cache()

    diagnostics = {"n_clean_records": len(records)}

    # =========================================================================
    # 1. Rank Global-Attention `o_proj` Thought-Start Detectors for Circuit 1 (`imm_edit_ok` -> check)!
    # =========================================================================
    o_c1_fits = {}
    c1_issues = sorted({r["issue"] for r in records if r["clean_role"] in ("imm_edit_ok", "post_edit_prem_submit")})
    for li in (17, 23, 35):
        def pack_o_c1(recs):
            on_z, ver_z, rd_z, err_z, init_z, bg_z = [], [], [], [], [], []
            healthy_y, unhealth_y, yn = [], [], []
            for r in recs:
                z_ts = r["cap"][(li, "oz_ts")].float()
                y_ts = r["cap"][(li, "oy_ts")].float()
                yn.append(r["cap"][(li, "oy_norm")])
                bg_z.append(r["cap"][(li, "oz_bg")].float())
                rl = r["clean_role"]
                if rl in ("imm_edit_ok", "post_edit_prem_submit"):
                    on_z.append(z_ts)
                    if r["nat_action"] == "run_command":
                        healthy_y.append(y_ts)
                    elif r["nat_action"] in ("submit_patch", "edit_file"):
                        unhealth_y.append(y_ts)
                elif rl == "imm_test_pass":
                    ver_z.append(z_ts.unsqueeze(0))
                elif rl in ("imm_read_full", "imm_read_trunc"):
                    rd_z.append(z_ts.unsqueeze(0))
                elif rl == "imm_edit_err":
                    err_z.append(z_ts.unsqueeze(0))
                else:
                    init_z.append(z_ts.unsqueeze(0))
            Xo = torch.stack(on_z)
            Xv = torch.cat(ver_z) if ver_z else torch.zeros(0, Xo.shape[1])
            Xr = torch.cat(rd_z) if rd_z else torch.zeros(0, Xo.shape[1])
            Xe = torch.cat(err_z) if err_z else torch.zeros(0, Xo.shape[1])
            Xi = torch.cat(init_z) if init_z else torch.zeros(0, Xo.shape[1])
            Xb = torch.cat(bg_z)
            mu_o = Xo.mean(0)
            cvecs = [
                mu_o - Xb.mean(0),
                mu_o - (Xv.mean(0) if Xv.shape[0] else Xb.mean(0)),
                mu_o - (Xr.mean(0) if Xr.shape[0] else Xb.mean(0)),
                mu_o - (Xe.mean(0) if Xe.shape[0] else Xb.mean(0)),
                mu_o,
            ]
            ogrps = [(Xv, 3.0), (Xr, 3.5), (Xe, 2.0), (Xi, 2.0), (Xb, 5.0)]
            # Empirical Thought-Start steering vector: push from unhealthy (submit/blind-edit) toward healthy (run_command)!
            u_ts = (torch.stack(healthy_y).mean(0) - torch.stack(unhealth_y).mean(0)) if (healthy_y and unhealth_y) else -u_c1_disp
            u_ts = u_ts / u_ts.norm().clamp_min(1e-8)
            return Xo, ogrps, cvecs, torch.cat([g[0] for g in ogrps if g[0].shape[0] > 0]), u_ts, sum(yn) / len(yn)

        oof_vals = []
        for iss in c1_issues:
            tr = [r for r in records if r["issue"] != iss]
            Xo_tr, og_tr, cv_tr, _, _, _ = pack_o_c1(tr)
            a_tr = solve_whitened_subspace_detector(Xo_tr, og_tr, cv_tr, k_pca=6)
            for r in records:
                if r["issue"] == iss and r["clean_role"] in ("imm_edit_ok", "post_edit_prem_submit"):
                    oof_vals.append(float(torch.dot(a_tr, r["cap"][(li, "oz_ts")].float())))

        Xo, og, cv, X_off_all, u_ts, yn = pack_o_c1(records)
        a_o = solve_whitened_subspace_detector(Xo, og, cv, k_pca=6)
        s_on = Xo @ a_o
        s_off = X_off_all @ a_o
        off_rms = float(s_off.pow(2).mean().sqrt().clamp_min(1e-6))
        oof_snr = (sum(oof_vals) / max(1, len(oof_vals))) / off_rms
        diagnostics[f"C1_oproj_L{li}_ts"] = {
            "n_on": int(Xo.shape[0]),
            "oof_pos_frac": round(sum(1 for v in oof_vals if v > 1.5 * off_rms) / max(1, len(oof_vals)), 3),
            "oof_snr": round(oof_snr, 2),
            "in_sample_snr": round(float(s_on.mean()) / off_rms, 2),
            "s_on_mean": round(float(s_on.mean()), 4),
            "s_off_rms": round(off_rms, 4),
        }
        o_c1_fits[li] = {"a": a_o, "b_dir": u_ts, "s_on_mean": float(s_on.mean().clamp_min(0.1)), "y_norm": yn, "oof_snr": oof_snr}

    # =========================================================================
    # 2. Fit Readout `down_proj` Detectors at `<|tool_call>call:`:
    #    - L41 (and L35): Circuit 1 (`imm_edit_ok` + `post_edit_prem_submit` @ `:` -> `run_command`)
    #    - L40: Circuit 2 (`imm_edit_err` @ `:` -> `read_file`, Experiment D!)
    #    - L39: Circuit 3 (`at_pytest` -> targeted `pytest -q tests/...`)
    # =========================================================================
    d_c1_fits = {}
    for li in (35, 41):
        def pack_d_c1(recs):
            on_h, ver_h, rd_h, err_h, bg_h, yn = [], [], [], [], [], []
            for r in recs:
                yn.append(r["cap"][(li, "dy_norm")])
                bg_h.append(r["cap"][(li, "dh_bg")].float())
                if not r["cap"]["has_colon"]:
                    continue
                hc = r["cap"][(li, "dh_col")].float()
                rl = r["clean_role"]
                if rl in ("imm_edit_ok", "post_edit_prem_submit"):
                    on_h.append(hc)
                elif rl == "imm_test_pass":
                    ver_h.append(hc.unsqueeze(0))
                elif rl in ("imm_read_full", "imm_read_trunc"):
                    rd_h.append(hc.unsqueeze(0))
                elif rl == "imm_edit_err":
                    err_h.append(hc.unsqueeze(0))
                else:
                    bg_h.append(hc.unsqueeze(0))
            Ho = torch.stack(on_h)
            Hv = torch.cat(ver_h) if ver_h else torch.zeros(0, 10240)
            Hr = torch.cat(rd_h) if rd_h else torch.zeros(0, 10240)
            He = torch.cat(err_h) if err_h else torch.zeros(0, 10240)
            Hb = torch.cat(bg_h)
            mu_o = Ho.mean(0)
            cvecs = [
                mu_o - Hb.mean(0),
                mu_o - (Hv.mean(0) if Hv.shape[0] else Hb.mean(0)),
                mu_o - (Hr.mean(0) if Hr.shape[0] else Hb.mean(0)),
                mu_o - (He.mean(0) if He.shape[0] else Hb.mean(0)),
                mu_o,
            ]
            ogrps = [(Hv, 3.5), (Hr, 4.0), (He, 2.5), (Hb, 5.0)]
            return Ho, ogrps, cvecs, torch.cat([g[0] for g in ogrps if g[0].shape[0] > 0]), Hv, Hr, sum(yn) / len(yn)

        oof_d = []
        for iss in c1_issues:
            tr = [r for r in records if r["issue"] != iss]
            Ho_tr, og_tr, cv_tr, _, _, _, _ = pack_d_c1(tr)
            a_tr = solve_whitened_subspace_detector(Ho_tr, og_tr, cv_tr, k_pca=10)
            for r in records:
                if r["issue"] == iss and r["clean_role"] in ("imm_edit_ok", "post_edit_prem_submit") and r["cap"]["has_colon"]:
                    oof_d.append(float(torch.dot(a_tr, r["cap"][(li, "dh_col")].float())))

        Ho, og, cv, H_off, Hv, Hr, yn = pack_d_c1(records)
        a_d = solve_whitened_subspace_detector(Ho, og, cv, k_pca=10)
        s_on = Ho @ a_d
        s_off = H_off @ a_d
        off_rms = float(s_off.pow(2).mean().sqrt().clamp_min(1e-6))
        ver_m = float((Hv @ a_d).abs().max()) if Hv.shape[0] else 0.0
        rd_m = float((Hr @ a_d).abs().max()) if Hr.shape[0] else 0.0
        diagnostics[f"C1_downproj_L{li}_colon"] = {
            "n_on": int(Ho.shape[0]),
            "oof_pos_frac": round(sum(1 for v in oof_d if v > 1.8 * off_rms) / max(1, len(oof_d)), 3),
            "oof_snr": round((sum(oof_d) / max(1, len(oof_d))) / off_rms, 2),
            "in_sample_snr": round(float(s_on.mean()) / off_rms, 2),
            "s_on_mean": round(float(s_on.mean()), 4),
            "s_off_rms": round(off_rms, 4),
            "ver_colon_ratio": round(ver_m / float(s_on.mean()), 3),
            "read_to_edit_colon_ratio": round(rd_m / float(s_on.mean()), 3),
        }
        d_c1_fits[li] = {"a": a_d, "b_dir": -u_c1_disp, "s_on_mean": float(s_on.mean().clamp_min(0.1)), "y_norm": yn}

    # Circuit 2 (Experiment D): L40 `down_proj` at `:` on `imm_edit_err` -> `read_file`!
    def pack_d_c2_l40(recs):
        on_h, rd_h, ok_h, ver_h, bg_h, yn = [], [], [], [], [], []
        for r in recs:
            yn.append(r["cap"][(40, "dy_norm")])
            bg_h.append(r["cap"][(40, "dh_bg")].float())
            if not r["cap"]["has_colon"]:
                continue
            hc = r["cap"][(40, "dh_col")].float()
            rl = r["clean_role"]
            if rl == "imm_edit_err":
                on_h.append(hc)
            elif rl in ("imm_read_full", "imm_read_trunc"):
                rd_h.append(hc.unsqueeze(0))   # CRITICAL OFF: never suppress edit_file right after read_file!
            elif rl in ("imm_edit_ok", "post_edit_prem_submit"):
                ok_h.append(hc.unsqueeze(0))
            elif rl == "imm_test_pass":
                ver_h.append(hc.unsqueeze(0))
            else:
                bg_h.append(hc.unsqueeze(0))
        Ho = torch.stack(on_h)
        Hr = torch.cat(rd_h) if rd_h else torch.zeros(0, 10240)
        Hok = torch.cat(ok_h) if ok_h else torch.zeros(0, 10240)
        Hv = torch.cat(ver_h) if ver_h else torch.zeros(0, 10240)
        Hb = torch.cat(bg_h)
        mu_o = Ho.mean(0)
        cvecs = [
            mu_o - Hb.mean(0),
            mu_o - (Hr.mean(0) if Hr.shape[0] else Hb.mean(0)),
            mu_o - (Hok.mean(0) if Hok.shape[0] else Hb.mean(0)),
            mu_o - (Hv.mean(0) if Hv.shape[0] else Hb.mean(0)),
            mu_o,
        ]
        ogrps = [(Hr, 5.0), (Hok, 2.5), (Hv, 3.0), (Hb, 5.0)]
        return Ho, ogrps, cvecs, torch.cat([g[0] for g in ogrps if g[0].shape[0] > 0]), Hr, sum(yn) / len(yn)

    c2_issues = sorted({r["issue"] for r in records if r["clean_role"] == "imm_edit_err" and r["cap"]["has_colon"]})
    oof_c2 = []
    for iss in c2_issues:
        tr = [r for r in records if r["issue"] != iss]
        Ho_tr, og_tr, cv_tr, _, _, _ = pack_d_c2_l40(tr)
        a_tr = solve_whitened_subspace_detector(Ho_tr, og_tr, cv_tr, k_pca=10)
        for r in records:
            if r["issue"] == iss and r["clean_role"] == "imm_edit_err" and r["cap"]["has_colon"]:
                oof_c2.append(float(torch.dot(a_tr, r["cap"][(40, "dh_col")].float())))

    Ho40, og40, cv40, Hoff40, Hr40, yn40 = pack_d_c2_l40(records)
    a40_c2 = solve_whitened_subspace_detector(Ho40, og40, cv40, k_pca=10)
    s40_on = Ho40 @ a40_c2
    s40_off = Hoff40 @ a40_c2
    rms40 = float(s40_off.pow(2).mean().sqrt().clamp_min(1e-6))
    diagnostics["C2_downproj_L40_editerr_to_read"] = {
        "n_on": int(Ho40.shape[0]),
        "oof_pos_frac": round(sum(1 for v in oof_c2 if v > 1.8 * rms40) / max(1, len(oof_c2)), 3),
        "oof_snr": round((sum(oof_c2) / max(1, len(oof_c2))) / rms40, 2),
        "in_sample_snr": round(float(s40_on.mean()) / rms40, 2),
        "read_to_edit_colon_ratio": round(float((Hr40 @ a40_c2).abs().max()) / float(s40_on.mean()), 3),
    }

    # Circuit 3: L39 Anti-Bare-Pytest
    h39_on, h39_off, yn39 = [], [], []
    for r in records:
        yn39.append(r["cap"][(39, "dy_norm")])
        h39_off.append(r["cap"][(39, "dh_bg")].float())
        if r["cap"]["has_colon"] and not r["cap"]["has_pytest"]:
            h39_off.append(r["cap"][(39, "dh_col")].float().unsqueeze(0))
        if r["cap"]["has_pytest"]:
            h39_on.append(r["cap"][(39, "dh_pt")].float())
    for bc in bare_caps:
        h39_on.append(bc[(39, "dh_pt")].float())
        h39_off.append(bc[(39, "dh_bg")].float())
    H39_on = torch.stack(h39_on)
    H39_off = torch.cat(h39_off)
    mu39 = H39_on.mean(0)
    hard39 = H39_off[(H39_off @ (mu39 / mu39.norm().clamp_min(1e-8))).topk(min(96, H39_off.shape[0])).indices]
    a39_c3 = solve_whitened_subspace_detector(H39_on, [(hard39, 3.0), (H39_off, 5.0)], [mu39 - H39_off.mean(0), mu39 - hard39.mean(0), mu39], k_pca=10)
    s39_on = H39_on @ a39_c3
    s39_off = H39_off @ a39_c3
    diagnostics["C3_downproj_L39_anti_bare"] = {
        "n_on": int(H39_on.shape[0]),
        "snr": round(float(s39_on.mean() / s39_off.pow(2).mean().sqrt().clamp_min(1e-6)), 2),
    }

    print("V14 Multi-Circuit Clean-State Diagnostics:", json.dumps(diagnostics, indent=2), flush=True)

    # Pick top-2 global-attention o_proj layers for Thought-Start Circuit 1
    best_o_layers = sorted([17, 23, 35], key=lambda l: o_c1_fits[l]["oof_snr"], reverse=True)[:2]
    print("Top-2 Global-Attn o_proj layers by OOF SNR:", best_o_layers, flush=True)

    ad_root = out_dir / "adapters"
    compiled = []

    # --- Family 1: Pure `down_proj` Clean Multi-Circuit ({39, 40, 41} and {35, 39, 40, 41}) ---
    def add_down_mc(name: str, c1_dlayers: list[int], rho_c1: float, rho_c2: float, rho_c3: float):
        ft = {}
        for li in c1_dlayers:
            f = d_c1_fits[li]
            g = rho_c1 * f["y_norm"] / f["s_on_mean"]
            ft[f"{li}.A"] = f["a"].bfloat16()
            ft[f"{li}.B"] = (g * f["b_dir"]).bfloat16()
        if rho_c2 > 0:
            g40 = rho_c2 * (sum(yn40_l := [r["cap"][(40, "dy_norm")] for r in records]) / len(yn40_l)) / float(s40_on.mean().clamp_min(0.1))
            ft["40.A"] = a40_c2.bfloat16()
            ft["40.B"] = (-g40 * u_c2_err).bfloat16()
        if rho_c3 > 0:
            g39 = rho_c3 * (sum(yn39) / len(yn39)) / float(s39_on.mean().clamp_min(0.1))
            ft["39.A"] = a39_c3.bfloat16()
            ft["39.B"] = (-g39 * u_c3_ab).bfloat16()
        fac_p = out_dir / f"fac_{name}.safetensors"
        save_file(ft, str(fac_p))
        zlbl = f"V14_dzero_{'_'.join(sorted(k.split('.')[0] for k in ft if k.endswith('.A')))}"
        if not (ad_root / zlbl).exists():
            zi = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / zlbl, "zero", rank=1)
            compiled.append({"adapter": zlbl, "sha256": zi["adapter_sha256"], "control": "zero", "family": "down_mc"})
        for ctrl in ("candidate", "random"):
            lbl = f"{name}_{ctrl}"
            inf = compile_explicit(ROOT / "champion.zip", fac_p, ad_root / lbl, ctrl, rank=1)
            compiled.append({"adapter": lbl, "sha256": inf["adapter_sha256"], "control": ctrl, "family": "down_mc"})

    add_down_mc("V14_dmc_39_40_41_gentle", [41], rho_c1=0.22, rho_c2=0.18, rho_c3=0.22)
    add_down_mc("V14_dmc_39_40_41_mid", [41], rho_c1=0.34, rho_c2=0.28, rho_c3=0.30)
    add_down_mc("V14_dmc_35_39_40_41_mid", [35, 41], rho_c1=0.20, rho_c2=0.24, rho_c3=0.28)

    # --- Family 2: Hybrid Global-Attn Thought-Start `o_proj` + Late Readout `down_proj` (r=1) ---
    def add_hybrid_oproj(name: str, rho_o: float, rho_c1_d: float, rho_c2_d: float, rho_c3_d: float):
        ospecs = {}
        for li in best_o_layers:
            fo = o_c1_fits[li]
            go = rho_o * fo["y_norm"] / fo["s_on_mean"]
            ospecs[li] = (fo["a"], go * fo["b_dir"])
        dspecs = {}
        f41 = d_c1_fits[41]
        dspecs[41] = (f41["a"], (rho_c1_d * f41["y_norm"] / f41["s_on_mean"]) * f41["b_dir"])
        if rho_c2_d > 0:
            g40 = rho_c2_d * yn40 / float(s40_on.mean().clamp_min(0.1))
            dspecs[40] = (a40_c2, -g40 * u_c2_err)
        if rho_c3_d > 0:
            g39 = rho_c3_d * (sum(yn39) / len(yn39)) / float(s39_on.mean().clamp_min(0.1))
            dspecs[39] = (a39_c3, -g39 * u_c3_ab)
        zlbl = "V14_hybrid_zero"
        if not (ad_root / zlbl).exists():
            compiled.append(compile_oproj_downproj_r1(ROOT / "champion.zip", ospecs, dspecs, oproj_in_dims, ad_root / zlbl, "zero"))
        for ctrl in ("candidate", "random"):
            compiled.append(compile_oproj_downproj_r1(ROOT / "champion.zip", ospecs, dspecs, oproj_in_dims, ad_root / f"{name}_{ctrl}", ctrl))

    add_hybrid_oproj("V14_hyb_oproj_gentle", rho_o=0.14, rho_c1_d=0.18, rho_c2_d=0.18, rho_c3_d=0.22)
    add_hybrid_oproj("V14_hyb_oproj_mid", rho_o=0.24, rho_c1_d=0.26, rho_c2_d=0.26, rho_c3_d=0.28)
    add_hybrid_oproj("V14_hyb_oproj_hi", rho_o=0.36, rho_c1_d=0.34, rho_c2_d=0.30, rho_c3_d=0.32)

    (out_dir / "manifest.json").write_text(json.dumps({"diagnostics": diagnostics, "compiled": compiled}, indent=2))
    print(f"Compiled {len(compiled)} V14 Multi-Circuit Clean-State adapters!", flush=True)


if __name__ == "__main__":
    main()
