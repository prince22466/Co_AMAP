"""Remote V4 synthesizer: exact ML2(35,41) + downstream L39/L40 post-edit cmd_open python3 -c targeter."""
import json
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load, load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import ROOT, attach_adapter, harvest_forward
from synthesize_multilayer_v2 import fit_window_selective
from synthesize_targeted_dispatch_v3 import find_subseq_end


def main():
    hdir = Path("/content/wsp/harvest32")
    out_dir = hdir / "v4_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    pres_multi = torch.load(hdir / "pres_multiwindow.pt", map_location="cpu", weights_only=False)
    cases = json.loads((hdir / "cases.json").read_text())
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
    # Attach L35 from ML2 so downstream L39/L40 sees exact ML2-active hidden states!
    attach_adapter(
        layers,
        {
            "base_model.model.model.language_model.layers.35.mlp.down_proj.lora_A.weight": ml2["35.A"].unsqueeze(0),
            "base_model.model.model.language_model.layers.35.mlp.down_proj.lora_B.weight": ml2["35.B"].unsqueeze(1),
        },
    )

    with torch.inference_mode():
        model(input_ids=torch.tensor([[tokenizer.bos_token_id]], device=model.device), use_cache=False, logits_to_keep=1)

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    cmd_open_seq = enc('<|tool_call>call:run_command{command:<|"|>')
    tok_py = enc('<|tool_call>call:run_command{command:<|"|>python3 -c "')[len(cmd_open_seq)]
    tok_pytest = enc('<|tool_call>call:run_command{command:<|"|>pytest')[len(cmd_open_seq)]
    print(
        f"cmd_open first tokens: py={tok_py} ({tokenizer.decode([tok_py])!r}) vs pytest={tok_pytest} ({tokenizer.decode([tok_pytest])!r})",
        flush=True,
    )

    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)
    # u_py points toward `pytest` - `python`, so -gain * u_py suppresses `pytest` and boosts `python`!
    u_py = (lm_w[tok_pytest] - lm_w[tok_py]) * scale_w
    u_py = u_py / u_py.norm().clamp_min(1e-10)

    on_rows = []
    for issue in sorted({c["issue"] for c in cases}):
        opts = [c for c in cases if c["issue"] == issue and c["role"] in ("unchecked", "check")]
        for case in sorted(opts, key=lambda c: (c["role"] != "unchecked", len(c["tokens"]))):
            suf = enc(
                '<|channel>thought\nI will now verify the fix before submitting.\n<channel|><|tool_call>call:run_command{command:<|"|>python3 -c "import sys"<|"|>}<tool_call|>'
            )
            pos = find_subseq_end(suf, cmd_open_seq)
            assert pos is not None
            pref = case["tokens"][-(4096 - len(suf)) :]
            hw = harvest_forward(model, layers, pref, suf, {"cmd_open": (pos - 1, pos)})
            on_rows.append({"issue": issue, "h": hw["cmd_open"][0], "y": hw["cmd_open"][1]})
            break

    off_crit_rows = []
    seen_loc = set()
    for c in cases:
        if c["role"] == "localize" and c["issue"] not in seen_loc:
            seen_loc.add(c["issue"])
            suf = enc(
                '<|channel>thought\nLet me search the codebase.\n<channel|><|tool_call>call:run_command{command:<|"|>grep -rn "def " .<|"|>}<tool_call|>'
            )
            pos = find_subseq_end(suf, cmd_open_seq)
            pref = c["tokens"][-(4096 - len(suf)) :]
            hw = harvest_forward(model, layers, pref, suf, {"loc_cmd_open": (pos - 1, pos)})
            off_crit_rows.append({"issue": c["issue"], "kind": "localize_cmd_open", "h": hw["loc_cmd_open"][0]})

    seen_ed = set()
    ed_open_seq = enc('<|tool_call>call:edit_file{filepath:<|"|>')
    for c in cases:
        if c["role"] == "edit" and c["issue"] not in seen_ed:
            seen_ed.add(c["issue"])
            suf = enc(
                '<|channel>thought\nLet me apply the edit.\n<channel|><|tool_call>call:edit_file{filepath:<|"|>a.py<|"|>,new_string:<|"|>x<|"|>,old_string:<|"|>y<|"|>}<tool_call|>'
            )
            pos = find_subseq_end(suf, ed_open_seq)
            pref = c["tokens"][-(4096 - len(suf)) :]
            hw = harvest_forward(model, layers, pref, suf, {"ed_open": (pos - 1, pos)})
            off_crit_rows.append({"issue": c["issue"], "kind": "edit_open", "h": hw["ed_open"][0]})

    print(
        f"V4 cmd_open dataset: {len(on_rows)} post-edit ON vs {len(off_crit_rows)} pre-edit/edit-open OFF + {len(pres_multi)} pres",
        flush=True,
    )

    py_rankings = []
    py_fits = {}
    n_on = len(on_rows)
    for l_idx in (39, 40):
        w_eff = layers[l_idx].mlp.down_proj.weight.detach().float().cpu()
        h_on = torch.stack([r["h"][l_idx].float() for r in on_rows])
        y_on = torch.stack([r["y"][l_idx].float() for r in on_rows])
        h_crit = torch.stack([r["h"][l_idx].float() for r in off_crit_rows])
        h_all_off = torch.cat([h_crit, torch.stack([r["h"][l_idx].float() for r in pres_multi])], dim=0)

        oof = []
        for i in range(n_on):
            tr = torch.arange(n_on) != i
            _, row_tr, sel_tr = fit_window_selective(
                h_on[tr], u_py, h_all_off, h_crit, w_eff, k=16, alpha_off=0.9, lam_ridge=3.0
            )
            oof.append(float(h_on[i] @ row_tr) if sel_tr.numel() > 0 else -1.0)

        _, row_full, sel_full = fit_window_selective(
            h_on, u_py, h_all_off, h_crit, w_eff, k=16, alpha_off=0.9, lam_ridge=3.0
        )
        if sel_full.numel() == 0:
            _, row_full, sel_full = fit_window_selective(
                h_on, u_py, h_all_off, h_crit, w_eff, k=16, alpha_off=0.5, lam_ridge=2.0
            )
        sig_on = h_on @ row_full
        sig_loc = torch.stack([r["h"][l_idx].float() for r in off_crit_rows if r["kind"] == "localize_cmd_open"]) @ row_full
        sig_off = h_all_off @ row_full
        on_mean = float(sig_on.mean())
        loc_rms = float(sig_loc.pow(2).mean().sqrt().clamp_min(1e-6))
        off_rms = float(sig_off.pow(2).mean().sqrt().clamp_min(1e-6))
        rec = {
            "layer": l_idx,
            "k": int(sel_full.numel()),
            "oof_pos_rate": round(sum(1 for v in oof if v > 0) / max(1, n_on), 3),
            "on_mean": round(on_mean, 4),
            "loc_cmd_rms": round(loc_rms, 4),
            "post_vs_loc_ratio": round(on_mean / loc_rms, 3),
            "selectivity_rms": round(on_mean / off_rms, 3),
        }
        py_rankings.append(rec)
        py_fits[l_idx] = {
            "u": u_py,
            "row": row_full,
            "y_norm": float(y_on.norm(dim=1).mean()),
            "on_act": max(on_mean, 1e-6),
        }

    print("V4 downstream L39/L40 rankings:", json.dumps(py_rankings, indent=2), flush=True)

    specs = []
    for l_idx in (39, 40):
        if py_fits[l_idx]["row"].abs().sum() == 0:
            continue
        for tag, rho_py in (("mid", 0.16), ("hi", 0.28)):
            f = py_fits[l_idx]
            g = rho_py * f["y_norm"] / f["on_act"]
            cname = f"V4_ml2_pyL{l_idx}_{tag}"
            tensors = {
                "35.A": ml2["35.A"],
                "35.B": ml2["35.B"],
                "41.A": ml2["41.A"],
                "41.B": ml2["41.B"],
                f"{l_idx}.A": f["row"].bfloat16(),
                f"{l_idx}.B": (-g * f["u"]).bfloat16(),
            }
            save_file(tensors, str(out_dir / f"{cname}.safetensors"))
            specs.append({"name": cname, "layers": sorted([35, l_idx, 41]), "py_layer": l_idx, "rho_py": rho_py})

    cname_g = "V4_ml2_35_41_g080"
    save_file(
        {
            "35.A": ml2["35.A"],
            "35.B": (ml2["35.B"].float() * 0.80).bfloat16(),
            "41.A": ml2["41.A"],
            "41.B": (ml2["41.B"].float() * 0.80).bfloat16(),
        },
        str(out_dir / f"{cname_g}.safetensors"),
    )
    specs.append({"name": cname_g, "layers": [35, 41], "py_layer": None, "rho_py": 0.0})

    (out_dir / "manifest.json").write_text(json.dumps({"rankings": py_rankings, "candidates": specs}, indent=2))
    print("Synthesized V4 candidates:", [s["name"] for s in specs], flush=True)


if __name__ == "__main__":
    main()
