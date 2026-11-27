"""Synthesize v3 r=1 disjoint adapters: {35,40,41} unchecked->run_command dispatch + {36/39} surgical anti-bare-pytest targeter."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from harvest_and_rank_v2 import (
    CHAMP_LAYERS,
    ROOT,
    attach_adapter,
    harvest_forward,
    resolve_targeted_check,
)
from synthesize_multilayer_v2 import compute_unembed_contrast, fit_window_selective


def find_subseq_end(seq: list[int], sub: list[int]) -> int | None:
    """Return index of last token of `sub` inside `seq` (exclusive slice end), or None."""
    m = len(sub)
    for i in range(len(seq) - m + 1):
        if seq[i : i + m] == sub:
            return i + m
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harvest-dir", type=Path, required=True)
    p.add_argument("--native-replays", type=Path, required=True)
    args = p.parse_args()

    hdir = args.harvest_dir
    out_dir = hdir / "v3_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = torch.load(hdir / "summaries_v2.pt", map_location="cpu", weights_only=False)
    paired_rows = saved["paired"]
    pres_multi = torch.load(hdir / "pres_multiwindow.pt", map_location="cpu", weights_only=False)
    cases = json.loads((hdir / "cases.json").read_text())
    native = {r["prompt_sha"]: r for r in json.loads(args.native_replays.read_text())}

    assets = json.loads((ROOT / "assets.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        assets["model"], dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    lm_root = getattr(model.model, "language_model", model.model)
    layers = lm_root.layers

    with zipfile.ZipFile(ROOT / "champion.zip") as z:
        blob = z.read("adapters/wsp_r1/adapter_model.safetensors")
        assert hashlib.sha256(blob).hexdigest() == "8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815"
        champ = load(blob)
    attach_adapter(layers, champ)

    with torch.inference_mode():
        model(input_ids=torch.tensor([[tokenizer.bos_token_id]], device=model.device), use_cache=False, logits_to_keep=1)

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

    # 1. Inspect exact tokenization of `<|tool_call>call:run_command{command:<|"|>pytest` vs `pytest -q tests/test_foo.py<|"|>}`
    sample_bare = enc('<|tool_call>call:run_command{command:<|"|>pytest<|"|>}<tool_call|>')
    sample_targ = enc('<|tool_call>call:run_command{command:<|"|>pytest -q tests/test_panel.py<|"|>}<tool_call|>')
    prefix_pytest = enc('<|tool_call>call:run_command{command:<|"|>pytest')
    quote_tok_id = enc('<|"|>')[0]

    # Verify exact next token after `pytest` in sample_bare vs sample_targ
    end_b = find_subseq_end(sample_bare, prefix_pytest)
    end_t = find_subseq_end(sample_targ, prefix_pytest)
    assert end_b is not None and end_t is not None, f"Tokenization mismatch: {sample_bare} vs {prefix_pytest}"
    bare_next_tok = sample_bare[end_b]
    targ_next_tok_q = sample_targ[end_t]
    sample_targ2 = enc('<|tool_call>call:run_command{command:<|"|>pytest tests/test_panel.py<|"|>}<tool_call|>')
    end_t2 = find_subseq_end(sample_targ2, prefix_pytest)
    targ_next_tok_tests = sample_targ2[end_t2]

    print(
        f"Exact tokens after `pytest`: bare={bare_next_tok} ({tokenizer.decode([bare_next_tok])!r}), "
        f"targ_q={targ_next_tok_q} ({tokenizer.decode([targ_next_tok_q])!r}), "
        f"targ_tests={targ_next_tok_tests} ({tokenizer.decode([targ_next_tok_tests])!r}), "
        f"quote_id={quote_tok_id}",
        flush=True,
    )

    # Pure weight-space unembedding direction at `pytest` token:
    # u_anti_bare points toward `bare_next_tok` (<|"|>) minus (` -q` + ` tests/test_`),
    # so writing -gain * u_anti_bare suppresses bare `<|"|>` and boosts ` -q` / ` tests`!
    lm_w = model.get_output_embeddings().weight.detach().float().cpu()
    norm_w = lm_root.norm.weight.detach().float().cpu()
    scale_w = norm_w if norm_w.mean().abs() > 0.25 else (1.0 + norm_w)
    w_neg_bare = lm_w[bare_next_tok]
    w_pos_targ = 0.60 * lm_w[targ_next_tok_q] + 0.40 * lm_w[targ_next_tok_tests]
    u_anti_bare = (w_neg_bare - w_pos_targ) * scale_w
    u_anti_bare = u_anti_bare / u_anti_bare.norm().clamp_min(1e-10)

    # 2. Harvest exact states at `at_bare_pytest` (ON) vs `at_targeted_end_before_quote` & `at_edit_before_quote` (CRITICAL OFF)!
    anti_bare_cache = hdir / "anti_bare_states.pt"
    if anti_bare_cache.is_file():
        ab_data = torch.load(anti_bare_cache, map_location="cpu", weights_only=False)
    else:
        on_rows = []
        off_quote_rows = []

        # Collect ON (`...command:<|"|>pytest`) and OFF (`...pytest -q tests/test_<stem>.py` right before `<|"|>`) on discovery snapshots
        for issue in sorted({c["issue"] for c in cases}):
            opts = [c for c in cases if c["issue"] == issue and c["role"] in ("unchecked", "check")]
            for case in sorted(opts, key=lambda c: (c["role"] != "unchecked", len(c["tokens"]))):
                act = native.get(case["prompt_sha"])
                if not act or act["prompt_tokens"] != len(case["tokens"]):
                    continue
                cmd = resolve_targeted_check(case)
                if not cmd:
                    continue
                if not cmd.startswith("pytest"):
                    stem = Path(case.get("filepath") or "test.py").stem.lstrip("_")
                    cmd_py = f"pytest -q tests/test_{stem}.py"
                else:
                    cmd_py = cmd

                thought_txt = "I will run pytest to verify the changes before submitting."
                suf_str = f"<|channel>thought\n{thought_txt}\n<channel|><|tool_call>call:run_command{{command:<|\"|>{cmd_py}<|\"|>}}<tool_call|>"
                suf_toks = enc(suf_str)
                pos_pytest = find_subseq_end(suf_toks, prefix_pytest)
                # Position right before the closing `<|"|>` of the targeted command!
                quote_positions = [idx for idx, t in enumerate(suf_toks) if t == quote_tok_id]
                if pos_pytest is None or len(quote_positions) < 2:
                    continue
                pos_before_close_quote = quote_positions[-1]  # exclusive end = token right before closing `<|"|>`!
                wins = {
                    "at_pytest": (pos_pytest - 1, pos_pytest),
                    "before_close_quote": (pos_before_close_quote - 1, pos_before_close_quote),
                }
                pref = case["tokens"][-(4096 - len(suf_toks)):]
                hw = harvest_forward(model, layers, pref, suf_toks, wins)
                on_rows.append({"issue": issue, "h": hw["at_pytest"][0], "y": hw["at_pytest"][1]})
                off_quote_rows.append({"issue": issue, "kind": "cmd_close_quote", "h": hw["before_close_quote"][0], "y": hw["before_close_quote"][1]})
                break

        # Also collect exact pre-closing-quote tokens inside `edit_file` calls so `<|"|>` in edit_file is 100% protected!
        seen_ed = set()
        for c in cases:
            if c["role"] != "edit" or c["issue"] in seen_ed:
                continue
            act = native.get(c["prompt_sha"])
            if not act or not act.get("text"):
                continue
            suf_toks = enc(act["text"])[:256]
            q_pos = [idx for idx, t in enumerate(suf_toks) if t == quote_tok_id and idx >= 4]
            if len(q_pos) < 2:
                continue
            seen_ed.add(c["issue"])
            # Take up to 2 pre-quote positions (e.g. closing quote of filepath and old_string/new_string)
            wins = {f"ed_q{k}": (qp - 1, qp) for k, qp in enumerate(q_pos[1::2][:2])}
            if not wins:
                continue
            pref = c["tokens"][-(4096 - len(suf_toks)):]
            hw = harvest_forward(model, layers, pref, suf_toks, wins)
            for k, (hq, yq) in hw.items():
                off_quote_rows.append({"issue": c["issue"], "kind": f"edit_{k}", "h": hq, "y": yq})

        ab_data = {"on_pytest": on_rows, "off_quotes": off_quote_rows}
        torch.save(ab_data, anti_bare_cache)

    on_pytest = ab_data["on_pytest"]
    off_quotes = ab_data["off_quotes"]
    print(f"Anti-bare-pytest dataset: {len(on_pytest)} ON (`at_pytest`) vs {len(off_quotes)} pre-quote OFF + {len(pres_multi)} pres states", flush=True)

    # 3. Fit & LOO-CV evaluate anti-bare-pytest detector on disjoint late layers [33, 36, 39]
    ab_cand_layers = [33, 36, 39]
    ab_rankings = []
    ab_fits = {}
    n_ab = len(on_pytest)

    for l_idx in ab_cand_layers:
        w_eff = layers[l_idx].mlp.down_proj.weight.detach().float().cpu()
        h_on = torch.stack([r["h"][l_idx].float() for r in on_pytest])
        y_on = torch.stack([r["y"][l_idx].float() for r in on_pytest])
        h_off_q = torch.stack([r["h"][l_idx].float() for r in off_quotes])
        h_off_all = torch.cat([h_off_q, torch.stack([r["h"][l_idx].float() for r in pres_multi])], dim=0)

        oof_on = []
        for i in range(n_ab):
            tr = torch.arange(n_ab) != i
            _, row_tr, sel_tr = fit_window_selective(
                h_on[tr], u_anti_bare, h_off_all, h_off_q, w_eff, k=16, alpha_off=1.2, lam_ridge=4.0
            )
            oof_on.append(float(h_on[i] @ row_tr) if sel_tr.numel() > 0 else -1.0)

        _, row_ab, sel_ab = fit_window_selective(
            h_on, u_anti_bare, h_off_all, h_off_q, w_eff, k=16, alpha_off=1.2, lam_ridge=4.0
        )
        if sel_ab.numel() == 0:
            continue
        sig_on = h_on @ row_ab
        sig_q = h_off_q @ row_ab
        sig_all = h_off_all @ row_ab
        oof_pos = sum(1 for v in oof_on if v > 0) / max(1, n_ab)
        on_mean = float(sig_on.mean())
        q_max = float(sig_q.abs().max().clamp_min(1e-6))
        off_rms = float(sig_all.pow(2).mean().sqrt().clamp_min(1e-6))

        rec = {
            "layer": l_idx,
            "k": int(sel_ab.numel()),
            "oof_pos_rate": round(oof_pos, 3),
            "on_pytest_mean": round(on_mean, 4),
            "on_pytest_min": round(float(sig_on.min()), 4),
            "pre_quote_max_abs": round(q_max, 4),
            "pytest_vs_close_quote_ratio": round(on_mean / q_max, 3),
            "selectivity_rms": round(on_mean / off_rms, 3),
        }
        ab_rankings.append(rec)
        ab_fits[l_idx] = {
            "u": u_anti_bare,
            "row": row_ab,
            "y_norm": float(y_on.norm(dim=1).mean()),
            "on_act": on_mean,
            "rec": rec,
        }

    ab_rankings.sort(key=lambda r: (r["oof_pos_rate"], r["pytest_vs_close_quote_ratio"], r["selectivity_rms"]), reverse=True)
    print("Anti-bare-pytest layer rankings:", json.dumps(ab_rankings, indent=2), flush=True)

    # 4. Fit unchecked->run_command tool_pre dispatch on disjoint layers [35, 40, 41]
    u_unembed_dispatch = compute_unembed_contrast(model, tokenizer)
    tool_rows = [r for r in paired_rows if r["window"] == "tool_pre"]
    tp_fits = {}
    for l_idx in (35, 40, 41):
        w_eff = layers[l_idx].mlp.down_proj.weight.detach().float().cpu()
        hp_all = torch.cat(
            [
                torch.stack([r["h"][l_idx].float() for r in pres_multi]),
                torch.stack([r["h"][l_idx].float() for r in off_quotes]),
            ],
            dim=0,
        )
        hp_crit = torch.stack([
            r["h"][l_idx].float()
            for r in pres_multi
            if r["window"] == "tool_pre" or r["role"] in ("verified", "edit")
        ])
        hu_tp = torch.stack([r["unchecked_h"][l_idx].float() for r in tool_rows])
        yu_tp = torch.stack([r["unchecked_y"][l_idx].float() for r in tool_rows])
        dy_tp = yu_tp - torch.stack([r["useful_y"][l_idx].float() for r in tool_rows])

        u_emp, row_tp, sel_tp = fit_window_selective(
            hu_tp, dy_tp.mean(0), hp_all, hp_crit, w_eff, k=16, alpha_off=1.0, lam_ridge=3.5
        )
        assert sel_tp.numel() > 0, f"Empty dispatch support on L{l_idx}"
        if l_idx >= 40:
            u_b = 0.60 * u_emp + 0.40 * (u_unembed_dispatch if torch.dot(u_emp, u_unembed_dispatch) >= 0 else -u_unembed_dispatch)
            u_b = u_b / u_b.norm().clamp_min(1e-10)
        else:
            u_b = u_emp
        tp_fits[l_idx] = {
            "u": u_b,
            "row": row_tp,
            "y_norm": float(yu_tp.norm(dim=1).mean()),
            "on_act": float((hu_tp @ row_tp).mean()),
        }

    best_ab_1 = ab_rankings[0]["layer"]
    best_ab_2 = [r["layer"] for r in ab_rankings[:2]]

    specs = []

    def add_spec(cname: str, tp_layers: list[int], rho_tp: float, ab_layers: list[int], rho_ab: float):
        assert not (set(tp_layers) & set(ab_layers)), "Dispatch and anti-bare layers must be disjoint"
        assert not ((set(tp_layers) | set(ab_layers)) & CHAMP_LAYERS), "Must be disjoint from champion"
        tensors = {}
        for li in tp_layers:
            f = tp_fits[li]
            g = rho_tp * f["y_norm"] / max(f["on_act"], 1e-6)
            tensors[f"{li}.A"] = f["row"].bfloat16()
            tensors[f"{li}.B"] = (-g * f["u"]).bfloat16()
        for li in ab_layers:
            f = ab_fits[li]
            g = rho_ab * f["y_norm"] / max(f["on_act"], 1e-6)
            tensors[f"{li}.A"] = f["row"].bfloat16()
            tensors[f"{li}.B"] = (-g * f["u"]).bfloat16()
        save_file(tensors, str(out_dir / f"{cname}.safetensors"))
        specs.append({
            "name": cname,
            "layers": sorted(tp_layers + ab_layers),
            "tp_layers": tp_layers,
            "rho_tp": rho_tp,
            "ab_layers": ab_layers,
            "rho_ab": rho_ab,
        })

    # 6 focused r=1 disjoint candidates spanning {35, 41} / {35, 40, 41} dispatch + surgical anti-bare-pytest!
    add_spec(f"V3_tp35_41_ab{best_ab_1}_mid", [35, 41], 0.14, [best_ab_1], 0.16)
    add_spec(f"V3_tp35_41_ab{best_ab_1}_hi", [35, 41], 0.16, [best_ab_1], 0.26)
    add_spec(f"V3_tp35_40_41_ab{best_ab_1}_mid", [35, 40, 41], 0.12, [best_ab_1], 0.18)
    add_spec(f"V3_tp35_40_41_ab{best_ab_1}_hi", [35, 40, 41], 0.15, [best_ab_1], 0.28)
    if len(best_ab_2) == 2:
        ab_tag = "_".join(map(str, best_ab_2))
        add_spec(f"V3_tp35_40_41_ab{ab_tag}_mid", [35, 40, 41], 0.13, best_ab_2, 0.16)
        add_spec(f"V3_tp35_41_ab{ab_tag}_hi", [35, 41], 0.15, best_ab_2, 0.22)

    manifest = {
        "anti_bare_rankings": ab_rankings,
        "candidates": specs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Synthesized {len(specs)} v3 targeted-dispatch + anti-bare-pytest r=1 candidates!", flush=True)


if __name__ == "__main__":
    main()
