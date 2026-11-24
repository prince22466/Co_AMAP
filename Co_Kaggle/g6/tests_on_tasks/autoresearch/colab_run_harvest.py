import glob, importlib, json, os, shutil, subprocess, sys, time, gc
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["KAGGLE_CONFIG_DIR"] = "/content"

# Ensure Kaggle config directory is set
if Path("/content/kaggle.json").exists():
    os.environ["KAGGLE_CONFIG_DIR"] = "/content"
elif Path("/root/.kaggle/kaggle.json").exists():
    os.environ["KAGGLE_CONFIG_DIR"] = "/root/.kaggle"

print("=" * 70)
print("1. LOADING ASSETS & CONFIGURING ACCELERATE OFFLOAD")
print("=" * 70)

import kagglehub

whl_dir = Path(kagglehub.dataset_download("metric/gemma-4-developer-agent-wheelhouse"))
model_dir = Path(kagglehub.model_download("google/gemma-4/Other/gemma-4-e4b-it-qat-w4a16-ct/2"))
print(f"Model directory: {model_dir}")

for p in glob.glob("/usr/local/lib/python*/dist-packages/*cutlass*.pth"):
    try: os.unlink(p)
    except OSError: pass

tmp_whl = Path("/tmp/wheelhouse")
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in whl_dir.glob("*.whl"):
    if "cutlass" in w.name.lower(): continue
    target_name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
    target = tmp_whl / target_name
    if not target.exists():
        try: os.symlink(w, target)
        except OSError: shutil.copyfile(w, target)

wheels = sorted(
    str(w) for w in tmp_whl.glob("*.whl")
    if any(w.name.startswith(p) for p in ("transformers-", "tokenizers-", "safetensors-", "compressed_tensors-", "loguru-"))
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels], check=True)
importlib.invalidate_caches()

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
print(f"Initial CUDA check: available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
if torch.cuda.is_available():
    print(f"Initial Free VRAM: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

probes_path = Path("/content/dla_probes.json")
probes = json.loads(probes_path.read_text(encoding="utf-8"))
print(f"Loaded {sum(len(v) for v in probes.values())} decision probes.")

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
print(f"Loading Gemma 4 E4B on GPU (dtype={dtype}, device_map='cuda')...")
tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    dtype=dtype,
    device_map="cuda",
    local_files_only=True
)
model.eval()
dev = next(model.parameters()).device

# Explicitly trigger decompression with warmup and clean up cache
print(f"Warming up model forward pass on {dev}...")
dummy = tokenizer("warmup", return_tensors="pt")
dummy = {k: v.to(dev) for k, v in dummy.items()}
with torch.no_grad():
    _ = model(**dummy, use_cache=False)
gc.collect()
torch.cuda.empty_cache()

free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
print(f"Model decompressed and ready on GPU! Free VRAM: {free_gb:.2f} GB")

def to_thought_start(text):
    marker = "<|thought>\n<thought|><|tool_call>call:"
    return (text[:-len(marker)] + "<|thought>\n") if text.endswith(marker) else text

GROUPS = ("prem_submit", "valid_submit", "run_after_edit", "edit_loop_fail", "healthy_read", "healthy_edit")
all_items = []
for g in GROUPS:
    for item in probes.get(g, []):
        all_items.append({**item, "group": g, "thought_text": to_thought_start(item["text"])})
idx_by_group = {g: [i for i, x in enumerate(all_items) if x["group"] == g] for g in GROUPS}
prem_idx = idx_by_group["prem_submit"]
val_idx = idx_by_group["valid_submit"]
run_idx = idx_by_group["run_after_edit"]
hread_idx = idx_by_group["healthy_read"]
hedit_idx = idx_by_group["healthy_edit"]
pres_idx = val_idx + run_idx + hread_idx + hedit_idx

lm = getattr(model, "model", model)
if hasattr(lm, "language_model"):
    lm = lm.language_model
layers = lm.layers
CAND_LAYERS = [16, 20, 24, 28, 30, 32, 35, 36]

print("\n" + "=" * 70)
print("2. HARVESTING ACTIVATIONS ACROSS CANDIDATE LAYERS")
print("=" * 70)

cap_x, cap_g = {}, {}
handles = []
for l_idx in CAND_LAYERS:
    def mk(li):
        def _h(mod, inp, out):
            cap_x[li] = inp[0][0, -1, :].detach().float().cpu()
            cap_g[li] = out[0, -1, :].detach().float().cpu()
        return _h
    handles.append(layers[l_idx].mlp.gate_proj.register_forward_hook(mk(l_idx)))

acts_x = {l: [] for l in CAND_LAYERS}
acts_g = {l: [] for l in CAND_LAYERS}
t0 = time.time()
with torch.no_grad():
    for i, item in enumerate(all_items):
        enc = tokenizer(item["thought_text"], return_tensors="pt", truncation=True, max_length=1024)
        enc = {k: v.to(dev) for k, v in enc.items()}
        cap_x.clear(); cap_g.clear()
        _ = model(**enc, use_cache=False)
        for l in CAND_LAYERS:
            acts_x[l].append(cap_x[l])
            acts_g[l].append(cap_g[l])
        if (i + 1) % 35 == 0 or (i + 1) == len(all_items):
            print(f"  Harvested {i+1}/{len(all_items)} ({time.time()-t0:.1f}s)", flush=True)

for h in handles: h.remove()
for l in CAND_LAYERS:
    acts_x[l] = torch.stack(acts_x[l], dim=0)
    acts_g[l] = torch.stack(acts_g[l], dim=0)

lm_head_w = model.get_output_embeddings().weight.detach().float().cpu()
def tok_vec(words):
    vecs = []
    for w in words:
        for variant in (w, " " + w):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                v = lm_head_w[ids[0]]
                vecs.append(v / (v.norm() + 1e-8))
    m = torch.stack(vecs, dim=0).mean(dim=0)
    return m / (m.norm() + 1e-8)

u_thought = tok_vec(["pytest", "test", "tests", "verify", "check", "run"]) - tok_vec(["submit", "done", "complete", "finished", "ready", "fixed"])
u_thought = u_thought / (u_thought.norm() + 1e-8)

def solve_signed_gate(H, target_idxs, pres_idxs, beta=3.0, lam=25.0):
    X = torch.cat([H[target_idxs], (beta ** 0.5) * H[pres_idxs]], dim=0)
    y = torch.cat([torch.full((len(target_idxs),), 3.0), torch.full((len(pres_idxs),), -3.0 * (beta ** 0.5))], dim=0)
    G = X @ X.T + lam * torch.eye(X.shape[0])
    w = X.T @ torch.linalg.solve(G, y)
    return w

all_tids = sorted(set(x["tid"] for x in all_items))
print("\n=== NONLINEAR GELU SWITCH LEAVE-TASK-OUT CV (ON=prem_submit, OFF=valid/healthy) ===", flush=True)
switch_basis = {"u_thought": u_thought.to(torch.bfloat16), "layers": {}}
cv_rows = []
for l in CAND_LAYERS:
    H = acts_x[l]
    oof_p, oof_v, oof_o = [], [], []
    for htid in all_tids:
        tr_t = [i for i in prem_idx if all_items[i]["tid"] != htid]
        tr_p = [i for i in pres_idx if all_items[i]["tid"] != htid]
        te_t = [i for i in prem_idx if all_items[i]["tid"] == htid]
        te_v = [i for i in val_idx if all_items[i]["tid"] == htid]
        te_o = [i for i in (run_idx + hread_idx + hedit_idx) if all_items[i]["tid"] == htid]
        if not (te_t or te_v or te_o): continue
        w = solve_signed_gate(H, tr_t, tr_p)
        for i in te_t: oof_p.append(F.gelu(torch.dot(w, H[i])).item())
        for i in te_v: oof_v.append(F.gelu(torch.dot(w, H[i])).item())
        for i in te_o: oof_o.append(F.gelu(torch.dot(w, H[i])).item())
    mp = sum(oof_p)/max(1,len(oof_p)); mv = sum(oof_v)/max(1,len(oof_v)); mo = sum(oof_o)/max(1,len(oof_o))
    ratio = mp / (max(abs(mv), abs(mo)) + 1e-4)
    cv_rows.append({"layer": l, "gelu_prem": round(mp,3), "gelu_val": round(mv,3), "gelu_other": round(mo,3), "on_off_ratio": round(ratio,2)})
    print(f"  L{l:2d}: GELU(prem)={mp:+.3f} | GELU(val)={mv:+.3f} | GELU(oth)={mo:+.3f} | ON/OFF={ratio:.2f}x", flush=True)

    w_full = solve_signed_gate(H, prem_idx, pres_idx)
    mean_g_pres = acts_g[l][pres_idx].mean(dim=0)
    quiet_idx = torch.topk(-mean_g_pres, k=16).indices  # 16 naturally-OFF neurons
    mu_prem_x = H[prem_idx].mean(dim=0)
    mu_prem_x = mu_prem_x / (mu_prem_x.norm() + 1e-8)
    switch_basis["layers"][l] = {
        "w_gate": w_full.to(torch.bfloat16),
        "w_up": mu_prem_x.to(torch.bfloat16),
        "neuron_idx": quiet_idx.to(torch.int64),
        "mean_x_norm": round(H[prem_idx].norm(dim=1).mean().item(), 3),
    }

cv_rows.sort(key=lambda x: x["gelu_prem"] - max(x["gelu_val"], x["gelu_other"]), reverse=True)
top_l = [r["layer"] for r in cv_rows[:3]]
print("\nTop 3 nonlinear switch layers:", top_l, flush=True)

print("\n" + "=" * 70)
print("3. LIVE GPU GENERATION VERIFICATION")
print("=" * 70)

test_prem = [i for i in prem_idx if all_items[i]["run"] == "iter_04"][:5]
test_val = [i for i in val_idx if all_items[i]["run"] == "iter_04"][:3]

def eval_transistor(label, target_layers, gain_down):
    gh = []
    for l in target_layers:
        e = switch_basis["layers"][l]
        wg = e["w_gate"].float()
        wu = e["w_up"].float()
        nidx = e["neuron_idx"]
        def mk_gate(wg_, nidx_):
            def _h(mod, inp, out):
                x = inp[0]
                s = (x @ wg_.to(x.device, x.dtype)).unsqueeze(-1)
                out = out.clone()
                out[..., nidx_.to(x.device)] += s
                return out
            return _h
        def mk_up(wu_, nidx_):
            def _h(mod, inp, out):
                x = inp[0]
                s = (x @ wu_.to(x.device, x.dtype)).unsqueeze(-1)
                out = out.clone()
                out[..., nidx_.to(x.device)] += s
                return out
            return _h
        def mk_down(nidx_, u_, g_):
            def _h(mod, inp, out):
                h = inp[0]
                sig = h[..., nidx_.to(h.device)].mean(dim=-1, keepdim=True)
                return out + sig * (g_ * u_.to(h.device, h.dtype))
            return _h
        gh.append(layers[l].mlp.gate_proj.register_forward_hook(mk_gate(wg, nidx)))
        gh.append(layers[l].mlp.up_proj.register_forward_hook(mk_up(wu, nidx)))
        gh.append(layers[l].mlp.down_proj.register_forward_hook(mk_down(nidx, u_thought, gain_down)))

    res = {"label": label, "layers": target_layers, "gain": gain_down, "prem": [], "val": []}
    with torch.no_grad():
        for tag, idxs in (("prem", test_prem), ("val", test_val)):
            for i in idxs:
                enc = tokenizer(all_items[i]["thought_text"], return_tensors="pt", truncation=True, max_length=1024)
                enc = {k: v.to(dev) for k, v in enc.items()}
                gen = model.generate(**enc, max_new_tokens=96, do_sample=False)
                new_txt = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
                res[tag].append({
                    "tid": all_items[i]["tid"],
                    "has_run": ("run_command" in new_txt or "pytest" in new_txt),
                    "has_sub": ("submit_patch" in new_txt),
                    "sample": new_txt[:150],
                })
    for h in gh: h.remove()
    p_run = sum(1 for x in res["prem"] if x["has_run"])
    p_sub = sum(1 for x in res["prem"] if x["has_sub"])
    v_sub = sum(1 for x in res["val"] if x["has_sub"])
    print(f"  [{label:16s}] prem_run={p_run}/{len(test_prem)} prem_sub={p_sub}/{len(test_prem)} | val_sub={v_sub}/{len(test_val)}", flush=True)
    for x in res["prem"]:
        print(f"    PREM {x['tid']:14s}: run={x['has_run']} sub={x['has_sub']} | {x['sample']!r}", flush=True)
    return res

gen_res = [
    eval_transistor("geglu_g0.5", top_l, 0.5),
    eval_transistor("geglu_g1.5", top_l, 1.5),
]

output_basis = Path("/content/geglu_switch_basis.pt")
output_report = Path("/content/geglu_switch_report.json")
torch.save(switch_basis, output_basis)
output_report.write_text(json.dumps({"cv": cv_rows, "top_layers": top_l, "gen": gen_res}, indent=2) + "\n", encoding="utf-8")
print("\n" + "=" * 70)
print(f"EXPERIMENT COMPLETE! Saved {output_basis} ({output_basis.stat().st_size} bytes)")
print(f"Saved {output_report} ({output_report.stat().st_size} bytes)")
print("=" * 70)
