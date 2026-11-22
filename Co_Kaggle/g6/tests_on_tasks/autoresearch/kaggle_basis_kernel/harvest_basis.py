# DLA Neuron/Head Spotter + AlphaEdit Null-Space Closed-Form LoRA Lab for Gemma 4 E4B
import glob, importlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
for pth_pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth', '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pth_pattern):
        try: os.unlink(pth)
        except OSError: pass

tmp_whl = Path('/tmp/wheelhouse')
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower(): continue
    target_name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    target = tmp_whl / target_name
    if not target.exists():
        try: os.symlink(w, target)
        except OSError: shutil.copyfile(w, target)

wheels = sorted(
    str(w) for w in tmp_whl.glob('*.whl')
    if any(w.name.startswith(p) for p in ('transformers-', 'tokenizers-', 'safetensors-', 'compressed_tensors-', 'loguru-'))
)
print(f'Installing {len(wheels)} core wheels', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')
PROBES_PATH = Path('/kaggle/input/datasets/prince22466/g6-dla-probes/dla_probes.json')
if not PROBES_PATH.exists():
    PROBES_PATH = Path('/kaggle/input/g6-dla-probes/dla_probes.json')
assert PROBES_PATH.is_file(), f'Missing {PROBES_PATH}'
probes = json.loads(PROBES_PATH.read_text(encoding='utf-8'))

print('Loading tokenizer and Gemma 4 E4B on GPU...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=dtype,
    device_map='auto',
    local_files_only=True,
)
model.eval()

# Token IDs for exact tool-routing logits at `<|tool_call>call:`
TOK_IDS = {w: tokenizer.encode(w, add_special_tokens=False)[0] for w in ('submit', 'run', 'read', 'edit', 'pytest', 'patch')}
print('Tool token IDs:', TOK_IDS, flush=True)

# Unembedding vectors (lm_head or tied embed_tokens)
lm_head_w = model.get_output_embeddings().weight.detach().float().cpu()
E = {w: lm_head_w[i] for w, i in TOK_IDS.items()}
E_unit = {w: v / (v.norm() + 1e-8) for w, v in E.items()}
u_run_minus_sub = E['run'] - E['submit']
u_run_minus_sub_unit = u_run_minus_sub / (u_run_minus_sub.norm() + 1e-8)
u_read_minus_edit = E['read'] - E['edit']
u_read_minus_edit_unit = u_read_minus_edit / (u_read_minus_edit.norm() + 1e-8)

lm = getattr(model, 'model', model)
if hasattr(lm, 'language_model'):
    lm = lm.language_model
layers = lm.layers
num_layers = len(layers)
# Analyze upper half of network (layers 20..41) where tool routing happens
CAND_LAYERS = list(range(20, num_layers))
print(f'Num layers: {num_layers}, hooking o_proj & down_proj on layers {CAND_LAYERS[0]}..{CAND_LAYERS[-1]}', flush=True)

# Inspect dimensions of o_proj and down_proj
sample_o_in = {}
sample_d_in = {}
cap_o_in, cap_o_out, cap_d_in, cap_d_out = {}, {}, {}, {}
handles = []

for l_idx in CAND_LAYERS:
    mod_o = layers[l_idx].self_attn.o_proj
    mod_d = layers[l_idx].mlp.down_proj

    def make_hook(li, store_in, store_out):
        def _hook(module, inp, out):
            store_in[li] = inp[0][0, -1, :].detach().float().cpu()
            store_out[li] = out[0, -1, :].detach().float().cpu()
        return _hook

    handles.append(mod_o.register_forward_hook(make_hook(l_idx, cap_o_in, cap_o_out)))
    handles.append(mod_d.register_forward_hook(make_hook(l_idx, cap_d_in, cap_d_out)))

# Flatten all probes
GROUPS = ('prem_submit', 'valid_submit', 'run_after_edit', 'edit_loop_fail', 'healthy_read', 'healthy_edit')
all_items = []
for g in GROUPS:
    for item in probes.get(g, []):
        all_items.append({**item, 'group': g})
print(f'Total probes to profile: {len(all_items)}', flush=True)

# Collect activations and baseline logits
acts_o_in = {l: [] for l in CAND_LAYERS}
acts_o_out = {l: [] for l in CAND_LAYERS}
acts_d_in = {l: [] for l in CAND_LAYERS}
acts_d_out = {l: [] for l in CAND_LAYERS}
base_logits = []

t0 = time.time()
with torch.no_grad():
    for idx, item in enumerate(all_items):
        enc = tokenizer(item['text'], return_tensors='pt', truncation=True, max_length=2560)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        cap_o_in.clear(); cap_o_out.clear(); cap_d_in.clear(); cap_d_out.clear()
        out = model(**enc, use_cache=False)
        last_logits = out.logits[0, -1, :].detach().float().cpu()
        row = {w: round(last_logits[i].item(), 3) for w, i in TOK_IDS.items()}
        # Best among the 4 main tools
        row['pred'] = max(('submit', 'run', 'read', 'edit'), key=lambda w: row[w])
        base_logits.append(row)
        for l in CAND_LAYERS:
            acts_o_in[l].append(cap_o_in[l])
            acts_o_out[l].append(cap_o_out[l])
            acts_d_in[l].append(cap_d_in[l])
            acts_d_out[l].append(cap_d_out[l])
        if (idx + 1) % 20 == 0 or (idx + 1) == len(all_items):
            print(f'  [{idx+1}/{len(all_items)}] elapsed={time.time()-t0:.1f}s', flush=True)

for h in handles:
    h.remove()

# Stack tensors: [N, dim]
for l in CAND_LAYERS:
    acts_o_in[l] = torch.stack(acts_o_in[l], dim=0)
    acts_o_out[l] = torch.stack(acts_o_out[l], dim=0)
    acts_d_in[l] = torch.stack(acts_d_in[l], dim=0)
    acts_d_out[l] = torch.stack(acts_d_out[l], dim=0)

print(f'o_proj in dim: {acts_o_in[CAND_LAYERS[0]].shape[1]}, down_proj in dim: {acts_d_in[CAND_LAYERS[0]].shape[1]}', flush=True)

# 1. Print baseline tool prediction breakdown per group
idx_by_group = {g: [i for i, x in enumerate(all_items) if x['group'] == g] for g in GROUPS}
print('\n=== BASELINE LOGIT SCORECARD ===', flush=True)
base_summary = {}
for g in GROUPS:
    idxs = idx_by_group[g]
    preds = {}
    mean_sub_minus_run = sum(base_logits[i]['submit'] - base_logits[i]['run'] for i in idxs) / max(1, len(idxs))
    mean_edit_minus_read = sum(base_logits[i]['edit'] - base_logits[i]['read'] for i in idxs) / max(1, len(idxs))
    for i in idxs:
        p = base_logits[i]['pred']
        preds[p] = preds.get(p, 0) + 1
    base_summary[g] = {'n': len(idxs), 'preds': preds, 'sub-run': round(mean_sub_minus_run, 3), 'edit-read': round(mean_edit_minus_read, 3)}
    print(f'  {g:16s} (n={len(idxs):2d}): preds={preds} | sub-run={mean_sub_minus_run:+.2f} | edit-read={mean_edit_minus_read:+.2f}', flush=True)

# 2. Direct Logit Attribution (DLA) per layer for o_proj and down_proj on (submit - run) and (edit - read)
# Compare prem_submit vs valid_submit AND prem_submit vs run_after_edit!
dla_layers = []
prem_idx = idx_by_group['prem_submit']
val_idx = idx_by_group['valid_submit']
run_idx = idx_by_group['run_after_edit']
eloop_idx = idx_by_group['edit_loop_fail']
hread_idx = idx_by_group['healthy_read']
hedit_idx = idx_by_group['healthy_edit']

u_sub_run = E['submit'] - E['run']
u_edit_read = E['edit'] - E['read']

for l in CAND_LAYERS:
    # DLA on (submit - run)
    o_dla_prem = (acts_o_out[l][prem_idx] @ u_sub_run).mean().item()
    o_dla_val = (acts_o_out[l][val_idx] @ u_sub_run).mean().item()
    o_dla_run = (acts_o_out[l][run_idx] @ u_sub_run).mean().item()
    d_dla_prem = (acts_d_out[l][prem_idx] @ u_sub_run).mean().item()
    d_dla_val = (acts_d_out[l][val_idx] @ u_sub_run).mean().item()
    d_dla_run = (acts_d_out[l][run_idx] @ u_sub_run).mean().item()

    # DLA on (edit - read)
    o_dla_eloop = (acts_o_out[l][eloop_idx] @ u_edit_read).mean().item()
    d_dla_eloop = (acts_d_out[l][eloop_idx] @ u_edit_read).mean().item()

    # Separation power (cosine / linear separability of prem_submit vs valid_submit in input space!)
    o_sep = (acts_o_in[l][prem_idx].mean(0) - acts_o_in[l][val_idx].mean(0)).norm().item() / (acts_o_in[l].std(0).norm().item() + 1e-6)
    d_sep = (acts_d_in[l][prem_idx].mean(0) - acts_d_in[l][val_idx].mean(0)).norm().item() / (acts_d_in[l].std(0).norm().item() + 1e-6)

    entry = {
        'layer': l,
        'o_prem_sub_run': round(o_dla_prem, 3),
        'o_val_sub_run': round(o_dla_val, 3),
        'o_run_sub_run': round(o_dla_run, 3),
        'd_prem_sub_run': round(d_dla_prem, 3),
        'd_val_sub_run': round(d_dla_val, 3),
        'd_run_sub_run': round(d_dla_run, 3),
        'o_eloop_edit_read': round(o_dla_eloop, 3),
        'd_eloop_edit_read': round(d_dla_eloop, 3),
        'o_prem_vs_val_sep': round(o_sep, 4),
        'd_prem_vs_val_sep': round(d_sep, 4),
    }
    dla_layers.append(entry)

print('\n=== TOP LAYERS BY DLA ON PREMATURE SUBMIT (submit - run) ===', flush=True)
for e in sorted(dla_layers, key=lambda x: abs(x['o_prem_sub_run']) + abs(x['d_prem_sub_run']), reverse=True)[:10]:
    print(f"  L{e['layer']:2d}: o_proj(prem={e['o_prem_sub_run']:+.2f}, val={e['o_val_sub_run']:+.2f}, run={e['o_run_sub_run']:+.2f}, sep={e['o_prem_vs_val_sep']:.3f}) | "
          f"down_proj(prem={e['d_prem_sub_run']:+.2f}, val={e['d_val_sub_run']:+.2f}, run={e['d_run_sub_run']:+.2f}, sep={e['d_prem_vs_val_sep']:.3f})", flush=True)

# 3. Closed-Form AlphaEdit (Null-Space Constrained Ridge) Solver + Leave-Task-Out Cross-Validation!
# Minimize: sum_{i in target} (w^T h_i - 1)^2 + beta * sum_{j in preserve} (w^T h_j)^2 + lam * ||w||^2
# Exact solution via Woodbury / primal-dual:
# Let X = [H_target; sqrt(beta) * H_pres] ([N, d]), y = [1_target; 0_pres] ([N])
# Then w = X^T (X X^T + lam * I)^{-1} y   (O(N^3) where N <= 141 -> 1 millisecond!)

def solve_alphaedit_detector(H_all, target_idxs, pres_idxs, beta=4.0, lam=10.0):
    """Return unit-normalized detector w in R^d and raw scale s such that s * (w^T h) ~ 1 on target and ~ 0 on pres."""
    H_t = H_all[target_idxs]
    H_p = H_all[pres_idxs]
    X = torch.cat([H_t, (beta ** 0.5) * H_p], dim=0)  # [N, d]
    y = torch.cat([torch.ones(len(target_idxs)), torch.zeros(len(pres_idxs))], dim=0)  # [N]
    G = X @ X.T + lam * torch.eye(X.shape[0])
    alpha = torch.linalg.solve(G, y)
    w = X.T @ alpha  # [d]
    norm = w.norm().item()
    return w / (norm + 1e-8), norm

# Check Leave-Task-Out Cross-Validation separation for prem_submit vs (valid_submit + run_after_edit + healthy_read + healthy_edit)
pres_for_prem = val_idx + run_idx + hread_idx + hedit_idx
pres_for_eloop = val_idx + run_idx + hread_idx + hedit_idx

print('\n=== ALPHAEDIT LEAVE-TASK-OUT CV SEPARATION (prem_submit -> 1, preserve -> 0) ===', flush=True)
cv_scores = []
all_tids = sorted(set(x['tid'] for x in all_items))

for l in CAND_LAYERS:
    for mod_name, H in (('o_proj', acts_o_in[l]), ('down_proj', acts_d_in[l])):
        # Leave-one-task-out CV for prem_submit detector
        oof_prem, oof_val, oof_other = [], [], []
        for hold_tid in all_tids:
            tr_t = [i for i in prem_idx if all_items[i]['tid'] != hold_tid]
            tr_p = [i for i in pres_for_prem if all_items[i]['tid'] != hold_tid]
            te_t = [i for i in prem_idx if all_items[i]['tid'] == hold_tid]
            te_v = [i for i in val_idx if all_items[i]['tid'] == hold_tid]
            te_o = [i for i in (run_idx + hread_idx + hedit_idx) if all_items[i]['tid'] == hold_tid]
            if not (te_t or te_v or te_o):
                continue
            w_u, sc = solve_alphaedit_detector(H, tr_t, tr_p, beta=4.0, lam=20.0)
            w = w_u * sc
            for i in te_t: oof_prem.append(torch.dot(w, H[i]).item())
            for i in te_v: oof_val.append(torch.dot(w, H[i]).item())
            for i in te_o: oof_other.append(torch.dot(w, H[i]).item())
        m_prem = sum(oof_prem) / max(1, len(oof_prem))
        m_val = sum(oof_val) / max(1, len(oof_val))
        m_oth = sum(oof_other) / max(1, len(oof_other))
        gap = m_prem - max(m_val, m_oth)
        cv_scores.append({'layer': l, 'module': mod_name, 'oof_prem': round(m_prem, 3), 'oof_val': round(m_val, 3), 'oof_other': round(m_oth, 3), 'gap': round(gap, 3)})

cv_scores.sort(key=lambda x: x['gap'], reverse=True)
for c in cv_scores[:12]:
    print(f"  L{c['layer']:2d} {c['module']:9s}: oof_prem={c['oof_prem']:+.3f} | oof_val={c['oof_val']:+.3f} | oof_other={c['oof_other']:+.3f} | gap={c['gap']:+.3f}", flush=True)

# 4. Build AlphaEdit Multi-Module Basis (both o_proj and down_proj for ALL layers 20..41)
#    AND run LIVE GPU LOGIT-FLIP VERIFICATION with forward hooks!
alphaedit_basis = {
    'model': 'gemma-4-e4b-it-qat-w4a16-ct',
    'source': 'dla_alphaedit_v5',
    'u_run_minus_sub': u_run_minus_sub_unit.to(torch.bfloat16),
    'u_read_minus_edit': u_read_minus_edit_unit.to(torch.bfloat16),
    'o_proj': {},
    'down_proj': {},
}

for l in CAND_LAYERS:
    w_o_prem, s_o_prem = solve_alphaedit_detector(acts_o_in[l], prem_idx, pres_for_prem, beta=4.0, lam=20.0)
    w_o_eloop, s_o_eloop = solve_alphaedit_detector(acts_o_in[l], eloop_idx, pres_for_eloop, beta=4.0, lam=20.0)
    alphaedit_basis['o_proj'][l] = {
        'v_prem': w_o_prem.to(torch.bfloat16), 'scale_prem': round(s_o_prem, 5),
        'v_eloop': w_o_eloop.to(torch.bfloat16), 'scale_eloop': round(s_o_eloop, 5),
        'd_in': acts_o_in[l].shape[1],
    }
    w_d_prem, s_d_prem = solve_alphaedit_detector(acts_d_in[l], prem_idx, pres_for_prem, beta=4.0, lam=20.0)
    w_d_eloop, s_d_eloop = solve_alphaedit_detector(acts_d_in[l], eloop_idx, pres_for_eloop, beta=4.0, lam=20.0)
    alphaedit_basis['down_proj'][l] = {
        'v_prem': w_d_prem.to(torch.bfloat16), 'scale_prem': round(s_d_prem, 5),
        'v_eloop': w_d_eloop.to(torch.bfloat16), 'scale_eloop': round(s_d_eloop, 5),
        'd_in': acts_d_in[l].shape[1],
    }

# Live GPU Logit-Flip Verification on top-6 o_proj + top-6 down_proj layers!
top_o_layers = [c['layer'] for c in cv_scores if c['module'] == 'o_proj'][:6]
top_d_layers = [c['layer'] for c in cv_scores if c['module'] == 'down_proj'][:6]
print(f'\n=== LIVE GPU LOGIT-FLIP VERIFICATION (top_o={top_o_layers}, top_d={top_d_layers}) ===', flush=True)

# Also run Leave-Task-Out CV for edit_loop_fail
eloop_cv = []
for l in CAND_LAYERS:
    for mod_name, H in (('o_proj', acts_o_in[l]), ('down_proj', acts_d_in[l])):
        oof_e, oof_h = [], []
        for hold_tid in all_tids:
            tr_t = [i for i in eloop_idx if all_items[i]['tid'] != hold_tid]
            tr_p = [i for i in pres_for_eloop if all_items[i]['tid'] != hold_tid]
            te_t = [i for i in eloop_idx if all_items[i]['tid'] == hold_tid]
            te_p = [i for i in pres_for_eloop if all_items[i]['tid'] == hold_tid]
            if not (te_t or te_p):
                continue
            w_u, sc = solve_alphaedit_detector(H, tr_t, tr_p, beta=4.0, lam=20.0)
            w = w_u * sc
            for i in te_t: oof_e.append(torch.dot(w, H[i]).item())
            for i in te_p: oof_h.append(torch.dot(w, H[i]).item())
        me = sum(oof_e) / max(1, len(oof_e))
        mh = sum(oof_h) / max(1, len(oof_h))
        eloop_cv.append({'layer': l, 'module': mod_name, 'oof_eloop': round(me, 3), 'oof_healthy': round(mh, 3), 'gap': round(me - mh, 3)})
eloop_cv.sort(key=lambda x: x['gap'], reverse=True)
print('\n=== ALPHAEDIT LEAVE-TASK-OUT CV SEPARATION (edit_loop_fail -> 1, healthy -> 0) ===', flush=True)
for c in eloop_cv[:10]:
    print(f"  L{c['layer']:2d} {c['module']:9s}: oof_eloop={c['oof_eloop']:+.3f} | oof_healthy={c['oof_healthy']:+.3f} | gap={c['gap']:+.3f}", flush=True)

def verify_live_config(o_layers, d_layers, gain_prem, gain_eloop):
    v_handles = []
    for l in o_layers:
        entry = alphaedit_basis['o_proj'][l]
        vp_cpu = (entry['v_prem'].float() * entry['scale_prem'])
        ve_cpu = (entry['v_eloop'].float() * entry['scale_eloop'])
        up_cpu = (gain_prem * u_run_minus_sub_unit)
        ue_cpu = (gain_eloop * u_read_minus_edit_unit)
        def mk_o(vp_, ve_, up_, ue_):
            def _h(mod, inp, out):
                h = inp[0]
                dev, dt = h.device, h.dtype
                delta = (h @ vp_.to(dev, dt)).unsqueeze(-1) * up_.to(dev, dt) + (h @ ve_.to(dev, dt)).unsqueeze(-1) * ue_.to(dev, dt)
                return out + delta
            return _h
        v_handles.append(layers[l].self_attn.o_proj.register_forward_hook(mk_o(vp_cpu, ve_cpu, up_cpu, ue_cpu)))
    for l in d_layers:
        entry = alphaedit_basis['down_proj'][l]
        vp_cpu = (entry['v_prem'].float() * entry['scale_prem'])
        ve_cpu = (entry['v_eloop'].float() * entry['scale_eloop'])
        up_cpu = (gain_prem * u_run_minus_sub_unit)
        ue_cpu = (gain_eloop * u_read_minus_edit_unit)
        def mk_d(vp_, ve_, up_, ue_):
            def _h(mod, inp, out):
                h = inp[0]
                dev, dt = h.device, h.dtype
                delta = (h @ vp_.to(dev, dt)).unsqueeze(-1) * up_.to(dev, dt) + (h @ ve_.to(dev, dt)).unsqueeze(-1) * ue_.to(dev, dt)
                return out + delta
            return _h
        v_handles.append(layers[l].mlp.down_proj.register_forward_hook(mk_d(vp_cpu, ve_cpu, up_cpu, ue_cpu)))

    # Evaluate on a balanced subset (all 24 prem_submit, all 22 valid_submit, 10 run_after_edit, 10 edit_loop_fail, 10 healthy_read, 10 healthy_edit)
    eval_idxs = prem_idx + val_idx + run_idx[:10] + eloop_idx[:10] + hread_idx[:10] + hedit_idx[:10]
    res_by_g = {g: [] for g in GROUPS}
    margins_prem = []
    margins_val = []
    with torch.no_grad():
        for i in eval_idxs:
            enc = tokenizer(all_items[i]['text'], return_tensors='pt', truncation=True, max_length=2560)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, use_cache=False)
            ll = out.logits[0, -1, :].detach().float().cpu()
            scores = {w: ll[TOK_IDS[w]].item() for w in ('submit', 'run', 'read', 'edit')}
            pred = max(scores, key=scores.get)
            g = all_items[i]['group']
            res_by_g[g].append(pred)
            if g == 'prem_submit':
                margins_prem.append(scores['run'] - scores['submit'])
            elif g == 'valid_submit':
                margins_val.append(scores['run'] - scores['submit'])
    for h in v_handles:
        h.remove()

    prem_flip = sum(1 for p in res_by_g['prem_submit'] if p == 'run') / max(1, len(res_by_g['prem_submit']))
    eloop_flip = sum(1 for p in res_by_g['edit_loop_fail'] if p == 'read') / max(1, len(res_by_g['edit_loop_fail']))
    read_keep = sum(1 for p in res_by_g['healthy_read'] if p == 'read') / max(1, len(res_by_g['healthy_read']))
    edit_keep = sum(1 for p in res_by_g['healthy_edit'] if p == 'edit') / max(1, len(res_by_g['healthy_edit']))
    summary = {
        'gain_prem': gain_prem, 'gain_eloop': gain_eloop,
        'prem_run_minus_sub': round(sum(margins_prem) / max(1, len(margins_prem)), 3),
        'val_run_minus_sub': round(sum(margins_val) / max(1, len(margins_val)), 3),
        'prem_to_run': round(prem_flip, 3),
        'eloop_to_read': round(eloop_flip, 3),
        'read_keep': round(read_keep, 3),
        'edit_keep': round(edit_keep, 3),
    }
    print('  VERIFY:', json.dumps(summary), flush=True)
    return summary

verify_results = []
# Test o_proj-only (top 4 CV layers) with two gains!
for gp, ge in [(1.0, 0.8), (2.0, 1.2)]:
    verify_results.append(verify_live_config(top_o_layers[:4], [], gp, ge))

WORKING = Path('/kaggle/working')
torch.save(alphaedit_basis, WORKING / 'alphaedit_basis.pt')
report = {
    'baseline_summary': base_summary,
    'top_dla_layers': sorted(dla_layers, key=lambda x: abs(x['o_prem_sub_run']) + abs(x['d_prem_sub_run']), reverse=True)[:15],
    'top_cv_components': cv_scores[:20],
    'top_eloop_cv_components': eloop_cv[:15],
    'top_o_layers': top_o_layers,
    'top_d_layers': top_d_layers,
    'live_verify_results': verify_results,
}
(WORKING / 'dla_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('ALL DONE! Saved alphaedit_basis.pt and dla_report.json', flush=True)
