# Thought-Start (<|thought>) DLA + AlphaEdit Weight-Space Programming Lab with Real Generation Verification
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
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')
PROBES_PATH = Path('/kaggle/input/datasets/prince22466/g6-dla-probes/dla_probes.json')
if not PROBES_PATH.exists():
    PROBES_PATH = Path('/kaggle/input/g6-dla-probes/dla_probes.json')
probes = json.loads(PROBES_PATH.read_text(encoding='utf-8'))

print('Loading Gemma 4 E4B on GPU...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
tokenizer.truncation_side = 'left'  # CRITICAL: never chop off the recent edit_file & <|thought> decision!
dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=dtype, device_map='auto', local_files_only=True)
model.eval()

# Convert probe prefixes from `<|thought>\n<thought|><|tool_call>call:` to `<|thought>\n` (START OF THOUGHT!)
def to_thought_start(text):
    marker = '<|thought>\n<thought|><|tool_call>call:'
    if text.endswith(marker):
        return text[:-len(marker)] + '<|thought>\n'
    return text

GROUPS = ('prem_submit', 'valid_submit', 'run_after_edit', 'edit_loop_fail', 'healthy_read', 'healthy_edit')
all_items = []
for g in GROUPS:
    for item in probes.get(g, []):
        all_items.append({**item, 'group': g, 'thought_text': to_thought_start(item['text'])})
idx_by_group = {g: [i for i, x in enumerate(all_items) if x['group'] == g] for g in GROUPS}
print('Loaded probes:', {g: len(idx_by_group[g]) for g in GROUPS}, flush=True)

# 1. Harvest thought-start activations on o_proj and down_proj for layers 16..41
lm = getattr(model, 'model', model)
if hasattr(lm, 'language_model'):
    lm = lm.language_model
layers = lm.layers
CAND_LAYERS = list(range(16, len(layers)))

cap_o_in, cap_d_in = {}, {}
handles = []
for l_idx in CAND_LAYERS:
    def mk(li, store):
        def _h(mod, inp, out):
            store[li] = inp[0][0, -1, :].detach().float().cpu()
        return _h
    handles.append(layers[l_idx].self_attn.o_proj.register_forward_hook(mk(l_idx, cap_o_in)))
    handles.append(layers[l_idx].mlp.down_proj.register_forward_hook(mk(l_idx, cap_d_in)))

acts_o = {l: [] for l in CAND_LAYERS}
acts_d = {l: [] for l in CAND_LAYERS}
t0 = time.time()
with torch.no_grad():
    for i, item in enumerate(all_items):
        enc = tokenizer(item['thought_text'], return_tensors='pt', truncation=True, max_length=2560)
        if i == 0:
            print('  SANITY CHECK input tail:', repr(tokenizer.decode(enc['input_ids'][0, -40:], skip_special_tokens=False)), flush=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        cap_o_in.clear(); cap_d_in.clear()
        _ = model(**enc, use_cache=False)
        for l in CAND_LAYERS:
            acts_o[l].append(cap_o_in[l])
            acts_d[l].append(cap_d_in[l])
        if (i + 1) % 30 == 0 or (i + 1) == len(all_items):
            print(f'  Harvested {i+1}/{len(all_items)} ({time.time()-t0:.1f}s)', flush=True)
for h in handles: h.remove()
for l in CAND_LAYERS:
    acts_o[l] = torch.stack(acts_o[l], dim=0)
    acts_d[l] = torch.stack(acts_d[l], dim=0)

# 2. Compute Thought-Start Steering Vector u_thought:
# Contrastive unembedding direction pushing verification reasoning ("pytest", "test", "verify", "check", "run")
# over premature-stop reasoning ("submit", "done", "complete", "finished", "ready")
lm_head_w = model.get_output_embeddings().weight.detach().float().cpu()
def tok_vec(words):
    vecs = []
    for w in words:
        for variant in (w, ' ' + w):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                v = lm_head_w[ids[0]]
                vecs.append(v / (v.norm() + 1e-8))
    m = torch.stack(vecs, dim=0).mean(dim=0)
    return m / (m.norm() + 1e-8)

u_verify = tok_vec(['pytest', 'test', 'tests', 'verify', 'check', 'run'])
u_stop = tok_vec(['submit', 'done', 'complete', 'finished', 'ready', 'fixed'])
u_thought = u_verify - u_stop
u_thought = u_thought / (u_thought.norm() + 1e-8)
print('Built u_thought (verify - stop), cos(verify, stop)=', round(torch.dot(u_verify, u_stop).item(), 3), flush=True)

# 3. Solve AlphaEdit Null-Space Detector at <|thought> start (prem_submit -> +1, valid_submit + healthy -> 0)
def solve_alphaedit(H, target_idxs, pres_idxs, beta=4.0, lam=20.0):
    X = torch.cat([H[target_idxs], (beta ** 0.5) * H[pres_idxs]], dim=0)
    y = torch.cat([torch.ones(len(target_idxs)), torch.zeros(len(pres_idxs))], dim=0)
    G = X @ X.T + lam * torch.eye(X.shape[0])
    w = X.T @ torch.linalg.solve(G, y)
    n = w.norm().item()
    return w / (n + 1e-8), n

prem_idx = idx_by_group['prem_submit']
val_idx = idx_by_group['valid_submit']
run_idx = idx_by_group['run_after_edit']
hread_idx = idx_by_group['healthy_read']
hedit_idx = idx_by_group['healthy_edit']
pres_idx = val_idx + run_idx + hread_idx + hedit_idx
all_tids = sorted(set(x['tid'] for x in all_items))

cv_rows = []
for l in CAND_LAYERS:
    for mod, H in (('o_proj', acts_o[l]), ('down_proj', acts_d[l])):
        oof_p, oof_v, oof_o = [], [], []
        for htid in all_tids:
            tr_t = [i for i in prem_idx if all_items[i]['tid'] != htid]
            tr_p = [i for i in pres_idx if all_items[i]['tid'] != htid]
            te_t = [i for i in prem_idx if all_items[i]['tid'] == htid]
            te_v = [i for i in val_idx if all_items[i]['tid'] == htid]
            te_o = [i for i in (run_idx + hread_idx + hedit_idx) if all_items[i]['tid'] == htid]
            if not (te_t or te_v or te_o): continue
            wu, sc = solve_alphaedit(H, tr_t, tr_p)
            w = wu * sc
            for i in te_t: oof_p.append(torch.dot(w, H[i]).item())
            for i in te_v: oof_v.append(torch.dot(w, H[i]).item())
            for i in te_o: oof_o.append(torch.dot(w, H[i]).item())
        mp = sum(oof_p)/max(1,len(oof_p)); mv = sum(oof_v)/max(1,len(oof_v)); mo = sum(oof_o)/max(1,len(oof_o))
        cv_rows.append({'layer': l, 'module': mod, 'oof_prem': round(mp,3), 'oof_val': round(mv,3), 'oof_other': round(mo,3), 'gap': round(mp - max(mv, mo), 3)})
cv_rows.sort(key=lambda x: x['gap'], reverse=True)
print('\n=== THOUGHT-START LEAVE-TASK-OUT CV TOP 10 ===', flush=True)
for r in cv_rows[:10]:
    print(f"  L{r['layer']:2d} {r['module']:9s}: prem={r['oof_prem']:+.3f} val={r['oof_val']:+.3f} oth={r['oof_other']:+.3f} gap={r['gap']:+.3f}", flush=True)

# Save full basis
thought_basis = {'u_thought': u_thought.to(torch.bfloat16), 'o_proj': {}, 'down_proj': {}, 'cv_top': cv_rows[:15]}
for l in CAND_LAYERS:
    wo, so = solve_alphaedit(acts_o[l], prem_idx, pres_idx)
    wd, sd = solve_alphaedit(acts_d[l], prem_idx, pres_idx)
    thought_basis['o_proj'][l] = {'v': wo.to(torch.bfloat16), 'scale': round(so, 5), 'd_in': acts_o[l].shape[1]}
    thought_basis['down_proj'][l] = {'v': wd.to(torch.bfloat16), 'scale': round(sd, 5), 'd_in': acts_d[l].shape[1]}

# 4. REAL AUTOREGRESSIVE GENERATION TEST on 5 iter_04 premature-submit traces + 3 valid-submit traces!
test_prem = [i for i in prem_idx if all_items[i]['run'] == 'iter_04'][:5]
test_val = [i for i in val_idx if all_items[i]['run'] == 'iter_04'][:3]
print(f'\n=== REAL GENERATION TEST (5 iter_04 prem_submit + 3 valid_submit) ===', flush=True)

def eval_gen(label, o_layers, gain):
    gh = []
    for l in o_layers:
        e = thought_basis['o_proj'][l]
        vc = e['v'].float() * e['scale']
        uc = gain * u_thought
        def mk_h(v_, u_):
            def _h(mod, inp, out):
                h = inp[0]
                return out + (h @ v_.to(h.device, h.dtype)).unsqueeze(-1) * u_.to(h.device, h.dtype)
            return _h
        gh.append(layers[l].self_attn.o_proj.register_forward_hook(mk_h(vc, uc)))

    res = {'label': label, 'o_layers': o_layers, 'gain': gain, 'prem': [], 'val': []}
    with torch.no_grad():
        for tag, idxs in (('prem', test_prem), ('val', test_val)):
            for i in idxs:
                enc = tokenizer(all_items[i]['thought_text'], return_tensors='pt', truncation=True, max_length=2560)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                gen = model.generate(**enc, max_new_tokens=96, do_sample=False)
                new_txt = tokenizer.decode(gen[0, enc['input_ids'].shape[1]:], skip_special_tokens=False)
                has_run = ('run_command' in new_txt or 'pytest' in new_txt)
                has_sub = ('submit_patch' in new_txt)
                res[tag].append({'tid': all_items[i]['tid'], 'has_run': has_run, 'has_sub': has_sub, 'sample': new_txt[:160]})
    for h in gh: h.remove()
    p_run = sum(1 for x in res['prem'] if x['has_run'])
    p_sub = sum(1 for x in res['prem'] if x['has_sub'])
    v_sub = sum(1 for x in res['val'] if x['has_sub'])
    print(f"  [{label:16s}] prem_run={p_run}/{len(test_prem)} prem_sub={p_sub}/{len(test_prem)} | val_sub={v_sub}/{len(test_val)}", flush=True)
    for x in res['prem']:
        print(f"    PREM {x['tid']:14s}: run={x['has_run']} sub={x['has_sub']} | {x['sample']!r}", flush=True)
    return res

top_o = [r['layer'] for r in cv_rows if r['module'] == 'o_proj'][:4]
gen_results = [
    eval_gen('baseline', [], 0.0),
    eval_gen('thought_g1.5', top_o, 1.5),
    eval_gen('thought_g3.0', top_o, 3.0),
]

WORKING = Path('/kaggle/working')
torch.save(thought_basis, WORKING / 'thought_basis.pt')
(WORKING / 'thought_report.json').write_text(json.dumps({'cv_top': cv_rows[:15], 'gen_results': gen_results}, indent=2) + '\n', encoding='utf-8')
print('ALL DONE!', flush=True)
