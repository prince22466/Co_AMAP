# Real Forward-Pass Contrastive Activation Harvester for Gemma 4 E4B (WSP Basis v2)
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
print(f'Installing {len(wheels)} core wheels: {[Path(w).name for w in wheels]}', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')
DS_DIR = Path('/kaggle/input/datasets/prince22466/g6-autoresearch-sub')
if not DS_DIR.exists():
    DS_DIR = Path('/kaggle/input/g6-autoresearch-sub')

prompts_path = DS_DIR / 'contrastive_prompts.json'
assert prompts_path.is_file(), f'Missing {prompts_path}'
bundle = json.loads(prompts_path.read_text(encoding='utf-8'))

print('Loading tokenizer and Gemma 4 E4B model on GPU...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=dtype,
    device_map='auto',
    local_files_only=True,
)
model.eval()
print('Model loaded! Device map:', getattr(model, 'hf_device_map', 'single'), flush=True)

# Locate language_model.layers
lm = getattr(model, 'model', model)
if hasattr(lm, 'language_model'):
    lm = lm.language_model
layers = lm.layers
num_layers = len(layers)
candidate_layers = list(range(16, num_layers))
print(f'Num layers: {num_layers}, hooking candidate layers {candidate_layers[0]}..{candidate_layers[-1]}', flush=True)

# Hook down_proj inputs (v_l in R^10240) and outputs (u_l in R^2560) at the last token position
captured_in = {}
captured_out = {}
handles = []

for l_idx in candidate_layers:
    mod = layers[l_idx].mlp.down_proj
    def make_hook(li):
        def _hook(module, inp, out):
            # inp[0]: [1, seq, 10240], out: [1, seq, 2560]
            captured_in[li] = inp[0][0, -1, :].detach().to(torch.float32).cpu()
            captured_out[li] = out[0, -1, :].detach().to(torch.float32).cpu()
        return _hook
    handles.append(mod.register_forward_hook(make_hook(l_idx)))

@torch.no_grad()
def collect_group(items, label):
    in_vecs = {l: [] for l in candidate_layers}
    out_vecs = {l: [] for l in candidate_layers}
    for i, item in enumerate(items):
        text = item['text']
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        captured_in.clear()
        captured_out.clear()
        _ = model(**enc, use_cache=False)
        for l in candidate_layers:
            in_vecs[l].append(captured_in[l])
            out_vecs[l].append(captured_out[l])
        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            print(f'  [{label}] {i+1}/{len(items)} done', flush=True)
    return {l: torch.stack(v, dim=0) for l, v in in_vecs.items()}, {l: torch.stack(v, dim=0) for l, v in out_vecs.items()}

basis_cache = {
    'key_prefix_fmt': 'base_model.model.model.language_model.layers.{layer}.mlp.down_proj',
    'd_in': 10240,
    'd_out': 2560,
    'num_layers': num_layers,
    'source': 'live_forward_activations_v2',
    'faults': {
        'edit_file_loop': {'layers': {}},
        'premature_submit': {'layers': {}},
    }
}
diag_layers = {'edit_file_loop': [], 'premature_submit': []}

for fault in ('edit_file_loop', 'premature_submit'):
    print(f'=== Harvesting fault={fault} ===', flush=True)
    in_fail, out_fail = collect_group(bundle[fault]['fail'], f'{fault}:fail')
    in_ok, out_ok = collect_group(bundle[fault]['healthy'], f'{fault}:healthy')

    for l in candidate_layers:
        # 1. Detector v_l (row of lora_A): contrastive MLP intermediate activation (fail - healthy)
        # Standardized Cohen's d per neuron so high-signal failure neurons dominate top-k!
        mu_f = in_fail[l].mean(dim=0)
        mu_h = in_ok[l].mean(dim=0)
        std_pooled = torch.sqrt(0.5 * (in_fail[l].var(dim=0, unbiased=False) + in_ok[l].var(dim=0, unbiased=False)) + 1e-6)
        t_stat = (mu_f - mu_h) / std_pooled

        # Layer separation metric: mean |Cohen's d| of top-64 most contrastive neurons
        top64_d = torch.topk(t_stat.abs(), 64).values.mean().item()
        v_dir = t_stat / (t_stat.norm() + 1e-8)

        # 2. Residual steering direction u_l (col of lora_B): healthy - fail output of down_proj
        u_diff = out_ok[l].mean(dim=0) - out_fail[l].mean(dim=0)
        u_dir = u_diff / (u_diff.norm() + 1e-8)

        basis_cache['faults'][fault]['layers'][l] = {
            'u': u_dir.to(torch.bfloat16),
            'v': v_dir.to(torch.bfloat16),
            'cohen_d': round(top64_d, 4),
        }
        diag_layers[fault].append({'layer': l, 'cohen_d': round(top64_d, 4), 'u_diff_norm': round(u_diff.norm().item(), 4)})

for h in handles:
    h.remove()

for fault in diag_layers:
    diag_layers[fault].sort(key=lambda x: x['cohen_d'], reverse=True)

WORKING = Path('/kaggle/working')
torch.save(basis_cache, WORKING / 'basis_cache.pt')
diag = {
    'model': 'gemma-4-e4b-it-qat-w4a16-ct',
    'source': 'live_forward_activations_v2',
    'key_prefix_fmt': basis_cache['key_prefix_fmt'],
    'd_in': 10240,
    'd_out': 2560,
    'num_hidden_layers': num_layers,
    'candidate_layers_analyzed': candidate_layers,
    'fault_rankings': diag_layers,
    'recommended_layers_edit': [x['layer'] for x in diag_layers['edit_file_loop'][:6]],
    'recommended_layers_premature': [x['layer'] for x in diag_layers['premature_submit'][:6]],
}
(WORKING / 'basis_diagnostics.json').write_text(json.dumps(diag, indent=2) + '\n', encoding='utf-8')
print('DONE! Top edit_file_loop layers:', diag['recommended_layers_edit'], flush=True)
print('Top premature_submit layers:', diag['recommended_layers_premature'], flush=True)
