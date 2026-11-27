"""Wait for remaining screen_ml32 adapters, download results, and inspect Gemma4 post_feedforward_layernorm & logit margins."""
import json
import subprocess
import time
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, remote_exec


def main():
    # 1. Wait up to 600s for screen_ml32.rc to finish
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        st = remote_exec("""
from pathlib import Path
rc_p = Path('/content/wsp/screen_ml32.rc')
log = Path('/content/wsp/screen_ml32.log').read_text(errors='replace')[-2500:] if Path('/content/wsp/screen_ml32.log').exists() else ''
for l in log.splitlines():
    if 'ML_SCREEN_RESULT' in l or 'Error' in l or 'Traceback' in l:
        print(l)
if rc_p.is_file() and rc_p.read_text().strip() != '':
    print('ML_DONE_RC=' + rc_p.read_text().strip())
""", timeout=90)
        print(st, flush=True)
        if "ML_DONE_RC=" in st:
            break
        time.sleep(20)

    subprocess.run(
        [CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_multilayer_r1/summary.json", str(HERE / "screen_multilayer_summary.json")],
        check=True,
    )
    print("\n=== FULL MULTI-LAYER SCREEN SUMMARY ===", flush=True)
    print((HERE / "screen_multilayer_summary.json").read_text(encoding="utf-8"), flush=True)

    # 2. Extract detailed unchecked generations + inspect Gemma4 layer structure & iter_04 perturbation scale!
    probe = remote_exec("""
import inspect, json, zipfile
from pathlib import Path
import torch
from safetensors.torch import load
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Inspect unchecked outputs from champion_r1, ML2_late_35_41_hi_candidate, ML6_coherent_21_25_32_33_35_41_hi_candidate
d = Path('/content/wsp/harvest32/screen_multilayer_r1')
out = {}
for p in sorted(d.glob('*.json')):
    if p.name == 'summary.json':
        continue
    rows = json.loads(p.read_text())
    unch = [r for r in rows if r.get('role') == 'unchecked']
    flips = sum(1 for r in unch if r.get('flipped_to_targeted_check'))
    if flips >= 1 or p.stem in ('champion_r1', 'ML2_late_35_41_hi_candidate', 'ML6_coherent_21_25_32_33_35_41_hi_candidate'):
        out[p.stem] = [
            {
                'issue': r['issue'],
                'action': r.get('action'),
                'command': r.get('command'),
                'defect_catch': r.get('discriminating_defect_catch'),
                'agent_exit': r.get('on_returned_patch_exit'),
                'ref_exit': r.get('on_reference_fix_exit'),
                'tail': r.get('text_tail', '')[-220:],
            }
            for r in unch
        ]
Path('/content/wsp/harvest32/screen_multilayer_r1/unchecked_details.json').write_text(json.dumps(out, indent=2))

# Inspect Gemma4 decoder layer module names & iter_04 tensor norms!
assets = json.loads(Path('/content/wsp/assets.json').read_text())
cfg = AutoConfig.from_pretrained(assets['model'], local_files_only=True)
print('Config architectures:', cfg.architectures)

with zipfile.ZipFile('/content/wsp/champion.zip') as z:
    champ = load(z.read('adapters/wsp_r1/adapter_model.safetensors'))
for k in sorted(champ):
    t = champ[k].float()
    print(f'CHAMP {k}: shape={tuple(t.shape)} norm={t.norm().item():.4f} nnz={(t.abs()>0).sum().item()}')

saved = torch.load('/content/wsp/harvest32/summaries_v2.pt', map_location='cpu', weights_only=False)
for l in [29, 31, 34, 37, 38]:
    a = champ[f'base_model.model.model.language_model.layers.{l}.mlp.down_proj.lora_A.weight'].float()[0]
    b = champ[f'base_model.model.model.language_model.layers.{l}.mlp.down_proj.lora_B.weight'].float()[:, 0]
    p0_rows = [r for r in saved['paired'] if r['window'] == 'prefix_end']
    acts = [float((r['unchecked_h'][l].float() @ a).abs() * b.norm() / r['unchecked_y'][l].float().norm()) for r in p0_rows]
    print(f'CHAMP L{l} rho at prefix_end:', [round(x, 4) for x in acts])
""", timeout=180)
    print("\n=== PROBE OUTPUT ===", flush=True)
    print(probe, flush=True)

    subprocess.run(
        [CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_multilayer_r1/unchecked_details.json", str(HERE / "screen_multilayer_unchecked_details.json")],
        check=True,
    )
    print("\n=== UNCHECKED DETAILS ===", flush=True)
    print((HERE / "screen_multilayer_unchecked_details.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
