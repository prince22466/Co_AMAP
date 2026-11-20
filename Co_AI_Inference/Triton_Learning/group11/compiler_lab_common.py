"""Backend-independent helpers for Group11 Triton compiler A/B experiments."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import platform
import re
import socket
import statistics
import sys
from pathlib import Path


def require_runtime():
    try:
        import torch
        import triton
        import triton.testing
    except ImportError as exc:
        raise SystemExit('Requires torch + Triton on the target GPU (e.g. BI-V150).') from exc
    if not torch.cuda.is_available():
        raise SystemExit('No compatible GPU visible through torch.cuda.')
    torch.cuda.set_device(0)
    return torch, triton


def parse_ints(value, *, label, powers_of_two=False):
    try:
        vals = list(dict.fromkeys(int(s.strip()) for s in value.split(',')))
    except ValueError as exc:
        raise ValueError(f'Invalid {label}: {value}') from exc
    if not vals or any(v <= 0 or (powers_of_two and v & (v - 1)) for v in vals):
        raise ValueError(f'{label} must contain positive ' + ('powers of two' if powers_of_two else 'integers'))
    return vals


def check_tensor(torch, got, expected, *, atol=0.002, rtol=0.001):
    """Reference must already have been computed on CPU; GPU conversion is copy only."""
    assert expected.device.type == 'cpu', 'Reference must originate from CPU'
    torch.cuda.synchronize()
    observed = got.detach().cpu()
    if observed.shape != expected.shape:
        raise AssertionError(f'Wrong shape: {tuple(observed.shape)} != {tuple(expected.shape)}')
    if not torch.isfinite(observed).all():
        raise AssertionError('GPU output contains NaN/Inf (possibly unwritten values)')
    delta = (observed.float() - expected.float()).abs()
    max_err = float(delta.max().item()) if delta.numel() else 0.0
    if not torch.allclose(observed, expected, atol=atol, rtol=rtol):
        flat = delta.reshape(-1)
        k = int(flat.argmax().item())
        raise AssertionError(f'CPU-reference mismatch max_abs={max_err:.6g} at flat_index={k}; '
                             f'got={float(observed.reshape(-1)[k]):.7g}, '
                             f'expected={float(expected.reshape(-1)[k]):.7g}')
    return max_err


def measure(triton, fn, *, warmup_ms, rep_ms, trials, single_run):
    if single_run:
        return {'latency_us': '', 'trial_medians_us': ''}
    medians = []
    for _ in range(trials):
        q = triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms,
                                    quantiles=[0.2, 0.5, 0.8])
        val = float(q[1])
        if not math.isfinite(val) or val <= 0:
            raise RuntimeError(f'Invalid GPU event timing {val}')
        medians.append(val)
    return {'latency_us': round(statistics.median(medians)*1000, 6),
            'trial_medians_us': ';'.join(f'{v*1000:.3f}' for v in medians)}


def compiler_metadata(kernel):
    md = getattr(kernel, 'metadata', None)
    return {
        'registers_per_thread': getattr(kernel, 'n_regs', '') or '',
        'spills': getattr(kernel, 'n_spills', '') if getattr(kernel, 'n_spills', None) is not None else '',
        'shared_bytes': getattr(md, 'shared', '') if md else '',
        'backend_warp_size': getattr(md, 'warp_size', '') if md else '',
    }


def save_compiler_artifacts(kernel, outdir, stem):
    """Only files the loaded backend exposes; no assumptions about CUDA/PTX."""
    asm = getattr(kernel, 'asm', None) or {}
    stages = {}
    for stage, obj in asm.items():
        if isinstance(obj, str):
            raw = obj.encode('utf-8')
            kind = 'text'
        elif isinstance(obj, (bytes, bytearray)):
            raw = bytes(obj)
            kind = 'binary'
        else:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        rec = {'kind': kind, 'sha256': digest, 'bytes': len(raw)}
        if kind == 'text':
            rec['lines'] = obj.count('\n') + 1
            # Text heuristics only; neither op count nor a claim about ISA.
            rec['heuristic_load_mentions'] = len(re.findall(r'\b(?:load|ld\.|global_load)\b', obj))
            rec['heuristic_convert_layout_mentions'] = obj.count('convert_layout')
        stages[str(stage)] = rec
        if outdir:
            dest = Path(outdir)
            dest.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(stage))
            (dest/f'{stem}.{safe}.{"txt" if kind == "text" else "bin"}').write_bytes(raw)
    if outdir:
        dest = Path(outdir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest/f'{stem}.stages.json').write_text(json.dumps(stages, indent=2) + '\n', encoding='utf-8')
    # Prefer actual binaries; otherwise identify the highest available textual stage.
    for stage in ('cubin','hsaco','spvbin','amdgcn','sass','ptx','llir','ttgir','ttir'):
        if stage in stages:
            return {'asm_stage': stage, 'asm_sha256': stages[stage]['sha256'],
                    'asm_kind': stages[stage]['kind'], 'compiler_stages_json': json.dumps(stages, sort_keys=True)}
    return {'asm_stage': '', 'asm_sha256': '', 'asm_kind': '',
            'compiler_stages_json': json.dumps(stages, sort_keys=True)}


def write_csv(path, rows):
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f'Wrote {p} ({len(rows)} records)')


def write_environment(csv_path, torch, triton, args):
    p = Path(csv_path).with_suffix('.env.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    info = {'hostname': socket.gethostname(), 'system': platform.platform(),
            'python': sys.version.split()[0], 'torch': torch.__version__,
            'triton': triton.__version__, 'gpu': torch.cuda.get_device_name(0),
            'arguments': vars(args)}
    p.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')


def summarise(rows):
    valid = [r for r in rows if r.get('status') == 'ok']
    for lab in sorted({r['lab'] for r in valid}):
        subset = [r for r in valid if r['lab'] == lab]
        print(f'\n{lab} compiler/timing comparison:')
        base = next((r for r in subset if r.get('latency_us') not in ('',None)), None)
        for r in subset:
            ratio = '' if not base or not r.get('latency_us') else f'{base["latency_us"]/r["latency_us"]:.3f}x'
            same = ('same' if r.get('asm_stage') == base.get('asm_stage') and r.get('asm_sha256') == base.get('asm_sha256')
                    else 'different') if base and base.get('asm_sha256') and r.get('asm_sha256') else 'unknown'
            print(f'  {r["variant"]:22s} latency={str(r.get("latency_us","")):>10s} us '
                  f'speedup_vs_first={ratio or "N/A":>8s} regs={str(r.get("registers_per_thread","")):>4s} '
                  f'spills={str(r.get("spills","")):>4s} '
                  f'artifact_vs_first={same} ({r.get("asm_stage", "none")})')
