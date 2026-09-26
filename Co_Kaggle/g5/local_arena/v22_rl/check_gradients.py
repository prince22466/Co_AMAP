"""Inspect real-replay FP16 gradients without modifying training checkpoints."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

import train_v21_static_history as train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=train.DEFAULT_OUTPUT_DIR / 'checkpoints/latest.pt')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('gradient_audit.json'))
    parser.add_argument('--batches', type=int, default=16)
    cli = parser.parse_args()
    # Read once so an active trainer cannot change the checkpoint during inspection.
    import io
    raw = cli.checkpoint.read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    args = train.build_parser().parse_args([])
    for key, value in payload['args'].items():
        setattr(args, key, value)
    args.current_epsilon = 0.0
    args.gradient_steps_per_update = cli.batches
    args.replay_warmup = args.batch_size
    device = torch.device('cpu')
    online = train.QuantizedResidualQ(len(train.TASK_FEATURE_NAMES), len(train.GLOBAL_FEATURE_NAMES), args.hidden, args.quantization)
    target = train.QuantizedResidualQ(len(train.TASK_FEATURE_NAMES), len(train.GLOBAL_FEATURE_NAMES), args.hidden, args.quantization)
    online.load_state_dict(payload['online_state_dict'])
    target.load_state_dict(payload['target_state_dict'])
    paths = train._history_paths(Path(args.history_dir))
    training, _ = train._split_histories(paths, args.validation_fraction, args.split_seed)
    history = training[0]
    print(f'Collecting {history.name}; checkpoint update={payload["update"]}, lr={args.learning_rate}', flush=True)
    result, transitions = train.run_static_episode(history, online, device, Path(args.base_executor), args, random.Random(123), deterministic=True, collect=True)
    if not result['ok']:
        raise RuntimeError(result['error'])
    print(f'Collected {len(transitions)} transitions', flush=True)
    replay = train.ReplayBuffer(max(len(transitions), args.batch_size))
    replay.extend(transitions)
    rows = []
    norms = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, *a, **kw):
        norm = original_clip(parameters, *a, **kw)
        norms.append(float(norm))
        return norm

    class AuditSGD(torch.optim.SGD):
        @torch.no_grad()
        def step(self, closure=None):
            for name, param in online.named_parameters():
                grad = param.grad
                g = grad.float()
                row = dict(batch=len(norms)-1, layer=name, count=param.numel(),
                           gradient_dtype=str(grad.dtype), finite=bool(torch.isfinite(g).all()),
                           nonzero=int(torch.count_nonzero(g)), mean_abs=float(g.abs().mean()),
                           max_abs=float(g.abs().max()), changed_by_lr={})
                for lr in sorted(set([1e-4, 1e-3, 1e-2, float(args.learning_rate)])):
                    updated = param.detach().clone().add_(grad, alpha=-lr)
                    row['changed_by_lr'][str(lr)] = int(torch.count_nonzero(updated != param))
                rows.append(row)
            return super().step(closure)

    optimizer = AuditSGD(online.parameters(), lr=args.learning_rate)
    torch.nn.utils.clip_grad_norm_ = capture_clip
    try:
        stats, _ = train.q_update_v21(online, target, optimizer, replay, device, args, random.Random(123), 0)
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    summary = []
    for name, _ in online.named_parameters():
        group = [r for r in rows if r['layer'] == name]
        summary.append(dict(layer=name, count=group[0]['count'],
            mean_abs_gradient=float(np.mean([r['mean_abs'] for r in group])),
            max_abs_gradient=max(r['max_abs'] for r in group),
            mean_nonzero=float(np.mean([r['nonzero'] for r in group])),
            mean_changed_by_lr={lr:float(np.mean([r['changed_by_lr'][lr] for r in group])) for lr in group[0]['changed_by_lr']}))
    report = dict(checkpoint=str(cli.checkpoint), checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                  update=payload['update'], optimizer_steps=payload['optimizer_steps'],
                  learning_rate=args.learning_rate, quantization=args.quantization,
                  episode=result, transition_count=len(transitions), batches=cli.batches,
                  stats=stats, gradient_norms_before_clip=norms, summary=summary, details=rows)
    cli.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(summary=summary, gradient_norm_range=[min(norms),max(norms)], stats=stats),indent=2))
    print(f'Saved {cli.output}', flush=True)


if __name__ == '__main__':
    main()
