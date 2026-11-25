"""Train P3 on simulator macro-transitions; evaluate with unmodified V23."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import math

import numpy as np
import torch

from ..p0.benchmark import default_data_root, discover_scenarios, verify_training_manifest
from ..common.evaluation import (assert_initial_parity, comparison, mean_loss,
                                 save_json, split_scenarios)
from ..common.profiles import N_ACTIONS
from ..common.rollout import rollout, run_jobs, snapshot
from .learning import Replay, exploration, optimize
from .model import QPolicy, signature


def select_scenarios(training, rng):
    """One random climate per training building, in deterministic order."""
    buildings = sorted({s.split("/")[0] for s in training})
    return [str(rng.choice(sorted(s for s in training if s.split("/")[0] == b)))
            for b in buildings]


def save_checkpoint(path, model, interval, update, loss, steps):
    model.validate()
    if not math.isfinite(loss):
        raise ValueError("Nonfinite validation loss")
    weights = snapshot(model)
    weights["epsilon"].zero_()
    tmp = path.with_suffix(".tmp")
    torch.save({"model": weights, "signature": signature(), "interval": interval,
                "update": update, "validation_loss": loss,
                "validation_steps": steps}, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--steps", type=int, default=864)
    parser.add_argument("--interval", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--reward-scale", type=float, default=100.)
    parser.add_argument("--epsilon-start", type=float, default=.4)
    parser.add_argument("--epsilon-end", type=float, default=.1)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-steps", type=int, default=32)
    parser.add_argument("--target-sync", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.updates, args.steps, args.interval, args.batch_size,
           args.gradient_steps, args.target_sync, args.eval_every) < 1:
        parser.error("Counts must be positive")
    if args.steps > 8640 or args.replay_capacity < args.batch_size:
        parser.error("Invalid horizon or replay capacity")
    if not args.checkpoint and (args.steps % args.interval or args.steps < 2 * args.interval):
        parser.error("Training requires at least two complete decision intervals")
    if not (0 < args.gamma < 1 and 0 < args.epsilon_end <= args.epsilon_start < 1
            and math.isfinite(args.reward_scale) and args.reward_scale > 0
            and math.isfinite(args.lr) and args.lr > 0):
        parser.error("Invalid learning settings")
    if not 0 <= args.seed < 2**32 - args.updates - 30:
        parser.error("Seed exceeds simulator RNG range")
    if args.full and not args.checkpoint:
        parser.error("--full requires --checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")

    root = args.data_root.resolve()
    scenarios = discover_scenarios(root)
    verify_training_manifest(root, scenarios)
    groups = split_scenarios(scenarios)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    scenario_rng = np.random.default_rng(args.seed)
    replay_rng = np.random.default_rng(args.seed + 1)
    model = QPolicy().to(args.device)
    target = QPolicy().to(args.device)
    target.load_state_dict(model.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    memory = Replay(args.replay_capacity)
    worker = partial(rollout, model_factory=QPolicy, gamma=args.gamma,
                     reward_scale=args.reward_scale)
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"))
            if args.workers > 1 else None)

    def evaluate(selected, mode="eval", steps=args.steps, seed=args.seed):
        model.validate()
        return run_jobs(pool, root, selected, steps, args.interval, model,
                        mode, seed, worker=worker)

    try:
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if state.get("signature") != signature() or state.get("interval") != args.interval:
                parser.error("Checkpoint signature or decision interval mismatch")
            model.load_state_dict(state["model"])
            model.validate()
            model.epsilon.zero_()
            selected = groups[args.split]
            horizon = "full" if args.full else f"{args.steps}steps"
            destination = out / f"evaluation_{args.split}_{horizon}.json"
            if destination.exists():
                parser.error("Evaluation already exists; use a new --out")
            steps = None if args.full else args.steps
            report = comparison(evaluate(selected, "baseline", steps),
                                evaluate(selected, "eval", steps))
            report.update(split=args.split, full=args.full, interval=args.interval,
                          seed=args.seed, checkpoint=str(args.checkpoint.resolve()))
            save_json(destination, report)
            print(f"V23={report['baseline_loss']:.6f} RL={report['rl_loss']:.6f}",
                  flush=True)
            return

        if any((out / name).exists() for name in ("best.pt", "history.json")):
            parser.error("Use a new output directory")
        validation = groups["validation"]
        baseline = evaluate(validation, "baseline")
        report = comparison(baseline, evaluate(validation))
        assert_initial_parity(report)
        best = report["rl_loss"]
        save_checkpoint(out / "best.pt", model, args.interval, 0, best, args.steps)
        history = [{"update": 0, "validation": report}]
        metadata = {"split": groups, "settings": vars(args).copy(), "history": history}
        metadata["settings"].update(data_root=str(root), out=str(out), checkpoint=None)
        save_json(out / "history.json", metadata)
        gradient_count = 0
        for i in range(1, args.updates + 1):
            epsilon = exploration(i, args.updates, args.epsilon_start, args.epsilon_end)
            model.epsilon.fill_(epsilon)
            selected = select_scenarios(groups["train"], scenario_rng)
            rows = evaluate(selected, "train", seed=args.seed + i)
            for row in rows:
                memory.extend(row, args.gamma)
            td_losses = []
            if len(memory) >= args.batch_size:
                for _ in range(args.gradient_steps):
                    batch = memory.sample(args.batch_size, replay_rng, args.device)
                    td_losses.append(optimize(model, target, optimizer, batch))
                    gradient_count += 1
                    if gradient_count % args.target_sync == 0:
                        target.load_state_dict(model.state_dict())
            model.validate()
            entry = {"update": i, "scenarios": selected, "epsilon": epsilon,
                     "train_loss": mean_loss(rows), "replay_size": len(memory),
                     "gradient_count": gradient_count,
                     "td_loss": float(np.mean(td_losses)) if td_losses else None,
                     "simulator_steps": sum(r["steps"] for r in rows),
                     "train_scenarios": [{k: v for k, v in r.items()
                                          if k not in {"trajectory", "bootstrap"}}
                                         for r in rows],
                     "action_counts": np.bincount(
                         [a for r in rows for a in r["trajectory"]["a"]],
                         minlength=N_ACTIONS).tolist()}
            if i % args.eval_every == 0 or i == args.updates:
                model.epsilon.zero_()
                report = comparison(baseline, evaluate(validation))
                entry["validation"] = report
                if report["rl_loss"] < best - 1e-9:
                    best = report["rl_loss"]
                    save_checkpoint(out / "best.pt", model, args.interval, i, best, args.steps)
            history.append(entry)
            save_json(out / "history.json", metadata)
            print(f"update={i} train={entry['train_loss']:.6f} "
                  f"replay={len(memory)} td={entry['td_loss']} best_val={best:.6f}",
                  flush=True)
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
