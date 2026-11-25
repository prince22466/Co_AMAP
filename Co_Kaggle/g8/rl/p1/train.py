"""P1: train/evaluate residual-profile PPO on the official P0 simulator."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from ..p0.benchmark import default_data_root, discover_scenarios, verify_training_manifest
from ..common.evaluation import (SPLIT, split_scenarios, mean_loss, comparison,
                                 assert_initial_parity, save_json)
from ..common.rollout import rollout as collect_rollout, run_jobs as dispatch_jobs, snapshot
from .model import ActorCritic, PROFILES, N_FEATURES, advantages

GAMMA = .999
REWARD_SCALE = 10.  # Positive constant only; evaluation always uses raw reward.
CHECKPOINT_SCHEMA = 1


def rollout(task):
    """PPO-specific processing on the shared simulator trajectory."""
    row = collect_rollout(task, model_factory=ActorCritic,
                          gamma=GAMMA, reward_scale=REWARD_SCALE)
    if task[5] == "train":
        trajectory = row["trajectory"]
        adv, target = advantages(trajectory["r"], trajectory["v"],
                                 trajectory["duration"], row.pop("bootstrap"), GAMMA)
        trajectory.update(adv=adv, target=target)
    return row


def run_jobs(pool, root, scenarios, steps, interval, model, mode, seed):
    return dispatch_jobs(pool, root, scenarios, steps, interval, model, mode, seed,
                         worker=rollout)


def update(model, optimizer, rows, device, epochs, batch_size):
    """Clipped categorical PPO; all old probabilities are from collected policy."""
    trajectories = [row["trajectory"] for row in rows]
    def tensor(key, dtype=torch.float32):
        return torch.as_tensor(np.concatenate([np.asarray(t[key]) for t in trajectories]),
                               dtype=dtype, device=device)
    x, a, old_logp = tensor("x"), tensor("a", torch.long), tensor("logp")
    adv, target = tensor("adv"), tensor("target")
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    losses = []
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=device).split(batch_size):
            logits, value = model(x[idx])
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(a[idx]) - old_logp[idx]).exp()
            surrogate = torch.minimum(ratio * adv[idx],
                                      ratio.clamp(.8, 1.2) * adv[idx])
            policy_loss = -surrogate.mean()
            value_loss = (value - target[idx]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + .5 * value_loss - .01 * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def compare(pool, root, scenarios, steps, interval, model, seed):
    baseline = run_jobs(pool, root, scenarios, steps, interval, model, "baseline", seed)
    learned = run_jobs(pool, root, scenarios, steps, interval, model, "eval", seed)
    return comparison(baseline, learned)


def checkpoint_signature():
    return {"schema": CHECKPOINT_SCHEMA, "features": N_FEATURES,
            "profiles": [[name, list(factors)] for name, factors in PROFILES],
            "baseline_version": 23}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=default_data_root())
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--steps", type=int, default=864)
    p.add_argument("--interval", type=int, default=12)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--scenarios-per-update", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, help="Evaluate saved best.pt only")
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--full", action="store_true", help="Full evaluation only")
    args = p.parse_args()
    if min(args.workers, args.steps, args.interval, args.updates, args.scenarios_per_update,
           args.eval_every, args.epochs, args.batch_size) < 1 or args.steps > 8640:
        p.error("Positive counts required; --steps <= 8640")
    if not args.checkpoint and args.steps % args.interval:
        p.error("Training --steps must be divisible by --interval")
    if not 0 <= args.seed < 2**32 - 1000 * args.updates - 30:
        p.error("Seed/update count exceeds the simulator RNG seed range")
    if args.full and not args.checkpoint:
        p.error("--full requires --checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA not available")
    root = args.data_root.resolve()
    scenarios = discover_scenarios(root)
    verify_training_manifest(root, scenarios)
    groups = split_scenarios(scenarios)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ActorCritic().to(args.device)
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"))
            if args.workers > 1 else None)
    try:
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            if state.get("signature") != checkpoint_signature():
                p.error("Checkpoint schema or operating profiles do not match this code")
            model.load_state_dict(state["model"])
            if args.interval != state["interval"]:
                p.error("--interval must match checkpoint")
            horizon = "full" if args.full else f"{args.steps}steps"
            destination = args.out / f"evaluation_{args.split}_{horizon}.json"
            if destination.exists():
                p.error(f"Evaluation already exists: {destination}; use a new --out")
            report = compare(pool, root, groups[args.split], None if args.full else args.steps,
                             args.interval, model, args.seed)
            report.update(split=args.split, full=args.full, interval=args.interval,
                          checkpoint=str(args.checkpoint.resolve()), seed=args.seed)
            save_json(destination, report)
            print(f"V23={report['baseline_loss']:.6f} RL={report['rl_loss']:.6f}", flush=True)
            return
        if (args.out / "best.pt").exists():
            p.error("Use a new output directory for a new training run")
        if args.scenarios_per_update > len(groups["train"]):
            p.error("Too many scenarios per update")
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        baseline = run_jobs(pool, root, groups["validation"], args.steps, args.interval,
                            model, "baseline", args.seed)
        initial = run_jobs(pool, root, groups["validation"], args.steps, args.interval,
                           model, "eval", args.seed)
        report = comparison(baseline, initial)
        assert_initial_parity(report)
        best = report["rl_loss"]
        def checkpoint(update_number, score):
            tmp = args.out / "best.tmp"
            torch.save({"model": snapshot(model), "interval": args.interval,
                        "update": update_number, "validation_loss": score,
                        "signature": checkpoint_signature()}, tmp)
            tmp.replace(args.out / "best.pt")
        checkpoint(0, best)
        history = [{"update": 0, "validation": report}]
        save_json(args.out / "history.json", {"split": groups, "seed": args.seed,
                  "steps": args.steps, "interval": args.interval, "history": history})
        for i in range(1, args.updates + 1):
            selected = rng.choice(groups["train"], args.scenarios_per_update,
                                  replace=False).tolist()
            rows = run_jobs(pool, root, selected, args.steps, args.interval,
                            model, "train", args.seed + 1000 * i)
            loss = update(model, optimizer, rows, args.device, args.epochs, args.batch_size)
            entry = {"update": i, "train_loss": mean_loss(rows), "ppo_loss": loss,
                     "scenarios": selected, "transitions": sum(r["steps"] for r in rows),
                     "profile_steps": np.sum([r["profile_steps"] for r in rows],
                                             axis=0).tolist()}
            if i % args.eval_every == 0 or i == args.updates:
                learned = run_jobs(pool, root, groups["validation"], args.steps,
                                   args.interval, model, "eval", args.seed)
                entry["validation"] = comparison(baseline, learned)
                score = entry["validation"]["rl_loss"]
                if score < best - 1e-9:
                    best = score
                    checkpoint(i, best)
            history.append(entry)
            save_json(args.out / "history.json", {"split": groups, "seed": args.seed,
                      "steps": args.steps, "interval": args.interval, "history": history})
            print(f"update={i} train={entry['train_loss']:.6f} best_val={best:.6f}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
