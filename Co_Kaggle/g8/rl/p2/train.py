"""Train a deterministic RL policy using paired simulator returns.

No gradient, critic, reward shaping, or simulator-internal policy inputs.
"""
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
from ..common.profiles import N_ACTIONS
from ..common.evaluation import (assert_initial_parity, comparison, mean_loss,
                                 save_json, split_scenarios)
from ..common.rollout import rollout, run_jobs as dispatch_jobs, snapshot
from .model import BOUND, SearchPolicy, candidates, signature


def select_scenarios(training, per_building, rng):
    """Sample equally from each train building, never validation or test."""
    buildings = sorted({s.split("/")[0] for s in training})
    selected = []
    for building in buildings:
        available = sorted(s for s in training if s.split("/")[0] == building)
        selected.extend(rng.choice(available, per_building, replace=False).tolist())
    return sorted(selected)


def run_jobs(pool, root, scenarios, steps, interval, model, mode, seed):
    """Use the same ordered simulator dispatcher as all profile models."""
    if mode not in {"baseline", "eval"}:
        raise ValueError("P2 uses deterministic evaluation rollouts only")
    worker = partial(rollout, model_factory=SearchPolicy)
    return dispatch_jobs(pool, root, scenarios, steps, interval, model, mode, seed,
                         worker=worker)


def choose_candidate(parent, trials, min_gain, max_regression):
    """Return a trial index only for paired, sufficiently improved returns."""
    if (not parent or not all(math.isfinite(x) and x >= 0
                              for x in (min_gain, max_regression))):
        raise ValueError("Invalid candidate comparison settings")
    eligible = []
    for i, trial in enumerate(trials):
        report = comparison(parent, trial)
        differences = [t["mean_loss"] - p["mean_loss"] for p, t in zip(parent, trial)]
        delta = report["delta"]
        if not math.isfinite(delta) or not all(map(math.isfinite, differences)):
            raise ValueError("Nonfinite candidate comparison")
        worst = max(differences)
        if delta < -min_gain and worst <= max_regression:
            eligible.append((delta, i))
    return min(eligible)[1] if eligible else None


def save_checkpoint(path, model, interval, update, loss, steps):
    model.validate_coefficients()
    if not math.isfinite(loss):
        raise ValueError("Nonfinite checkpoint loss")
    tmp = path.with_suffix(".tmp")
    torch.save({"model": snapshot(model), "signature": signature(),
                "interval": interval, "update": update,
                "validation_loss": loss, "validation_steps": steps}, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=default_data_root())
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", choices=("cpu",), default="cpu")
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--steps", type=int, default=864)
    p.add_argument("--interval", type=int, default=12)
    p.add_argument("--scenarios-per-building", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=1,
                   help="Validation cadence; default checks every accepted policy")
    p.add_argument("--sigma", type=float, default=.5)
    p.add_argument("--min-gain", type=float, default=1e-6)
    p.add_argument("--max-regression", type=float, default=.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    if min(args.workers, args.updates, args.steps, args.interval, args.eval_every) < 1:
        p.error("Counts must be positive")
    if args.steps > 8640 or not 1 <= args.scenarios_per_building <= 5:
        p.error("Invalid horizon or number of scenarios per building")
    if not args.checkpoint and args.steps % args.interval:
        p.error("Training steps must be divisible by interval")
    if (not all(math.isfinite(x) for x in
                (args.sigma, args.min_gain, args.max_regression)) or
            not 0 < args.sigma <= BOUND or args.min_gain < 0 or args.max_regression < 0):
        p.error("Invalid search settings")
    if not 0 <= args.seed < 2**32 - 30:
        p.error("Seed exceeds simulator RNG range")
    if args.full and not args.checkpoint:
        p.error("--full requires --checkpoint")
    root = args.data_root.resolve()
    scenarios = discover_scenarios(root)
    verify_training_manifest(root, scenarios)
    groups = split_scenarios(scenarios)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    model = SearchPolicy()
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"))
            if args.workers > 1 else None)

    def evaluate(policy, selected, mode="eval", steps=args.steps):
        return run_jobs(pool, root, selected, steps, args.interval, policy, mode, args.seed)

    try:
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if state.get("signature") != signature() or state.get("interval") != args.interval:
                p.error("Checkpoint policy schema or interval mismatch")
            model.load_state_dict(state["model"])
            model.validate_coefficients()
            horizon = "full" if args.full else f"{args.steps}steps"
            destination = out / f"evaluation_{args.split}_{horizon}.json"
            if destination.exists():
                p.error("Evaluation already exists; use a new --out")
            steps = None if args.full else args.steps
            selected = groups[args.split]
            report = comparison(evaluate(model, selected, "baseline", steps),
                                evaluate(model, selected, "eval", steps))
            report.update(split=args.split, full=args.full, interval=args.interval,
                          seed=args.seed, checkpoint=str(args.checkpoint.resolve()))
            save_json(destination, report)
            print(f"V23={report['baseline_loss']:.6f} RL={report['rl_loss']:.6f}", flush=True)
            return

        if any((out / name).exists() for name in ("best.pt", "history.json")):
            p.error("Use a new output directory")
        validation = groups["validation"]
        baseline = evaluate(model, validation, "baseline")
        initial = evaluate(model, validation)
        report = comparison(baseline, initial)
        assert_initial_parity(report)
        best = report["rl_loss"]
        save_checkpoint(out / "best.pt", model, args.interval, 0, best, args.steps)
        history = [{"update": 0, "validation": report}]
        metadata = {"split": groups, "seed": args.seed, "steps": args.steps,
                    "interval": args.interval, "sigma": args.sigma,
                    "min_gain": args.min_gain, "max_regression": args.max_regression,
                    "scenarios_per_building": args.scenarios_per_building,
                    "eval_every": args.eval_every, "history": history}
        save_json(out / "history.json", metadata)
        # Cache validation of the unchanged incumbent; do not rerun physics
        # for rejected mutations. A dirty incumbent needs fresh validation.
        validation_dirty = False
        for i in range(1, args.updates + 1):
            selected = select_scenarios(groups["train"], args.scenarios_per_building, rng)
            profile = 1 + (i - 1) % (N_ACTIONS - 1)
            proposals = candidates(model, profile, args.sigma, rng)
            parent = evaluate(model, selected)
            trials = [evaluate(candidate, selected) for candidate in proposals]
            accepted = choose_candidate(parent, trials, args.min_gain, args.max_regression)
            entry = {"update": i, "profile_mutated": profile, "scenarios": selected,
                     "parent_loss": mean_loss(parent),
                     "candidates": [comparison(parent, trial) for trial in trials],
                     "accepted": accepted,
                     "candidate_coefficients": [c.coefficients.tolist() for c in proposals],
                     "train_loss": mean_loss(parent if accepted is None else trials[accepted]),
                     "simulator_steps": sum(r["steps"] for rows in [parent, *trials] for r in rows)}
            if accepted is not None:
                model = proposals[accepted]
                validation_dirty = True
            if i % args.eval_every == 0 or i == args.updates:
                entry["validation_reused"] = not validation_dirty
                if validation_dirty:
                    report = comparison(baseline, evaluate(model, validation))
                    validation_dirty = False
                entry["validation"] = report
                score = report["rl_loss"]
                if score < best - 1e-9:
                    best = score
                    save_checkpoint(out / "best.pt", model, args.interval, i, best, args.steps)
            history.append(entry)
            save_json(out / "history.json", metadata)
            print(f"update={i} parent={entry['parent_loss']:.6f} "
                  f"train={entry['train_loss']:.6f} accepted={accepted} "
                  f"best_val={best:.6f}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
