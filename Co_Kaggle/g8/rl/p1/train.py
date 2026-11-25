"""P1: train/evaluate residual-profile PPO on the official P0 simulator."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from multiprocessing import get_context
import os
import random
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.distributions import Categorical

from ..p0.benchmark import default_data_root, discover_scenarios, verify_training_manifest
from ..p0.controllers import load_controller, validate_action
from ..p0.metrics import weighted_losses
from ..p0.runner import simulator_working_directory
from .model import ActorCritic, ProfileController, PROFILES, N_FEATURES, advantages

GAMMA = .999
REWARD_SCALE = 10.  # Positive constant only; evaluation always uses raw reward.
CHECKPOINT_SCHEMA = 1
SPLIT = {"train": (1, 2, 4, 7), "validation": (5,), "test": (8,)}


def split_scenarios(scenarios):
    groups = {key: [s for s in scenarios
                    if s.split("/")[0] in {f"building_train_{i}" for i in ids}]
              for key, ids in SPLIT.items()}
    if sorted(s for group in groups.values() for s in group) != sorted(scenarios):
        raise ValueError("Unassigned or duplicated scenario")
    if [len(groups[k]) for k in SPLIT] != [20, 5, 5]:
        raise ValueError("Expected 20/5/5 building-disjoint scenarios")
    return groups


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rollout(task, model_factory=ActorCritic):
    """A fresh environment per job; no threads because Gin/cwd are global."""
    root_text, scenario, steps, interval, weights, mode, seed = task
    torch.set_num_threads(1)
    if mode not in {"train", "eval", "baseline"}:
        raise ValueError(f"Unknown rollout mode: {mode}")
    if interval < 1 or (steps is not None and steps < 1):
        raise ValueError("Invalid rollout horizon or interval")
    if mode == "train" and steps is not None and steps % interval:
        raise ValueError("Training horizon must contain whole decision intervals")
    # Rollout sampling must not reset the optimizer's CPU/CUDA RNG stream.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    root = Path(root_text).resolve()
    cache = Path(tempfile.gettempdir()) / "g8_rl_p1_cache" / str(os.getpid())
    for key, name in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "mpl"),
                      ("XDG_CACHE_HOME", "xdg")):
        path = cache / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Initializing a temporary model consumes CPU RNG; isolate that too.
    with torch.random.fork_rng(devices=[]):
        model = model_factory()
    model.load_state_dict(weights)
    model.eval()
    config = root / "smart_buildings" / "training_scenarios" / scenario / "config.gin"
    with simulator_working_directory(root):
        from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
        env = load_hybrid_environment(config)
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0  # Match gateway/P0 typo.
        try:
            spec = env.action_spec()
            metadata = create_building_meta_data(env, config.parent)
            baseline = load_controller(23)(metadata, spec)
            controller = ProfileController(baseline)
            obs, _ = env.reset()
            configured = int(env._num_timesteps_in_episode)
            if configured != 8640:
                raise ValueError("Unexpected simulator horizon")
            limit = steps if steps is not None else configured + 2
            count = 0
            raw_sum = 0.
            parts_sum = dict.fromkeys(("comfort", "energy", "carbon"), 0.)
            histogram = [0] * len(PROFILES)
            trajectory = {k: [] for k in ("x", "a", "logp", "v", "r", "duration")}
            done = terminated = truncated = False
            while count < limit and not done:
                if mode == "baseline":
                    # Independent, unmodified V23 path for a genuine parity check.
                    profile = 0
                else:
                    x = controller.features(obs)
                    with torch.no_grad():
                        logits, value = model(torch.from_numpy(x))
                        distribution = Categorical(logits=logits)
                        action = (torch.multinomial(distribution.probs, 1,
                                                    generator=generator).squeeze(0)
                                  if mode == "train" else logits.argmax())
                        logp = distribution.log_prob(action)
                    profile = int(action)
                    controller.select(profile)
                macro_reward = 0.
                duration = 0
                for j in range(min(interval, limit - count)):
                    policy = baseline if mode == "baseline" else controller
                    command = validate_action(policy.get_action(obs), spec)
                    obs, reward, terminated, truncated, _ = env.step(command)
                    reward = float(reward)
                    if not math.isfinite(reward):
                        raise ValueError(f"Nonfinite reward: {scenario}/{count}")
                    macro_reward += GAMMA ** j * REWARD_SCALE * reward
                    raw_sum += reward
                    if mode != "train":
                        _, response = env.get_reward_info_and_response()
                        parts = weighted_losses(
                            response.normalized_productivity_regret,
                            response.normalized_energy_cost,
                            response.normalized_carbon_emission,
                            env.reward_function.weights)
                        # Same per-step tolerance as P0 diagnostics_core: the
                        # simulator reward passes through float32 protos.
                        if not math.isclose(-reward, sum(parts.values()),
                                            abs_tol=1e-8, rel_tol=1e-7):
                            raise ValueError(
                                f"Reward decomposition mismatch {scenario}/{count}: "
                                f"{-reward!r} vs {sum(parts.values())!r}")
                        for k in parts_sum:
                            parts_sum[k] += parts[k]
                    count += 1
                    duration += 1
                    histogram[profile] += 1
                    done = bool(terminated or truncated)
                    if done:
                        break
                if mode == "train":
                    for key, val in zip(trajectory, (x, profile, float(logp), float(value),
                                                     macro_reward, duration)):
                        trajectory[key].append(val)
            if (steps is None and (not terminated or count < configured)) or (
                    steps is not None and count != steps) or (terminated and count < configured):
                raise RuntimeError(f"Incomplete rollout {scenario}: {count}")
            row = {"scenario": scenario, "steps": count, "mean_loss": -raw_sum / count,
                   "full_episode": bool(terminated and count >= configured),
                   "terminated": bool(terminated), "truncated": bool(truncated),
                   "profile_steps": histogram}
            if mode != "train":
                row["loss_components"] = {k: v / count for k, v in parts_sum.items()}
            else:
                # Partial windows are time limits, not simulator terminal states.
                with torch.no_grad():
                    _, next_value = model(torch.from_numpy(controller.features(obs)))
                adv, target = advantages(trajectory["r"], trajectory["v"],
                                         trajectory["duration"],
                                         0. if terminated else float(next_value), GAMMA)
                row["trajectory"] = {**trajectory, "adv": adv, "target": target}
            return row
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def run_jobs(pool, root, scenarios, steps, interval, model, mode, seed):
    weights = snapshot(model)
    tasks = [(str(root), s, steps, interval, weights, mode, seed + i)
             for i, s in enumerate(scenarios)]
    return list(map(rollout, tasks)) if pool is None else list(pool.map(rollout, tasks))


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


def mean_loss(rows):
    return float(np.mean([r["mean_loss"] for r in rows]))


def compare(pool, root, scenarios, steps, interval, model, seed):
    baseline = run_jobs(pool, root, scenarios, steps, interval, model, "baseline", seed)
    learned = run_jobs(pool, root, scenarios, steps, interval, model, "eval", seed)
    return comparison(baseline, learned)


def comparison(baseline, learned):
    if not baseline or [(r["scenario"], r["steps"], r["full_episode"]) for r in baseline] != [
            (r["scenario"], r["steps"], r["full_episode"]) for r in learned]:
        raise ValueError("Unpaired evaluation")
    return {"baseline_loss": mean_loss(baseline), "rl_loss": mean_loss(learned),
            "delta": mean_loss(learned) - mean_loss(baseline),
            "baseline": baseline, "rl": learned}


def assert_initial_parity(report):
    """Compare every scenario and component, not just canceling mean errors."""
    for baseline, learned in zip(report["baseline"], report["rl"]):
        same_loss = math.isclose(baseline["mean_loss"], learned["mean_loss"],
                                 abs_tol=1e-9, rel_tol=1e-9)
        same_parts = all(math.isclose(v, learned["loss_components"][k],
                                      abs_tol=1e-9, rel_tol=1e-9)
                         for k, v in baseline["loss_components"].items())
        if not same_loss or not same_parts or learned["profile_steps"][0] != learned["steps"]:
            raise RuntimeError(f"Initial greedy policy differs from V23: {baseline['scenario']}")


def checkpoint_signature():
    return {"schema": CHECKPOINT_SCHEMA, "features": N_FEATURES,
            "profiles": [[name, list(factors)] for name, factors in PROFILES],
            "baseline_version": 23}


def save_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


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
