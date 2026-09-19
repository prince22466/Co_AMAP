#!/usr/bin/env python3
"""Train v20 on real Kaggriculture game-history seeds against frozen v19.

Seeds are read from replay JSON configuration/info, never inferred from filenames.
A deterministic train/validation split is created once. PPO updates use only
training seeds. After every update the current policy is evaluated
deterministically against v19 on held-out seeds from both seats. Training stops
when held-out win rate is strictly greater than the requested target.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from train_v20_ppo import (
    ActorCritic,
    DEFAULT_EXECUTOR,
    GLOBAL_FEATURE_NAMES,
    TASK_FEATURE_NAMES,
    choose_device,
    discounted_returns,
    ppo_update,
    prepare_opponents,
    run_episode,
    save_checkpoint,
    terminal_reward,
    write_jsonl,
)

HERE = Path(__file__).resolve().parent
G5_ROOT = HERE.parent.parent
DEFAULT_HISTORY_ROOT = G5_ROOT / "game_history"


def seed_hint(history: dict):
    info = history.get("info") or {}
    cfg = history.get("configuration") or {}
    for obj in (info, cfg):
        for key in ("seed", "randomSeed", "random_seed"):
            if key in obj and obj[key] is not None:
                return int(obj[key])
    return None


def load_history_seeds(root: Path) -> tuple[list[int], list[dict]]:
    rows = []
    seeds = set()
    for path in sorted(root.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                history = json.load(f)
            seed = seed_hint(history)
            rows.append({"path": str(path), "seed": seed})
            if seed is not None:
                seeds.add(int(seed))
        except Exception as exc:
            rows.append({"path": str(path), "seed": None, "error": f"{type(exc).__name__}: {exc}"})
    return sorted(seeds), rows


def split_seeds(seeds: list[int], validation_fraction: float, split_seed: int):
    if len(seeds) < 2:
        raise ValueError("need at least two unique replay seeds")
    xs = list(seeds)
    random.Random(split_seed).shuffle(xs)
    n_val = max(1, int(round(len(xs) * validation_fraction)))
    n_val = min(n_val, len(xs) - 1)
    val = sorted(xs[:n_val])
    train = sorted(xs[n_val:])
    return train, val


def evaluate(model, device, executor, opponent, seeds, episode_steps, baseline_scale):
    rows = []
    for seed in seeds:
        for seat in (0, 1):
            result, _ = run_episode(
                model=model,
                device=device,
                executor_path=executor,
                opponent=opponent,
                seed=seed,
                seat=seat,
                episode_steps=episode_steps,
                baseline_scale=baseline_scale,
                deterministic=True,
                forced_baseline=False,
                collect_steps=False,
            )
            rows.append(result)
    ok = [r for r in rows if r.ok and r.margin is not None]
    if not ok:
        return {
            "games": len(rows),
            "games_ok": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "win_rate": 0.0,
            "mean_margin": None,
            "rows": rows,
        }
    margins = np.asarray([float(r.margin) for r in ok], dtype=np.float64)
    return {
        "games": len(rows),
        "games_ok": len(ok),
        "wins": int((margins > 0).sum()),
        "ties": int((margins == 0).sum()),
        "losses": int((margins < 0).sum()),
        "win_rate": float((margins > 0).mean()),
        "mean_margin": float(margins.mean()),
        "rows": rows,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    p.add_argument("--executor", type=Path, default=DEFAULT_EXECUTOR)
    p.add_argument("--output-dir", type=Path, default=HERE / "runs" / "history_vs_v19")
    p.add_argument("--target-win-rate", type=float, default=0.70)
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument("--split-seed", type=int, default=20260919)
    p.add_argument("--training-seed", type=int, default=32020)
    p.add_argument("--max-updates", type=int, default=1000)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--episode-steps", type=int, default=720)
    p.add_argument("--device", default="auto")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--baseline-scale", type=float, default=1.0)
    p.add_argument("--resume", type=Path)
    return p


def main():
    args = parser().parse_args()
    if not (0.0 < args.validation_fraction < 1.0):
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not (0.0 < args.target_win_rate <= 1.0):
        raise SystemExit("--target-win-rate must be in (0,1]")

    history_root = args.history_root.expanduser().resolve()
    executor = args.executor.expanduser().resolve()
    if not history_root.is_dir():
        raise SystemExit(f"history root not found: {history_root}")
    if not executor.is_file():
        raise SystemExit(f"v19 executor not found: {executor}")

    all_seeds, source_rows = load_history_seeds(history_root)
    if len(all_seeds) < 2:
        raise SystemExit("fewer than two usable real game-history seeds found")
    train_seeds, val_seeds = split_seeds(
        all_seeds, args.validation_fraction, args.split_seed
    )

    device = choose_device(args.device)
    random.seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    rng = random.Random(args.training_seed)

    model = ActorCritic(
        len(TASK_FEATURE_NAMES), len(GLOBAL_FEATURE_NAMES), args.hidden
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start_update = 0
    if args.resume:
        from train_v20_ppo import load_checkpoint
        start_update, _ = load_checkpoint(
            args.resume.expanduser().resolve(), model, optimizer, device
        )

    out = args.output_dir.expanduser().resolve()
    ckpts = out / "checkpoints"
    ckpts.mkdir(parents=True, exist_ok=True)

    split_payload = {
        "history_root": str(history_root),
        "source_files": source_rows,
        "unique_seed_count": len(all_seeds),
        "train_seed_count": len(train_seeds),
        "validation_seed_count": len(val_seeds),
        "train_seeds": train_seeds,
        "validation_seeds": val_seeds,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
    }
    (out / "history_seed_split.json").write_text(
        json.dumps(split_payload, indent=2) + "\n", encoding="utf-8"
    )
    (out / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "executor": str(executor),
                "device_resolved": str(device),
                "opponent": "frozen v19",
                "target_condition": "held_out_win_rate > target_win_rate",
                "evaluation_seats": [0, 1],
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory(prefix="v20_history_") as tmp:
        opponent = prepare_opponents([str(executor)], Path(tmp))[0]

        # Baseline measurement before any update.
        baseline_eval = evaluate(
            model, device, executor, opponent, val_seeds,
            args.episode_steps, args.baseline_scale
        )
        baseline_summary = {k: v for k, v in baseline_eval.items() if k != "rows"}
        baseline_summary["update"] = -1
        baseline_summary["phase"] = "initial"
        write_jsonl(out / "validation.jsonl", baseline_summary)
        print(json.dumps(baseline_summary, sort_keys=True))

        if (
            baseline_eval["games_ok"] == baseline_eval["games"]
            and baseline_eval["win_rate"] > args.target_win_rate
        ):
            save_checkpoint(ckpts / "target.pt", model, optimizer, -1, args)
            print(
                f"target already reached: {baseline_eval['win_rate']:.3f} "
                f"> {args.target_win_rate:.3f}"
            )
            return 0

        for update_no in range(start_update, args.max_updates):
            started = time.perf_counter()
            results = []
            steps = []
            return_chunks = []
            attempts = 0

            while len(results) < args.episodes_per_update:
                attempts += 1
                if attempts > args.episodes_per_update * 4:
                    raise RuntimeError("too many failed training episodes")
                seed = rng.choice(train_seeds)
                seat = rng.randrange(2)
                result, episode_steps = run_episode(
                    model=model,
                    device=device,
                    executor_path=executor,
                    opponent=opponent,
                    seed=seed,
                    seat=seat,
                    episode_steps=args.episode_steps,
                    baseline_scale=args.baseline_scale,
                    deterministic=False,
                    forced_baseline=False,
                    collect_steps=True,
                )
                write_jsonl(
                    out / "episodes.jsonl",
                    {"update": update_no, "split": "train", **asdict(result)},
                )
                if (
                    not result.ok
                    or result.terminal_reward is None
                    or not episode_steps
                ):
                    continue
                steps.extend(episode_steps)
                return_chunks.append(
                    discounted_returns(
                        episode_steps,
                        result.terminal_reward,
                        args.gamma,
                        args.episode_steps,
                    )
                )
                results.append(result)

            returns = np.concatenate(return_chunks)
            stats = ppo_update(
                model, optimizer, device, steps, returns,
                args.baseline_scale, args.ppo_epochs, args.minibatch_size,
                args.clip_ratio, args.value_coef, args.entropy_coef,
                args.max_grad_norm,
            )
            margins = np.asarray([float(r.margin) for r in results], dtype=np.float64)
            train_metrics = {
                "update": update_no,
                "episodes": len(results),
                "decisions": len(steps),
                "train_wins": int((margins > 0).sum()),
                "train_ties": int((margins == 0).sum()),
                "train_losses": int((margins < 0).sum()),
                "train_win_rate": float((margins > 0).mean()),
                "train_mean_margin": float(margins.mean()),
                "elapsed_train_seconds": round(time.perf_counter() - started, 3),
                **stats,
            }
            write_jsonl(out / "metrics.jsonl", train_metrics)

            numbered = ckpts / f"update_{update_no:04d}.pt"
            save_checkpoint(numbered, model, optimizer, update_no, args)
            save_checkpoint(ckpts / "latest.pt", model, optimizer, update_no, args)

            eval_started = time.perf_counter()
            val = evaluate(
                model, device, executor, opponent, val_seeds,
                args.episode_steps, args.baseline_scale
            )
            val_summary = {k: v for k, v in val.items() if k != "rows"}
            val_summary.update(
                update=update_no,
                phase="validation",
                elapsed_eval_seconds=round(time.perf_counter() - eval_started, 3),
            )
            write_jsonl(out / "validation.jsonl", val_summary)
            for row in val["rows"]:
                write_jsonl(
                    out / "validation_games.jsonl",
                    {"update": update_no, "split": "validation", **asdict(row)},
                )

            print(json.dumps({**train_metrics, **val_summary}, sort_keys=True))

            if (
                val["games_ok"] == val["games"]
                and val["win_rate"] > args.target_win_rate
            ):
                save_checkpoint(ckpts / "target.pt", model, optimizer, update_no, args)
                (out / "TARGET_REACHED.json").write_text(
                    json.dumps(
                        {
                            "update": update_no,
                            "target_win_rate": args.target_win_rate,
                            "observed_validation_win_rate": val["win_rate"],
                            "wins": val["wins"],
                            "ties": val["ties"],
                            "losses": val["losses"],
                            "validation_games": val["games_ok"],
                            "validation_seeds": val_seeds,
                            "both_seats": True,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    f"TARGET REACHED at update {update_no}: "
                    f"{val['win_rate']:.3f} > {args.target_win_rate:.3f}"
                )
                return 0

    raise SystemExit(
        f"target not reached within {args.max_updates} updates; "
        "resume from checkpoints/latest.pt to continue"
    )


if __name__ == "__main__":
    raise SystemExit(main())
