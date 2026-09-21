#!/usr/bin/env python3
"""Evaluate v21 on v20 loss histories with static recorded opponent replies."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from train_v21_static_history import (
    DEFAULT_BASE_EXECUTOR,
    DEFAULT_HISTORY_DIR,
    DEFAULT_V20_SUBMISSION,
    GLOBAL_FEATURE_NAMES,
    TASK_FEATURE_NAMES,
    QuantizedResidualQ,
    _infer_v20_seat,
    _load_history,
    recorded_action_parity,
    run_static_episode,
)

HERE = Path(__file__).resolve().parent
DEFAULT_V21_CHECKPOINT = HERE / "runs" / "static_v20_history" / "checkpoints" / "latest.pt"
DEFAULT_OUTPUT = HERE / "runs" / "v21_v20_static_replay.json"


def _load_model(checkpoint, expected_algorithm, device, hidden):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != expected_algorithm:
        raise ValueError(
            f"{checkpoint}: expected {expected_algorithm}, got {payload.get('algorithm')!r}"
        )
    quantization = payload.get("quantization", "fp16")
    model = QuantizedResidualQ(
        len(TASK_FEATURE_NAMES),
        len(GLOBAL_FEATURE_NAMES),
        hidden,
        quantization,
    ).to(device)
    model.load_state_dict(payload["online_state_dict"])
    model.eval()
    return model, payload


def evaluate_one(path, v21_model, device, args):
    history = _load_history(path)
    original_rewards = [
        float(history["steps"][-1][0]["reward"]),
        float(history["steps"][-1][1]["reward"]),
    ]
    v20_seat = _infer_v20_seat(history)
    opponent_seat = 1 - v20_seat
    original_margin = original_rewards[v20_seat] - original_rewards[opponent_seat]
    row = {
        "episode": path.stem,
        "valid": False,
        "v20_seat": v20_seat,
        "original_rewards": original_rewards,
        "original_v20_margin": original_margin,
    }

    control = recorded_action_parity(history)
    row["recorded_action_control"] = control
    if not control.get("exact"):
        row["error"] = "recorded-action parity failed"
        return row

    v21, _ = run_static_episode(
        path, v21_model, device, args.base_executor, args, random.Random(1),
        deterministic=True, collect=False,
    )
    if not v21["ok"]:
        row["error"] = v21["error"]
        return row

    v21_margin = float(v21["margin"])
    row.update(
        valid=True,
        v21_rewards=v21["rewards"],
        v21_margin=v21_margin,
        margin_improvement=v21_margin - original_margin,
        candidate_reward_delta=float(v21["rewards"][v20_seat]) - original_rewards[v20_seat],
        opponent_reward_delta=float(v21["rewards"][opponent_seat]) - original_rewards[opponent_seat],
        result="WIN" if v21_margin > 0 else "LOSS" if v21_margin < 0 else "TIE",
    )
    return row


def summarize(rows):
    valid = [row for row in rows if row.get("valid")]
    if not valid:
        return {"games_total": len(rows), "games_valid": 0, "games_invalid": len(rows)}
    margins = [float(row["v21_margin"]) for row in valid]
    improvements = [float(row["margin_improvement"]) for row in valid]
    return {
        "games_total": len(rows),
        "games_valid": len(valid),
        "games_invalid": len(rows) - len(valid),
        "wins": sum(m > 0 for m in margins),
        "ties": sum(m == 0 for m in margins),
        "losses": sum(m < 0 for m in margins),
        "loss_cases_repaired": sum(m > 0 for m in margins),
        "repair_rate": sum(m > 0 for m in margins) / len(margins),
        "margin_improved_cases": sum(x > 0 for x in improvements),
        "margin_worsened_cases": sum(x < 0 for x in improvements),
        "margin_unchanged_cases": sum(x == 0 for x in improvements),
        "mean_v21_margin": float(np.mean(margins)),
        "mean_margin_improvement": float(np.mean(improvements)),
        "worst_v21_margin": min(margins),
        "best_v21_margin": max(margins),
    }


def write_csv(path, rows):
    fields = [
        "episode", "valid", "v20_seat", "original_v20_margin", "v21_margin",
        "margin_improvement", "candidate_reward_delta", "opponent_reward_delta",
        "result", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    p.add_argument("--base-executor", type=Path, default=DEFAULT_BASE_EXECUTOR)
    p.add_argument("--v20-submission", type=Path, default=DEFAULT_V20_SUBMISSION)
    p.add_argument("--v21-checkpoint", type=Path, default=DEFAULT_V21_CHECKPOINT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--episodes", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--prior-scale", type=float, default=1.0)
    p.add_argument("--explore-top-k", type=int, default=3)
    p.add_argument("--bootstrap-candidates", type=int, default=32)
    p.add_argument("--reward-scale", type=float, default=10000.0)
    p.add_argument("--reward-clip", type=float, default=2.0)
    return p


def main():
    args = build_parser().parse_args()
    args.current_epsilon = 0.0
    args.history_dir = args.history_dir.expanduser().resolve()
    args.base_executor = args.base_executor.expanduser().resolve()
    args.v20_submission = args.v20_submission.expanduser().resolve()
    args.v21_checkpoint = args.v21_checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    wanted = {x.strip() for x in args.episodes.split(",") if x.strip()}
    paths = sorted(args.history_dir.glob("*.json"))
    if wanted:
        paths = [path for path in paths if path.stem in wanted]
    if not paths:
        raise SystemExit("no matching histories")

    device = torch.device(args.device)
    v21_model, v21_payload = _load_model(
        args.v21_checkpoint, "v21_static_pure_q_delivered_value_double_dqn", device, args.hidden
    )

    rows = []
    for index, path in enumerate(paths, 1):
        try:
            row = evaluate_one(path, v21_model, device, args)
        except Exception as exc:
            row = {"episode": path.stem, "valid": False, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if row.get("valid"):
            print(
                f"[{index:2d}/{len(paths)}] {row['episode']} "
                f"v20={row['original_v20_margin']:+.0f} "
                f"v21={row['v21_margin']:+.0f} delta={row['margin_improvement']:+.0f} "
                f"{row['result']}"
            )
        else:
            print(f"[{index:2d}/{len(paths)}] {row['episode']} INVALID {row.get('error', '')}")

    result = {
        "protocol": {
            "mode": "v21_static_replacement_replay_on_v20_losses",
            "history_dir": str(args.history_dir),
            "base_executor": str(args.base_executor),
            "v20_submission": str(args.v20_submission),
            "v20_reference": "recorded game_history/v20 terminal rewards and action stream",
            "v21_checkpoint": str(args.v21_checkpoint),
            "v21_checkpoint_update": v21_payload.get("update"),
            "recorded_action_parity_required": True,
            "v20_policy_parity_required": False,
            "opponent_behavior": "recorded action stream; non-adaptive",
        },
        "summary": summarize(rows),
        "matches": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"].get("games_invalid", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
