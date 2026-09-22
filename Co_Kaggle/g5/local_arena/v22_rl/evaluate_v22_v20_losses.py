#!/usr/bin/env python3
"""Evaluate a v22 SELL-only PPO checkpoint on held/static v20 loss histories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_v22_selling_history import (
    CHECKPOINT_ALGORITHM,
    DEFAULT_HISTORY_DIR,
    DEFAULT_V20_SUBMISSION,
    STATE_DIM,
    SellActorCritic,
    _infer_v20_seat,
    _load_history,
    choose_device,
    recorded_action_parity,
    run_static_episode,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "runs" / "static_v20_history" / "checkpoints" / "latest.pt"
DEFAULT_OUTPUT = HERE / "runs" / "v22_v20_static_replay.json"


def _load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("algorithm") != CHECKPOINT_ALGORITHM:
        raise ValueError(
            f"{path}: expected {CHECKPOINT_ALGORITHM}, got {payload.get('algorithm')!r}"
        )
    if payload.get("precision") != "fp16":
        raise ValueError(f"{path}: v22 evaluator requires fp16 checkpoint")
    hidden = int(payload.get("hidden", payload.get("args", {}).get("hidden", 128)))
    state_dim = int(payload.get("state_dim", STATE_DIM))
    if state_dim != STATE_DIM:
        raise ValueError(f"state dimension mismatch: checkpoint={state_dim} code={STATE_DIM}")
    model = SellActorCritic(STATE_DIM, hidden).to(device).half()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _args_from_checkpoint(payload, cli):
    saved = dict(payload.get("args") or {})
    cli.gamma = float(saved.get("gamma", cli.gamma))
    cli.margin_bonus = float(saved.get("margin_bonus", cli.margin_bonus))
    cli.margin_scale = float(saved.get("margin_scale", cli.margin_scale))
    cli.price_shaping = float(saved.get("price_shaping", cli.price_shaping))
    cli.overflow_penalty = float(saved.get("overflow_penalty", cli.overflow_penalty))
    return cli


def evaluate_one(path, model, device, args):
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

    result, _, _ = run_static_episode(
        path, model, device, args.v20_submission, args,
        deterministic=True, collect=False,
    )
    if not result["ok"]:
        row["error"] = result["error"]
        return row

    margin = float(result["margin"])
    row.update(
        valid=True,
        v22_rewards=result["rewards"],
        v22_margin=margin,
        margin_improvement=margin - original_margin,
        candidate_reward_delta=float(result["rewards"][v20_seat]) - original_rewards[v20_seat],
        opponent_reward_delta=float(result["rewards"][opponent_seat]) - original_rewards[opponent_seat],
        result="WIN" if margin > 0 else "LOSS" if margin < 0 else "TIE",
        sell_decisions=result["sell_decisions"],
        sale_orders=result["sale_orders"],
        quoted_units_sold=result["quoted_units_sold"],
        quoted_sale_value=result["quoted_sale_value"],
        expected_overflow_units=result["expected_overflow_units"],
        sold_units_by_product=result["sold_units_by_product"],
        quoted_value_by_product=result["quoted_value_by_product"],
        produced_value=result["produced_value"],
        transport_progress_value=result["transport_progress_value"],
        delivered_value=result["delivered_value"],
        produced_units=result["produced_units"],
        delivered_units=result["delivered_units"],
        produced_value_by_product=result["produced_value_by_product"],
        delivered_value_by_product=result["delivered_value_by_product"],
    )
    return row


def summarize(rows):
    valid = [row for row in rows if row.get("valid")]
    if not valid:
        return {"games_total": len(rows), "games_valid": 0, "games_invalid": len(rows)}
    margins = np.asarray([row["v22_margin"] for row in valid], dtype=np.float64)
    improvements = np.asarray([row["margin_improvement"] for row in valid], dtype=np.float64)
    return {
        "games_total": len(rows),
        "games_valid": len(valid),
        "games_invalid": len(rows) - len(valid),
        "wins": int((margins > 0).sum()),
        "ties": int((margins == 0).sum()),
        "losses": int((margins < 0).sum()),
        "repair_rate": float((margins > 0).mean()),
        "margin_improved_cases": int((improvements > 0).sum()),
        "margin_worsened_cases": int((improvements < 0).sum()),
        "margin_unchanged_cases": int((improvements == 0).sum()),
        "mean_v22_margin": float(margins.mean()),
        "mean_margin_improvement": float(improvements.mean()),
        "worst_v22_margin": float(margins.min()),
        "best_v22_margin": float(margins.max()),
        "mean_quoted_sale_value": float(np.mean([row["quoted_sale_value"] for row in valid])),
        "mean_expected_overflow_units": float(np.mean([row["expected_overflow_units"] for row in valid])),
        "mean_produced_value": float(np.mean([row["produced_value"] for row in valid])),
        "mean_delivered_value": float(np.mean([row["delivered_value"] for row in valid])),
        "mean_transport_progress_value": float(np.mean([row["transport_progress_value"] for row in valid])),
    }


def write_csv(path, rows):
    fields = [
        "episode", "valid", "v20_seat", "original_v20_margin", "v22_margin",
        "margin_improvement", "candidate_reward_delta", "opponent_reward_delta",
        "result", "sell_decisions", "sale_orders", "quoted_units_sold",
        "quoted_sale_value", "expected_overflow_units", "produced_value",
        "delivered_value", "transport_progress_value", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    p.add_argument("--v20-submission", type=Path, default=DEFAULT_V20_SUBMISSION)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--episodes", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--margin-bonus", type=float, default=0.25)
    p.add_argument("--margin-scale", type=float, default=10000.0)
    p.add_argument("--price-shaping", type=float, default=0.0005)
    p.add_argument("--overflow-penalty", type=float, default=0.01)
    return p


def main():
    args = build_parser().parse_args()
    args.history_dir = args.history_dir.expanduser().resolve()
    args.v20_submission = args.v20_submission.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    wanted = {x.strip() for x in args.episodes.split(",") if x.strip()}
    paths = sorted(args.history_dir.glob("*.json"))
    if wanted:
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise SystemExit("no matching histories")

    device = choose_device(args.device)
    model, payload = _load_model(args.checkpoint, device)
    args = _args_from_checkpoint(payload, args)
    rows = []
    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            try:
                row = evaluate_one(path, model, device, args)
            except Exception as exc:
                row = {"episode": path.stem, "valid": False, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            if row.get("valid"):
                print(
                    f"[{index:2d}/{len(paths)}] {row['episode']} "
                    f"v20={row['original_v20_margin']:+.0f} "
                    f"v22={row['v22_margin']:+.0f} delta={row['margin_improvement']:+.0f} "
                    f"{row['result']} sale_value={row['quoted_sale_value']:.0f}"
                )
            else:
                print(f"[{index:2d}/{len(paths)}] {row['episode']} INVALID {row.get('error', '')}")

    result = {
        "protocol": {
            "mode": "v22_sell_only_static_replacement_replay_on_v20_losses",
            "history_dir": str(args.history_dir),
            "v20_submission": str(args.v20_submission),
            "checkpoint": str(args.checkpoint),
            "checkpoint_update": payload.get("update"),
            "precision": "fp16",
            "opponent_behavior": "recorded action stream; non-adaptive",
            "control_scope": "SELL only; all other v20 decisions frozen",
            "worker_telemetry": "production/transport/delivery diagnostics only",
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
