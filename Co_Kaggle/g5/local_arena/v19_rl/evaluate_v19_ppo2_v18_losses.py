#!/usr/bin/env python3
"""Evaluate train_v19_ppo_2.py against v18's downloaded loss histories.

For each history this performs:
1. Recorded-action game-state parity, ignoring only remainingOverageTime.
2. Exact v18 executor parity with HERD_THRESHOLD=500.
3. Counterfactual fixed HERD_THRESHOLD=800 replay.
4. Counterfactual deterministic update-99 actor replay (no exploration).

The opponent remains on its historical recorded action stream, so stages 3/4
are regression/failure-scenario tests, not adaptive live rematches.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import kaggle_environments
import torch

from evaluate_v19_v18_losses import (
    _agent_observation,
    _environment_from_history,
    _field,
    _final_rewards_from_states,
    _first_diff,
    _infer_v18_seat,
    _plain,
    _recorded_step_actions,
    _saved_final_rewards,
    _seed_hint,
    _sha256,
)
from train_v19_ppo_2 import (
    BASELINE_THRESHOLD,
    DEFAULT_EXECUTOR,
    FEATURE_NAMES,
    HERD_THRESHOLDS,
    HerdController,
    ThresholdActor,
)

HERE = Path(__file__).resolve().parent
G5_ROOT = HERE.parent.parent
DEFAULT_HISTORY_DIR = G5_ROOT / "game_history" / "v18"
DEFAULT_CHECKPOINT = (
    HERE / "runs" / "herd_differentiable_2" / "checkpoints" / "update_0099.pt"
)
DEFAULT_OUTPUT = (
    HERE / "runs" / "herd_differentiable_2" / "v18_loss_replay_update_0099_compare.json"
)
STATIC_TEST_THRESHOLD = 800.0


def _state_core_without_timing(state: Any) -> dict[str, Any]:
    """Game-semantic replay state, excluding wall-clock budget bookkeeping."""
    obs = _plain(_field(state, "observation", {}))
    if isinstance(obs, dict):
        obs.pop("remainingOverageTime", None)
    return {
        "observation": obs,
        "reward": _plain(_field(state, "reward", None)),
        "status": str(_field(state, "status", "")),
    }


def recorded_action_game_parity(history: dict[str, Any]) -> dict[str, Any]:
    """Require identical game state at every step under both recorded streams."""
    env = _environment_from_history(history)
    saved_steps = history["steps"]

    for seat in (0, 1):
        diff = _first_diff(
            _state_core_without_timing(saved_steps[0][seat]),
            _state_core_without_timing(env.steps[0][seat]),
        )
        if diff:
            return {
                "exact": False,
                "mismatch_step": 0,
                "mismatch_seat": seat,
                "mismatch": diff,
            }

    for t in range(1, len(saved_steps)):
        actions = _recorded_step_actions(history, t)
        if any(a is None for a in actions):
            return {
                "exact": False,
                "mismatch_step": t,
                "mismatch_seat": None,
                "mismatch": f"recorded action is None at replay step {t}: {actions}",
            }
        env.step(actions)
        produced = env.steps[-1]
        for seat in (0, 1):
            diff = _first_diff(
                _state_core_without_timing(saved_steps[t][seat]),
                _state_core_without_timing(produced[seat]),
            )
            if diff:
                return {
                    "exact": False,
                    "mismatch_step": t,
                    "mismatch_seat": seat,
                    "mismatch": diff,
                }

    return {
        "exact": True,
        "steps": len(saved_steps),
        "rewards": _final_rewards_from_states(env.steps[-1]),
        "ignored_fields": ["observation.remainingOverageTime"],
    }


def load_actor(
    checkpoint: Path, device: torch.device
) -> tuple[ThresholdActor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != "differentiable_threshold_twin_q_mc":
        raise ValueError(
            f"unexpected checkpoint algorithm: {payload.get('algorithm')!r}"
        )
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema mismatch")
    if tuple(payload.get("herd_thresholds", ())) != HERD_THRESHOLDS:
        raise ValueError("checkpoint threshold-anchor schema mismatch")

    saved_args = payload.get("args", {}) or {}
    actor = ThresholdActor(
        input_dim=len(FEATURE_NAMES),
        hidden=int(saved_args.get("hidden", 64)),
        baseline_bias=float(saved_args.get("baseline_logit_bias", 3.0)),
    ).to(device)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    return actor, payload


def run_replacement(
    history: dict[str, Any],
    candidate_seat: int,
    actor: ThresholdActor,
    device: torch.device,
    executor: Path,
    forced_threshold: float | None,
) -> dict[str, Any]:
    """Candidate policy vs fixed historical opponent actions."""
    env = _environment_from_history(history)
    opponent_seat = 1 - candidate_seat
    saved_steps = history["steps"]

    controller = HerdController(
        executor_path=executor,
        actor=actor,
        device=device,
        exploration_std=0.0,
        forced_threshold=forced_threshold,
    )

    divergences = 0
    first_divergence = None

    for t in range(1, len(saved_steps)):
        obs = _agent_observation(env, candidate_seat)
        candidate_action = controller(obs)
        recorded = _recorded_step_actions(history, t)
        opponent_action = recorded[opponent_seat]
        original_v18_action = recorded[candidate_seat]

        if candidate_action != original_v18_action:
            divergences += 1
            if first_divergence is None:
                first_divergence = {
                    "replay_step": t,
                    "day": int(_field(obs, "day", -1)),
                    "hour": int(_field(obs, "hour", -1)),
                    "recorded_v18_action": original_v18_action,
                    "candidate_action": candidate_action,
                }

        actions = [None, None]
        actions[candidate_seat] = candidate_action
        actions[opponent_seat] = opponent_action
        env.step(actions)

    final = env.steps[-1]
    rewards = _final_rewards_from_states(final)
    statuses = [str(_field(s, "status", "")) for s in final]
    thresholds = [
        {
            "day": int(step.day),
            "threshold": float(step.threshold),
            "anchor_probs": [float(x) for x in step.anchor_probs],
        }
        for step in controller.steps
    ]

    return {
        "rewards": rewards,
        "statuses": statuses,
        "margin": rewards[candidate_seat] - rewards[opponent_seat],
        "action_divergences": divergences,
        "first_action_divergence": first_divergence,
        "thresholds": thresholds,
    }


def _policy_fields(
    prefix: str,
    run: dict[str, Any],
    original_rewards: list[float],
    original_margin: float,
    candidate_seat: int,
) -> dict[str, Any]:
    opponent_seat = 1 - candidate_seat
    margin = float(run["margin"])
    trace = [float(x["threshold"]) for x in run["thresholds"]]
    return {
        f"{prefix}_rewards": run["rewards"],
        f"{prefix}_margin": margin,
        f"{prefix}_margin_improvement": margin - original_margin,
        f"{prefix}_candidate_reward_delta":
            float(run["rewards"][candidate_seat]) - float(original_rewards[candidate_seat]),
        f"{prefix}_opponent_reward_delta":
            float(run["rewards"][opponent_seat]) - float(original_rewards[opponent_seat]),
        f"{prefix}_result": "WIN" if margin > 0 else "LOSS" if margin < 0 else "TIE",
        f"{prefix}_action_divergences": int(run["action_divergences"]),
        f"{prefix}_first_action_divergence": run["first_action_divergence"],
        f"{prefix}_thresholds": run["thresholds"],
        f"{prefix}_threshold_mean": statistics.mean(trace) if trace else None,
        f"{prefix}_threshold_min": min(trace) if trace else None,
        f"{prefix}_threshold_max": max(trace) if trace else None,
    }


def evaluate_one(
    path: Path,
    checkpoint: Path,
    executor: Path,
    actor: ThresholdActor,
    device: torch.device,
    static_threshold: float,
) -> dict[str, Any]:
    history = json.loads(path.read_text(encoding="utf-8"))
    original_rewards = _saved_final_rewards(history)
    row: dict[str, Any] = {
        "episode": path.stem,
        "seed": _seed_hint(history),
        "original_rewards": original_rewards,
        "valid": False,
    }

    control = recorded_action_game_parity(history)
    row["recorded_action_control"] = control
    if not control.get("exact"):
        row["error"] = "recorded-action game-state parity failed"
        return row

    seat = _infer_v18_seat(history)
    opp = 1 - seat
    original_margin = original_rewards[seat] - original_rewards[opp]
    row["v18_seat"] = seat
    row["original_v18_margin"] = original_margin

    if original_margin >= 0:
        row["error"] = f"expected v18 loss but margin is {original_margin}"
        return row

    baseline = run_replacement(
        history, seat, actor, device, executor, forced_threshold=BASELINE_THRESHOLD
    )
    row["baseline_500_parity"] = {
        "rewards": baseline["rewards"],
        "statuses": baseline["statuses"],
        "margin": baseline["margin"],
        "action_divergences": baseline["action_divergences"],
        "first_action_divergence": baseline["first_action_divergence"],
    }
    if (
        baseline["statuses"] != ["DONE", "DONE"]
        or baseline["rewards"] != original_rewards
        or baseline["action_divergences"] != 0
    ):
        row["error"] = "v18 executor parity failed under threshold=500"
        return row

    static_run = run_replacement(
        history, seat, actor, device, executor, forced_threshold=static_threshold
    )
    learned = run_replacement(
        history, seat, actor, device, executor, forced_threshold=None
    )

    if static_run["statuses"] != ["DONE", "DONE"]:
        row["error"] = f"static-{static_threshold:g} replay failed: {static_run['statuses']}"
        return row
    if learned["statuses"] != ["DONE", "DONE"]:
        row["error"] = f"learned replay failed: {learned['statuses']}"
        return row

    row.update(
        _policy_fields(
            "static_test", static_run, original_rewards, original_margin, seat
        )
    )
    row.update(
        _policy_fields(
            "learned", learned, original_rewards, original_margin, seat
        )
    )
    row["learned_minus_static_test_margin"] = (
        float(learned["margin"]) - float(static_run["margin"])
    )
    row["learned_same_actions_as_static_test"] = (
        learned["rewards"] == static_run["rewards"]
        and learned["margin"] == static_run["margin"]
        and learned["action_divergences"] == static_run["action_divergences"]
    )
    row["valid"] = True
    return row


def summarize_policy(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid")]
    margins = [float(r[f"{prefix}_margin"]) for r in valid]
    deltas = [float(r[f"{prefix}_margin_improvement"]) for r in valid]
    divs = [int(r[f"{prefix}_action_divergences"]) for r in valid]
    if not valid:
        return {}
    return {
        "games": len(valid),
        "wins": sum(x > 0 for x in margins),
        "ties": sum(x == 0 for x in margins),
        "losses": sum(x < 0 for x in margins),
        "loss_cases_repaired": sum(x > 0 for x in margins),
        "margin_improved_cases": sum(x > 0 for x in deltas),
        "margin_worsened_cases": sum(x < 0 for x in deltas),
        "margin_unchanged_cases": sum(x == 0 for x in deltas),
        "mean_margin": statistics.mean(margins),
        "median_margin": statistics.median(margins),
        "mean_margin_improvement": statistics.mean(deltas),
        "median_margin_improvement": statistics.median(deltas),
        "mean_action_divergences": statistics.mean(divs),
        "games_with_action_divergence": sum(x > 0 for x in divs),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid")]
    invalid = [r for r in rows if not r.get("valid")]
    out: dict[str, Any] = {
        "games_total": len(rows),
        "games_valid": len(valid),
        "games_invalid": len(invalid),
        "static_test": summarize_policy(rows, "static_test"),
        "learned": summarize_policy(rows, "learned"),
    }
    if valid:
        comparisons = [float(r["learned_minus_static_test_margin"]) for r in valid]
        out["learned_vs_static_test"] = {
            "learned_better_cases": sum(x > 0 for x in comparisons),
            "static_test_better_cases": sum(x < 0 for x in comparisons),
            "equal_margin_cases": sum(x == 0 for x in comparisons),
            "mean_learned_minus_static_test_margin": statistics.mean(comparisons),
            "same_action_outcome_cases": sum(
                bool(r["learned_same_actions_as_static_test"]) for r in valid
            ),
            "mean_learned_threshold": statistics.mean(
                float(r["learned_threshold_mean"])
                for r in valid
                if r["learned_threshold_mean"] is not None
            ),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode", "valid", "seed", "v18_seat", "original_v18_margin",
        "static_test_margin", "static_test_margin_improvement",
        "static_test_action_divergences", "static_test_result",
        "learned_margin", "learned_margin_improvement",
        "learned_action_divergences", "learned_result",
        "learned_threshold_mean", "learned_threshold_min", "learned_threshold_max",
        "learned_minus_static_test_margin", "learned_same_actions_as_static_test",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--executor", type=Path, default=DEFAULT_EXECUTOR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--static-threshold",
        type=float,
        default=DEFAULT_STATIC_TEST_THRESHOLD,
        help="fixed HERD_THRESHOLD to compare with baseline=500 and learned actor",
    )
    p.add_argument("--episodes", default="")
    p.add_argument("--fail-fast", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    history_dir = args.history_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    executor = args.executor.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.output == DEFAULT_OUTPUT and args.static_threshold != DEFAULT_STATIC_TEST_THRESHOLD:
        output = output.with_name(
            f"v18_loss_replay_update_0099_static_{args.static_threshold:g}_compare.json"
        )
    csv_output = output.with_suffix(".csv")

    wanted = {x.strip() for x in args.episodes.split(",") if x.strip()}
    paths = sorted(history_dir.glob("*.json"))
    if wanted:
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise SystemExit(f"no histories found in {history_dir}")
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not executor.is_file():
        raise SystemExit(f"executor not found: {executor}")

    device = torch.device(args.device)
    actor, payload = load_actor(checkpoint, device)

    print(f"engine=kaggle-environments {getattr(kaggle_environments, '__version__', 'unknown')}")
    print(f"histories={len(paths)}")
    print(f"checkpoint={checkpoint}")
    print(f"checkpoint_update={payload.get('update')}")
    print(f"checkpoint_sha256={_sha256(checkpoint)}")
    print(f"executor={executor}")
    print(f"executor_sha256={_sha256(executor)}")
    print("policies=baseline500, static_test, learned deterministic update99")
    print("exploration=disabled")
    print()

    rows: list[dict[str, Any]] = []
    for i, path in enumerate(paths, 1):
        try:
            row = evaluate_one(
                path, checkpoint, executor, actor, device, args.static_threshold
            )
        except Exception as exc:
            row = {
                "episode": path.stem,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)

        if row.get("valid"):
            print(
                f"[{i:2d}/{len(paths)}] {row['episode']} "
                f"v18={row['original_v18_margin']:+.0f} "
                f"static{args.static_threshold:g}={row['static_test_margin']:+.0f} "
                f"Δstatic={row['static_test_margin_improvement']:+.0f} "
                f"learned={row['learned_margin']:+.0f} "
                f"ΔL={row['learned_margin_improvement']:+.0f} "
                f"tau={row['learned_threshold_mean']:.1f} "
                f"divStatic={row['static_test_action_divergences']} "
                f"divL={row['learned_action_divergences']}"
            )
        else:
            print(
                f"[{i:2d}/{len(paths)}] {row['episode']} INVALID "
                f"{row.get('error', '')}"
            )
            control = row.get("recorded_action_control") or {}
            if not control.get("exact", True):
                print(
                    "    parity mismatch:",
                    f"step={control.get('mismatch_step')}",
                    f"seat={control.get('mismatch_seat')}",
                    control.get("mismatch"),
                )
            if args.fail_fast:
                break

        payload_out = {
            "protocol": {
                "mode": "v18_loss_replacement_replay_ppo2_compare",
                "note": (
                    "Recorded opponent actions do not adapt. This is a historical "
                    "failure-scenario regression test, not a live rematch."
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_update": payload.get("update"),
                "checkpoint_sha256": _sha256(checkpoint),
                "executor": str(executor),
                "executor_sha256": _sha256(executor),
                "deterministic_learned_policy": True,
                "exploration_std": 0.0,
                "comparison_thresholds": [500.0, float(args.static_threshold)],
                "replay_parity_ignored_fields": [
                    "observation.remainingOverageTime"
                ],
            },
            "summary": summarize(rows),
            "matches": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
        write_csv(csv_output, rows)

    summary = summarize(rows)
    print()
    print("Summary")
    print(
        f"  valid/total: {summary['games_valid']}/{summary['games_total']} "
        f"(invalid {summary['games_invalid']})"
    )
    for key, label in (
        ("static_test", f"static {args.static_threshold:g}"),
        ("learned", "learned"),
    ):
        s = summary.get(key) or {}
        if not s:
            continue
        print(
            f"  {label}: W/T/L={s['wins']}/{s['ties']}/{s['losses']} "
            f"repaired={s['loss_cases_repaired']} "
            f"mean_margin={s['mean_margin']:+.1f} "
            f"mean_delta={s['mean_margin_improvement']:+.1f} "
            f"games_with_divergence={s['games_with_action_divergence']}"
        )
    comp = summary.get("learned_vs_static_test") or {}
    if comp:
        print(
            f"  learned vs static {args.static_threshold:g}: "
            f"better/equal/worse={comp['learned_better_cases']}/"
            f"{comp['equal_margin_cases']}/"
            f"{comp['static_test_better_cases']} "
            f"mean_delta={comp['mean_learned_minus_static_test_margin']:+.1f} "
            f"same_action_outcomes={comp['same_action_outcome_cases']}"
        )
        print(f"  mean learned threshold={comp['mean_learned_threshold']:.2f}")
    print(f"  JSON: {output}")
    print(f"  CSV:  {csv_output}")
    return 0 if summary["games_invalid"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
