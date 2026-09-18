#!/usr/bin/env python3
"""Strict v19 counterfactual evaluation on v18's downloaded loss histories.

This evaluator is intentionally conservative.

For each replay JSON under game_history/v18 it performs three stages:

A) RECORDED-ACTION PARITY CONTROL
   Recreate Kaggriculture using the replay's name/configuration/info, then feed
   the exact actions stored in replay.steps[1:] back into the environment.
   Every produced observation/reward/status must exactly match the saved replay.
   This validates action indexing, seed/random scenario, market evolution, shop
   sequence, seat state, and current-engine compatibility.

B) V18 EXECUTOR PARITY
   Infer v18 as the losing seat (this directory is explicitly v18 loss history),
   replace that recorded action stream with the frozen v18 executor forced to
   HERD_THRESHOLD=500, and keep the opponent on its recorded actions.
   The candidate's generated action sequence and final rewards must reproduce
   the original game. This validates that the executor/wrapper is the same v18
   policy that produced the saved loss.

C) V19 REPLACEMENT REPLAY
   Run the PPO checkpoint deterministically (argmax) in the v18 seat while the
   opponent remains on its historical recorded action stream.

Stage C is a counterfactual regression test, NOT a live rematch. Once v19
changes the state trajectory, the historical opponent does not adapt. Results
therefore answer "does v19 repair this failure scenario against the recorded
opponent behavior?", not "would v19 beat the original opponent agent live?".
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import kaggle_environments
import torch
from kaggle_environments import make

from train_v19_ppo import (
    ActorCritic,
    BASELINE_ACTION_INDEX,
    DEFAULT_EXECUTOR,
    FEATURE_NAMES,
    HERD_THRESHOLDS,
    HerdPolicyController,
)


HERE = Path(__file__).resolve().parent
G5_ROOT = HERE.parent.parent
DEFAULT_HISTORY_DIR = G5_ROOT / "game_history" / "v18"
DEFAULT_CHECKPOINT = HERE / "runs" / "herd_ppo" / "checkpoints" / "update_0099.pt"
DEFAULT_OUTPUT = HERE / "runs" / "herd_ppo" / "v18_loss_replay_update_0099.json"


# ---------- generic helpers ----------

def _field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(obj: Any) -> Any:
    """Convert Kaggle Struct/list values into plain JSON-compatible values."""
    return json.loads(json.dumps(obj))


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _final_rewards_from_states(states: list[Any]) -> list[float]:
    if len(states) != 2:
        raise ValueError(f"expected two final player states, got {len(states)}")
    rewards = [_number(_field(s, "reward", None)) for s in states]
    if any(x is None for x in rewards):
        raise ValueError(f"missing/non-finite final rewards: {rewards}")
    return [float(rewards[0]), float(rewards[1])]


def _saved_final_rewards(history: dict[str, Any]) -> list[float]:
    steps = history.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("replay must contain at least two steps")
    return _final_rewards_from_states(steps[-1])


def _environment_from_history(history: dict[str, Any]):
    name = history.get("name") or "kaggriculture"
    configuration = copy.deepcopy(history.get("configuration") or {})
    info = copy.deepcopy(history.get("info") or {})
    return make(
        name,
        configuration=configuration,
        info=info,
        debug=False,
    )


def _recorded_step_actions(history: dict[str, Any], replay_step: int) -> list[Any]:
    """Actions that produced history['steps'][replay_step].

    Kaggle stores the initial state in steps[0]. For each later replay state,
    steps[t][seat]['action'] is the action applied to the preceding state.
    """
    states = history["steps"][replay_step]
    if not isinstance(states, list) or len(states) != 2:
        raise ValueError(f"step {replay_step}: expected two player states")
    return [copy.deepcopy(_field(states[0], "action", None)),
            copy.deepcopy(_field(states[1], "action", None))]


def _state_core(state: Any) -> dict[str, Any]:
    """Fields that define game parity; ignore runtime-only info/log metadata."""
    return {
        "observation": _plain(_field(state, "observation", {})),
        "reward": _plain(_field(state, "reward", None)),
        "status": str(_field(state, "status", "")),
    }


def _first_diff(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        # JSON numeric int/float equality is semantically fine.
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and expected == actual:
            return None
        return f"{path}: type/value expected={expected!r}, actual={actual!r}"
    if isinstance(expected, dict):
        ek, ak = set(expected), set(actual)
        if ek != ak:
            return f"{path}: keys expected-only={sorted(ek-ak)}, actual-only={sorted(ak-ek)}"
        for key in sorted(ek):
            diff = _first_diff(expected[key], actual[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length expected={len(expected)}, actual={len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            diff = _first_diff(e, a, f"{path}[{i}]")
            if diff:
                return diff
        return None
    if expected != actual:
        return f"{path}: expected={expected!r}, actual={actual!r}"
    return None


# ---------- strict replay parity ----------

def recorded_action_parity(history: dict[str, Any]) -> dict[str, Any]:
    """Replay both historical action streams and compare every saved state."""
    env = _environment_from_history(history)
    saved_steps = history["steps"]

    # The freshly initialized state must already agree with replay step 0.
    for seat in (0, 1):
        diff = _first_diff(_state_core(saved_steps[0][seat]), _state_core(env.steps[0][seat]))
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
            diff = _first_diff(_state_core(saved_steps[t][seat]), _state_core(produced[seat]))
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
    }


# ---------- policy/model helpers ----------

def _load_model(checkpoint: Path, device: torch.device) -> tuple[ActorCritic, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)

    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema does not match train_v19_ppo.py")
    if tuple(payload.get("herd_thresholds", ())) != HERD_THRESHOLDS:
        raise ValueError("checkpoint action schema does not match train_v19_ppo.py")

    saved_args = payload.get("args", {}) or {}
    hidden = int(saved_args.get("hidden", 64))
    baseline_bias = float(saved_args.get("baseline_logit_bias", 3.0))

    model = ActorCritic(
        len(FEATURE_NAMES),
        hidden=hidden,
        baseline_bias=baseline_bias,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _farm_counts(farm: Any) -> tuple[dict[str, int], dict[str, int]]:
    animals = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    crops = {"WHEAT": 0, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0}
    for row in _field(farm, "tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            crop = tile.get("crop")
            if animal in animals:
                animals[animal] += 1
            if crop in crops:
                crops[crop] += 1
    return animals, crops


def _snapshot(controller: HerdPolicyController, obs: Any) -> dict[str, Any]:
    player = int(_field(obs, "player", 0))
    opponent = 1 - player
    farms = _field(obs, "farms", [])
    own = farms[player]
    opp = farms[opponent]
    own_animals, own_crops = _farm_counts(own)
    opp_animals, opp_crops = _farm_counts(opp)

    private = _field(obs, "private", {})
    try:
        totals = controller.executor.totals(private)
    except Exception:
        totals = {}

    market = _field(obs, "market", {})
    prices = _field(market, "prices", {}) or {}
    town = _field(obs, "town", {})
    return {
        "day": int(_field(obs, "day", -1)),
        "hour": int(_field(obs, "hour", -1)),
        "own_money": float(_field(own, "money", 0)),
        "opponent_money": float(_field(opp, "money", 0)),
        "money_margin": float(_field(own, "money", 0)) - float(_field(opp, "money", 0)),
        "own_animals": own_animals,
        "opponent_animals": opp_animals,
        "own_crops": own_crops,
        "opponent_crops": opp_crops,
        "own_private_totals": {
            key: float(totals.get(key, 0))
            for key in ("WHEAT", "FERTILIZER", "MILK", "WOOL", "EGG",
                        "CARROT", "TOMATO", "STRAWBERRY", "MELON")
        },
        "prices": {
            key: float(prices.get(key, 0))
            for key in ("WHEAT", "MILK", "WOOL", "EGG",
                        "CARROT", "TOMATO", "STRAWBERRY", "MELON", "FERTILIZER")
        },
        "shops": list(_field(town, "unlocked_shops", []) or []),
    }


def _count_action(counter_unit: Counter, counter_market: Counter, action: Any) -> None:
    if not isinstance(action, dict):
        return
    farmer = action.get("farmer")
    if isinstance(farmer, list) and farmer:
        counter_unit[str(farmer[0])] += 1
    for hand in action.get("hands", []) or []:
        if isinstance(hand, list) and hand:
            counter_unit[str(hand[0])] += 1
    for order in action.get("market", []) or []:
        if not isinstance(order, list) or not order:
            continue
        op = str(order[0])
        key = op
        if len(order) >= 2 and isinstance(order[1], str):
            key += ":" + order[1]
        counter_market[key] += 1


def _agent_observation(env: Any, seat: int) -> Any:
    """Return exactly the runtime-visible observation Kaggle would pass to an agent.

    Environment.__agent_runner uses the private __get_shared_state projection,
    which removes schema fields marked hidden and applies shared fields. Calling
    the controller on raw env.state would risk exposing replay-only information.
    The project pins kaggle-environments==1.32.7; fail loudly if this projection
    hook is unavailable instead of silently changing evaluation semantics.
    """
    getter = getattr(env, "_Environment__get_shared_state", None)
    if getter is None:
        raise RuntimeError(
            "kaggle-environments no longer exposes Environment.__get_shared_state; "
            "cannot guarantee runtime-equivalent agent observations"
        )
    return getter(seat).observation


def _run_replacement(
    history: dict[str, Any],
    candidate_seat: int,
    controller: HerdPolicyController,
    compare_to_recorded_candidate: bool,
) -> dict[str, Any]:
    """Manually step candidate vs fixed recorded opponent actions."""
    env = _environment_from_history(history)
    opponent_seat = 1 - candidate_seat
    saved_steps = history["steps"]

    action_divergences = 0
    first_divergence = None
    daily = []
    seen_days: set[int] = set()
    unit_ops: Counter = Counter()
    market_ops: Counter = Counter()

    for t in range(1, len(saved_steps)):
        obs = _agent_observation(env, candidate_seat)
        day = int(_field(obs, "day", -1))
        if day not in seen_days:
            daily.append(_snapshot(controller, obs))
            seen_days.add(day)

        candidate_action = controller(obs)
        recorded_actions = _recorded_step_actions(history, t)
        opponent_action = recorded_actions[opponent_seat]
        original_candidate_action = recorded_actions[candidate_seat]

        _count_action(unit_ops, market_ops, candidate_action)

        if compare_to_recorded_candidate and candidate_action != original_candidate_action:
            action_divergences += 1
            if first_divergence is None:
                first_divergence = {
                    "replay_step": t,
                    "day": day,
                    "hour": int(_field(obs, "hour", -1)),
                    "recorded_v18_action": original_candidate_action,
                    "generated_action": candidate_action,
                }

        actions = [None, None]
        actions[candidate_seat] = candidate_action
        actions[opponent_seat] = opponent_action
        env.step(actions)

    final_states = env.steps[-1]
    rewards = _final_rewards_from_states(final_states)
    statuses = [str(_field(s, "status", "")) for s in final_states]

    # Add one terminal snapshot if it represents a day not already captured.
    try:
        terminal_obs = final_states[candidate_seat].observation
        terminal_day = int(_field(terminal_obs, "day", -1))
        if terminal_day not in seen_days:
            daily.append(_snapshot(controller, terminal_obs))
    except Exception:
        pass

    return {
        "rewards": rewards,
        "statuses": statuses,
        "candidate_margin": rewards[candidate_seat] - rewards[opponent_seat],
        "action_divergences": action_divergences,
        "first_action_divergence": first_divergence,
        "thresholds": [
            {
                "day": int(step.day),
                "action_index": int(step.action),
                "threshold": float(step.threshold),
            }
            for step in controller.steps
        ],
        "unit_ops": dict(sorted(unit_ops.items())),
        "market_ops": dict(sorted(market_ops.items())),
        "daily": daily,
    }


def _infer_v18_seat(history: dict[str, Any]) -> int:
    """game_history/v18 is documented as v18 LOSS history."""
    rewards = _saved_final_rewards(history)
    if rewards[0] == rewards[1]:
        raise ValueError("saved history is tied; cannot infer losing v18 seat")
    return 0 if rewards[0] < rewards[1] else 1


def _seed_hint(history: dict[str, Any]) -> Any:
    info = history.get("info") or {}
    cfg = history.get("configuration") or {}
    for obj in (info, cfg):
        for key in ("seed", "randomSeed", "random_seed"):
            if key in obj:
                return obj[key]
    return None


# ---------- one episode ----------

def evaluate_one(
    path: Path,
    checkpoint: Path,
    executor: Path,
    model: ActorCritic,
    device: torch.device,
) -> dict[str, Any]:
    history = json.loads(path.read_text(encoding="utf-8"))
    original_rewards = _saved_final_rewards(history)

    result: dict[str, Any] = {
        "episode": path.stem,
        "valid": False,
        "seed": _seed_hint(history),
        "original_rewards": original_rewards,
    }

    # Gate 1: exact replay parity of BOTH recorded action streams.
    control = recorded_action_parity(history)
    result["recorded_action_control"] = control
    if not control.get("exact"):
        result["error"] = (
            "recorded-action parity failed; current engine cannot safely "
            "counterfactually evaluate this replay"
        )
        return result

    v18_seat = _infer_v18_seat(history)
    opponent_seat = 1 - v18_seat
    original_margin = original_rewards[v18_seat] - original_rewards[opponent_seat]
    result["v18_seat"] = v18_seat
    result["original_v18_margin"] = original_margin

    if original_margin >= 0:
        result["error"] = (
            "history directory is expected to contain v18 losses, but inferred "
            f"v18 margin is {original_margin}"
        )
        return result

    # Gate 2: dynamically regenerate v18 itself with fixed threshold 500.
    baseline_controller = HerdPolicyController(
        executor_path=executor,
        model=model,
        device=device,
        deterministic=True,
        forced_action=BASELINE_ACTION_INDEX,
    )
    baseline = _run_replacement(
        history,
        candidate_seat=v18_seat,
        controller=baseline_controller,
        compare_to_recorded_candidate=True,
    )
    result["v18_executor_parity"] = {
        "rewards": baseline["rewards"],
        "statuses": baseline["statuses"],
        "margin": baseline["candidate_margin"],
        "action_divergences": baseline["action_divergences"],
        "first_action_divergence": baseline["first_action_divergence"],
        "thresholds": baseline["thresholds"],
    }

    if (
        baseline["statuses"] != ["DONE", "DONE"]
        or baseline["rewards"] != original_rewards
        or baseline["action_divergences"] != 0
    ):
        result["error"] = (
            "v18 executor parity failed; v18_c258_compiled.py / wrapper does "
            "not exactly reproduce the saved v18 action stream"
        )
        return result

    # Stage 3: actual v19 deterministic counterfactual.
    v19_controller = HerdPolicyController(
        executor_path=executor,
        model=model,
        device=device,
        deterministic=True,
        forced_action=None,
    )
    v19 = _run_replacement(
        history,
        candidate_seat=v18_seat,
        controller=v19_controller,
        compare_to_recorded_candidate=True,
    )

    if v19["statuses"] != ["DONE", "DONE"]:
        result["error"] = f"v19 replacement replay did not finish: {v19['statuses']}"
        return result

    v19_margin = float(v19["candidate_margin"])
    candidate_reward_delta = (
        float(v19["rewards"][v18_seat]) - float(original_rewards[v18_seat])
    )
    opponent_reward_delta = (
        float(v19["rewards"][opponent_seat]) - float(original_rewards[opponent_seat])
    )

    result.update(
        valid=True,
        v19_rewards=v19["rewards"],
        v19_margin=v19_margin,
        margin_improvement=v19_margin - original_margin,
        candidate_reward_delta=candidate_reward_delta,
        opponent_reward_delta=opponent_reward_delta,
        result="WIN" if v19_margin > 0 else "LOSS" if v19_margin < 0 else "TIE",
        v19_thresholds=v19["thresholds"],
        action_divergences_from_original_v18=v19["action_divergences"],
        first_action_divergence=v19["first_action_divergence"],
        v19_unit_ops=v19["unit_ops"],
        v19_market_ops=v19["market_ops"],
        baseline_unit_ops=baseline["unit_ops"],
        baseline_market_ops=baseline["market_ops"],
        daily=v19["daily"],
    )
    return result


# ---------- aggregate/reporting ----------

def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid")]
    invalid = [r for r in rows if not r.get("valid")]
    if not valid:
        return {
            "games_total": len(rows),
            "games_valid": 0,
            "games_invalid": len(invalid),
        }

    margins = [float(r["v19_margin"]) for r in valid]
    original = [float(r["original_v18_margin"]) for r in valid]
    improvements = [float(r["margin_improvement"]) for r in valid]
    own_delta = [float(r["candidate_reward_delta"]) for r in valid]
    opp_delta = [float(r["opponent_reward_delta"]) for r in valid]

    threshold_counts: Counter = Counter()
    for r in valid:
        threshold_counts.update(str(int(x["threshold"])) for x in r["v19_thresholds"])

    return {
        "games_total": len(rows),
        "games_valid": len(valid),
        "games_invalid": len(invalid),
        "wins": sum(m > 0 for m in margins),
        "ties": sum(m == 0 for m in margins),
        "losses": sum(m < 0 for m in margins),
        "loss_cases_repaired": sum(m > 0 for m in margins),
        "margin_improved_cases": sum(d > 0 for d in improvements),
        "margin_worsened_cases": sum(d < 0 for d in improvements),
        "margin_unchanged_cases": sum(d == 0 for d in improvements),
        "mean_original_v18_margin": statistics.mean(original),
        "mean_v19_margin": statistics.mean(margins),
        "median_v19_margin": statistics.median(margins),
        "mean_margin_improvement": statistics.mean(improvements),
        "median_margin_improvement": statistics.median(improvements),
        "mean_candidate_reward_delta": statistics.mean(own_delta),
        "mean_recorded_opponent_reward_delta": statistics.mean(opp_delta),
        "worst_v19_margin": min(margins),
        "best_v19_margin": max(margins),
        "threshold_counts": dict(sorted(threshold_counts.items())),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode", "valid", "seed", "v18_seat",
        "original_v18_margin", "v19_margin", "margin_improvement",
        "candidate_reward_delta", "opponent_reward_delta", "result",
        "action_divergences_from_original_v18", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--executor", type=Path, default=DEFAULT_EXECUTOR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", default="", help="optional comma-separated replay IDs")
    p.add_argument("--fail-fast", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    history_dir = args.history_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    executor = args.executor.resolve()
    output = args.output.resolve()
    csv_output = output.with_suffix(".csv")

    wanted = {x.strip() for x in args.episodes.split(",") if x.strip()}
    paths = sorted(history_dir.glob("*.json"))
    if wanted:
        paths = [p for p in paths if p.stem in wanted]

    if not paths:
        raise SystemExit(f"no matching JSON histories found in {history_dir}")
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not executor.is_file():
        raise SystemExit(f"executor not found: {executor}")

    device = torch.device(args.device)
    model, payload = _load_model(checkpoint, device)

    print(f"engine=kaggle-environments {getattr(kaggle_environments, '__version__', 'unknown')}")
    print(f"histories={len(paths)}")
    print(f"checkpoint={checkpoint}")
    print(f"checkpoint_sha256={_sha256(checkpoint)}")
    print(f"checkpoint_update={payload.get('update')}")
    print(f"executor={executor}")
    print(f"executor_sha256={_sha256(executor)}")
    print("v19_policy=deterministic argmax")
    print("protocol=exact recorded-action parity -> exact v18 parity -> v19 replacement replay")
    print()

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, 1):
        try:
            row = evaluate_one(path, checkpoint, executor, model, device)
        except Exception as exc:
            row = {
                "episode": path.stem,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)

        if row.get("valid"):
            print(
                f"[{index:2d}/{len(paths)}] {row['episode']} "
                f"seat={row['v18_seat']} "
                f"v18={row['original_v18_margin']:+.0f} "
                f"v19={row['v19_margin']:+.0f} "
                f"delta={row['margin_improvement']:+.0f} "
                f"ownΔ={row['candidate_reward_delta']:+.0f} "
                f"oppΔ={row['opponent_reward_delta']:+.0f} "
                f"div={row['action_divergences_from_original_v18']} "
                f"{row['result']}"
            )
        else:
            print(
                f"[{index:2d}/{len(paths)}] {row['episode']} INVALID "
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

        # Save incrementally so long runs retain completed work.
        result = {
            "protocol": {
                "mode": "replacement_replay_with_strict_parity_gates",
                "note": (
                    "Recorded opponent actions are fixed and do not adapt to v19. "
                    "This is a regression/failure-scenario test, not a live rematch."
                ),
                "history_dir": str(history_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_update": payload.get("update"),
                "executor": str(executor),
                "executor_sha256": _sha256(executor),
                "kaggle_environments_version": getattr(
                    kaggle_environments, "__version__", "unknown"
                ),
                "deterministic_policy": True,
                "recorded_action_indexing": "history.steps[1:]",
                "passes_replay_info": True,
                "full_state_parity_required": True,
                "v18_executor_action_parity_required": True,
                "agent_observation_projection": "Environment.__get_shared_state",
                "raw_replay_state_not_exposed_to_policy": True,
            },
            "summary": summarize(rows),
            "matches": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_csv(csv_output, rows)

    summary = summarize(rows)
    print()
    print("Summary")
    print(f"  valid/total: {summary.get('games_valid', 0)}/{summary.get('games_total', 0)}")
    print(f"  invalid: {summary.get('games_invalid', 0)}")
    if summary.get("games_valid", 0):
        print(
            f"  v19 W/T/L on v18 loss scenarios: "
            f"{summary['wins']}/{summary['ties']}/{summary['losses']}"
        )
        print(f"  repaired losses: {summary['loss_cases_repaired']}")
        print(
            f"  margin improved/worsened/unchanged: "
            f"{summary['margin_improved_cases']}/"
            f"{summary['margin_worsened_cases']}/"
            f"{summary['margin_unchanged_cases']}"
        )
        print(f"  mean original v18 margin: {summary['mean_original_v18_margin']:+.1f}")
        print(f"  mean v19 margin: {summary['mean_v19_margin']:+.1f}")
        print(f"  mean margin improvement: {summary['mean_margin_improvement']:+.1f}")
        print(f"  mean own reward delta: {summary['mean_candidate_reward_delta']:+.1f}")
        print(
            "  mean recorded-opponent reward delta: "
            f"{summary['mean_recorded_opponent_reward_delta']:+.1f}"
        )
        print(f"  threshold counts: {summary['threshold_counts']}")
    print(f"  JSON: {output}")
    print(f"  CSV:  {csv_output}")
    return 0 if summary.get("games_invalid", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
