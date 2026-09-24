#!/usr/bin/env python3
"""Isolated static-replay process for v23 candidate policies."""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

from static_replay import (
    _agent_observation,
    _environment_from_history,
    _field,
    _load_notebook_agent,
    _recorded_step_actions,
    _saved_final_rewards,
    recorded_action_parity,
)

HERE = Path(__file__).resolve().parent
HISTORY_DIR = HERE / "working_files" / "loss_games_v20"


def _inside_v23(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.relative_to(HERE)
    return path


def _load_python_agent(path: Path):
    source = path.read_text(encoding="utf-8")
    module = types.ModuleType(f"v23_candidate_{id(source)}")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise ValueError(f"{path}: expected callable agent(obs)")
    return agent


def evaluate(candidate: Path, episodes: list[str], max_episodes: int):
    candidate = _inside_v23(candidate)
    if candidate.suffix not in {".py", ".ipynb"} or not candidate.is_file():
        raise ValueError("candidate must be an existing .py or .ipynb under v23")

    all_paths = sorted(HISTORY_DIR.glob("*.json"))
    wanted = {x.strip() for x in episodes if x.strip()}
    paths = [p for p in all_paths if not wanted or p.stem in wanted][:max_episodes]
    if not paths:
        raise ValueError("no matching v20 loss histories")

    if candidate.suffix == ".ipynb":
        _, candidate_agent = _load_notebook_agent(candidate, "v23_candidate")
    else:
        candidate_agent = _load_python_agent(candidate)

    rows = []
    for history_path in paths:
        case_started = time.monotonic()
        history = json.loads(history_path.read_text(encoding="utf-8"))
        control = recorded_action_parity(history)
        original = _saved_final_rewards(history)

        if not control.get("exact"):
            rows.append({
                "episode": history_path.stem,
                "valid": False,
                "error": "recorded-action parity failed",
                "recorded_action_control": control,
                "elapsed_seconds": round(time.monotonic() - case_started, 6),
            })
            continue

        if original[0] == original[1]:
            rows.append({
                "episode": history_path.stem,
                "valid": False,
                "error": "recorded v20 history is tied; cannot infer v20 seat",
                "elapsed_seconds": round(time.monotonic() - case_started, 6),
            })
            continue

        candidate_seat = 0 if original[0] < original[1] else 1
        opponent_seat = 1 - candidate_seat
        original_margin = float(original[candidate_seat] - original[opponent_seat])
        env = _environment_from_history(history)
        action_divergences = 0
        first_divergence = None

        try:
            for replay_step in range(1, len(history["steps"])):
                obs = _agent_observation(env, candidate_seat)
                action = candidate_agent(obs)
                recorded = _recorded_step_actions(history, replay_step)
                opponent_action = recorded[opponent_seat]
                if opponent_action is None:
                    raise RuntimeError(f"recorded opponent action is None at step {replay_step}")

                if action != recorded[candidate_seat]:
                    action_divergences += 1
                    if first_divergence is None:
                        first_divergence = {
                            "replay_step": replay_step,
                            "day": int(_field(obs, "day", -1)),
                            "hour": int(_field(obs, "hour", -1)),
                        }

                actions = [None, None]
                actions[candidate_seat] = action
                actions[opponent_seat] = opponent_action
                env.step(actions)

            final_states = env.steps[-1]
            rewards = [float(_field(state, "reward")) for state in final_states]
            statuses = [str(_field(state, "status", "")) for state in final_states]
            margin = rewards[candidate_seat] - rewards[opponent_seat]
            valid = statuses == ["DONE", "DONE"]
            rows.append({
                "episode": history_path.stem,
                "valid": valid,
                "v20_seat": candidate_seat,
                "original_rewards": original,
                "candidate_rewards": rewards,
                "original_v20_margin": original_margin,
                "candidate_margin": margin,
                "margin_improvement": margin - original_margin,
                "result": "WIN" if margin > 0 else "LOSS" if margin < 0 else "TIE",
                "action_divergences": action_divergences,
                "first_action_divergence": first_divergence,
                "error": "" if valid else f"non-DONE status: {statuses}",
                "elapsed_seconds": round(time.monotonic() - case_started, 6),
            })
        except BaseException as exc:
            rows.append({
                "episode": history_path.stem,
                "valid": False,
                "original_v20_margin": original_margin,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - case_started, 6),
            })

    valid_rows = [row for row in rows if row.get("valid")]
    if valid_rows:
        margins = [float(row["candidate_margin"]) for row in valid_rows]
        improvements = [float(row["margin_improvement"]) for row in valid_rows]
        summary = {
            "games_total": len(rows),
            "games_valid": len(valid_rows),
            "games_invalid": len(rows) - len(valid_rows),
            "wins": sum(x > 0 for x in margins),
            "ties": sum(x == 0 for x in margins),
            "losses": sum(x < 0 for x in margins),
            "loss_cases_repaired": sum(x > 0 for x in margins),
            "repair_rate": sum(x > 0 for x in margins) / len(margins),
            "margin_improved_cases": sum(x > 0 for x in improvements),
            "margin_worsened_cases": sum(x < 0 for x in improvements),
            "margin_unchanged_cases": sum(x == 0 for x in improvements),
            "mean_candidate_margin": sum(margins) / len(margins),
            "mean_margin_improvement": sum(improvements) / len(improvements),
            "worst_candidate_margin": min(margins),
            "best_candidate_margin": max(margins),
            "best_margin_improvement": max(improvements),
        }
    else:
        summary = {
            "games_total": len(rows),
            "games_valid": 0,
            "games_invalid": len(rows),
        }

    return {
        "protocol": {
            "mode": "v23_static_replacement_replay_on_v20_losses",
            "candidate": str(candidate.relative_to(HERE)),
            "candidate_seat": "losing v20 seat",
            "opponent_behavior": "recorded historical commands; non-adaptive",
            "recorded_action_parity_required": True,
        },
        "summary": summary,
        "matches": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--episodes-json", default="[]")
    parser.add_argument("--max-episodes", type=int, default=25)
    args = parser.parse_args()

    episodes = json.loads(args.episodes_json)
    if not isinstance(episodes, list) or not all(isinstance(x, str) for x in episodes):
        raise SystemExit("--episodes-json must be a JSON array of strings")
    max_episodes = max(1, min(int(args.max_episodes), 50))

    try:
        result = evaluate(HERE / args.candidate, episodes, max_episodes)
    except BaseException as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}

    print("__V23_RESULT__" + json.dumps(result, sort_keys=True, default=str))
    return 0 if "error" not in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
