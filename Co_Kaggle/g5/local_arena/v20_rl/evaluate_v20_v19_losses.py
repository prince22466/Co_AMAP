#!/usr/bin/env python3
"""Static v20 counterfactual replay on v19's downloaded loss histories.

For each replay JSON under game_history/v19 this evaluator runs three stages:

A) RECORDED-ACTION PARITY CONTROL
   Recreate Kaggriculture from the replay metadata and feed both recorded action
   streams back into the environment. Every produced observation/reward/status
   must exactly match the saved replay.

B) V19 EXECUTOR PARITY
   Infer v19 as the losing seat (game_history/v19 is documented as v19 loss
   history), run the actual v19 submission notebook in that seat, and keep the
   opponent on its recorded historical actions. v19 must reproduce the saved
   v19 action stream and final rewards exactly.

C) V20 STATIC REPLACEMENT REPLAY
   Replace v19 with the actual v20 submission notebook while the opponent stays
   on its original recorded action stream.

Stage C is intentionally STATIC. Once v20 changes the trajectory, the recorded
opponent does not adapt. The result answers:

    "Does v20 repair this v19 failure scenario against the opponent behavior
     that was actually recorded?"

It is not a live rematch and must not be interpreted as one.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
G5_ROOT = HERE.parent.parent
SUBMISSION_DIR = G5_ROOT / "submission_nb"

DEFAULT_HISTORY_DIR = G5_ROOT / "game_history" / "v19"
DEFAULT_V19 = SUBMISSION_DIR / "kaggriculture-sub_v19.ipynb"
DEFAULT_V20 = SUBMISSION_DIR / "kaggriculture-sub_v20.ipynb"
DEFAULT_OUTPUT = HERE / "runs" / "v20_v19_static_replay.json"

_KAGGLE_ENVIRONMENTS = None


@contextmanager
def _silence_stderr_during_kaggle_init():
    """Suppress noisy optional OpenSpiel probes from Kaggle initialization.

    Some OpenSpiel builds write long "Unknown game ..." diagnostics directly
    to native stderr (file descriptor 2), so redirect both Python stderr and
    fd 2 only while Kaggle imports/creates an environment.
    """
    original_stderr = sys.stderr
    devnull = open(os.devnull, "w")
    saved_fd = None
    stderr_fd = None
    try:
        try:
            original_stderr.flush()
            stderr_fd = original_stderr.fileno()
            saved_fd = os.dup(stderr_fd)
            os.dup2(devnull.fileno(), stderr_fd)
        except (AttributeError, OSError, ValueError):
            stderr_fd = None
            saved_fd = None

        sys.stderr = devnull
        yield
    finally:
        sys.stderr = original_stderr
        if saved_fd is not None and stderr_fd is not None:
            try:
                os.dup2(saved_fd, stderr_fd)
            finally:
                os.close(saved_fd)
        devnull.close()


def _get_kaggle_environments():
    """Import kaggle_environments lazily without OpenSpiel discovery noise."""
    global _KAGGLE_ENVIRONMENTS
    if _KAGGLE_ENVIRONMENTS is None:
        with _silence_stderr_during_kaggle_init():
            import kaggle_environments as module
        _KAGGLE_ENVIRONMENTS = module
    return _KAGGLE_ENVIRONMENTS


# ---------- generic helpers ----------

def _field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(obj: Any) -> Any:
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
    kaggle_environments = _get_kaggle_environments()
    with _silence_stderr_during_kaggle_init():
        return kaggle_environments.make(
            history.get("name") or "kaggriculture",
            configuration=copy.deepcopy(history.get("configuration") or {}),
            info=copy.deepcopy(history.get("info") or {}),
            debug=False,
        )


def _recorded_step_actions(history: dict[str, Any], replay_step: int) -> list[Any]:
    states = history["steps"][replay_step]
    if not isinstance(states, list) or len(states) != 2:
        raise ValueError(f"step {replay_step}: expected two player states")
    return [
        copy.deepcopy(_field(states[0], "action", None)),
        copy.deepcopy(_field(states[1], "action", None)),
    ]


def _state_core(state: Any) -> dict[str, Any]:
    return {
        "observation": _plain(_field(state, "observation", {})),
        "reward": _plain(_field(state, "reward", None)),
        "status": str(_field(state, "status", "")),
    }


def _first_diff(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        if (
            isinstance(expected, (int, float))
            and isinstance(actual, (int, float))
            and expected == actual
        ):
            return None
        return f"{path}: type/value expected={expected!r}, actual={actual!r}"

    if isinstance(expected, dict):
        ek, ak = set(expected), set(actual)
        if ek != ak:
            return (
                f"{path}: keys expected-only={sorted(ek-ak)}, "
                f"actual-only={sorted(ak-ek)}"
            )
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


def _seed_hint(history: dict[str, Any]) -> Any:
    info = history.get("info") or {}
    cfg = history.get("configuration") or {}
    for obj in (info, cfg):
        for key in ("seed", "randomSeed", "random_seed"):
            if key in obj:
                return obj[key]
    return None


# ---------- notebook submission loading ----------

def _extract_notebook_main(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        lines = "".join(cell.get("source", [])).splitlines()
        if not lines:
            continue
        match = re.match(r"^\s*%%writefile\s+(.+?)\s*$", lines[0])
        if match and Path(match.group(1).strip("'\"")).name == "main.py":
            found.append("\n".join(lines[1:]) + "\n")
    if len(found) != 1:
        raise ValueError(
            f"{path}: expected exactly one %%writefile main.py cell, found {len(found)}"
        )
    return found[0]


def _load_notebook_agent(path: Path, label: str):
    source = _extract_notebook_main(path)
    module = types.ModuleType(f"static_replay_{label}_{id(source)}")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise ValueError(f"{path}: main.py does not define callable agent(obs)")
    return module, agent


def _agent_observation(env: Any, seat: int) -> Any:
    """Return the same shared/private projection Kaggle passes to an agent."""
    getter = getattr(env, "_Environment__get_shared_state", None)
    if getter is None:
        raise RuntimeError(
            "kaggle-environments no longer exposes Environment.__get_shared_state; "
            "cannot guarantee runtime-equivalent observations"
        )
    return getter(seat).observation


# ---------- strict parity gates ----------

def recorded_action_parity(history: dict[str, Any]) -> dict[str, Any]:
    env = _environment_from_history(history)
    saved_steps = history["steps"]

    for seat in (0, 1):
        diff = _first_diff(
            _state_core(saved_steps[0][seat]),
            _state_core(env.steps[0][seat]),
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
                _state_core(saved_steps[t][seat]),
                _state_core(produced[seat]),
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
    }


def _infer_v19_seat(history: dict[str, Any]) -> int:
    rewards = _saved_final_rewards(history)
    if rewards[0] == rewards[1]:
        raise ValueError("saved history is tied; cannot infer losing v19 seat")
    return 0 if rewards[0] < rewards[1] else 1


def _run_submission_replacement(
    history: dict[str, Any],
    candidate_seat: int,
    notebook: Path,
    label: str,
    compare_to_recorded_candidate: bool,
) -> dict[str, Any]:
    """Run one submission while keeping the opponent's recorded actions fixed."""
    env = _environment_from_history(history)
    _, agent = _load_notebook_agent(notebook, label)
    opponent_seat = 1 - candidate_seat
    saved_steps = history["steps"]

    action_divergences = 0
    first_divergence = None

    for t in range(1, len(saved_steps)):
        obs = _agent_observation(env, candidate_seat)
        candidate_action = agent(obs)

        recorded_actions = _recorded_step_actions(history, t)
        opponent_action = recorded_actions[opponent_seat]
        original_candidate_action = recorded_actions[candidate_seat]

        if compare_to_recorded_candidate and candidate_action != original_candidate_action:
            action_divergences += 1
            if first_divergence is None:
                first_divergence = {
                    "replay_step": t,
                    "day": int(_field(obs, "day", -1)),
                    "hour": int(_field(obs, "hour", -1)),
                    "recorded_v19_action": original_candidate_action,
                    "generated_action": candidate_action,
                }

        actions = [None, None]
        actions[candidate_seat] = candidate_action
        actions[opponent_seat] = opponent_action
        env.step(actions)

    final_states = env.steps[-1]
    rewards = _final_rewards_from_states(final_states)
    statuses = [str(_field(s, "status", "")) for s in final_states]

    return {
        "rewards": rewards,
        "statuses": statuses,
        "candidate_margin": rewards[candidate_seat] - rewards[opponent_seat],
        "action_divergences": action_divergences,
        "first_action_divergence": first_divergence,
    }


# ---------- per-episode evaluation ----------

def evaluate_one(path: Path, v19_notebook: Path, v20_notebook: Path) -> dict[str, Any]:
    history = json.loads(path.read_text(encoding="utf-8"))
    original_rewards = _saved_final_rewards(history)

    result: dict[str, Any] = {
        "episode": path.stem,
        "seed": _seed_hint(history),
        "valid": False,
        "original_rewards": original_rewards,
    }

    # Gate A: the replay must still reproduce exactly under this engine.
    control = recorded_action_parity(history)
    result["recorded_action_control"] = control
    if not control.get("exact"):
        result["error"] = (
            "recorded-action parity failed; current engine cannot safely "
            "counterfactually evaluate this replay"
        )
        return result

    v19_seat = _infer_v19_seat(history)
    opponent_seat = 1 - v19_seat
    original_margin = original_rewards[v19_seat] - original_rewards[opponent_seat]
    result["v19_seat"] = v19_seat
    result["original_v19_margin"] = original_margin

    if original_margin >= 0:
        result["error"] = (
            "game_history/v19 is expected to contain v19 losses, but inferred "
            f"v19 margin is {original_margin}"
        )
        return result

    # Gate B: the checked-in v19 notebook must reproduce the recorded v19.
    v19 = _run_submission_replacement(
        history=history,
        candidate_seat=v19_seat,
        notebook=v19_notebook,
        label="v19",
        compare_to_recorded_candidate=True,
    )
    result["v19_executor_parity"] = {
        "rewards": v19["rewards"],
        "statuses": v19["statuses"],
        "margin": v19["candidate_margin"],
        "action_divergences": v19["action_divergences"],
        "first_action_divergence": v19["first_action_divergence"],
    }

    if (
        v19["statuses"] != ["DONE", "DONE"]
        or v19["rewards"] != original_rewards
        or v19["action_divergences"] != 0
    ):
        result["error"] = (
            "v19 executor parity failed; checked-in v19 submission does not "
            "exactly reproduce the saved v19 action stream"
        )
        return result

    # Stage C: replace v19 with v20; opponent stays on recorded actions.
    v20 = _run_submission_replacement(
        history=history,
        candidate_seat=v19_seat,
        notebook=v20_notebook,
        label="v20",
        compare_to_recorded_candidate=True,
    )

    if v20["statuses"] != ["DONE", "DONE"]:
        result["error"] = f"v20 replacement replay did not finish: {v20['statuses']}"
        return result

    v20_margin = float(v20["candidate_margin"])
    own_delta = float(v20["rewards"][v19_seat]) - float(original_rewards[v19_seat])
    opp_delta = float(v20["rewards"][opponent_seat]) - float(original_rewards[opponent_seat])

    result.update(
        valid=True,
        v20_rewards=v20["rewards"],
        v20_margin=v20_margin,
        margin_improvement=v20_margin - original_margin,
        candidate_reward_delta=own_delta,
        opponent_reward_delta=opp_delta,
        result="WIN" if v20_margin > 0 else "LOSS" if v20_margin < 0 else "TIE",
        action_divergences_from_original_v19=v20["action_divergences"],
        first_action_divergence=v20["first_action_divergence"],
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

    margins = [float(r["v20_margin"]) for r in valid]
    original = [float(r["original_v19_margin"]) for r in valid]
    improvements = [float(r["margin_improvement"]) for r in valid]
    own_delta = [float(r["candidate_reward_delta"]) for r in valid]
    opp_delta = [float(r["opponent_reward_delta"]) for r in valid]

    return {
        "games_total": len(rows),
        "games_valid": len(valid),
        "games_invalid": len(invalid),
        "wins": sum(m > 0 for m in margins),
        "ties": sum(m == 0 for m in margins),
        "losses": sum(m < 0 for m in margins),
        "loss_cases_repaired": sum(m > 0 for m in margins),
        "repair_rate": sum(m > 0 for m in margins) / len(margins),
        "margin_improved_cases": sum(d > 0 for d in improvements),
        "margin_worsened_cases": sum(d < 0 for d in improvements),
        "margin_unchanged_cases": sum(d == 0 for d in improvements),
        "mean_original_v19_margin": statistics.mean(original),
        "mean_v20_margin": statistics.mean(margins),
        "median_v20_margin": statistics.median(margins),
        "mean_margin_improvement": statistics.mean(improvements),
        "median_margin_improvement": statistics.median(improvements),
        "mean_candidate_reward_delta": statistics.mean(own_delta),
        "mean_recorded_opponent_reward_delta": statistics.mean(opp_delta),
        "worst_v20_margin": min(margins),
        "best_v20_margin": max(margins),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode",
        "valid",
        "seed",
        "v19_seat",
        "original_v19_margin",
        "v20_margin",
        "margin_improvement",
        "candidate_reward_delta",
        "opponent_reward_delta",
        "result",
        "action_divergences_from_original_v19",
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
    p.add_argument("--v19", type=Path, default=DEFAULT_V19)
    p.add_argument("--v20", type=Path, default=DEFAULT_V20)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--episodes",
        default="",
        help="optional comma-separated replay IDs, e.g. 110783673,110787062",
    )
    p.add_argument("--fail-fast", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    history_dir = args.history_dir.expanduser().resolve()
    v19_notebook = args.v19.expanduser().resolve()
    v20_notebook = args.v20.expanduser().resolve()
    output = args.output.expanduser().resolve()
    csv_output = output.with_suffix(".csv")

    wanted = {x.strip() for x in args.episodes.split(",") if x.strip()}
    paths = sorted(history_dir.glob("*.json"))
    if wanted:
        paths = [p for p in paths if p.stem in wanted]

    if not paths:
        raise SystemExit(f"no matching JSON histories found in {history_dir}")
    if not v19_notebook.is_file():
        raise SystemExit(f"v19 notebook not found: {v19_notebook}")
    if not v20_notebook.is_file():
        raise SystemExit(f"v20 notebook not found: {v20_notebook}")

    kaggle_environments = _get_kaggle_environments()
    print(
        "engine=kaggle-environments "
        f"{getattr(kaggle_environments, '__version__', 'unknown')}"
    )
    print(f"histories={len(paths)}")
    print(f"v19={v19_notebook}")
    print(f"v19_sha256={_sha256(v19_notebook)}")
    print(f"v20={v20_notebook}")
    print(f"v20_sha256={_sha256(v20_notebook)}")
    print(
        "protocol=exact recorded-action parity -> exact v19 executor parity "
        "-> v20 static replacement replay"
    )
    print("opponent=historical recorded action stream (non-adaptive)")
    print()

    rows: list[dict[str, Any]] = []

    for index, path in enumerate(paths, 1):
        try:
            row = evaluate_one(path, v19_notebook, v20_notebook)
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
                f"seat={row['v19_seat']} "
                f"v19={row['original_v19_margin']:+.0f} "
                f"v20={row['v20_margin']:+.0f} "
                f"delta={row['margin_improvement']:+.0f} "
                f"ownΔ={row['candidate_reward_delta']:+.0f} "
                f"oppΔ={row['opponent_reward_delta']:+.0f} "
                f"div={row['action_divergences_from_original_v19']} "
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

        result = {
            "protocol": {
                "mode": "v20_static_replacement_replay_on_v19_losses",
                "note": (
                    "The opponent action stream is copied from the historical "
                    "v19 replay and does not adapt after v20 changes the state. "
                    "This is a static failure-scenario regression test, not a "
                    "live rematch."
                ),
                "history_dir": str(history_dir),
                "v19_submission": str(v19_notebook),
                "v19_sha256": _sha256(v19_notebook),
                "v20_submission": str(v20_notebook),
                "v20_sha256": _sha256(v20_notebook),
                "kaggle_environments_version": getattr(
                    kaggle_environments, "__version__", "unknown"
                ),
                "recorded_action_indexing": "history.steps[1:]",
                "full_state_parity_required": True,
                "v19_executor_action_parity_required": True,
                "agent_observation_projection": "Environment.__get_shared_state",
                "raw_replay_state_not_exposed_to_policy": True,
                "opponent_adapts": False,
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
    print(
        f"  valid/total: "
        f"{summary.get('games_valid', 0)}/{summary.get('games_total', 0)}"
    )
    print(f"  invalid: {summary.get('games_invalid', 0)}")

    if summary.get("games_valid", 0):
        print(
            "  v20 W/T/L on v19 static loss scenarios: "
            f"{summary['wins']}/{summary['ties']}/{summary['losses']}"
        )
        print(
            f"  repaired v19 losses: {summary['loss_cases_repaired']} "
            f"({summary['repair_rate']:.1%})"
        )
        print(
            "  margin improved/worsened/unchanged: "
            f"{summary['margin_improved_cases']}/"
            f"{summary['margin_worsened_cases']}/"
            f"{summary['margin_unchanged_cases']}"
        )
        print(
            f"  mean original v19 margin: "
            f"{summary['mean_original_v19_margin']:+.1f}"
        )
        print(f"  mean v20 margin: {summary['mean_v20_margin']:+.1f}")
        print(
            f"  mean margin improvement: "
            f"{summary['mean_margin_improvement']:+.1f}"
        )

    print(f"  JSON: {output}")
    print(f"  CSV:  {csv_output}")
    return 0 if summary.get("games_invalid", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
