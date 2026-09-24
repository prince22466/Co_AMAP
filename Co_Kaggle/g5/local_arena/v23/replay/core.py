"""Self-contained generic static-replay helpers for v23.

No v20/v21/v22 imports. Candidate replaces the losing v20 seat while the
historical opponent action commands are submitted verbatim.
"""
from __future__ import annotations

import copy
import json
import re
import types
from pathlib import Path
from typing import Any


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _environment_from_history(history: dict[str, Any]):
    import kaggle_environments
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


def _final_rewards_from_states(states: list[Any]) -> list[float]:
    if len(states) != 2:
        raise ValueError(f"expected two final states, got {len(states)}")
    rewards = []
    for state in states:
        value = _field(state, "reward", None)
        if value is None:
            raise ValueError("missing terminal reward")
        rewards.append(float(value))
    return rewards


def _saved_final_rewards(history: dict[str, Any]) -> list[float]:
    steps = history.get("steps") or []
    if not steps:
        raise ValueError("history has no steps")
    return _final_rewards_from_states(steps[-1])


def _state_core(state: Any) -> dict[str, Any]:
    return {
        "observation": _plain(_field(state, "observation", {})),
        "reward": _plain(_field(state, "reward", None)),
        "status": str(_field(state, "status", "")),
    }


def _first_diff(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
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
        for index, (e, a) in enumerate(zip(expected, actual)):
            diff = _first_diff(e, a, f"{path}[{index}]")
            if diff:
                return diff
        return None
    return None if expected == actual else f"{path}: expected={expected!r}, actual={actual!r}"


def _agent_observation(env: Any, seat: int) -> Any:
    getter = getattr(env, "_Environment__get_shared_state", None)
    if getter is None:
        raise RuntimeError(
            "kaggle-environments no longer exposes Environment.__get_shared_state; "
            "cannot guarantee runtime-equivalent observations"
        )
    return getter(seat).observation


def recorded_action_parity(history: dict[str, Any]) -> dict[str, Any]:
    env = _environment_from_history(history)
    saved_steps = history["steps"]
    for seat in (0, 1):
        diff = _first_diff(_state_core(saved_steps[0][seat]), _state_core(env.steps[0][seat]))
        if diff:
            return {"exact": False, "mismatch_step": 0, "mismatch_seat": seat, "mismatch": diff}

    for replay_step in range(1, len(saved_steps)):
        actions = _recorded_step_actions(history, replay_step)
        if any(action is None for action in actions):
            return {
                "exact": False,
                "mismatch_step": replay_step,
                "mismatch_seat": None,
                "mismatch": f"recorded action is None at replay step {replay_step}: {actions}",
            }
        env.step(actions)
        produced = env.steps[-1]
        for seat in (0, 1):
            diff = _first_diff(_state_core(saved_steps[replay_step][seat]), _state_core(produced[seat]))
            if diff:
                return {
                    "exact": False,
                    "mismatch_step": replay_step,
                    "mismatch_seat": seat,
                    "mismatch": diff,
                }

    return {
        "exact": True,
        "steps": len(saved_steps),
        "rewards": _final_rewards_from_states(env.steps[-1]),
    }


def _extract_notebook_main(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        lines = "".join(cell.get("source", [])).splitlines()
        if not lines:
            continue
        match = re.match(r"^\s*%%writefile\s+(.+?)\s*$", lines[0])
        if match and Path(match.group(1).strip("'\"")).name == "main.py":
            found.append("\n".join(lines[1:]) + "\n")
    if len(found) != 1:
        raise ValueError(f"{path}: expected exactly one %%writefile main.py cell, found {len(found)}")
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
