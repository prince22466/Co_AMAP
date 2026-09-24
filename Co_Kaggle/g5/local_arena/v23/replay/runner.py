#!/usr/bin/env python3
"""Isolated static-replay process for v23 candidate policies."""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
import types
from pathlib import Path

try:
    from .core import (
        _agent_observation,
        _environment_from_history,
        _field,
        _load_notebook_agent,
        _recorded_step_actions,
        _saved_final_rewards,
        recorded_action_parity,
    )
except ImportError:  # direct script execution: python replay/runner.py
    from core import (
        _agent_observation,
        _environment_from_history,
        _field,
        _load_notebook_agent,
        _recorded_step_actions,
        _saved_final_rewards,
        recorded_action_parity,
    )

HERE = Path(__file__).resolve().parent
V23_ROOT = HERE.parent
HISTORY_DIR = V23_ROOT / "working_files" / "loss_games_v20"


def _inside_v23(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.relative_to(V23_ROOT)
    return path


FORBIDDEN_FLOAT_PATTERNS = (
    r"torch\.float32\b", r"torch\.float64\b", r"torch\.double\b", r"torch\.bfloat16\b",
    r"np\.float32\b", r"np\.float64\b", r"numpy\.float32\b", r"numpy\.float64\b",
    r"dtype\s*=\s*[\"']float32[\"']", r"dtype\s*=\s*[\"']float64[\"']",
    r"dtype\s*=\s*[\"']double[\"']", r"dtype\s*=\s*[\"']bfloat16[\"']",
    r"\.float\(\)", r"\.double\(\)",
)


def _candidate_source(path: Path) -> str:
    if path.suffix == ".ipynb":
        try:
            from .core import _extract_notebook_main
        except ImportError:
            from core import _extract_notebook_main
        return _extract_notebook_main(path)
    return path.read_text(encoding="utf-8")


def _audit_fp16_source(path: Path) -> dict:
    source = _candidate_source(path)
    violations = []
    for pattern in FORBIDDEN_FLOAT_PATTERNS:
        for match in re.finditer(pattern, source):
            line = source.count("\n", 0, match.start()) + 1
            violations.append({"line": line, "pattern": pattern, "text": match.group(0)})
    return {"ok": not violations, "violations": violations[:50]}


def _set_fp16_defaults() -> dict:
    info = {"torch": "not-imported"}
    try:
        import torch
        torch.set_default_dtype(torch.float16)
        info["torch"] = str(torch.get_default_dtype())
    except Exception as exc:
        info["torch"] = f"unavailable: {type(exc).__name__}: {exc}"
    return info


@contextlib.contextmanager
def _numpy_fp16_candidate_defaults():
    """Force candidate-created NumPy floating arrays to FP16 during agent(obs)."""
    try:
        import numpy as np
    except Exception:
        yield
        return

    originals = {}
    names = ("array", "asarray", "zeros", "ones", "empty", "full")
    for name in names:
        originals[name] = getattr(np, name)

    def _cast_float_array(value):
        try:
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                if value.dtype != np.float16:
                    return value.astype(np.float16, copy=False)
        except Exception:
            pass
        return value

    def array(*args, **kwargs):
        if kwargs.get("dtype") is not None:
            return originals["array"](*args, **kwargs)
        return _cast_float_array(originals["array"](*args, **kwargs))

    def asarray(*args, **kwargs):
        if kwargs.get("dtype") is not None:
            return originals["asarray"](*args, **kwargs)
        return _cast_float_array(originals["asarray"](*args, **kwargs))

    def zeros(*args, **kwargs):
        if kwargs.get("dtype") is None:
            kwargs["dtype"] = np.float16
        return originals["zeros"](*args, **kwargs)

    def ones(*args, **kwargs):
        if kwargs.get("dtype") is None:
            kwargs["dtype"] = np.float16
        return originals["ones"](*args, **kwargs)

    def empty(*args, **kwargs):
        if kwargs.get("dtype") is None:
            kwargs["dtype"] = np.float16
        return originals["empty"](*args, **kwargs)

    def full(*args, **kwargs):
        if kwargs.get("dtype") is not None:
            return originals["full"](*args, **kwargs)
        return _cast_float_array(originals["full"](*args, **kwargs))

    np.array, np.asarray = array, asarray
    np.zeros, np.ones, np.empty, np.full = zeros, ones, empty, full
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(np, name, fn)


def _audit_runtime_globals(module: types.ModuleType) -> dict:
    violations = []
    try:
        import torch
    except Exception:
        torch = None
    try:
        import numpy as np
    except Exception:
        np = None

    for name, value in module.__dict__.items():
        if name.startswith("__"):
            continue
        try:
            if torch is not None and isinstance(value, torch.Tensor):
                if value.is_floating_point() and value.dtype != torch.float16:
                    violations.append({"name": name, "kind": "torch_tensor", "dtype": str(value.dtype)})
            elif torch is not None and isinstance(value, torch.nn.Module):
                for pname, param in value.named_parameters(recurse=True):
                    if param.is_floating_point() and param.dtype != torch.float16:
                        violations.append({"name": f"{name}.{pname}", "kind": "parameter", "dtype": str(param.dtype)})
                for bname, buf in value.named_buffers(recurse=True):
                    if buf.is_floating_point() and buf.dtype != torch.float16:
                        violations.append({"name": f"{name}.{bname}", "kind": "buffer", "dtype": str(buf.dtype)})
            elif np is not None and isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.floating) and value.dtype != np.float16:
                    violations.append({"name": name, "kind": "numpy_array", "dtype": str(value.dtype)})
        except Exception:
            continue
    return {"ok": not violations, "violations": violations[:50]}


def _load_python_agent(path: Path):
    source = path.read_text(encoding="utf-8")
    module = types.ModuleType(f"v23_candidate_{id(source)}")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise ValueError(f"{path}: expected callable agent(obs)")
    return module, agent


def evaluate(candidate: Path, episodes: list[str], max_episodes: int, record_dir: Path | None = None):
    candidate = _inside_v23(candidate)
    if candidate.suffix not in {".py", ".ipynb"} or not candidate.is_file():
        raise ValueError("candidate must be an existing .py or .ipynb under v23")

    all_paths = sorted(HISTORY_DIR.glob("*.json"))
    wanted = {x.strip() for x in episodes if x.strip()}
    paths = [p for p in all_paths if not wanted or p.stem in wanted][:max_episodes]
    if not paths:
        raise ValueError("no matching v20 loss histories")

    precision_audit = _audit_fp16_source(candidate)
    if not precision_audit["ok"]:
        return {
            "error": "candidate violates FP16 precision contract",
            "precision_audit": precision_audit,
        }

    fp16_defaults = _set_fp16_defaults()
    if candidate.suffix == ".ipynb":
        candidate_module, candidate_agent = _load_notebook_agent(candidate, "v23_candidate")
    else:
        candidate_module, candidate_agent = _load_python_agent(candidate)

    runtime_audit = _audit_runtime_globals(candidate_module)
    if not runtime_audit["ok"]:
        return {
            "error": "candidate runtime globals violate FP16 precision contract",
            "precision_audit": precision_audit,
            "runtime_precision_audit": runtime_audit,
            "fp16_defaults": fp16_defaults,
        }

    if record_dir is not None:
        record_dir = _inside_v23(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)

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
        step_trace = []

        try:
            for replay_step in range(1, len(history["steps"])):
                obs = _agent_observation(env, candidate_seat)
                with _numpy_fp16_candidate_defaults():
                    action = candidate_agent(obs)
                runtime_audit = _audit_runtime_globals(candidate_module)
                if not runtime_audit["ok"]:
                    raise RuntimeError(
                        "candidate created non-FP16 floating runtime state: "
                        + json.dumps(runtime_audit["violations"][:10], sort_keys=True)
                    )
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

                step_trace.append({
                    "replay_step": replay_step,
                    "day": int(_field(obs, "day", -1)),
                    "hour": int(_field(obs, "hour", -1)),
                    "candidate_action": _plain(action),
                    "recorded_v20_action": _plain(recorded[candidate_seat]),
                    "recorded_opponent_action": _plain(opponent_action),
                    "diverged_from_v20": action != recorded[candidate_seat],
                })

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
            "candidate": str(candidate.relative_to(V23_ROOT)),
            "candidate_seat": "losing v20 seat",
            "opponent_behavior": "recorded historical commands; non-adaptive",
            "recorded_action_parity_required": True,
            "precision_policy": "candidate floating model/tensor compute defaults to FP16",
            "fp16_defaults": fp16_defaults,
        },
        "precision_audit": precision_audit,
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
        result = evaluate(V23_ROOT / args.candidate, episodes, max_episodes)
    except BaseException as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}

    print("__V23_RESULT__" + json.dumps(result, sort_keys=True, default=str))
    return 0 if "error" not in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
