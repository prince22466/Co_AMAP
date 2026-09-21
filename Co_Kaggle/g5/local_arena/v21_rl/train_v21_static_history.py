#!/usr/bin/env python3
"""Train v21 from the v20 residual Double-DQN using v20 loss histories.

The model architecture and constrained worker-task action space are unchanged
from v20. The only training-protocol change is the environment opponent:

* candidate: v21 policy initialized from the embedded weights in the checked-in v20 submission notebook
* opponent: recorded opponent actions from each game_history/v20 replay

Because the opponent action stream is replayed verbatim, this is STATIC
counterfactual training. It is useful for repairing known v20 failure
trajectories, but it is not a live rematch against an adaptive opponent.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import random
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LOCAL_ARENA = HERE.parent
G5_ROOT = LOCAL_ARENA.parent
V20_DIR = LOCAL_ARENA / "v20_rl"
if str(V20_DIR) not in sys.path:
    sys.path.insert(0, str(V20_DIR))

from evaluate_v20_v19_losses import (
    _agent_observation,
    _environment_from_history,
    _field,
    _recorded_step_actions,
    _saved_final_rewards,
    _seed_hint,
    _run_submission_replacement,
    recorded_action_parity,
)
from train_v20_ppo import (
    GLOBAL_FEATURE_NAMES,
    TASK_FEATURE_NAMES,
    _extract_notebook_main,
    choose_device,
    write_jsonl,
)
from train_v20_q_history import (
    QController,
    ReplayBuffer,
    ResidualQ,
    build_transitions,
    epsilon_for_update,
    q_update,
)
from quantization import (
    QUANTIZATION_CHOICES,
    QuantizedResidualQ,
    quantized_state_dict,
)

DEFAULT_HISTORY_DIR = G5_ROOT / "game_history" / "v20"
DEFAULT_BASE_EXECUTOR = G5_ROOT / "submission_nb" / "kaggriculture-sub_v19.ipynb"
DEFAULT_V20_SUBMISSION = G5_ROOT / "submission_nb" / "kaggriculture-sub_v20.ipynb"
DEFAULT_OUTPUT_DIR = HERE / "runs" / "static_v20_history"


def _infer_v20_seat(history: dict[str, Any]) -> int:
    rewards = _saved_final_rewards(history)
    if rewards[0] == rewards[1]:
        raise ValueError("saved v20 history is tied; cannot infer the v20 seat")
    return 0 if rewards[0] < rewards[1] else 1


def _load_history(path: Path) -> dict[str, Any]:
    history = json.loads(path.read_text(encoding="utf-8"))
    steps = history.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError(f"{path}: replay must contain at least two steps")
    return history


def _final_rewards(states) -> list[float]:
    if len(states) != 2:
        raise ValueError(f"expected two final player states, got {len(states)}")
    rewards = []
    for state in states:
        reward = _field(state, "reward", None)
        if reward is None:
            raise ValueError("missing terminal reward")
        rewards.append(float(reward))
    return rewards


def run_static_episode(
    history_path: Path,
    model: ResidualQ,
    device: torch.device,
    base_executor: Path,
    args,
    explore_rng: random.Random,
    *,
    deterministic: bool,
    collect: bool,
    compare_to_recorded_candidate: bool = False,
) -> tuple[dict[str, Any], list[Any]]:
    history = _load_history(history_path)
    candidate_seat = _infer_v20_seat(history)
    opponent_seat = 1 - candidate_seat
    original_rewards = _saved_final_rewards(history)

    controller = QController(
        path=base_executor,
        model=model,
        device=device,
        prior_scale=args.prior_scale,
        epsilon=0.0 if deterministic else args.current_epsilon,
        explore_top_k=args.explore_top_k,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        bootstrap_candidates=args.bootstrap_candidates,
        rng=explore_rng,
        deterministic=deterministic,
        collect=collect,
    )

    env = _environment_from_history(history)
    action_divergences = 0
    first_divergence = None

    try:
        for replay_step in range(1, len(history["steps"])):
            obs = _agent_observation(env, candidate_seat)
            candidate_action = controller(obs)
            recorded_actions = _recorded_step_actions(history, replay_step)
            opponent_action = recorded_actions[opponent_seat]
            recorded_candidate_action = recorded_actions[candidate_seat]

            if opponent_action is None:
                raise RuntimeError(
                    f"recorded opponent action is None at replay step {replay_step}"
                )

            if compare_to_recorded_candidate and candidate_action != recorded_candidate_action:
                action_divergences += 1
                if first_divergence is None:
                    first_divergence = {
                        "replay_step": replay_step,
                        "day": int(_field(obs, "day", -1)),
                        "hour": int(_field(obs, "hour", -1)),
                        "recorded_v20_action": recorded_candidate_action,
                        "generated_action": candidate_action,
                    }

            actions = [None, None]
            actions[candidate_seat] = candidate_action
            actions[opponent_seat] = opponent_action
            env.step(actions)

        final_states = env.steps[-1]
        rewards = _final_rewards(final_states)
        statuses = [str(_field(s, "status", "")) for s in final_states]
        margin = rewards[candidate_seat] - rewards[opponent_seat]
        ok = statuses == ["DONE", "DONE"]

        transitions = (
            build_transitions(
                controller.selector.records,
                margin,
                args.gamma,
                args.reward_scale,
                args.reward_clip,
            )
            if ok and collect
            else []
        )

        return {
            "episode": history_path.stem,
            "seed": _seed_hint(history),
            "ok": ok,
            "v20_seat": candidate_seat,
            "original_rewards": original_rewards,
            "rewards": rewards,
            "margin": margin,
            "result": "WIN" if margin > 0 else "LOSS" if margin < 0 else "TIE",
            "decisions": controller.selector.decisions,
            "explorations": controller.selector.explorations,
            "candidate_mean": (
                float(np.mean(controller.selector.candidate_counts))
                if controller.selector.candidate_counts else 0.0
            ),
            "candidate_max": (
                int(max(controller.selector.candidate_counts))
                if controller.selector.candidate_counts else 0
            ),
            "action_divergences": action_divergences,
            "first_action_divergence": first_divergence,
            "error": "" if ok else f"non-DONE status: {statuses}",
        }, transitions
    except Exception as exc:
        return {
            "episode": history_path.stem,
            "seed": _seed_hint(history),
            "ok": False,
            "v20_seat": candidate_seat,
            "original_rewards": original_rewards,
            "rewards": None,
            "margin": None,
            "result": "ERROR",
            "decisions": controller.selector.decisions,
            "explorations": controller.selector.explorations,
            "candidate_mean": 0.0,
            "candidate_max": 0,
            "action_divergences": action_divergences,
            "first_action_divergence": first_divergence,
            "error": f"{type(exc).__name__}: {exc}",
        }, []


def _history_paths(history_dir: Path) -> list[Path]:
    paths = sorted(history_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no JSON histories found in {history_dir}")
    return paths


def _split_histories(paths, validation_fraction, split_seed):
    if len(paths) < 2:
        raise ValueError("need at least two v20 histories for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    shuffled = list(paths)
    random.Random(split_seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    return sorted(shuffled[validation_count:]), sorted(shuffled[:validation_count])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _literal_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"submission main.py does not define {name}")


def _load_v20_submission_weights(
    submission: Path,
    online: ResidualQ,
    target: ResidualQ,
    device: torch.device,
    hidden: int,
):
    """Decode the exact residual-Q parameters embedded in kaggriculture-sub_v20.ipynb."""
    if hidden != 64:
        raise ValueError(
            "the checked-in v20 submission embeds a 38->64->64->1 network; "
            "--hidden must remain 64 when initializing v21 from that submission"
        )

    source = _extract_notebook_main(submission)
    encoded = _literal_assignment(source, "_Q_WEIGHTS_B64")
    raw = base64.b64decode(encoded)
    values = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if values.size != 6721:
        raise ValueError(
            f"{submission}: expected 6721 embedded Q parameters, got {values.size}"
        )

    expected_input = len(GLOBAL_FEATURE_NAMES) + len(TASK_FEATURE_NAMES)
    if expected_input != 38:
        raise ValueError(f"unexpected v20 Q input dimension: {expected_input}")

    with torch.no_grad():
        online.net[0].weight.copy_(
            torch.from_numpy(values[0:2432].reshape(64, 38)).to(device)
        )
        online.net[0].bias.copy_(
            torch.from_numpy(values[2432:2496]).to(device)
        )
        online.net[2].weight.copy_(
            torch.from_numpy(values[2496:6592].reshape(64, 64)).to(device)
        )
        online.net[2].bias.copy_(
            torch.from_numpy(values[6592:6656]).to(device)
        )
        online.net[4].weight.copy_(
            torch.from_numpy(values[6656:6720].reshape(1, 64)).to(device)
        )
        online.net[4].bias.copy_(
            torch.from_numpy(values[6720:6721]).to(device)
        )

    target.load_state_dict(online.state_dict())
    return {
        "source": str(submission),
        "sha256": _sha256(submission),
        "weight_count": int(values.size),
        "architecture": "38->64->64->1",
        "embedded_symbol": "_Q_WEIGHTS_B64",
    }


def _save_checkpoint(
    path, online, target, optimizer, update, optimizer_steps, args,
    sample_rng, explore_rng, replay_rng,
):
    payload = {
        "algorithm": "v21_static_residual_double_dqn",
        "update": int(update),
        "optimizer_steps": int(optimizer_steps),
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "deployment_state_dict": quantized_state_dict(online, args.quantization),
        "quantization": args.quantization,
        "optimizer_state_dict": optimizer.state_dict(),
        "task_feature_names": TASK_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "parent_v20_submission": str(args.v20_submission),
        "parent_v20_submission_sha256": args.parent_v20_submission_sha256,
        "training_protocol": "static recorded-opponent reply from game_history/v20",
        "args": vars(args),
        "sample_rng_state": sample_rng.getstate(),
        "explore_rng_state": explore_rng.getstate(),
        "replay_rng_state": replay_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "note": "Replay-buffer contents are intentionally not checkpointed.",
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def _load_v21_checkpoint(
    path, online, target, optimizer, device, sample_rng, explore_rng, replay_rng,
):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("algorithm") != "v21_static_residual_double_dqn":
        raise ValueError(
            f"{path}: expected v21_static_residual_double_dqn, "
            f"got {payload.get('algorithm')!r}"
        )
    saved_quantization = payload.get("quantization", "fp16")
    if getattr(online, "quantization", saved_quantization) != saved_quantization:
        raise ValueError(
            f"checkpoint quantization {saved_quantization!r} does not match "
            f"requested {getattr(online, 'quantization', None)!r}"
        )
    online.load_state_dict(payload["online_state_dict"])
    target.load_state_dict(payload.get("target_state_dict", payload["online_state_dict"]))
    if payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if payload.get("sample_rng_state") is not None:
        sample_rng.setstate(payload["sample_rng_state"])
    if payload.get("explore_rng_state") is not None:
        explore_rng.setstate(payload["explore_rng_state"])
    if payload.get("replay_rng_state") is not None:
        replay_rng.setstate(payload["replay_rng_state"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"])
    if payload.get("numpy_rng_state") is not None:
        np.random.set_state(payload["numpy_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return int(payload.get("update", -1)) + 1, int(payload.get("optimizer_steps", 0)), payload


def _preflight(paths, v20_submission, v20_model, device, base_executor, args):
    rows = []
    for index, path in enumerate(paths, 1):
        history = _load_history(path)
        control = recorded_action_parity(history)
        row = {
            "episode": path.stem,
            "recorded_action_parity": bool(control.get("exact")),
        }
        if not control.get("exact"):
            row["error"] = control.get("mismatch")
            rows.append(row)
            print(f"[preflight {index}/{len(paths)}] {path.stem}: replay parity FAILED", flush=True)
            continue

        if args.skip_v20_policy_parity:
            row["v20_policy_parity"] = None
            rows.append(row)
            print(f"[preflight {index}/{len(paths)}] {path.stem}: replay parity OK", flush=True)
            continue

        original = _saved_final_rewards(history)
        candidate_seat = _infer_v20_seat(history)

        submitted_v20 = _run_submission_replacement(
            history=history,
            candidate_seat=candidate_seat,
            notebook=v20_submission,
            label="v20",
            compare_to_recorded_candidate=True,
        )
        submission_exact = (
            submitted_v20["statuses"] == ["DONE", "DONE"]
            and submitted_v20["rewards"] == original
            and submitted_v20["action_divergences"] == 0
        )

        args.current_epsilon = 0.0
        reconstructed, _ = run_static_episode(
            path, v20_model, device, base_executor, args, random.Random(0),
            deterministic=True, collect=False, compare_to_recorded_candidate=True,
        )
        reconstructed_exact = (
            reconstructed["ok"]
            and reconstructed["action_divergences"] == 0
            and reconstructed["rewards"] == original
        )
        exact = submission_exact and reconstructed_exact
        row.update(
            v20_policy_parity=exact,
            v20_submission_parity=submission_exact,
            reconstructed_v20_parity=reconstructed_exact,
            submission_rewards=submitted_v20["rewards"],
            submission_action_divergences=submitted_v20["action_divergences"],
            reconstructed_rewards=reconstructed["rewards"],
            reconstructed_action_divergences=reconstructed["action_divergences"],
            first_action_divergence=(
                submitted_v20["first_action_divergence"]
                or reconstructed["first_action_divergence"]
            ),
            original_rewards=original,
            error="" if exact else (
                "checked-in v20 submission or its decoded ResidualQ reconstruction "
                "does not reproduce the recorded v20 trajectory"
            ),
        )
        rows.append(row)
        print(
            f"[preflight {index}/{len(paths)}] {path.stem}: "
            f"submission={'OK' if submission_exact else 'FAILED'} "
            f"decoded_model={'OK' if reconstructed_exact else 'FAILED'}",
            flush=True,
        )

    failed = [
        row for row in rows
        if not row["recorded_action_parity"] or row.get("v20_policy_parity") is False
    ]
    if failed:
        examples = ", ".join(row["episode"] for row in failed[:5])
        raise RuntimeError(
            f"preflight failed for {len(failed)} histories ({examples}). "
            "Refusing to train on mismatched static histories. "
            "Use --skip-v20-policy-parity only if the recorded v20 histories "
            "intentionally differ from the checked-in kaggriculture-sub_v20.ipynb."
        )
    return rows


def evaluate_histories(paths, model, device, base_executor, args, phase):
    model.eval()
    rows = []
    wins = ties = losses = errors = 0
    margins = []
    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            args.current_epsilon = 0.0
            result, _ = run_static_episode(
                path, model, device, base_executor, args, random.Random(index),
                deterministic=True, collect=False,
            )
            rows.append(result)
            if not result["ok"]:
                errors += 1
            elif result["margin"] > 0:
                wins += 1; margins.append(float(result["margin"]))
            elif result["margin"] < 0:
                losses += 1; margins.append(float(result["margin"]))
            else:
                ties += 1; margins.append(0.0)
    valid = wins + ties + losses
    summary = {
        "phase": phase, "games_total": len(paths), "games_valid": valid,
        "errors": errors, "wins": wins, "ties": ties, "losses": losses,
        "win_rate": wins / valid if valid else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "rows": rows,
    }
    print(
        f"[{phase}] W/T/L/E={wins}/{ties}/{losses}/{errors} "
        f"win_rate={summary['win_rate']:.3f} mean_margin={summary['mean_margin']}",
        flush=True,
    )
    model.train()
    return summary


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    p.add_argument("--base-executor", type=Path, default=DEFAULT_BASE_EXECUTOR)
    p.add_argument("--v20-submission", type=Path, default=DEFAULT_V20_SUBMISSION)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument("--split-seed", type=int, default=20260921)
    p.add_argument("--training-seed", type=int, default=32121)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--validate-every-updates", type=int, default=1)
    p.add_argument("--checkpoint-every-updates", type=int, default=1)
    p.add_argument("--target-win-rate", type=float, default=0.70)
    p.add_argument("--max-training-hours", type=float, default=8.0)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--skip-v20-policy-parity", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--quantization",
        choices=QUANTIZATION_CHOICES,
        default="fp16",
        help="QAT/deployment weight format; FP16 is the default, FP8 E4M3FN is experimental",
    )
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gradient-steps-per-update", type=int, default=64)
    p.add_argument("--replay-capacity", type=int, default=50000)
    p.add_argument("--replay-warmup", type=int, default=4000)
    p.add_argument("--target-sync-steps", type=int, default=500)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--prior-scale", type=float, default=1.0)
    p.add_argument("--epsilon-start", type=float, default=0.02)
    p.add_argument("--epsilon-end", type=float, default=0.005)
    p.add_argument("--epsilon-decay-updates", type=int, default=40)
    p.add_argument("--explore-top-k", type=int, default=3)
    p.add_argument("--bootstrap-candidates", type=int, default=32)
    p.add_argument("--reward-scale", type=float, default=10000.0)
    p.add_argument("--reward-clip", type=float, default=2.0)
    return p


def main():
    args = build_parser().parse_args()
    history_dir = args.history_dir.expanduser().resolve()
    base_executor = args.base_executor.expanduser().resolve()
    v20_submission = args.v20_submission.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    args.history_dir = history_dir
    args.base_executor = base_executor
    args.v20_submission = v20_submission
    args.output_dir = output_dir

    if not history_dir.is_dir():
        raise SystemExit(f"history directory not found: {history_dir}")
    if not base_executor.is_file():
        raise SystemExit(f"base executor not found: {base_executor}")
    if not v20_submission.is_file():
        raise SystemExit(f"v20 submission notebook not found: {v20_submission}")

    paths = _history_paths(history_dir)
    train_paths, validation_paths = _split_histories(paths, args.validation_fraction, args.split_seed)
    device = choose_device(args.device)
    print(f"device={device}", flush=True)
    print(f"v20 histories={len(paths)} train/validation={len(train_paths)}/{len(validation_paths)}", flush=True)

    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    sample_rng = random.Random(args.training_seed)
    explore_rng = random.Random(args.training_seed ^ 0x21)
    replay_rng = random.Random(args.training_seed ^ 0xD0D1)

    v20_reference = ResidualQ(
        len(TASK_FEATURE_NAMES), len(GLOBAL_FEATURE_NAMES), args.hidden
    ).to(device)
    v20_reference_target = ResidualQ(
        len(TASK_FEATURE_NAMES), len(GLOBAL_FEATURE_NAMES), args.hidden
    ).to(device)
    parent_payload = _load_v20_submission_weights(
        v20_submission, v20_reference, v20_reference_target, device, args.hidden
    )
    args.parent_v20_submission_sha256 = parent_payload["sha256"]
    v20_reference.eval()

    online = QuantizedResidualQ(
        len(TASK_FEATURE_NAMES), len(GLOBAL_FEATURE_NAMES), args.hidden, args.quantization
    ).to(device)
    target = QuantizedResidualQ(
        len(TASK_FEATURE_NAMES), len(GLOBAL_FEATURE_NAMES), args.hidden, args.quantization
    ).to(device)
    online.load_state_dict(v20_reference.state_dict())
    target.load_state_dict(v20_reference.state_dict())
    online.train(); target.eval()

    preflight_rows = _preflight(
        paths, v20_submission, v20_reference, device, base_executor, args
    )
    if args.preflight_only:
        print("preflight-only: all selected histories passed", flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    (output_dir / "history_split.json").write_text(json.dumps({
        "history_dir": str(history_dir),
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "training": [p.name for p in train_paths],
        "validation": [p.name for p in validation_paths],
    }, indent=2) + "\n", encoding="utf-8")
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight_rows, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(json.dumps({
        **vars(args),
        "algorithm": "v21 static residual Double-DQN",
        "device_resolved": str(device),
        "parent_v20_submission": parent_payload["source"],
        "parent_v20_submission_sha256": parent_payload["sha256"],
        "parent_v20_embedded_weight_count": parent_payload["weight_count"],
        "parent_v20_architecture": parent_payload["architecture"],
        "training_protocol": "candidate actions from v21; opponent actions replayed verbatim from v20 loss histories",
        "validation_protocol": "held-out v20 loss histories with the same static recorded-opponent protocol",
        "model_structure": "same ResidualQ(task_features + global_state, hidden=64) as v20",
        "quantization": args.quantization,
        "qat": "FP32 master parameters with straight-through fake-quantized weights/biases on every forward pass",
        "action_value": "normalized v19 learned_task_score prior + neural residual",
        "reward": "delta money margin / reward_scale + terminal win/tie/loss",
        "replay_checkpointed": False,
    }, indent=2, default=str) + "\n", encoding="utf-8")

    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    start_update = 0
    optimizer_steps = 0
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise SystemExit(f"resume checkpoint not found: {resume_path}")
        start_update, optimizer_steps, _ = _load_v21_checkpoint(
            resume_path, online, target, optimizer, device,
            sample_rng, explore_rng, replay_rng,
        )

    replay = ReplayBuffer(args.replay_capacity)
    started = time.monotonic()
    initial_validation = evaluate_histories(
        validation_paths, online, device, base_executor, args, "initial_validation"
    )
    write_jsonl(output_dir / "validation.jsonl", {
        "update": start_update - 1,
        **{k: v for k, v in initial_validation.items() if k != "rows"},
    })

    if initial_validation["games_valid"] > 0 and initial_validation["win_rate"] > args.target_win_rate:
        _save_checkpoint(
            checkpoints / "target.pt", online, target, optimizer, start_update - 1,
            optimizer_steps, args, sample_rng, explore_rng, replay_rng,
        )
        (output_dir / "TARGET_REACHED.json").write_text(
            json.dumps(initial_validation, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return 0

    for update in range(start_update, start_update + args.updates):
        if (time.monotonic() - started) / 3600.0 >= args.max_training_hours:
            _save_checkpoint(
                checkpoints / "timeout.pt", online, target, optimizer, update - 1,
                optimizer_steps, args, sample_rng, explore_rng, replay_rng,
            )
            return 0

        epsilon = epsilon_for_update(args, update)
        args.current_epsilon = epsilon
        episode_rows = []
        added_transitions = 0
        for episode_index in range(args.episodes_per_update):
            path = sample_rng.choice(train_paths)
            result, transitions = run_static_episode(
                path, online, device, base_executor, args, explore_rng,
                deterministic=False, collect=True,
            )
            replay.extend(transitions)
            added_transitions += len(transitions)
            row = {
                "update": update, "episode_index": episode_index,
                "epsilon": epsilon, **result, "transitions": len(transitions),
            }
            episode_rows.append(row)
            write_jsonl(output_dir / "episodes.jsonl", row)

        train_ok = [row for row in episode_rows if row["ok"]]
        train_wins = sum(row["margin"] > 0 for row in train_ok)
        q_stats, optimizer_steps = q_update(
            online, target, optimizer, replay, device, args, replay_rng, optimizer_steps
        )
        metrics = {
            "update": update,
            "epsilon": epsilon,
            "episodes": len(episode_rows),
            "episodes_ok": len(train_ok),
            "training_win_rate": train_wins / len(train_ok) if train_ok else 0.0,
            "mean_training_margin": (
                float(np.mean([row["margin"] for row in train_ok])) if train_ok else None
            ),
            "new_transitions": added_transitions,
            "replay_size": len(replay),
            "optimizer_steps": optimizer_steps,
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
            **q_stats,
        }
        write_jsonl(output_dir / "metrics.jsonl", metrics)
        print(
            f"[update {update:04d}] train_ok={len(train_ok)}/{len(episode_rows)} "
            f"win_rate={metrics['training_win_rate']:.3f} replay={len(replay)} "
            f"td_loss={metrics['td_loss']}",
            flush=True,
        )

        _save_checkpoint(
            checkpoints / "latest.pt", online, target, optimizer, update,
            optimizer_steps, args, sample_rng, explore_rng, replay_rng,
        )
        if args.checkpoint_every_updates > 0 and (update + 1) % args.checkpoint_every_updates == 0:
            _save_checkpoint(
                checkpoints / f"update_{update:04d}.pt", online, target, optimizer, update,
                optimizer_steps, args, sample_rng, explore_rng, replay_rng,
            )

        if args.validate_every_updates > 0 and (update + 1) % args.validate_every_updates == 0:
            validation = evaluate_histories(
                validation_paths, online, device, base_executor, args, f"validation_{update:04d}"
            )
            write_jsonl(output_dir / "validation.jsonl", {
                "update": update,
                **{k: v for k, v in validation.items() if k != "rows"},
            })
            (output_dir / "validation_latest.json").write_text(
                json.dumps(validation, indent=2, default=str) + "\n", encoding="utf-8"
            )
            if validation["games_valid"] > 0 and validation["win_rate"] > args.target_win_rate:
                _save_checkpoint(
                    checkpoints / "target.pt", online, target, optimizer, update,
                    optimizer_steps, args, sample_rng, explore_rng, replay_rng,
                )
                (output_dir / "TARGET_REACHED.json").write_text(
                    json.dumps(validation, indent=2, default=str) + "\n", encoding="utf-8"
                )
                return 0

        if (time.monotonic() - started) / 3600.0 >= args.max_training_hours:
            _save_checkpoint(
                checkpoints / "timeout.pt", online, target, optimizer, update,
                optimizer_steps, args, sample_rng, explore_rng, replay_rng,
            )
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
