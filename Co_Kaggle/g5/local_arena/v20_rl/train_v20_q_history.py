#!/usr/bin/env python3
"""Train v20 with residual Double-DQN over v19-feasible worker/task choices.

The frozen v19 learned_task_score remains the action-ranking prior. A small
neural network learns a residual Q correction from real game-history rematches
against frozen v19. Training uses conservative top-k epsilon-greedy exploration,
experience replay, a target network, and held-out deterministic validation.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_v20_ppo import (
    DEFAULT_EXECUTOR,
    GLOBAL_FEATURE_NAMES,
    TASK_FEATURE_NAMES,
    EpisodeResult,
    _field,
    _load_executor,
    choose_device,
    global_state,
    norm_task,
    prepare_opponents,
    write_jsonl,
)
from train_v20_history import load_history_seeds, split_seeds

HERE = Path(__file__).resolve().parent
G5_ROOT = HERE.parent.parent
DEFAULT_HISTORY_ROOT = G5_ROOT / "game_history"


def normalized_prior(raw_scores: np.ndarray) -> np.ndarray:
    """Monotonic, per-state normalization of v19's ranking score.

    The v19 score is a ranking surrogate rather than a calibrated return. We
    preserve its ordering while keeping the fixed prior on a stable scale for
    Bellman updates. The best v19 candidate always has prior 0.
    """
    x = np.asarray(raw_scores, dtype=np.float32)
    if x.size == 0:
        return x
    scale = max(float(x.std()), 1.0)
    return np.clip((x - float(x.max())) / scale, -5.0, 0.0).astype(np.float32)


def money_margin(obs) -> float:
    p = int(obs["player"])
    own = obs["farms"][p]
    opp = obs["farms"][1 - p]
    return float(own["money"]) - float(opp["money"])


@dataclass
class Transition:
    state: np.ndarray
    action_feature: np.ndarray
    action_prior: float
    reward: float
    discount: float
    next_state: np.ndarray | None
    next_candidates: np.ndarray | None
    next_prior: np.ndarray | None
    done: bool


@dataclass
class DecisionRecord:
    state: np.ndarray
    candidates: np.ndarray
    prior: np.ndarray
    action_feature: np.ndarray
    action_prior: float
    margin: float
    turn: int


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: list[Transition] = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def add(self, transition: Transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def extend(self, transitions: list[Transition]):
        for transition in transitions:
            self.add(transition)

    def sample(self, n: int, rng: random.Random) -> list[Transition]:
        return rng.sample(self.data, n)


class ResidualQ(nn.Module):
    """Q(s,a) = fixed normalized v19 prior + trainable neural residual."""

    def __init__(self, task_dim: int, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_dim + state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        # Exact v19 ranking at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def residual(self, states: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, candidates], dim=-1)).squeeze(-1)

    def q_values(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        prior: torch.Tensor,
        prior_scale: float,
    ) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0).expand(candidates.shape[0], -1)
        elif state.shape[0] == 1 and candidates.shape[0] != 1:
            state = state.expand(candidates.shape[0], -1)
        return prior_scale * prior + self.residual(state, candidates)


class QSelector:
    def __init__(
        self,
        model: ResidualQ,
        device,
        prior_scale: float,
        epsilon: float,
        explore_top_k: int,
        gamma: float,
        reward_scale: float,
        reward_clip: float,
        bootstrap_candidates: int,
        rng: random.Random,
        deterministic: bool = False,
        collect: bool = True,
    ):
        self.model = model
        self.device = device
        self.prior_scale = float(prior_scale)
        self.epsilon = 0.0 if deterministic else float(epsilon)
        self.explore_top_k = int(explore_top_k)
        self.gamma = float(gamma)
        self.reward_scale = float(reward_scale)
        self.reward_clip = float(reward_clip)
        self.bootstrap_candidates = int(bootstrap_candidates)
        self.rng = rng
        self.deterministic = deterministic
        self.collect = collect
        self.module = None

        self.records: list[DecisionRecord] = []
        self.current_margin: float | None = None
        self.explorations = 0
        self.decisions = 0
        self.candidate_counts: list[int] = []

    def begin_observation(self, obs):
        self.current_margin = money_margin(obs)

    def _candidate_bundle(self, obs, free, tasks, positions, invs):
        choices = []
        for i in free:
            for k, (p, op, weight, resource) in enumerate(tasks):
                if resource == "NO_WHEAT":
                    if invs[i].get("WHEAT", 0):
                        continue
                elif resource == "NO_FERTILIZER":
                    if invs[i].get("FERTILIZER", 0):
                        continue
                elif resource and not invs[i].get(resource, 0):
                    continue

                distance = self.module.dist(positions[i], p)
                if distance >= 24 - obs["hour"]:
                    continue

                features = self.module.task_features(
                    obs, i, (p, op, weight, resource)
                )
                baseline = float(self.module.learned_task_score(features))
                choices.append((i, k, distance, features, baseline))

        if not choices:
            return choices, None, None, None

        candidates = np.stack(
            [norm_task(choice[3]) for choice in choices]
        ).astype(np.float32)
        raw_baseline = np.asarray(
            [choice[4] for choice in choices], dtype=np.float32
        )
        prior = normalized_prior(raw_baseline)
        state = global_state(self.module, obs, len(choices), len(free))
        return choices, candidates, prior, state

    def _bootstrap_subset(self, candidates, prior):
        if len(candidates) <= self.bootstrap_candidates:
            return (
                np.asarray(candidates, dtype=np.float16).copy(),
                np.asarray(prior, dtype=np.float16).copy(),
            )
        # v19's strongest candidates are the safest bootstrap set for this
        # constrained-improvement experiment.
        ids = np.argsort(-prior, kind="stable")[: self.bootstrap_candidates]
        return (
            np.asarray(candidates[ids], dtype=np.float16).copy(),
            np.asarray(prior[ids], dtype=np.float16).copy(),
        )

    def choose(self, obs, free, tasks, positions, invs):
        choices, candidates, prior, state = self._candidate_bundle(
            obs, free, tasks, positions, invs
        )
        if not choices:
            return None

        ct = torch.as_tensor(candidates, dtype=torch.float32, device=self.device)
        pt = torch.as_tensor(prior, dtype=torch.float32, device=self.device)
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q = (
                self.model.q_values(st, ct, pt, self.prior_scale)
                .detach()
                .cpu()
                .numpy()
            )

        # Exact v19 tie-breaking remains after the learned Q value.
        order = sorted(
            range(len(choices)),
            key=lambda z: (
                float(q[z]),
                choices[z][4],
                -choices[z][2],
                -choices[z][0],
                -choices[z][1],
            ),
            reverse=True,
        )

        selected = order[0]
        if (
            not self.deterministic
            and len(order) > 1
            and self.rng.random() < self.epsilon
        ):
            pool = order[: min(self.explore_top_k, len(order))]
            selected = self.rng.choice(pool)
            if selected != order[0]:
                self.explorations += 1

        self.decisions += 1
        self.candidate_counts.append(len(choices))

        if self.collect:
            bootstrap_candidates, bootstrap_prior = self._bootstrap_subset(
                candidates, prior
            )
            self.records.append(
                DecisionRecord(
                    state=np.asarray(state, dtype=np.float32).copy(),
                    candidates=bootstrap_candidates,
                    prior=bootstrap_prior,
                    action_feature=np.asarray(
                        candidates[selected], dtype=np.float16
                    ).copy(),
                    action_prior=float(prior[selected]),
                    margin=float(
                        self.current_margin
                        if self.current_margin is not None
                        else money_margin(obs)
                    ),
                    turn=int(obs["day"]) * 24 + int(obs["hour"]),
                )
            )

        return choices[selected][0], choices[selected][1]


def build_transitions(
    records: list[DecisionRecord],
    final_margin: float,
    gamma: float,
    reward_scale: float,
    reward_clip: float,
) -> list[Transition]:
    """Convert recorded decisions into TD transitions after the episode.

    Internal worker assignments in the same Kaggriculture hour receive zero
    immediate reward and discount 1. When the environment advances, reward is
    the clipped change in money margin and discount reflects elapsed hours.
    The final decision also receives the terminal win/tie/loss reward.
    """
    if not records:
        return []

    transitions: list[Transition] = []
    for idx, record in enumerate(records):
        if idx + 1 < len(records):
            nxt = records[idx + 1]
            env_steps = max(0, int(nxt.turn) - int(record.turn))
            dense = 0.0
            if env_steps > 0:
                dense = (float(nxt.margin) - float(record.margin)) / reward_scale
                dense = float(np.clip(dense, -reward_clip, reward_clip))
            transitions.append(
                Transition(
                    state=record.state,
                    action_feature=record.action_feature,
                    action_prior=record.action_prior,
                    reward=dense,
                    discount=float(gamma ** env_steps),
                    next_state=nxt.state,
                    next_candidates=nxt.candidates,
                    next_prior=nxt.prior,
                    done=False,
                )
            )
            continue

        dense = (float(final_margin) - float(record.margin)) / reward_scale
        dense = float(np.clip(dense, -reward_clip, reward_clip))
        terminal = (
            1.0 if final_margin > 0
            else -1.0 if final_margin < 0
            else 0.0
        )
        transitions.append(
            Transition(
                state=record.state,
                action_feature=record.action_feature,
                action_prior=record.action_prior,
                reward=dense + terminal,
                discount=0.0,
                next_state=None,
                next_candidates=None,
                next_prior=None,
                done=True,
            )
        )
    return transitions


class QController:
    def __init__(
        self,
        path: Path,
        model: ResidualQ,
        device,
        prior_scale: float,
        epsilon: float,
        explore_top_k: int,
        gamma: float,
        reward_scale: float,
        reward_clip: float,
        bootstrap_candidates: int,
        rng: random.Random,
        deterministic: bool,
        collect: bool,
    ):
        self.selector = QSelector(
            model=model,
            device=device,
            prior_scale=prior_scale,
            epsilon=epsilon,
            explore_top_k=explore_top_k,
            gamma=gamma,
            reward_scale=reward_scale,
            reward_clip=reward_clip,
            bootstrap_candidates=bootstrap_candidates,
            rng=rng,
            deterministic=deterministic,
            collect=collect,
        )
        self.executor = _load_executor(path, self.selector)
        self.selector.module = self.executor

    def __call__(self, obs):
        self.selector.begin_observation(obs)
        return self.executor.agent(obs)


def run_episode(
    model,
    device,
    executor_path,
    opponent,
    seed,
    seat,
    episode_steps,
    prior_scale,
    epsilon,
    explore_top_k,
    gamma,
    reward_scale,
    reward_clip,
    bootstrap_candidates,
    explore_rng,
    deterministic=False,
    collect=True,
):
    from kaggle_environments import make

    label, runner = opponent
    ctrl = QController(
        path=executor_path,
        model=model,
        device=device,
        prior_scale=prior_scale,
        epsilon=epsilon,
        explore_top_k=explore_top_k,
        gamma=gamma,
        reward_scale=reward_scale,
        reward_clip=reward_clip,
        bootstrap_candidates=bootstrap_candidates,
        rng=explore_rng,
        deterministic=deterministic,
        collect=collect,
    )

    def candidate(obs):
        return ctrl(obs)

    players = [None, None]
    players[seat] = candidate
    players[1 - seat] = runner

    try:
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": episode_steps, "seed": seed},
            debug=False,
        )
        env.run(players)
        final = env.steps[-1]
        ours, theirs = final[seat], final[1 - seat]
        status_ours = str(_field(ours, "status", ""))
        status_theirs = str(_field(theirs, "status", ""))
        reward_ours = _field(ours, "reward")
        reward_theirs = _field(theirs, "reward")
        if reward_ours is None or reward_theirs is None:
            raise RuntimeError("missing terminal reward")

        reward_ours = float(reward_ours)
        reward_theirs = float(reward_theirs)
        margin = reward_ours - reward_theirs
        ok = status_ours == "DONE" and status_theirs == "DONE"
        transitions = (
            build_transitions(
                ctrl.selector.records,
                margin,
                gamma,
                reward_scale,
                reward_clip,
            )
            if ok and collect
            else []
        )

        result = EpisodeResult(
            ok=ok,
            seed=seed,
            opponent=label,
            seat=seat,
            our_money=reward_ours,
            opponent_money=reward_theirs,
            margin=margin,
            terminal_reward=(
                1.0 if margin > 0 else -1.0 if margin < 0 else 0.0
            ) if ok else None,
            status_ours=status_ours,
            status_opponent=status_theirs,
            decisions=ctrl.selector.decisions,
            error="" if ok else "non-DONE status",
        )
        info = {
            "explorations": ctrl.selector.explorations,
            "candidate_mean": (
                float(np.mean(ctrl.selector.candidate_counts))
                if ctrl.selector.candidate_counts else 0.0
            ),
            "candidate_max": (
                int(max(ctrl.selector.candidate_counts))
                if ctrl.selector.candidate_counts else 0
            ),
        }
        return result, transitions, info
    except Exception as exc:
        result = EpisodeResult(
            ok=False,
            seed=seed,
            opponent=label,
            seat=seat,
            our_money=None,
            opponent_money=None,
            margin=None,
            terminal_reward=None,
            status_ours="ERROR",
            status_opponent="ERROR",
            decisions=ctrl.selector.decisions,
            error=f"{type(exc).__name__}: {exc}",
        )
        return result, [], {
            "explorations": ctrl.selector.explorations,
            "candidate_mean": 0.0,
            "candidate_max": 0,
        }


def epsilon_for_update(args, update: int) -> float:
    if args.epsilon_decay_updates <= 0:
        return args.epsilon_end
    frac = min(max(update, 0) / args.epsilon_decay_updates, 1.0)
    return args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)


def q_update(
    online: ResidualQ,
    target: ResidualQ,
    optimizer,
    replay: ReplayBuffer,
    device,
    args,
    replay_rng: random.Random,
    optimizer_steps: int,
):
    if len(replay) < max(args.replay_warmup, args.batch_size):
        return {
            "q_updates": 0,
            "td_loss": None,
            "q_mean": None,
            "target_mean": None,
            "mean_abs_td": None,
        }, optimizer_steps

    losses = []
    q_means = []
    target_means = []
    abs_tds = []

    for _ in range(args.gradient_steps_per_update):
        batch = replay.sample(args.batch_size, replay_rng)

        states = torch.as_tensor(
            np.stack([t.state for t in batch]),
            dtype=torch.float32,
            device=device,
        )
        action_features = torch.as_tensor(
            np.stack([t.action_feature for t in batch]).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        action_prior = torch.as_tensor(
            [t.action_prior for t in batch],
            dtype=torch.float32,
            device=device,
        )
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=device
        )
        discounts = torch.as_tensor(
            [t.discount for t in batch], dtype=torch.float32, device=device
        )

        q_pred = online.q_values(
            states, action_features, action_prior, args.prior_scale
        )

        next_values = torch.zeros(len(batch), dtype=torch.float32, device=device)
        flat_states = []
        flat_candidates = []
        flat_prior = []
        slices = []
        cursor = 0

        for batch_index, transition in enumerate(batch):
            if (
                transition.done
                or transition.next_state is None
                or transition.next_candidates is None
                or len(transition.next_candidates) == 0
            ):
                continue

            count = len(transition.next_candidates)
            flat_states.append(
                np.repeat(
                    transition.next_state[None, :], count, axis=0
                ).astype(np.float32)
            )
            flat_candidates.append(
                transition.next_candidates.astype(np.float32)
            )
            flat_prior.append(transition.next_prior.astype(np.float32))
            slices.append((batch_index, cursor, cursor + count))
            cursor += count

        if cursor:
            next_states_t = torch.as_tensor(
                np.concatenate(flat_states, axis=0),
                dtype=torch.float32,
                device=device,
            )
            next_candidates_t = torch.as_tensor(
                np.concatenate(flat_candidates, axis=0),
                dtype=torch.float32,
                device=device,
            )
            next_prior_t = torch.as_tensor(
                np.concatenate(flat_prior, axis=0),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                online_next = online.q_values(
                    next_states_t,
                    next_candidates_t,
                    next_prior_t,
                    args.prior_scale,
                )
                target_next = target.q_values(
                    next_states_t,
                    next_candidates_t,
                    next_prior_t,
                    args.prior_scale,
                )
                for batch_index, start, end in slices:
                    local = online_next[start:end]
                    best = int(torch.argmax(local).item())
                    next_values[batch_index] = target_next[start + best]

        bellman_target = rewards + discounts * next_values
        loss = F.smooth_l1_loss(q_pred, bellman_target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
        optimizer.step()

        optimizer_steps += 1
        if optimizer_steps % args.target_sync_steps == 0:
            target.load_state_dict(online.state_dict())

        td = (bellman_target - q_pred).detach()
        losses.append(float(loss.item()))
        q_means.append(float(q_pred.detach().mean().item()))
        target_means.append(float(bellman_target.detach().mean().item()))
        abs_tds.append(float(td.abs().mean().item()))

    return {
        "q_updates": len(losses),
        "td_loss": float(np.mean(losses)),
        "q_mean": float(np.mean(q_means)),
        "target_mean": float(np.mean(target_means)),
        "mean_abs_td": float(np.mean(abs_tds)),
    }, optimizer_steps


def save_q_checkpoint(
    path: Path,
    online: ResidualQ,
    target: ResidualQ,
    optimizer,
    update: int,
    optimizer_steps: int,
    args,
    sample_rng: random.Random,
    explore_rng: random.Random,
    replay_rng: random.Random,
):
    payload = {
        "algorithm": "v20_residual_double_dqn",
        "update": int(update),
        "optimizer_steps": int(optimizer_steps),
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "task_feature_names": TASK_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
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


def load_q_checkpoint(
    path: Path,
    online: ResidualQ,
    target: ResidualQ,
    optimizer,
    device,
    sample_rng: random.Random,
    explore_rng: random.Random,
    replay_rng: random.Random,
):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("algorithm") != "v20_residual_double_dqn":
        raise ValueError("checkpoint algorithm mismatch")

    online.load_state_dict(payload["online_state_dict"])
    target.load_state_dict(
        payload.get("target_state_dict", payload["online_state_dict"])
    )
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
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    if payload.get("numpy_rng_state") is not None:
        np.random.set_state(payload["numpy_rng_state"])

    return (
        int(payload.get("update", -1)) + 1,
        int(payload.get("optimizer_steps", 0)),
        payload,
    )


def evaluate(
    model,
    device,
    executor,
    opponent,
    seeds,
    args,
    phase="validation",
):
    rows = []
    total = len(seeds) * 2
    wins = ties = losses = 0
    eval_rng = random.Random(0)

    print(f"[{phase}] starting {total} games", flush=True)
    for seed in seeds:
        for seat in (0, 1):
            result, _, _ = run_episode(
                model=model,
                device=device,
                executor_path=executor,
                opponent=opponent,
                seed=seed,
                seat=seat,
                episode_steps=args.episode_steps,
                prior_scale=args.prior_scale,
                epsilon=0.0,
                explore_top_k=args.explore_top_k,
                gamma=args.gamma,
                reward_scale=args.reward_scale,
                reward_clip=args.reward_clip,
                bootstrap_candidates=args.bootstrap_candidates,
                explore_rng=eval_rng,
                deterministic=True,
                collect=False,
            )
            rows.append(result)

            if result.ok and result.margin is not None:
                if result.margin > 0:
                    wins += 1
                    outcome = "WIN"
                elif result.margin < 0:
                    losses += 1
                    outcome = "LOSS"
                else:
                    ties += 1
                    outcome = "TIE"
                completed = len(rows)
                print(
                    f"[{phase}] {completed}/{total} seed={seed} seat={seat} "
                    f"{outcome} margin={result.margin:+.0f} "
                    f"running W/T/L={wins}/{ties}/{losses} "
                    f"win_rate={wins / completed:.1%}",
                    flush=True,
                )
            else:
                print(
                    f"[{phase}] {len(rows)}/{total} seed={seed} seat={seat} "
                    f"ERROR {result.error}",
                    flush=True,
                )

    ok = [row for row in rows if row.ok and row.margin is not None]
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

    margins = np.asarray([float(row.margin) for row in ok], dtype=np.float64)
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
    p.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "runs" / "history_q_vs_v19",
    )
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument("--split-seed", type=int, default=20260919)
    p.add_argument("--training-seed", type=int, default=32020)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--episode-steps", type=int, default=720)
    p.add_argument("--device", default="auto")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gradient-steps-per-update", type=int, default=256)
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
    p.add_argument("--resume", type=Path)
    return p


def main():
    # ------------------------------------------------------------------
    # TRAINING BREAK CONDITIONS — edit these values for later runs.
    # Training stops when EITHER condition becomes true.
    # ------------------------------------------------------------------
    TARGET_WIN_RATE = 0.70
    MAX_TRAINING_HOURS = 2.0
    VALIDATE_EVERY_UPDATES = 1

    args = parser().parse_args()
    if not (0.0 < args.validation_fraction < 1.0):
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not (0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0):
        raise SystemExit("require 0 <= epsilon_end <= epsilon_start <= 1")
    if args.explore_top_k < 1:
        raise SystemExit("--explore-top-k must be >= 1")
    if args.bootstrap_candidates < 1:
        raise SystemExit("--bootstrap-candidates must be >= 1")
    if args.reward_scale <= 0:
        raise SystemExit("--reward-scale must be > 0")

    history_root = args.history_root.expanduser().resolve()
    executor = args.executor.expanduser().resolve()
    if not history_root.is_dir():
        raise SystemExit(f"history root not found: {history_root}")
    if not executor.is_file():
        raise SystemExit(f"v19 executor not found: {executor}")

    print("=== v20 residual Double-DQN training ===", flush=True)
    print(f"history_root={history_root}", flush=True)
    print(f"executor={executor}", flush=True)
    print(
        f"break conditions: validation win rate > {TARGET_WIN_RATE:.1%} "
        f"OR runtime >= {MAX_TRAINING_HOURS:.2f}h",
        flush=True,
    )
    print(
        f"exploration: top-{args.explore_top_k} epsilon-greedy "
        f"{args.epsilon_start:.2%} -> {args.epsilon_end:.2%}",
        flush=True,
    )

    all_seeds, source_rows = load_history_seeds(history_root)
    if len(all_seeds) < 2:
        raise SystemExit("fewer than two usable real game-history seeds found")
    train_seeds, val_seeds = split_seeds(
        all_seeds, args.validation_fraction, args.split_seed
    )

    print(
        f"loaded {len(all_seeds)} unique seeds: "
        f"{len(train_seeds)} train / {len(val_seeds)} validation",
        flush=True,
    )

    device = choose_device(args.device)
    print(f"device={device}", flush=True)

    random.seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)

    sample_rng = random.Random(args.training_seed)
    explore_rng = random.Random(args.training_seed + 1)
    replay_rng = random.Random(args.training_seed + 2)

    online = ResidualQ(
        len(TASK_FEATURE_NAMES), len(GLOBAL_FEATURE_NAMES), args.hidden
    ).to(device)
    target = copy.deepcopy(online).to(device)
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)

    start_update = 0
    optimizer_steps = 0
    if args.resume:
        start_update, optimizer_steps, _ = load_q_checkpoint(
            args.resume.expanduser().resolve(),
            online,
            target,
            optimizer,
            device,
            sample_rng,
            explore_rng,
            replay_rng,
        )
        print(
            f"resumed from {args.resume}: next update={start_update}, "
            f"optimizer_steps={optimizer_steps}",
            flush=True,
        )

    replay = ReplayBuffer(args.replay_capacity)

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
                "algorithm": "residual Double-DQN",
                "executor": str(executor),
                "device_resolved": str(device),
                "opponent": "frozen v19",
                "target_win_rate": TARGET_WIN_RATE,
                "max_training_hours": MAX_TRAINING_HOURS,
                "validate_every_updates": VALIDATE_EVERY_UPDATES,
                "reward": (
                    "delta money margin / reward_scale + terminal win/loss"
                ),
                "action_value": (
                    "normalized v19 learned_task_score prior + neural residual"
                ),
                "replay_checkpointed": False,
                "evaluation_seats": [0, 1],
            },
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    best_path = out / "BEST.json"
    best_win_rate = -1.0
    best_mean_margin = float("-inf")
    if best_path.exists():
        try:
            old_best = json.loads(best_path.read_text(encoding="utf-8"))
            best_win_rate = float(old_best.get("win_rate", -1.0))
            best_mean_margin = float(
                old_best.get("mean_margin", float("-inf"))
            )
        except Exception:
            pass

    with tempfile.TemporaryDirectory(prefix="v20_q_history_") as tmp:
        opponent = prepare_opponents([str(executor)], Path(tmp))[0]

        phase = "resume_initial" if args.resume else "initial"
        print(
            f"\n=== {phase} held-out validation vs frozen v19 "
            f"({len(val_seeds) * 2} games) ===",
            flush=True,
        )
        baseline_started = time.perf_counter()
        baseline = evaluate(
            online, device, executor, opponent, val_seeds, args, phase=phase
        )
        baseline_summary = {
            k: v for k, v in baseline.items() if k != "rows"
        }
        baseline_summary.update(
            update=start_update - 1 if args.resume else -1,
            phase=phase,
            elapsed_eval_seconds=round(
                time.perf_counter() - baseline_started, 3
            ),
        )
        write_jsonl(out / "validation.jsonl", baseline_summary)
        for row in baseline["rows"]:
            write_jsonl(
                out / "validation_games.jsonl",
                {
                    "update": baseline_summary["update"],
                    "phase": phase,
                    "split": "validation",
                    **asdict(row),
                },
            )

        print(
            "baseline summary: "
            + json.dumps(baseline_summary, sort_keys=True),
            flush=True,
        )

        baseline_margin = (
            float(baseline["mean_margin"])
            if baseline["mean_margin"] is not None
            else float("-inf")
        )
        if (
            baseline["win_rate"] > best_win_rate
            or (
                baseline["win_rate"] == best_win_rate
                and baseline_margin > best_mean_margin
            )
        ):
            best_win_rate = baseline["win_rate"]
            best_mean_margin = baseline_margin
            save_q_checkpoint(
                ckpts / "best.pt",
                online,
                target,
                optimizer,
                baseline_summary["update"],
                optimizer_steps,
                args,
                sample_rng,
                explore_rng,
                replay_rng,
            )
            best_path.write_text(
                json.dumps(
                    {
                        "update": baseline_summary["update"],
                        "win_rate": best_win_rate,
                        "mean_margin": baseline["mean_margin"],
                        "phase": phase,
                    },
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )

        if (
            baseline["games_ok"] == baseline["games"]
            and baseline["win_rate"] > TARGET_WIN_RATE
        ):
            save_q_checkpoint(
                ckpts / "target.pt",
                online,
                target,
                optimizer,
                baseline_summary["update"],
                optimizer_steps,
                args,
                sample_rng,
                explore_rng,
                replay_rng,
            )
            print(
                f"target already reached: {baseline['win_rate']:.3f} "
                f"> {TARGET_WIN_RATE:.3f}",
                flush=True,
            )
            return 0

        training_started = time.perf_counter()
        update_no = start_update

        while True:
            elapsed_hours = (
                time.perf_counter() - training_started
            ) / 3600.0
            if elapsed_hours >= MAX_TRAINING_HOURS:
                save_q_checkpoint(
                    ckpts / "timeout.pt",
                    online,
                    target,
                    optimizer,
                    update_no - 1,
                    optimizer_steps,
                    args,
                    sample_rng,
                    explore_rng,
                    replay_rng,
                )
                (out / "TRAINING_STOPPED.json").write_text(
                    json.dumps(
                        {
                            "reason": "timeout",
                            "elapsed_hours": elapsed_hours,
                            "max_training_hours": MAX_TRAINING_HOURS,
                            "last_completed_update": update_no - 1,
                            "best_win_rate": best_win_rate,
                        },
                        indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"TIMEOUT: {elapsed_hours:.3f}h >= "
                    f"{MAX_TRAINING_HOURS:.3f}h",
                    flush=True,
                )
                return 0

            epsilon = epsilon_for_update(args, update_no)
            print(
                f"\n=== Q update {update_no}: "
                f"{args.episodes_per_update} episodes, "
                f"epsilon={epsilon:.2%} ===",
                flush=True,
            )

            update_started = time.perf_counter()
            results = []
            transition_count = 0
            explorations = 0
            candidate_means = []
            candidate_max = 0
            attempts = 0

            while len(results) < args.episodes_per_update:
                attempts += 1
                if attempts > args.episodes_per_update * 4:
                    raise RuntimeError("too many failed training episodes")

                seed = sample_rng.choice(train_seeds)
                seat = sample_rng.randrange(2)
                result, transitions, info = run_episode(
                    model=online,
                    device=device,
                    executor_path=executor,
                    opponent=opponent,
                    seed=seed,
                    seat=seat,
                    episode_steps=args.episode_steps,
                    prior_scale=args.prior_scale,
                    epsilon=epsilon,
                    explore_top_k=args.explore_top_k,
                    gamma=args.gamma,
                    reward_scale=args.reward_scale,
                    reward_clip=args.reward_clip,
                    bootstrap_candidates=args.bootstrap_candidates,
                    explore_rng=explore_rng,
                    deterministic=False,
                    collect=True,
                )

                write_jsonl(
                    out / "episodes.jsonl",
                    {
                        "update": update_no,
                        "split": "train",
                        "epsilon": epsilon,
                        **asdict(result),
                        **info,
                    },
                )

                if not result.ok:
                    print(
                        f"[Q u{update_no}] attempt={attempts} seed={seed} "
                        f"seat={seat} GAME_ERROR status="
                        f"{result.status_ours}/{result.status_opponent} "
                        f"{result.error}",
                        flush=True,
                    )
                    continue
                if result.margin is None:
                    print(
                        f"[Q u{update_no}] attempt={attempts} seed={seed} "
                        f"seat={seat} MISSING_MARGIN",
                        flush=True,
                    )
                    continue
                if not transitions:
                    print(
                        f"[Q u{update_no}] attempt={attempts} seed={seed} "
                        f"seat={seat} NO_TRANSITIONS "
                        f"decisions={result.decisions} "
                        f"margin={result.margin:+.0f}",
                        flush=True,
                    )
                    continue

                replay.extend(transitions)
                results.append(result)
                transition_count += len(transitions)
                explorations += int(info["explorations"])
                candidate_means.append(float(info["candidate_mean"]))
                candidate_max = max(candidate_max, int(info["candidate_max"]))

                outcome = (
                    "WIN" if result.margin > 0
                    else "LOSS" if result.margin < 0
                    else "TIE"
                )
                print(
                    f"[Q u{update_no}] {len(results)}/"
                    f"{args.episodes_per_update} seed={seed} seat={seat} "
                    f"{outcome} margin={result.margin:+.0f} "
                    f"decisions={result.decisions} "
                    f"explore={info['explorations']} "
                    f"replay={len(replay)}",
                    flush=True,
                )

            print(
                f"[Q u{update_no}] collected {transition_count} transitions; "
                f"running {args.gradient_steps_per_update} replay updates...",
                flush=True,
            )
            q_stats, optimizer_steps = q_update(
                online,
                target,
                optimizer,
                replay,
                device,
                args,
                replay_rng,
                optimizer_steps,
            )

            margins = np.asarray(
                [float(result.margin) for result in results],
                dtype=np.float64,
            )
            decisions = sum(result.decisions for result in results)
            train_metrics = {
                "update": update_no,
                "episodes": len(results),
                "transitions": transition_count,
                "replay_size": len(replay),
                "epsilon": epsilon,
                "explorations": explorations,
                "exploration_rate": (
                    explorations / decisions if decisions else 0.0
                ),
                "candidate_mean": (
                    float(np.mean(candidate_means))
                    if candidate_means else 0.0
                ),
                "candidate_max": candidate_max,
                "train_wins": int((margins > 0).sum()),
                "train_ties": int((margins == 0).sum()),
                "train_losses": int((margins < 0).sum()),
                "train_win_rate": float((margins > 0).mean()),
                "train_mean_margin": float(margins.mean()),
                "optimizer_steps": optimizer_steps,
                "elapsed_train_seconds": round(
                    time.perf_counter() - update_started, 3
                ),
                **q_stats,
            }
            write_jsonl(out / "metrics.jsonl", train_metrics)

            save_q_checkpoint(
                ckpts / f"update_{update_no:04d}.pt",
                online,
                target,
                optimizer,
                update_no,
                optimizer_steps,
                args,
                sample_rng,
                explore_rng,
                replay_rng,
            )
            save_q_checkpoint(
                ckpts / "latest.pt",
                online,
                target,
                optimizer,
                update_no,
                optimizer_steps,
                args,
                sample_rng,
                explore_rng,
                replay_rng,
            )

            print(
                f"[Q u{update_no}] train W/T/L="
                f"{train_metrics['train_wins']}/"
                f"{train_metrics['train_ties']}/"
                f"{train_metrics['train_losses']} "
                f"win_rate={train_metrics['train_win_rate']:.1%} "
                f"mean_margin={train_metrics['train_mean_margin']:+.0f} "
                f"td_loss={train_metrics['td_loss']}",
                flush=True,
            )

            should_validate = (
                (update_no - start_update) % VALIDATE_EVERY_UPDATES == 0
            )
            if should_validate:
                eval_started = time.perf_counter()
                val = evaluate(
                    online,
                    device,
                    executor,
                    opponent,
                    val_seeds,
                    args,
                    phase=f"validation q{update_no}",
                )
                val_summary = {
                    k: v for k, v in val.items() if k != "rows"
                }
                val_summary.update(
                    update=update_no,
                    phase="validation",
                    elapsed_eval_seconds=round(
                        time.perf_counter() - eval_started, 3
                    ),
                )
                write_jsonl(out / "validation.jsonl", val_summary)
                for row in val["rows"]:
                    write_jsonl(
                        out / "validation_games.jsonl",
                        {
                            "update": update_no,
                            "phase": "validation",
                            "split": "validation",
                            **asdict(row),
                        },
                    )

                print(
                    f"[Q u{update_no}] validation W/T/L="
                    f"{val['wins']}/{val['ties']}/{val['losses']} "
                    f"win_rate={val['win_rate']:.1%} "
                    f"mean_margin={val['mean_margin']}",
                    flush=True,
                )

                val_margin = (
                    float(val["mean_margin"])
                    if val["mean_margin"] is not None
                    else float("-inf")
                )
                if (
                    val["win_rate"] > best_win_rate
                    or (
                        val["win_rate"] == best_win_rate
                        and val_margin > best_mean_margin
                    )
                ):
                    best_win_rate = val["win_rate"]
                    best_mean_margin = val_margin
                    save_q_checkpoint(
                        ckpts / "best.pt",
                        online,
                        target,
                        optimizer,
                        update_no,
                        optimizer_steps,
                        args,
                        sample_rng,
                        explore_rng,
                        replay_rng,
                    )
                    best_path.write_text(
                        json.dumps(
                            {
                                "update": update_no,
                                "win_rate": best_win_rate,
                                "mean_margin": val["mean_margin"],
                                "wins": val["wins"],
                                "ties": val["ties"],
                                "losses": val["losses"],
                            },
                            indent=2,
                        ) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"[Q u{update_no}] NEW BEST: "
                        f"{best_win_rate:.1%}",
                        flush=True,
                    )

                if (
                    val["games_ok"] == val["games"]
                    and val["win_rate"] > TARGET_WIN_RATE
                ):
                    save_q_checkpoint(
                        ckpts / "target.pt",
                        online,
                        target,
                        optimizer,
                        update_no,
                        optimizer_steps,
                        args,
                        sample_rng,
                        explore_rng,
                        replay_rng,
                    )
                    (out / "TARGET_REACHED.json").write_text(
                        json.dumps(
                            {
                                "update": update_no,
                                "target_win_rate": TARGET_WIN_RATE,
                                "observed_validation_win_rate": val["win_rate"],
                                "wins": val["wins"],
                                "ties": val["ties"],
                                "losses": val["losses"],
                                "validation_games": val["games_ok"],
                                "validation_seeds": val_seeds,
                                "both_seats": True,
                            },
                            indent=2,
                        ) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"TARGET REACHED at Q update {update_no}: "
                        f"{val['win_rate']:.3f} > "
                        f"{TARGET_WIN_RATE:.3f}",
                        flush=True,
                    )
                    return 0

            elapsed_hours = (
                time.perf_counter() - training_started
            ) / 3600.0
            if elapsed_hours >= MAX_TRAINING_HOURS:
                save_q_checkpoint(
                    ckpts / "timeout.pt",
                    online,
                    target,
                    optimizer,
                    update_no,
                    optimizer_steps,
                    args,
                    sample_rng,
                    explore_rng,
                    replay_rng,
                )
                (out / "TRAINING_STOPPED.json").write_text(
                    json.dumps(
                        {
                            "reason": "timeout",
                            "elapsed_hours": elapsed_hours,
                            "max_training_hours": MAX_TRAINING_HOURS,
                            "last_completed_update": update_no,
                            "best_win_rate": best_win_rate,
                        },
                        indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"TIMEOUT after Q update {update_no}: "
                    f"{elapsed_hours:.3f}h >= "
                    f"{MAX_TRAINING_HOURS:.3f}h",
                    flush=True,
                )
                return 0

            print(
                f"[Q u{update_no}] elapsed={elapsed_hours:.3f}h / "
                f"{MAX_TRAINING_HOURS:.3f}h; "
                f"best validation={best_win_rate:.1%}",
                flush=True,
            )
            update_no += 1


if __name__ == "__main__":
    raise SystemExit(main())
