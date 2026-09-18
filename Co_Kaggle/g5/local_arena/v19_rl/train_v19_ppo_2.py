#!/usr/bin/env python3
"""Differentiable-threshold v19 herd-controller experiment.

Control chain is unchanged:
    state -> HERD_THRESHOLD -> unchanged v18 executor -> game -> terminal reward

The real environment is discrete, so gradients cannot pass through Kaggriculture
itself. Instead this trainer learns a differentiable twin-Q surrogate
Q(s, threshold) from Monte-Carlo terminal returns, then updates the actor with
dQ/dthreshold * dthreshold/dtheta.

The actor retains the anchors {200,350,500,650,800}, but outputs their
softmax-weighted expectation as a continuous HERD_THRESHOLD.
"""
from __future__ import annotations

import argparse, json, math, random, shutil, tempfile, time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_v19_ppo import (
    BASELINE_ACTION_INDEX, DEFAULT_EXECUTOR, DEFAULT_OPPONENTS, FEATURE_NAMES,
    HERD_THRESHOLDS, RL_FIRST_DAY, RL_LAST_DAY, _field, _load_executor,
    choose_device, encode_state, prepare_opponents, terminal_reward, write_jsonl,
)

HERE = Path(__file__).resolve().parent
TMIN, TMAX = float(min(HERD_THRESHOLDS)), float(max(HERD_THRESHOLDS))
TMID, THALF = 0.5 * (TMIN + TMAX), 0.5 * (TMAX - TMIN)
BASELINE_THRESHOLD = float(HERD_THRESHOLDS[BASELINE_ACTION_INDEX])


def normalize_threshold(x: torch.Tensor) -> torch.Tensor:
    return (x - TMID) / THALF


@dataclass
class MacroStep:
    state: np.ndarray
    threshold: float
    anchor_probs: list[float]
    day: int


@dataclass
class EpisodeResult:
    ok: bool
    seed: int
    opponent: str
    seat: int
    our_money: float | None
    opponent_money: float | None
    margin: float | None
    terminal_reward: float | None
    status_ours: str
    status_opponent: str
    thresholds: list[float]
    days: list[int]
    error: str = ""


class ThresholdActor(nn.Module):
    """state -> soft anchor weights -> differentiable continuous threshold"""

    def __init__(self, input_dim: int, hidden: int = 64, baseline_bias: float = 3.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, len(HERD_THRESHOLDS))
        self.register_buffer("anchors", torch.tensor(HERD_THRESHOLDS, dtype=torch.float32))
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[BASELINE_ACTION_INDEX] = baseline_bias
            # With symmetric anchors and all other logits zero, expectation is
            # exactly 500. This preserves v18's baseline at initialization.

    def forward(self, x: torch.Tensor):
        logits = self.head(self.body(x))
        probs = torch.softmax(logits, dim=-1)
        threshold = (probs * self.anchors).sum(dim=-1)
        return threshold, probs, logits


class ThresholdQ(nn.Module):
    """Differentiable Q_phi(s, threshold) surrogate."""

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action_norm: torch.Tensor):
        if action_norm.ndim == 1:
            action_norm = action_norm.unsqueeze(-1)
        return self.net(torch.cat([state, action_norm], dim=-1)).squeeze(-1)


class HerdController:
    def __init__(self, executor_path: Path, actor: ThresholdActor, device: torch.device,
                 exploration_std: float = 0.0, forced_threshold: float | None = None,
                 rng: np.random.Generator | None = None):
        self.executor = _load_executor(executor_path)
        self.actor, self.device = actor, device
        self.exploration_std = float(exploration_std)
        self.forced_threshold = forced_threshold
        self.rng = rng or np.random.default_rng()
        self.last_day: int | None = None
        self.steps: list[MacroStep] = []
        self.executor.HERD_THRESHOLD = BASELINE_THRESHOLD

    def __call__(self, obs):
        day = int(obs["day"])
        if RL_FIRST_DAY <= day <= RL_LAST_DAY and day != self.last_day:
            state = encode_state(self.executor, obs)
            st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                threshold_t, probs_t, _ = self.actor(st)
            threshold = float(threshold_t.item())
            if self.forced_threshold is not None:
                threshold = float(self.forced_threshold)
            elif self.exploration_std > 0:
                threshold += float(self.rng.normal(0.0, self.exploration_std))
            threshold = float(np.clip(threshold, TMIN, TMAX))
            self.executor.HERD_THRESHOLD = threshold
            self.steps.append(MacroStep(
                state, threshold,
                [float(x) for x in probs_t.squeeze(0).cpu().tolist()], day,
            ))
            self.last_day = day
        return self.executor.agent(obs)


def run_episode(actor, device, executor_path, opponent, seed, seat, episode_steps,
                exploration_std=0.0, forced_threshold=None):
    from kaggle_environments import make
    label, opponent_runner = opponent
    controller = HerdController(
        executor_path, actor, device, exploration_std, forced_threshold,
        np.random.default_rng(seed ^ 0x5EED5EED),
    )
    players: list[Any] = [None, None]
    players[seat], players[1-seat] = controller, opponent_runner
    try:
        env = make("kaggriculture",
                   configuration={"episodeSteps": episode_steps, "seed": seed},
                   debug=False)
        env.run(players)
        final = env.steps[-1]
        ours, theirs = final[seat], final[1-seat]
        so, st = str(_field(ours, "status", "")), str(_field(theirs, "status", ""))
        ro, rt = _field(ours, "reward", None), _field(theirs, "reward", None)
        if ro is None or rt is None:
            raise RuntimeError("episode finished without numeric rewards")
        ro, rt = float(ro), float(rt)
        margin, ok = ro - rt, so == "DONE" and st == "DONE"
        return EpisodeResult(
            ok, seed, label, seat, ro, rt, margin,
            terminal_reward(margin) if ok else None, so, st,
            [x.threshold for x in controller.steps], [x.day for x in controller.steps],
            "" if ok else "non-DONE status",
        ), controller.steps
    except Exception as exc:
        return EpisodeResult(
            False, seed, label, seat, None, None, None, None, "ERROR", "ERROR",
            [x.threshold for x in controller.steps], [x.day for x in controller.steps],
            f"{type(exc).__name__}: {exc}",
        ), controller.steps


def mc_targets(steps: list[MacroStep], final_reward: float, gamma: float):
    n = len(steps)
    return np.asarray(
        [float(final_reward) * gamma ** (n - 1 - t) for t in range(n)],
        dtype=np.float32,
    )


def set_grad(module: nn.Module, enabled: bool):
    for p in module.parameters():
        p.requires_grad_(enabled)


def update(actor, q1, q2, actor_opt, critic_opt, device, states, thresholds,
           targets, epochs, minibatch, max_grad_norm, entropy_coef, baseline_coef):
    states = torch.as_tensor(states, dtype=torch.float32, device=device)
    thresholds = torch.as_tensor(thresholds, dtype=torch.float32, device=device)
    actions = normalize_threshold(thresholds)
    targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
    n = states.shape[0]
    c_losses, a_losses, q_means, entropies = [], [], [], []

    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, minibatch):
            idx = order[start:start+minibatch]
            s, a, y = states[idx], actions[idx], targets[idx]

            # Terminal reward supervises Q(s, threshold).
            p1, p2 = q1(s, a), q2(s, a)
            c_loss = 0.5 * (F.mse_loss(p1, y) + F.mse_loss(p2, y))
            critic_opt.zero_grad(set_to_none=True)
            c_loss.backward()
            nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), max_grad_norm)
            critic_opt.step()
            c_losses.append(float(c_loss.item()))

            # Actor gets dQ/dthreshold * dthreshold/dtheta.
            set_grad(q1, False); set_grad(q2, False)
            tau, probs, _ = actor(s)
            an = normalize_threshold(tau)
            q = torch.minimum(q1(s, an), q2(s, an))
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(-1).mean()
            baseline_penalty = (((tau - BASELINE_THRESHOLD) / THALF) ** 2).mean()
            a_loss = -q.mean() - entropy_coef * entropy + baseline_coef * baseline_penalty
            actor_opt.zero_grad(set_to_none=True)
            a_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_opt.step()
            set_grad(q1, True); set_grad(q2, True)
            a_losses.append(float(a_loss.item()))
            q_means.append(float(q.mean().item()))
            entropies.append(float(entropy.item()))

    # Diagnostics: if this stays ~0, the learned Q says threshold has no effect.
    probe = actions.detach().clone().requires_grad_(True)
    qp = 0.5 * (q1(states, probe) + q2(states, probe))
    dq_da = torch.autograd.grad(qp.sum(), probe)[0]

    with torch.no_grad():
        tau, probs, _ = actor(states)
        q_anchors = []
        for threshold in HERD_THRESHOLDS:
            a = torch.full(
                (n,), (float(threshold)-TMID)/THALF,
                dtype=torch.float32, device=device,
            )
            q_anchors.append(torch.minimum(q1(states, a), q2(states, a)))
        qm = torch.stack(q_anchors, -1)
        span = qm.max(-1).values - qm.min(-1).values
        mean_probs = probs.mean(0).cpu().numpy()

    return {
        "critic_loss": float(np.mean(c_losses)),
        "actor_loss": float(np.mean(a_losses)),
        "actor_q": float(np.mean(q_means)),
        "actor_entropy": float(np.mean(entropies)),
        "dq_da_abs_mean": float(dq_da.abs().mean().item()),
        "dq_da_abs_max": float(dq_da.abs().max().item()),
        "critic_anchor_q_span_mean": float(span.mean().item()),
        "actor_threshold_mean": float(tau.mean().item()),
        "actor_threshold_std": float(tau.std(unbiased=False).item()),
        "actor_anchor_probs": {
            str(int(t)): float(mean_probs[i]) for i, t in enumerate(HERD_THRESHOLDS)
        },
        "optimizer_minibatches": len(a_losses),
    }


def save_checkpoint(path, actor, q1, q2, actor_opt, critic_opt, update_no, args):
    torch.save({
        "algorithm": "differentiable_threshold_twin_q_mc",
        "update": update_no,
        "actor_state_dict": actor.state_dict(),
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
        "actor_optimizer_state_dict": actor_opt.state_dict(),
        "critic_optimizer_state_dict": critic_opt.state_dict(),
        "feature_names": FEATURE_NAMES,
        "herd_thresholds": HERD_THRESHOLDS,
        "threshold_range": (TMIN, TMAX),
        "args": vars(args),
    }, path)


def load_checkpoint(path, actor, q1, q2, actor_opt, critic_opt, device):
    p = torch.load(path, map_location=device, weights_only=False)
    if p.get("algorithm") != "differentiable_threshold_twin_q_mc":
        raise ValueError("checkpoint belongs to another trainer")
    if tuple(p.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema mismatch")
    actor.load_state_dict(p["actor_state_dict"])
    q1.load_state_dict(p["q1_state_dict"]); q2.load_state_dict(p["q2_state_dict"])
    actor_opt.load_state_dict(p["actor_optimizer_state_dict"])
    critic_opt.load_state_dict(p["critic_optimizer_state_dict"])
    return int(p.get("update", 0)) + 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executor", type=Path, default=DEFAULT_EXECUTOR)
    p.add_argument("--opponents", nargs="+", default=[str(x) for x in DEFAULT_OPPONENTS])
    p.add_argument("--output-dir", type=Path, default=HERE/"runs"/"herd_differentiable_2")
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--episodes-per-update", type=int, default=16)
    p.add_argument("--episode-steps", type=int, default=720)
    p.add_argument("--seed", type=int, default=29019)
    p.add_argument("--device", default="auto")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--actor-learning-rate", type=float, default=3e-4)
    p.add_argument("--critic-learning-rate", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--update-epochs", type=int, default=6)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--baseline-logit-bias", type=float, default=3.0)
    p.add_argument("--entropy-coef", type=float, default=0.001)
    p.add_argument("--baseline-coef", type=float, default=0.0)
    p.add_argument("--exploration-std", type=float, default=150.0)
    p.add_argument("--exploration-min-std", type=float, default=35.0)
    p.add_argument("--exploration-decay", type=float, default=0.985)
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    executor = args.executor.expanduser().resolve()
    if not executor.is_file():
        raise SystemExit(f"executor not found: {executor}")
    device = choose_device(args.device)
    out = args.output_dir.expanduser().resolve()
    ckpts = out/"checkpoints"; ckpts.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    actor = ThresholdActor(len(FEATURE_NAMES), args.hidden, args.baseline_logit_bias).to(device)
    q1 = ThresholdQ(len(FEATURE_NAMES), args.hidden).to(device)
    q2 = ThresholdQ(len(FEATURE_NAMES), args.hidden).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    critic_opt = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()), lr=args.critic_learning_rate
    )
    start = 0
    if args.resume:
        start = load_checkpoint(
            args.resume.expanduser().resolve(), actor, q1, q2,
            actor_opt, critic_opt, device,
        )

    (out/"config.json").write_text(json.dumps({
        **vars(args), "algorithm":"differentiable_threshold_twin_q_mc",
        "executor":str(executor), "device_resolved":str(device),
        "feature_names":FEATURE_NAMES, "threshold_anchors":HERD_THRESHOLDS,
        "threshold_range":[TMIN,TMAX], "baseline_threshold":BASELINE_THRESHOLD,
        "gradient_chain":"terminal_reward -> MC Q target -> Q(s,tau) -> dQ/dtau -> actor",
    }, indent=2, default=str)+"\n", encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="v19_dpg2_") as tmp:
        opponents = prepare_opponents(args.opponents, Path(tmp))
        if args.smoke_only:
            result, steps = run_episode(
                actor, device, executor, opponents[0], args.seed, 0,
                args.episode_steps, 0.0,
            )
            print(json.dumps(asdict(result), indent=2))
            if not result.ok:
                raise SystemExit(1)
            if any(abs(s.threshold-BASELINE_THRESHOLD)>1e-4 for s in steps):
                raise SystemExit("untrained actor did not emit threshold=500")
            return

        episodes_log, metrics_log = out/"episodes.jsonl", out/"metrics.jsonl"
        for update_no in range(start, args.updates):
            steps_all, targets_all, results = [], [], []
            exploration = max(
                args.exploration_min_std,
                args.exploration_std * args.exploration_decay**update_no,
            )
            attempts = 0; started = time.perf_counter()
            while len(results) < args.episodes_per_update:
                attempts += 1
                if attempts > args.episodes_per_update*3:
                    raise RuntimeError("too many failed episodes")
                opponent, seat = rng.choice(opponents), rng.randrange(2)
                seed = rng.randrange(1, 2_147_483_647)
                result, steps = run_episode(
                    actor, device, executor, opponent, seed, seat,
                    args.episode_steps, exploration,
                )
                write_jsonl(episodes_log, {"update":update_no, **asdict(result)})
                if not result.ok or result.terminal_reward is None or not steps:
                    continue
                steps_all.extend(steps)
                targets_all.append(mc_targets(steps, result.terminal_reward, args.gamma))
                results.append(result)

            states = np.stack([s.state for s in steps_all]).astype(np.float32)
            thresholds = np.asarray([s.threshold for s in steps_all], dtype=np.float32)
            targets = np.concatenate(targets_all).astype(np.float32)
            stats = update(
                actor,q1,q2,actor_opt,critic_opt,device,states,thresholds,targets,
                args.update_epochs,args.minibatch_size,args.max_grad_norm,
                args.entropy_coef,args.baseline_coef,
            )
            margins = np.asarray([r.margin for r in results], dtype=np.float64)
            rewards = np.asarray([r.terminal_reward for r in results], dtype=np.float64)
            nearest = Counter(
                min(range(len(HERD_THRESHOLDS)),
                    key=lambda i: abs(HERD_THRESHOLDS[i]-float(x)))
                for x in thresholds
            )
            metrics = {
                "update":update_no,"episodes":len(results),"macro_steps":len(steps_all),
                "wins":int((margins>0).sum()),"ties":int((margins==0).sum()),
                "losses":int((margins<0).sum()),"win_rate":float((margins>0).mean()),
                "mean_margin":float(margins.mean()),"median_margin":float(np.median(margins)),
                "mean_terminal_reward":float(rewards.mean()),"exploration_std":float(exploration),
                "sampled_threshold_mean":float(thresholds.mean()),
                "sampled_threshold_std":float(thresholds.std()),
                "nearest_anchor_counts":{
                    str(int(HERD_THRESHOLDS[i])):int(nearest.get(i,0))
                    for i in range(len(HERD_THRESHOLDS))
                },
                "elapsed_seconds":round(time.perf_counter()-started,3), **stats,
            }
            write_jsonl(metrics_log, metrics); print(json.dumps(metrics, sort_keys=True))
            path = ckpts/f"update_{update_no:04d}.pt"
            save_checkpoint(path,actor,q1,q2,actor_opt,critic_opt,update_no,args)
            shutil.copyfile(path, ckpts/"latest.pt")


if __name__ == "__main__":
    main()
