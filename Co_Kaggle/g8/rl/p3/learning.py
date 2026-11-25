"""Replay memory, variable-duration Bellman targets and Double DQN updates."""
from __future__ import annotations

from collections import deque
import math

import numpy as np
import torch
from torch.nn import functional as F

from ..common.profiles import N_ACTIONS, N_FEATURES


def transitions(row, gamma):
    """Convert one episode into (state, action, reward, next state, discount).

    common.rollout supplies the final observation explicitly. Time limits
    retain their bootstrap multiplier; only true termination clears it.
    Never join episodes.
    """
    if not 0 < gamma < 1:
        raise ValueError("Gamma must be between zero and one")
    t = row["trajectory"]
    x = np.asarray(t["x"], dtype=np.float32)
    actions = np.asarray(t["a"])
    rewards = np.asarray(t["r"], dtype=np.float32)
    duration = np.asarray(t["duration"])
    n = len(actions)
    if (x.shape != (n, N_FEATURES) or rewards.shape != (n,)
            or duration.shape != (n,) or actions.shape != (n,) or n == 0
            or not np.isfinite(x).all() or not np.isfinite(rewards).all()
            or not np.issubdtype(actions.dtype, np.integer)
            or not np.issubdtype(duration.dtype, np.integer)
            or np.any(duration <= 0) or duration.sum() != row["steps"]
            or np.any(actions < 0) or np.any(actions >= N_ACTIONS)):
        raise ValueError("Invalid trajectory")
    terminal = bool(row["terminated"])
    final = None
    if not terminal:
        final = np.asarray(row["final_features"], dtype=np.float32)
        if final.shape != (N_FEATURES,) or not np.isfinite(final).all():
            raise ValueError("Invalid final observation features")
    for i in range(n):
        done = terminal and i == n - 1
        if done:
            nxt = np.zeros(N_FEATURES, dtype=np.float32)
        elif i == n - 1:
            nxt = final.copy()
        else:
            nxt = x[i + 1].copy()
        # Scaling by (1-gamma) keeps Q on a per-step reward scale.
        yield (x[i].copy(), int(actions[i]), float((1 - gamma) * rewards[i]),
               nxt, 0.0 if done else float(gamma ** int(duration[i])))


class Replay:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.rows = deque(maxlen=capacity)

    def extend(self, row, gamma):
        self.rows.extend(transitions(row, gamma))

    def __len__(self):
        return len(self.rows)

    def sample(self, size, rng, device):
        indices = rng.choice(len(self.rows), size, replace=False)
        # Materialize once, avoiding repeated random deque indexing.
        rows = list(self.rows)
        x, a, r, nxt, discount = zip(*(rows[i] for i in indices))
        return tuple(torch.as_tensor(np.asarray(v), dtype=dtype, device=device)
                     for v, dtype in zip((x, a, r, nxt, discount),
                                         (torch.float32, torch.long, torch.float32,
                                          torch.float32, torch.float32)))


@torch.no_grad()
def bellman_target(online, target, reward, next_state, discount):
    """Online network selects the action; lagged target evaluates it.

    The ungated Q maximizer is the Double DQN learning target. The deployment
    margin is a separate conservative decision rule.
    """
    action = online.q_values(next_state).argmax(dim=-1, keepdim=True)
    value = target.q_values(next_state).gather(1, action).squeeze(1)
    return reward + discount * value


def optimize(online, target, optimizer, batch):
    state, action, reward, nxt, discount = batch
    expected = bellman_target(online, target, reward, nxt, discount)
    predicted = online.q_values(state).gather(1, action[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(predicted, expected)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite TD loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def exploration(update, updates, start, end):
    if not (1 <= update <= updates and all(map(math.isfinite, (start, end)))
            and 0 < end <= start < 1):
        raise ValueError("Invalid exploration schedule")
    fraction = (update - 1) / max(1, updates - 1)
    return start + fraction * (end - start)
