#!/usr/bin/env python3
"""Train a PPO residual policy for Kaggriculture v19 herd expansion.

The trainer leaves v18 unchanged. For every episode it loads a fresh copy of
``v18_c258_compiled.py`` as the complete low-level executor. Once per in-game
day, only while v18's animal planner is active (days 3..17), PPO selects one
value for the existing ``HERD_THRESHOLD`` from ``{200, 350, 500, 650, 800}``.

Typical use from ``Co_Kaggle/g5``::

    python local_arena/v19_rl/train_v19_ppo.py --updates 100 --episodes-per-update 16

Dependencies::

    pip install -r local_arena/v19_rl/requirements.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import shutil
import tarfile
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HERE = Path(__file__).resolve().parent
LOCAL_ARENA = HERE.parent
G5_ROOT = LOCAL_ARENA.parent
SUBMISSION_DIR = G5_ROOT / "submission_nb"

DEFAULT_EXECUTOR = LOCAL_ARENA / "v18_c258_compiled.py"
DEFAULT_OPPONENTS = [
    SUBMISSION_DIR / "kaggriculture-sub_v16.ipynb",
    SUBMISSION_DIR / "kaggriculture-sub_v17.ipynb",
    DEFAULT_EXECUTOR,
]

HERD_THRESHOLDS = (200.0, 350.0, 500.0, 650.0, 800.0)
RL_FIRST_DAY = 3
RL_LAST_DAY = 17
BASELINE_ACTION_INDEX = HERD_THRESHOLDS.index(500.0)

BASE_PRICES = {
    "WHEAT": 25.0,
    "CARROT": 35.0,
    "TOMATO": 60.0,
    "STRAWBERRY": 120.0,
    "MELON": 250.0,
    "MILK": 160.0,
    "WOOL": 200.0,
    "EGG": 50.0,
    "FERTILIZER": 100.0,
}

FEATURE_NAMES = (
    "day_fraction",
    "days_remaining_fraction",
    "own_money_100k",
    "opponent_money_100k",
    "money_margin_100k",
    "own_hands_12",
    "opponent_hands_12",
    "own_quadrants_4",
    "opponent_quadrants_4",
    "own_cows_15",
    "own_sheep_15",
    "own_geese_15",
    "opponent_cows_15",
    "opponent_sheep_15",
    "opponent_geese_15",
    "own_wheat_300",
    "own_fertilizer_200",
    "wheat_seed_stock_100",
    "own_wheat_plots_100",
    "own_carrot_plots_100",
    "own_tomato_plots_100",
    "own_strawberry_plots_100",
    "own_melon_plots_100",
    "price_wheat_ratio",
    "price_milk_ratio",
    "price_wool_ratio",
    "price_egg_ratio",
    "price_strawberry_ratio",
    "price_melon_ratio",
    "inventory_wheat_delta_1000",
    "inventory_milk_delta_500",
    "inventory_wool_delta_500",
    "inventory_egg_delta_500",
    "forecast_milk_d4_ratio",
    "forecast_milk_d8_ratio",
    "forecast_wool_d4_ratio",
    "forecast_wool_d8_ratio",
    "forecast_egg_d4_ratio",
    "forecast_egg_d8_ratio",
    "forecast_wheat_d4_ratio",
    "forecast_wheat_d8_ratio",
    "shop_demand_wheat_20",
    "shop_demand_milk_20",
    "shop_demand_wool_20",
    "shop_demand_egg_20",
)


@dataclass
class MacroStep:
    state: np.ndarray
    action: int
    old_log_prob: float
    old_value: float
    day: int
    threshold: float


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


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, baseline_bias: float = 3.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, len(HERD_THRESHOLDS))
        self.critic = nn.Linear(hidden, 1)
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        with torch.no_grad():
            self.actor.bias[BASELINE_ACTION_INDEX] = baseline_bias
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.body(x)
        return self.actor(z), self.critic(z).squeeze(-1)


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _load_executor(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"v18 executor not found: {path}")
    name = f"v19_v18_executor_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import executor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("agent", "forecast", "price", "totals", "HERD_THRESHOLD", "SHOPS")
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise RuntimeError(f"executor missing required symbols: {missing}")
    return module


def _farm_counts(farm: dict[str, Any]) -> tuple[Counter, Counter]:
    animals: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("animal"):
                animals[tile["animal"]] += 1
            if tile.get("crop"):
                crops[tile["crop"]] += 1
    return animals, crops


def _shop_demand(module: Any, shops: Iterable[str]) -> Counter:
    demand: Counter[str] = Counter()
    for shop in shops:
        for product in module.SHOPS.get(shop, ()):
            demand[product] += 1
    return demand


def _future_price(module: Any, projected: dict[str, list[float]], item: str, day: int, horizon: int) -> float:
    at = min(30, day + horizon)
    return float(module.price(item, projected[item][at]))


def encode_state(module: Any, obs: dict[str, Any]) -> np.ndarray:
    """Encode only live/public state plus our own private state."""
    player = int(obs["player"])
    opponent = 1 - player
    day = int(obs["day"])
    own_farm = obs["farms"][player]
    opp_farm = obs["farms"][opponent]
    own_animals, own_crops = _farm_counts(own_farm)
    opp_animals, _ = _farm_counts(opp_farm)
    own_total = module.totals(obs["private"])
    seeds = obs["private"]["seeds"]
    prices = obs["market"]["prices"]
    inventory = obs["market"]["inventory"]
    projected, _ = module.forecast(obs)
    shop_demand = _shop_demand(module, obs["town"]["unlocked_shops"])

    def price_ratio(item: str) -> float:
        return float(prices.get(item, BASE_PRICES[item])) / BASE_PRICES[item]

    values = [
        day / 29.0,
        (29 - day) / 29.0,
        float(own_farm["money"]) / 100_000.0,
        float(opp_farm["money"]) / 100_000.0,
        (float(own_farm["money"]) - float(opp_farm["money"])) / 100_000.0,
        len(own_farm["hands"]) / 12.0,
        len(opp_farm["hands"]) / 12.0,
        len(own_farm["unlocked_quadrants"]) / 4.0,
        len(opp_farm["unlocked_quadrants"]) / 4.0,
        own_animals["COW"] / 15.0,
        own_animals["SHEEP"] / 15.0,
        own_animals["GOOSE"] / 15.0,
        opp_animals["COW"] / 15.0,
        opp_animals["SHEEP"] / 15.0,
        opp_animals["GOOSE"] / 15.0,
        float(own_total.get("WHEAT", 0)) / 300.0,
        float(own_total.get("FERTILIZER", 0)) / 200.0,
        float(seeds.get("WHEAT", 0)) / 100.0,
        own_crops["WHEAT"] / 100.0,
        own_crops["CARROT"] / 100.0,
        own_crops["TOMATO"] / 100.0,
        own_crops["STRAWBERRY"] / 100.0,
        own_crops["MELON"] / 100.0,
        price_ratio("WHEAT"),
        price_ratio("MILK"),
        price_ratio("WOOL"),
        price_ratio("EGG"),
        price_ratio("STRAWBERRY"),
        price_ratio("MELON"),
        (float(inventory["WHEAT"]) - 10_000.0) / 1_000.0,
        (float(inventory["MILK"]) - 10_000.0) / 500.0,
        (float(inventory["WOOL"]) - 10_000.0) / 500.0,
        (float(inventory["EGG"]) - 10_000.0) / 500.0,
        _future_price(module, projected, "MILK", day, 4) / BASE_PRICES["MILK"],
        _future_price(module, projected, "MILK", day, 8) / BASE_PRICES["MILK"],
        _future_price(module, projected, "WOOL", day, 4) / BASE_PRICES["WOOL"],
        _future_price(module, projected, "WOOL", day, 8) / BASE_PRICES["WOOL"],
        _future_price(module, projected, "EGG", day, 4) / BASE_PRICES["EGG"],
        _future_price(module, projected, "EGG", day, 8) / BASE_PRICES["EGG"],
        _future_price(module, projected, "WHEAT", day, 4) / BASE_PRICES["WHEAT"],
        _future_price(module, projected, "WHEAT", day, 8) / BASE_PRICES["WHEAT"],
        shop_demand["WHEAT"] / 20.0,
        shop_demand["MILK"] / 20.0,
        shop_demand["WOOL"] / 20.0,
        shop_demand["EGG"] / 20.0,
    ]
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (len(FEATURE_NAMES),):
        raise AssertionError(f"feature shape {state.shape} != {(len(FEATURE_NAMES),)}")
    if not np.isfinite(state).all():
        raise ValueError("non-finite strategic feature encountered")
    return np.clip(state, -5.0, 5.0)


class HerdPolicyController:
    """PPO macro policy wrapped around an unchanged v18 executor."""
    def __init__(
        self,
        executor_path: Path,
        model: ActorCritic,
        device: torch.device,
        deterministic: bool = False,
        forced_action: int | None = None,
    ) -> None:
        self.executor = _load_executor(executor_path)
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.forced_action = forced_action
        self.last_decision_day: int | None = None
        self.steps: list[MacroStep] = []
        self.executor.HERD_THRESHOLD = HERD_THRESHOLDS[BASELINE_ACTION_INDEX]

    def __call__(self, obs: dict[str, Any]):
        day = int(obs["day"])
        if RL_FIRST_DAY <= day <= RL_LAST_DAY and day != self.last_decision_day:
            state = encode_state(self.executor, obs)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, value = self.model(state_t)
                dist = Categorical(logits=logits)
                if self.forced_action is not None:
                    action_t = torch.tensor([self.forced_action], device=self.device)
                elif self.deterministic:
                    action_t = logits.argmax(dim=-1)
                else:
                    action_t = dist.sample()
                log_prob = dist.log_prob(action_t)
            action = int(action_t.item())
            threshold = HERD_THRESHOLDS[action]
            self.executor.HERD_THRESHOLD = threshold
            self.steps.append(
                MacroStep(
                    state=state,
                    action=action,
                    old_log_prob=float(log_prob.item()),
                    old_value=float(value.item()),
                    day=day,
                    threshold=threshold,
                )
            )
            self.last_decision_day = day
        return self.executor.agent(obs)


def _extract_notebook_main(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = source.splitlines()
        if not lines:
            continue
        match = re.match(r"^\s*%%writefile\s+(.+?)\s*$", lines[0])
        if match and Path(match.group(1).strip("'\"")).name == "main.py":
            found.append("\n".join(lines[1:]) + "\n")
    if len(found) != 1:
        raise ValueError(f"{path}: expected exactly one %%writefile main.py cell; found {len(found)}")
    return found[0]


def _extract_archive_main(path: Path) -> str:
    with tarfile.open(path, "r:*") as archive:
        files = [m for m in archive.getmembers() if m.isfile()]
        roots = [m for m in files if m.name.replace("\\", "/") == "main.py"]
        if len(roots) != 1:
            raise ValueError(f"{path}: expected one root-level main.py")
        handle = archive.extractfile(roots[0])
        if handle is None:
            raise ValueError(f"{path}: could not read main.py")
        return handle.read().decode("utf-8")


def prepare_opponents(raw: list[str], directory: Path) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for index, value in enumerate(raw):
        if value in {"pass", "random", "starter"}:
            result.append((value, value))
            continue
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"opponent not found: {path}")
        lower = path.name.lower()
        if lower.endswith(".py"):
            result.append((path.stem, str(path)))
            continue
        if lower.endswith(".ipynb"):
            source = _extract_notebook_main(path)
        elif lower.endswith((".tar.gz", ".tgz", ".tar")):
            source = _extract_archive_main(path)
        else:
            raise ValueError(f"unsupported opponent format: {path}")
        materialized = directory / f"opponent_{index}_{path.stem}.py"
        materialized.write_text(source, encoding="utf-8")
        result.append((path.stem, str(materialized)))
    if not result:
        raise ValueError("at least one opponent is required")
    return result


def terminal_reward(margin: float) -> float:
    if margin > 0:
        outcome = 1.0
    elif margin < 0:
        outcome = -1.0
    else:
        outcome = 0.0
    return outcome + 0.05 * math.tanh(margin / 10_000.0)


def run_episode(
    model: ActorCritic,
    device: torch.device,
    executor_path: Path,
    opponent: tuple[str, str],
    seed: int,
    seat: int,
    episode_steps: int,
    deterministic: bool = False,
    forced_action: int | None = None,
) -> tuple[EpisodeResult, list[MacroStep]]:
    from kaggle_environments import make

    label, opponent_runner = opponent
    controller = HerdPolicyController(
        executor_path=executor_path,
        model=model,
        device=device,
        deterministic=deterministic,
        forced_action=forced_action,
    )

    def candidate_agent(obs):
        return controller(obs)

    players: list[Any] = [None, None]
    players[seat] = candidate_agent
    players[1 - seat] = opponent_runner
    try:
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": episode_steps, "seed": seed},
            debug=False,
        )
        env.run(players)
        final = env.steps[-1]
        ours = final[seat]
        theirs = final[1 - seat]
        status_ours = str(_field(ours, "status", ""))
        status_theirs = str(_field(theirs, "status", ""))
        our_reward = _field(ours, "reward", None)
        their_reward = _field(theirs, "reward", None)
        if our_reward is None or their_reward is None:
            raise RuntimeError("episode finished without numeric rewards")
        our_money = float(our_reward)
        opponent_money = float(their_reward)
        margin = our_money - opponent_money
        ok = status_ours == "DONE" and status_theirs == "DONE"
        result = EpisodeResult(
            ok=ok,
            seed=seed,
            opponent=label,
            seat=seat,
            our_money=our_money,
            opponent_money=opponent_money,
            margin=margin,
            terminal_reward=terminal_reward(margin) if ok else None,
            status_ours=status_ours,
            status_opponent=status_theirs,
            thresholds=[s.threshold for s in controller.steps],
            days=[s.day for s in controller.steps],
            error="" if ok else "non-DONE status",
        )
        return result, controller.steps
    except Exception as exc:
        return (
            EpisodeResult(
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
                thresholds=[s.threshold for s in controller.steps],
                days=[s.day for s in controller.steps],
                error=f"{type(exc).__name__}: {exc}",
            ),
            controller.steps,
        )


def episode_targets(
    steps: list[MacroStep],
    final_reward: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(steps)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    values = np.asarray([step.old_value for step in steps], dtype=np.float32)
    rewards = np.zeros(n, dtype=np.float32)
    rewards[-1] = float(final_reward)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        next_value = 0.0 if t == n - 1 else float(values[t + 1])
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    return advantages, advantages + values


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    states: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    clip_ratio: float,
    epochs: int,
    minibatch_size: int,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    target_kl: float,
) -> dict[str, float]:
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

    n = states_t.shape[0]
    stats: list[tuple[float, float, float, float, float]] = []
    stop = False
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            idx = order[start : start + minibatch_size]
            logits, values = model(states_t[idx])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t[idx])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
            unclipped = ratio * advantages_t[idx]
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_t[idx]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * torch.square(returns_t[idx] - values).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs_t[idx] - new_log_probs).mean().item()
                clip_fraction = (torch.abs(ratio - 1.0) > clip_ratio).float().mean().item()
            stats.append(
                (
                    float(policy_loss.item()),
                    float(value_loss.item()),
                    float(entropy.item()),
                    float(approx_kl),
                    float(clip_fraction),
                )
            )
            if target_kl > 0 and approx_kl > 1.5 * target_kl:
                stop = True
                break
        if stop:
            break

    arr = np.asarray(stats, dtype=np.float64)
    return {
        "policy_loss": float(arr[:, 0].mean()),
        "value_loss": float(arr[:, 1].mean()),
        "entropy": float(arr[:, 2].mean()),
        "approx_kl": float(arr[:, 3].mean()),
        "clip_fraction": float(arr[:, 4].mean()),
        "optimizer_minibatches": int(len(stats)),
    }


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    update: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "update": update,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature_names": FEATURE_NAMES,
            "herd_thresholds": HERD_THRESHOLDS,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema does not match this trainer")
    if tuple(payload.get("herd_thresholds", ())) != HERD_THRESHOLDS:
        raise ValueError("checkpoint action schema does not match this trainer")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("update", 0)) + 1


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", type=Path, default=DEFAULT_EXECUTOR)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=[str(path) for path in DEFAULT_OPPONENTS],
        help="Opponent .py/.ipynb/archive paths or built-ins: pass random starter",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs" / "herd_ppo")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=19019)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--baseline-logit-bias",
        type=float,
        default=3.0,
        help="Initial logit advantage for threshold=500, keeping early play near v18",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run one episode forced to threshold=500 and exit without training",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.updates <= 0 or args.episodes_per_update <= 0:
        raise SystemExit("--updates and --episodes-per-update must be positive")
    if args.minibatch_size <= 0:
        raise SystemExit("--minibatch-size must be positive")

    executor_path = args.executor.expanduser().resolve()
    if not executor_path.is_file():
        raise SystemExit(f"executor not found: {executor_path}")

    output_dir = args.output_dir.expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    model = ActorCritic(
        input_dim=len(FEATURE_NAMES),
        hidden=args.hidden,
        baseline_bias=args.baseline_logit_bias,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_update = 0
    if args.resume:
        start_update = load_checkpoint(args.resume.expanduser().resolve(), model, optimizer, device)

    config = {
        **vars(args),
        "executor": str(executor_path),
        "output_dir": str(output_dir),
        "resume": str(args.resume.resolve()) if args.resume else None,
        "device_resolved": str(device),
        "feature_names": FEATURE_NAMES,
        "herd_thresholds": HERD_THRESHOLDS,
        "rl_days": [RL_FIRST_DAY, RL_LAST_DAY],
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory(prefix="v19_ppo_opponents_") as temp_name:
        opponents = prepare_opponents(args.opponents, Path(temp_name))
        print(f"device={device} executor={executor_path}")
        print("opponents=" + ", ".join(label for label, _ in opponents))
        print(f"features={len(FEATURE_NAMES)} actions={HERD_THRESHOLDS}")

        if args.smoke_only:
            result, steps = run_episode(
                model=model,
                device=device,
                executor_path=executor_path,
                opponent=opponents[0],
                seed=args.seed,
                seat=0,
                episode_steps=args.episode_steps,
                deterministic=True,
                forced_action=BASELINE_ACTION_INDEX,
            )
            print(json.dumps(asdict(result), indent=2))
            if not result.ok:
                raise SystemExit(1)
            expected_days = list(range(RL_FIRST_DAY, RL_LAST_DAY + 1))
            if [s.day for s in steps] != expected_days:
                raise SystemExit(f"unexpected macro decision days: {[s.day for s in steps]}")
            if any(s.threshold != 500.0 for s in steps):
                raise SystemExit("smoke run did not stay at v18 HERD_THRESHOLD=500")
            return

        episodes_log = output_dir / "episodes.jsonl"
        metrics_log = output_dir / "metrics.jsonl"
        for update in range(start_update, args.updates):
            batch_steps: list[MacroStep] = []
            batch_advantages: list[np.ndarray] = []
            batch_returns: list[np.ndarray] = []
            episode_results: list[EpisodeResult] = []
            started = time.perf_counter()

            attempts = 0
            while len(episode_results) < args.episodes_per_update:
                attempts += 1
                if attempts > args.episodes_per_update * 3:
                    raise RuntimeError("too many failed episodes; inspect episodes.jsonl")
                opponent = rng.choice(opponents)
                seat = rng.randrange(2)
                env_seed = rng.randrange(1, 2_147_483_647)
                result, steps = run_episode(
                    model=model,
                    device=device,
                    executor_path=executor_path,
                    opponent=opponent,
                    seed=env_seed,
                    seat=seat,
                    episode_steps=args.episode_steps,
                )
                write_jsonl(episodes_log, {"update": update, **asdict(result)})
                if not result.ok or result.terminal_reward is None or not steps:
                    print(f"episode failed: {result.error}")
                    continue
                adv, ret = episode_targets(steps, result.terminal_reward, args.gamma, args.gae_lambda)
                batch_steps.extend(steps)
                batch_advantages.append(adv)
                batch_returns.append(ret)
                episode_results.append(result)

            states = np.stack([step.state for step in batch_steps]).astype(np.float32)
            actions = np.asarray([step.action for step in batch_steps], dtype=np.int64)
            old_log_probs = np.asarray([step.old_log_prob for step in batch_steps], dtype=np.float32)
            advantages = np.concatenate(batch_advantages).astype(np.float32)
            returns = np.concatenate(batch_returns).astype(np.float32)

            update_stats = ppo_update(
                model=model,
                optimizer=optimizer,
                device=device,
                states=states,
                actions=actions,
                old_log_probs=old_log_probs,
                advantages=advantages,
                returns=returns,
                clip_ratio=args.clip_ratio,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
            )

            margins = np.asarray([r.margin for r in episode_results], dtype=np.float64)
            rewards = np.asarray([r.terminal_reward for r in episode_results], dtype=np.float64)
            wins = int((margins > 0).sum())
            ties = int((margins == 0).sum())
            losses = int((margins < 0).sum())
            action_counts = Counter(step.action for step in batch_steps)
            elapsed = time.perf_counter() - started
            metrics = {
                "update": update,
                "episodes": len(episode_results),
                "macro_steps": len(batch_steps),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": wins / len(episode_results),
                "mean_margin": float(margins.mean()),
                "median_margin": float(np.median(margins)),
                "mean_terminal_reward": float(rewards.mean()),
                "action_counts": {
                    str(int(HERD_THRESHOLDS[index])): int(action_counts.get(index, 0))
                    for index in range(len(HERD_THRESHOLDS))
                },
                "elapsed_seconds": round(elapsed, 3),
                **update_stats,
            }
            write_jsonl(metrics_log, metrics)
            print(json.dumps(metrics, sort_keys=True))

            numbered = checkpoints_dir / f"update_{update:04d}.pt"
            latest = checkpoints_dir / "latest.pt"
            save_checkpoint(numbered, model, optimizer, update, args)
            shutil.copyfile(numbered, latest)


if __name__ == "__main__":
    main()
