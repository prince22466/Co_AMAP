#!/usr/bin/env python3
"""v22: FP16 PPO for SELL decisions only, trained on static v20 loss histories.

The checked-in v20 submission remains the executor for every non-selling decision:
worker/task allocation, crop/animal planning, movement, hiring, purchasing, land,
and all other market logic are unchanged. v22 surgically replaces only the SELL
loop inside v20 market_orders().

Opponent actions are replayed verbatim from game_history/v20, matching the static
counterfactual protocol used by v21. The v21 worker model is not imported or used.
Only its production/delivery telemetry methodology is reproduced here for logging.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HERE = Path(__file__).resolve().parent
LOCAL_ARENA = HERE.parent
G5_ROOT = LOCAL_ARENA.parent
V20_DIR = LOCAL_ARENA / "v20_rl"
if str(V20_DIR) not in sys.path:
    sys.path.insert(0, str(V20_DIR))

from evaluate_v20_v19_losses import (
    _agent_observation,
    _environment_from_history,
    _extract_notebook_main,
    _field,
    _recorded_step_actions,
    _run_submission_replacement,
    _saved_final_rewards,
    _seed_hint,
    recorded_action_parity,
)

DEFAULT_HISTORY_DIR = G5_ROOT / "game_history" / "v20"
DEFAULT_V20_SUBMISSION = G5_ROOT / "submission_nb" / "kaggriculture-sub_v20.ipynb"
DEFAULT_OUTPUT_DIR = HERE / "runs" / "static_v20_history"

PRODUCTS = (
    "MILK", "WOOL", "STRAWBERRY", "MELON", "EGG",
    "TOMATO", "CARROT", "FERTILIZER", "WHEAT",
)
PRICE_SCALES = {
    "WHEAT": 25.0, "CARROT": 35.0, "TOMATO": 60.0,
    "STRAWBERRY": 120.0, "MELON": 250.0, "EGG": 50.0,
    "MILK": 160.0, "WOOL": 200.0, "FERTILIZER": 100.0,
}
ACTION_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
NUM_ACTIONS = len(ACTION_FRACTIONS)
SHED_CAPACITY = 100
MOVE_ACTIONS = {"NORTH", "SOUTH", "EAST", "WEST"}

GLOBAL_FEATURE_NAMES = (
    "day", "hour", "turn", "remaining_turns", "money", "opp_money",
    "money_margin", "effective_stock", "capacity_remaining", "overflow",
    "sellable_value", "last_5_turns",
)
PRODUCT_FEATURE_NAMES = (
    "sellable_qty", "held_qty", "total_qty", "reserve_qty",
    "price_now", "ma_1d", "ma_3d", "ma_5d", "min_5d", "max_5d",
    "price_vs_ma_5d", "price_vs_max_5d", "delta_1_turn", "delta_1_day",
    "ema_1d", "ema_5d", "ema_spread", "percentile_5d",
)
STATE_DIM = len(GLOBAL_FEATURE_NAMES) + len(PRODUCTS) * len(PRODUCT_FEATURE_NAMES)
CHECKPOINT_ALGORITHM = "v22_fp16_sell_only_ppo_static_v20"


@dataclass
class SellStep:
    state: np.ndarray
    mask: np.ndarray
    actions: np.ndarray
    old_log_prob: float
    old_value: float
    turn: int
    shaping_reward: float
    quoted_units_sold: int
    quoted_sale_value: float
    expected_overflow_units: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _history_paths(history_dir: Path) -> list[Path]:
    paths = sorted(history_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no JSON histories found in {history_dir}")
    return paths


def _split_histories(paths, validation_fraction, split_seed):
    if len(paths) < 2:
        raise ValueError("need at least two v20 histories for train/validation")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    shuffled = list(paths)
    random.Random(split_seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * validation_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def _load_history(path: Path) -> dict[str, Any]:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history.get("steps"), list) or len(history["steps"]) < 2:
        raise ValueError(f"{path}: invalid replay")
    return history


def _infer_v20_seat(history: dict[str, Any]) -> int:
    rewards = _saved_final_rewards(history)
    if rewards[0] == rewards[1]:
        raise ValueError("saved v20 history is tied; cannot infer v20 seat")
    return 0 if rewards[0] < rewards[1] else 1


def _final_rewards(states) -> list[float]:
    if len(states) != 2:
        raise ValueError(f"expected two terminal states, got {len(states)}")
    out = []
    for state in states:
        reward = _field(state, "reward", None)
        if reward is None:
            raise ValueError("missing terminal reward")
        out.append(float(reward))
    return out


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


class SellActorCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, len(PRODUCTS) * NUM_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        with torch.no_grad():
            self.actor.bias.view(len(PRODUCTS), NUM_ACTIONS)[:, -1] = 2.0
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        self.half()

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        logits = self.actor(h).view(*state.shape[:-1], len(PRODUCTS), NUM_ACTIONS)
        return logits, self.critic(h).squeeze(-1)

    @staticmethod
    def masked_logits(logits: torch.Tensor, mask: torch.Tensor):
        floor = torch.tensor(-60000.0, dtype=torch.float16, device=logits.device)
        return torch.where(mask, logits, floor)


class SellingPolicy:
    def __init__(
        self, model, device, episode_steps, *,
        deterministic, collect, forced_v20_baseline,
        price_shaping, overflow_penalty,
    ):
        self.model = model
        self.device = device
        self.episode_steps = int(episode_steps)
        self.deterministic = deterministic
        self.collect = collect
        self.forced_v20_baseline = forced_v20_baseline
        self.price_shaping = float(price_shaping)
        self.overflow_penalty = float(overflow_penalty)
        self.price_history = []
        self.steps = []
        self.decisions = 0
        self.sale_orders = 0
        self.quoted_units_sold = 0
        self.quoted_sale_value = 0.0
        self.expected_overflow_units = 0
        self.sold_units_by_product = {p: 0 for p in PRODUCTS}
        self.quoted_value_by_product = {p: 0.0 for p in PRODUCTS}

    @staticmethod
    def _ema(values, span):
        if not values:
            return 0.0
        alpha = 2.0 / (span + 1.0)
        out = float(values[0])
        for value in values[1:]:
            out = alpha * float(value) + (1.0 - alpha) * out
        return out

    def _push_prices(self, turn, prices):
        if self.price_history and turn <= self.price_history[-1][0]:
            self.price_history.clear()
        self.price_history.append((turn, {p: float(prices[p]) for p in PRODUCTS}))
        self.price_history = self.price_history[-121:]

    def _series(self, product, n=None):
        rows = self.price_history if n is None else self.price_history[-n:]
        return [float(row[1][product]) for row in rows]

    def _reserves(self, obs):
        day, hour = int(obs["day"]), int(obs["hour"])
        turn = day * 24 + hour
        remaining = max(1, self.episode_steps - turn)
        final_five = remaining <= 5
        if final_five:
            return 0, 0, True
        farm = obs["farms"][int(obs["player"])]
        live = sum(
            1 for row in farm["tiles"] for tile in row
            if isinstance(tile, dict) and "animal" in tile
        )
        return max(4, live + 2), (0 if day < 10 else 4), False

    @staticmethod
    def _action_quantities(available):
        values = []
        for index, fraction in enumerate(ACTION_FRACTIONS):
            qty = int(available) if index == NUM_ACTIONS - 1 else int(math.floor(float(available) * fraction))
            values.append(max(0, min(int(available), qty)))
        return values

    def _action_mask(self, available):
        mask = np.zeros((len(PRODUCTS), NUM_ACTIONS), dtype=np.bool_)
        quantities = {}
        for p_idx, product in enumerate(PRODUCTS):
            q = self._action_quantities(available[product])
            quantities[product] = q
            seen = set()
            for a_idx, qty in enumerate(q):
                if qty in seen:
                    continue
                seen.add(qty)
                mask[p_idx, a_idx] = True
            mask[p_idx, 0] = True
        return mask, quantities

    def _state(self, obs, held, total, available, reserves, final_five):
        day, hour = int(obs["day"]), int(obs["hour"])
        turn = day * 24 + hour
        remaining = max(1, self.episode_steps - turn)
        player = int(obs["player"])
        own, opp = obs["farms"][player], obs["farms"][1 - player]
        prices = obs["market"]["prices"]
        effective_stock = int(sum(max(0, int(v)) for v in held.values()))
        sellable_value = sum(float(available[p]) * float(prices[p]) for p in PRODUCTS)
        global_features = [
            day / 29.0,
            hour / 23.0,
            turn / max(1.0, float(self.episode_steps - 1)),
            remaining / max(1.0, float(self.episode_steps)),
            float(own["money"]) / 100000.0,
            float(opp["money"]) / 100000.0,
            (float(own["money"]) - float(opp["money"])) / 100000.0,
            effective_stock / float(SHED_CAPACITY),
            (SHED_CAPACITY - effective_stock) / float(SHED_CAPACITY),
            max(0, effective_stock - SHED_CAPACITY) / float(SHED_CAPACITY),
            sellable_value / 100000.0,
            1.0 if final_five else 0.0,
        ]
        product_features = []
        percentiles = {}
        for product in PRODUCTS:
            scale = PRICE_SCALES[product]
            series = self._series(product)
            one_day, three_day, five_day = series[-24:], series[-72:], series[-120:]
            current = float(prices[product])
            ma1 = float(np.mean(one_day)) if one_day else current
            ma3 = float(np.mean(three_day)) if three_day else current
            ma5 = float(np.mean(five_day)) if five_day else current
            min5, max5 = (min(five_day), max(five_day)) if five_day else (current, current)
            previous = series[-2] if len(series) >= 2 else current
            previous_day = series[-25] if len(series) >= 25 else series[0]
            ema1, ema5 = self._ema(one_day, 24), self._ema(five_day, 120)
            percentile = (
                sum(value <= current for value in five_day) / len(five_day)
                if five_day else 0.5
            )
            percentiles[product] = float(percentile)
            reserve = int(reserves.get(product, 0))
            product_features.extend([
                available[product] / 100.0,
                max(0, int(held.get(product, 0))) / 100.0,
                max(0, int(total.get(product, 0))) / 100.0,
                reserve / 100.0,
                current / scale,
                ma1 / scale,
                ma3 / scale,
                ma5 / scale,
                min5 / scale,
                max5 / scale,
                current / max(1e-3, ma5) - 1.0,
                current / max(1e-3, max5),
                (current - previous) / scale,
                (current - previous_day) / scale,
                ema1 / scale,
                ema5 / scale,
                (ema1 - ema5) / scale,
                percentile,
            ])
        state = np.asarray(global_features + product_features, dtype=np.float16)
        state = np.clip(state, np.float16(-5.0), np.float16(5.0)).astype(np.float16)
        if state.size != STATE_DIM:
            raise RuntimeError(f"state feature mismatch: {state.size} != {STATE_DIM}")
        return state, percentiles

    def _baseline_orders(self, held, total, reserve_wheat, reserve_fert):
        orders = []
        for product in PRODUCTS:
            n = int(held.get(product, 0))
            if product == "WHEAT":
                n = min(n, max(0, int(total.get(product, 0)) - int(reserve_wheat)))
            if product == "FERTILIZER":
                n = min(n, max(0, int(total.get(product, 0)) - int(reserve_fert)))
            if n:
                orders.append(["SELL", product, n])
        return orders

    def orders(self, obs, held, total, reserve_wheat_v20, reserve_fert_v20):
        if self.forced_v20_baseline:
            return self._baseline_orders(held, total, reserve_wheat_v20, reserve_fert_v20)

        turn = int(obs["day"]) * 24 + int(obs["hour"])
        prices = obs["market"]["prices"]
        self._push_prices(turn, prices)
        reserve_wheat, reserve_fert, final_five = self._reserves(obs)
        reserves = {"WHEAT": reserve_wheat, "FERTILIZER": reserve_fert}
        available = {p: max(0, int(held.get(p, 0))) for p in PRODUCTS}
        available["WHEAT"] = min(
            available["WHEAT"], max(0, int(total.get("WHEAT", 0)) - reserve_wheat)
        )
        available["FERTILIZER"] = min(
            available["FERTILIZER"], max(0, int(total.get("FERTILIZER", 0)) - reserve_fert)
        )
        if not any(available.values()):
            return []

        state, percentiles = self._state(obs, held, total, available, reserves, final_five)
        mask, quantities = self._action_mask(available)
        st = torch.as_tensor(state, dtype=torch.float16, device=self.device).unsqueeze(0)
        mt = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(st)
            logits = self.model.masked_logits(logits, mt)
            dist = Categorical(logits=logits)
            actions = logits.argmax(dim=-1) if self.deterministic else dist.sample()
            log_prob = dist.log_prob(actions).sum(dim=-1)
        action_np = actions[0].detach().cpu().numpy().astype(np.int64)

        orders = []
        sold_units = 0
        quoted_value = 0.0
        quality_numerator = 0.0
        for p_idx, product in enumerate(PRODUCTS):
            qty = quantities[product][int(action_np[p_idx])]
            if qty <= 0:
                continue
            orders.append(["SELL", product, int(qty)])
            value_now = float(qty) * float(prices[product])
            sold_units += int(qty)
            quoted_value += value_now
            quality_numerator += float(qty) * (2.0 * percentiles[product] - 1.0)
            self.sold_units_by_product[product] += int(qty)
            self.quoted_value_by_product[product] += value_now

        effective_stock = int(sum(max(0, int(v)) for v in held.values()))
        expected_overflow = max(0, effective_stock - sold_units - SHED_CAPACITY)
        price_quality = quality_numerator / max(1, sold_units)
        shaping = (
            self.price_shaping * price_quality
            - self.overflow_penalty * min(1.0, expected_overflow / float(SHED_CAPACITY))
        )
        self.decisions += 1
        self.sale_orders += len(orders)
        self.quoted_units_sold += sold_units
        self.quoted_sale_value += quoted_value
        self.expected_overflow_units += expected_overflow
        if self.collect:
            self.steps.append(SellStep(
                state=state, mask=mask, actions=action_np,
                old_log_prob=float(log_prob[0].item()),
                old_value=float(value[0].item()),
                turn=turn, shaping_reward=float(shaping),
                quoted_units_sold=sold_units,
                quoted_sale_value=quoted_value,
                expected_overflow_units=expected_overflow,
            ))
        return orders


def _install_selling_policy(source: str) -> str:
    tree = ast.parse(source)
    market_orders = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "market_orders"),
        None,
    )
    loops = [] if market_orders is None else [
        node for node in market_orders.body
        if isinstance(node, ast.For) and any(
            isinstance(child, ast.Constant) and child.value == "SELL"
            for child in ast.walk(node)
        )
    ]
    if len(loops) != 1:
        raise RuntimeError(f"expected exactly one v20 market_orders SELL loop, found {len(loops)}")
    loop = loops[0]
    replacement = [
        "    for sale in SELLING_POLICY.orders(obs, held, total, reserve_wheat, reserve_fert):",
        "        orders.append(sale)",
        "        cash += sale[2] * max(1, prices[sale[1]] * .8)",
    ]
    lines = source.splitlines()
    return "\n".join(lines[:loop.lineno - 1] + replacement + lines[loop.end_lineno:]) + "\n"


def _load_v20_executor(path: Path, selling_policy):
    path = path.expanduser().resolve()
    source = _extract_notebook_main(path) if path.suffix == ".ipynb" else path.read_text(encoding="utf-8")
    source = _install_selling_policy(source)
    module = types.ModuleType(f"v22_sell_executor_{id(selling_policy)}")
    module.__file__ = str(path)
    module.SELLING_POLICY = selling_policy
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class WorkerTelemetry:
    """v21-equivalent production/transport/delivery metrics; never used as reward."""
    def __init__(self, executor):
        self.executor = executor
        self.eligible = []
        self.produced_value = 0.0
        self.transport_progress_value = 0.0
        self.delivered_value = 0.0
        self.produced_units = 0.0
        self.delivered_units = 0.0
        self.produced_value_by_product = {p: 0.0 for p in PRODUCTS}
        self.delivered_value_by_product = {p: 0.0 for p in PRODUCTS}

    def _ensure_workers(self, count):
        while len(self.eligible) < count:
            self.eligible.append({})

    @staticmethod
    def _worker_actions(candidate_action):
        if not isinstance(candidate_action, dict):
            return []
        return [candidate_action.get("farmer") or ["PASS"]] + list(candidate_action.get("hands") or [])

    @staticmethod
    def _positions(obs):
        farm = obs["farms"][int(obs["player"])]
        return [farm["farmer"]] + list(farm["hands"])

    @staticmethod
    def _tile(obs, position):
        if position is None:
            return None
        farm = obs["farms"][int(obs["player"])]
        x, y = int(position[0]), int(position[1])
        return farm["tiles"][y][x]

    @staticmethod
    def _nearest_shed_distance(position, shed):
        if position is None:
            return 0
        x, y = int(position[0]), int(position[1])
        return min(abs(x - sx) + abs(y - sy) for sx, sy in shed)

    def _produced_from_action(self, obs, position, action):
        if not action:
            return {}
        op = action[0]
        tile = self._tile(obs, position)
        if op == "HARVEST" and isinstance(tile, dict):
            units = float(tile.get("yield_units", 0))
            if units <= 0:
                return {}
            if tile.get("kind") == "PLANT":
                product = tile.get("crop")
            elif tile.get("animal"):
                spec = self.executor.ANIMALS.get(tile.get("animal"))
                product = spec[1] if spec else None
            else:
                product = None
            return {product: units} if product in PRODUCTS else {}
        if op == "COLLECT_FERTILIZER" and isinstance(tile, dict) and tile.get("fertilizer_available"):
            return {"FERTILIZER": 1.0}
        return {}

    def observe_step(self, obs_before, candidate_action, obs_after, shed_points):
        before_invs = list(obs_before["private"]["inventories"])
        after_invs = list(obs_after["private"]["inventories"])
        actions = self._worker_actions(candidate_action)
        before_pos = self._positions(obs_before)
        after_pos = self._positions(obs_after)
        shed = {tuple(point) for point in shed_points}
        prices = obs_before["market"]["prices"]
        day_rolled = int(obs_after["day"]) != int(obs_before["day"])
        count = max(len(before_invs), len(before_pos))
        self._ensure_workers(count)

        for i in range(count):
            binv = before_invs[i] if i < len(before_invs) else {}
            ainv = after_invs[i] if i < len(after_invs) else {}
            action = actions[i] if i < len(actions) else ["PASS"]
            op = action[0] if action else "PASS"
            bp = before_pos[i] if i < len(before_pos) else None
            ap = after_pos[i] if (not day_rolled and i < len(after_pos)) else None
            old_eligible = dict(self.eligible[i])
            produced = self._produced_from_action(obs_before, bp, action)
            if not day_rolled:
                confirmed = {}
                for product, units in produced.items():
                    gain = max(0.0, float(ainv.get(product, 0)) - float(binv.get(product, 0)))
                    if gain > 0:
                        confirmed[product] = min(float(units), gain)
                produced = confirmed

            for product, units in produced.items():
                value = float(units) * float(prices.get(product, 0.0))
                self.produced_units += float(units)
                self.produced_value += value
                self.produced_value_by_product[product] += value

            if op in MOVE_ACTIONS and bp is not None and ap is not None:
                progress = self._nearest_shed_distance(bp, shed) - self._nearest_shed_distance(ap, shed)
                if progress:
                    carried_value = sum(
                        float(qty) * float(prices.get(product, 0.0))
                        for product, qty in old_eligible.items()
                    )
                    self.transport_progress_value += carried_value * progress

            pool = dict(old_eligible)
            for product, units in produced.items():
                pool[product] = pool.get(product, 0.0) + float(units)
            if op == "FEED":
                pool["WHEAT"] = max(0.0, pool.get("WHEAT", 0.0) - 1.0)
            elif op == "FERTILIZE":
                pool["FERTILIZER"] = max(0.0, pool.get("FERTILIZER", 0.0) - 1.0)

            manual = {}
            at_shed = bp is not None and tuple(bp) in shed
            if at_shed and op == "DROP":
                for product in PRODUCTS:
                    removed = max(0.0, float(binv.get(product, 0)) - float(ainv.get(product, 0)))
                    if removed > 0:
                        manual[product] = removed
            elif at_shed and op == "PLACE" and len(action) >= 2 and action[1] in PRODUCTS:
                product = action[1]
                removed = max(0.0, float(binv.get(product, 0)) - float(ainv.get(product, 0)))
                if removed > 0:
                    manual[product] = removed

            delivered = {}
            for product, removed in manual.items():
                qty = min(float(removed), float(pool.get(product, 0.0)))
                if qty > 0:
                    delivered[product] = qty
                    pool[product] = max(0.0, pool.get(product, 0.0) - qty)
            if day_rolled:
                for product, qty in list(pool.items()):
                    if qty > 0:
                        delivered[product] = delivered.get(product, 0.0) + float(qty)
                pool = {}

            for product, units in delivered.items():
                value = float(units) * float(prices.get(product, 0.0))
                self.delivered_units += float(units)
                self.delivered_value += value
                self.delivered_value_by_product[product] += value

            next_eligible = {}
            if not day_rolled:
                for product, qty in pool.items():
                    qty = min(float(qty), float(ainv.get(product, 0)))
                    if qty > 0:
                        next_eligible[product] = qty
            self.eligible[i] = next_eligible

    def as_dict(self):
        return {
            "produced_value": float(self.produced_value),
            "transport_progress_value": float(self.transport_progress_value),
            "delivered_value": float(self.delivered_value),
            "produced_units": float(self.produced_units),
            "delivered_units": float(self.delivered_units),
            "produced_value_by_product": dict(self.produced_value_by_product),
            "delivered_value_by_product": dict(self.delivered_value_by_product),
        }


class SellingController:
    def __init__(self, path, model, device, episode_steps, args, *, deterministic, collect, forced_v20_baseline=False):
        self.policy = SellingPolicy(
            model, device, episode_steps,
            deterministic=deterministic,
            collect=collect,
            forced_v20_baseline=forced_v20_baseline,
            price_shaping=args.price_shaping,
            overflow_penalty=args.overflow_penalty,
        )
        self.executor = _load_v20_executor(path, self.policy)

    def __call__(self, obs):
        return self.executor.agent(obs)


def terminal_reward(margin, margin_bonus, margin_scale):
    outcome = 1.0 if margin > 0 else -1.0 if margin < 0 else 0.0
    return outcome + float(margin_bonus) * math.tanh(float(margin) / float(margin_scale))


def build_discounted_returns(steps, terminal, gamma, terminal_turn):
    if not steps:
        return np.empty((0,), dtype=np.float16)
    out = np.empty((len(steps),), dtype=np.float16)
    running = np.float16(terminal)
    next_turn = int(terminal_turn)
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        gap = max(0, next_turn - int(step.turn))
        discount = np.float16(float(gamma) ** gap)
        running = np.float16(step.shaping_reward) + np.float16(discount * running)
        out[index] = running
        next_turn = int(step.turn)
    return out


def run_static_episode(history_path, model, device, v20_submission, args, *, deterministic, collect, forced_v20_baseline=False, compare_to_recorded_candidate=False):
    history = _load_history(history_path)
    candidate_seat = _infer_v20_seat(history)
    opponent_seat = 1 - candidate_seat
    original_rewards = _saved_final_rewards(history)
    episode_steps = len(history["steps"]) - 1
    controller = SellingController(
        v20_submission, model, device, episode_steps, args,
        deterministic=deterministic, collect=collect,
        forced_v20_baseline=forced_v20_baseline,
    )
    env = _environment_from_history(history)
    telemetry = WorkerTelemetry(controller.executor)
    divergences = 0
    first_divergence = None

    try:
        for replay_step in range(1, len(history["steps"])):
            obs = _agent_observation(env, candidate_seat)
            candidate_action = controller(obs)
            recorded_actions = _recorded_step_actions(history, replay_step)
            opponent_action = recorded_actions[opponent_seat]
            recorded_candidate = recorded_actions[candidate_seat]
            if opponent_action is None:
                raise RuntimeError(f"recorded opponent action is None at replay step {replay_step}")
            if compare_to_recorded_candidate and candidate_action != recorded_candidate:
                divergences += 1
                if first_divergence is None:
                    first_divergence = {
                        "replay_step": replay_step,
                        "day": int(_field(obs, "day", -1)),
                        "hour": int(_field(obs, "hour", -1)),
                        "recorded_v20_action": recorded_candidate,
                        "generated_action": candidate_action,
                    }
            actions = [None, None]
            actions[candidate_seat] = candidate_action
            actions[opponent_seat] = opponent_action
            env.step(actions)
            post_obs = _agent_observation(env, candidate_seat)
            telemetry.observe_step(obs, candidate_action, post_obs, controller.executor.SHED)

        final_states = env.steps[-1]
        rewards = _final_rewards(final_states)
        statuses = [str(_field(s, "status", "")) for s in final_states]
        margin = rewards[candidate_seat] - rewards[opponent_seat]
        ok = statuses == ["DONE", "DONE"]
        tr = terminal_reward(margin, args.margin_bonus, args.margin_scale) if ok else None
        result = {
            "episode": history_path.stem,
            "seed": _seed_hint(history),
            "ok": ok,
            "v20_seat": candidate_seat,
            "original_rewards": original_rewards,
            "rewards": rewards,
            "margin": margin,
            "terminal_reward": tr,
            "result": "WIN" if margin > 0 else "LOSS" if margin < 0 else "TIE",
            "sell_decisions": controller.policy.decisions,
            "sale_orders": controller.policy.sale_orders,
            "quoted_units_sold": controller.policy.quoted_units_sold,
            "quoted_sale_value": controller.policy.quoted_sale_value,
            "expected_overflow_units": controller.policy.expected_overflow_units,
            "sold_units_by_product": dict(controller.policy.sold_units_by_product),
            "quoted_value_by_product": dict(controller.policy.quoted_value_by_product),
            "action_divergences": divergences,
            "first_action_divergence": first_divergence,
            "error": "" if ok else f"non-DONE status: {statuses}",
            **telemetry.as_dict(),
        }
        returns = (
            build_discounted_returns(controller.policy.steps, tr, args.gamma, episode_steps - 1)
            if ok and collect and tr is not None else np.empty((0,), dtype=np.float16)
        )
        return result, controller.policy.steps, returns
    except Exception as exc:
        return {
            "episode": history_path.stem,
            "seed": _seed_hint(history),
            "ok": False,
            "v20_seat": candidate_seat,
            "original_rewards": original_rewards,
            "rewards": None,
            "margin": None,
            "terminal_reward": None,
            "result": "ERROR",
            "sell_decisions": controller.policy.decisions,
            "sale_orders": controller.policy.sale_orders,
            "quoted_units_sold": controller.policy.quoted_units_sold,
            "quoted_sale_value": controller.policy.quoted_sale_value,
            "expected_overflow_units": controller.policy.expected_overflow_units,
            "sold_units_by_product": dict(controller.policy.sold_units_by_product),
            "quoted_value_by_product": dict(controller.policy.quoted_value_by_product),
            "action_divergences": divergences,
            "first_action_divergence": first_divergence,
            "error": f"{type(exc).__name__}: {exc}",
            **telemetry.as_dict(),
        }, [], np.empty((0,), dtype=np.float16)


def ppo_update(model, optimizer, device, steps, returns, args):
    if not steps:
        return {"ppo_updates": 0, "policy_loss": None, "value_loss": None, "entropy": None, "loss": None, "return_mean": None}
    old_values = torch.as_tensor(
        np.asarray([s.old_value for s in steps], dtype=np.float16),
        dtype=torch.float16, device=device,
    )
    returns_t = torch.as_tensor(returns, dtype=torch.float16, device=device)
    advantages = returns_t - old_values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + torch.tensor(1e-3, dtype=torch.float16, device=device)
        )

    stats = []
    for _ in range(args.ppo_epochs):
        order = np.random.permutation(len(steps))
        for start in range(0, len(order), args.minibatch_size):
            ids = order[start:start + args.minibatch_size]
            batch_steps = [steps[int(i)] for i in ids]
            states = torch.as_tensor(np.stack([s.state for s in batch_steps]), dtype=torch.float16, device=device)
            masks = torch.as_tensor(np.stack([s.mask for s in batch_steps]), dtype=torch.bool, device=device)
            actions = torch.as_tensor(np.stack([s.actions for s in batch_steps]), dtype=torch.long, device=device)
            old_log_prob = torch.as_tensor(
                np.asarray([s.old_log_prob for s in batch_steps], dtype=np.float16),
                dtype=torch.float16, device=device,
            )
            idx = torch.as_tensor(ids, dtype=torch.long, device=device)
            adv = advantages.index_select(0, idx)
            targets = returns_t.index_select(0, idx)

            logits, values = model(states)
            logits = model.masked_logits(logits, masks)
            dist = Categorical(logits=logits)
            new_log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            ratio = torch.exp(new_log_prob - old_log_prob)
            clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
            policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
            value_loss = torch.tensor(0.5, dtype=torch.float16, device=device) * ((values - targets) ** 2).mean()
            loss = (
                policy_loss
                + np.float16(args.value_coef) * value_loss
                - np.float16(args.entropy_coef) * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            stats.append((
                float(policy_loss.detach().float().item()),
                float(value_loss.detach().float().item()),
                float(entropy.detach().float().item()),
                float(loss.detach().float().item()),
            ))
    arr = np.asarray(stats, dtype=np.float64)
    return {
        "ppo_updates": len(stats),
        "policy_loss": float(arr[:, 0].mean()),
        "value_loss": float(arr[:, 1].mean()),
        "entropy": float(arr[:, 2].mean()),
        "loss": float(arr[:, 3].mean()),
        "return_mean": float(np.asarray(returns, dtype=np.float32).mean()),
    }


def _save_checkpoint(path, model, optimizer, update, args):
    torch.save({
        "algorithm": CHECKPOINT_ALGORITHM,
        "update": int(update),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "precision": "fp16",
        "state_dim": STATE_DIM,
        "hidden": args.hidden,
        "products": PRODUCTS,
        "action_fractions": ACTION_FRACTIONS,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "product_feature_names": PRODUCT_FEATURE_NAMES,
        "parent_v20_submission": str(args.v20_submission),
        "parent_v20_submission_sha256": args.parent_v20_submission_sha256,
        "args": vars(args),
    }, path)


def _load_checkpoint(path, model, optimizer, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("algorithm") != CHECKPOINT_ALGORITHM:
        raise ValueError(f"checkpoint algorithm mismatch: {payload.get('algorithm')!r}")
    if payload.get("precision") != "fp16":
        raise ValueError("v22 requires fp16 checkpoint")
    model.load_state_dict(payload["model_state_dict"])
    model.half()
    if optimizer is not None and payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("update", -1)) + 1, payload


def preflight(path, model, device, v20_submission, args):
    history = _load_history(path)
    original = _saved_final_rewards(history)
    control = recorded_action_parity(history)
    if not control.get("exact"):
        raise RuntimeError(f"recorded replay parity failed: {control}")
    v20_seat = _infer_v20_seat(history)
    direct = _run_submission_replacement(
        history=history,
        candidate_seat=v20_seat,
        notebook=v20_submission,
        label="v20",
        compare_to_recorded_candidate=True,
    )
    direct_ok = (
        direct["statuses"] == ["DONE", "DONE"]
        and direct["rewards"] == original
        and direct["action_divergences"] == 0
    )
    if not direct_ok:
        raise RuntimeError("checked-in v20 submission does not reproduce selected v20 history")

    replaced, _, _ = run_static_episode(
        path, model, device, v20_submission, args,
        deterministic=True, collect=False,
        forced_v20_baseline=True,
        compare_to_recorded_candidate=True,
    )
    replacement_ok = (
        replaced["ok"]
        and replaced["rewards"] == original
        and replaced["action_divergences"] == 0
    )
    if not replacement_ok:
        raise RuntimeError(
            "SELL-loop replacement is not behavior-preserving in forced baseline mode: "
            f"{replaced.get('first_action_divergence')}"
        )
    return {
        "episode": path.stem,
        "recorded_action_parity": True,
        "v20_submission_parity": True,
        "sell_loop_replacement_parity": True,
        "original_rewards": original,
    }


def evaluate_histories(paths, model, device, v20_submission, args, phase):
    model.eval()
    rows = []
    wins = ties = losses = errors = 0
    margins, produced, delivered, sale_values, overflows = [], [], [], [], []
    with torch.inference_mode():
        for path in paths:
            result, _, _ = run_static_episode(
                path, model, device, v20_submission, args,
                deterministic=True, collect=False,
            )
            rows.append(result)
            if not result["ok"]:
                errors += 1
                continue
            margins.append(float(result["margin"]))
            produced.append(float(result["produced_value"]))
            delivered.append(float(result["delivered_value"]))
            sale_values.append(float(result["quoted_sale_value"]))
            overflows.append(float(result["expected_overflow_units"]))
            if result["margin"] > 0:
                wins += 1
            elif result["margin"] < 0:
                losses += 1
            else:
                ties += 1
    valid = wins + ties + losses
    summary = {
        "phase": phase,
        "games_total": len(paths),
        "games_valid": valid,
        "errors": errors,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / valid if valid else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "mean_produced_value": float(np.mean(produced)) if produced else None,
        "mean_delivered_value": float(np.mean(delivered)) if delivered else None,
        "mean_quoted_sale_value": float(np.mean(sale_values)) if sale_values else None,
        "mean_expected_overflow_units": float(np.mean(overflows)) if overflows else None,
        "rows": rows,
    }
    print(
        f"[{phase}] W/T/L/E={wins}/{ties}/{losses}/{errors} "
        f"win_rate={summary['win_rate']:.3f} margin={summary['mean_margin']} "
        f"produced={summary['mean_produced_value']} delivered={summary['mean_delivered_value']} "
        f"sale_value={summary['mean_quoted_sale_value']} overflow={summary['mean_expected_overflow_units']}",
        flush=True,
    )
    model.train()
    return summary


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    p.add_argument("--v20-submission", type=Path, default=DEFAULT_V20_SUBMISSION)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument("--split-seed", type=int, default=20260922)
    p.add_argument("--training-seed", type=int, default=32222)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--validate-every-updates", type=int, default=1)
    p.add_argument("--checkpoint-every-updates", type=int, default=1)
    p.add_argument("--target-win-rate", type=float, default=0.60)
    p.add_argument("--max-training-hours", type=float, default=2.0)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--margin-bonus", type=float, default=0.25)
    p.add_argument("--margin-scale", type=float, default=10000.0)
    p.add_argument("--price-shaping", type=float, default=0.0005)
    p.add_argument("--overflow-penalty", type=float, default=0.01)
    return p


def main():
    args = build_parser().parse_args()
    args.history_dir = args.history_dir.expanduser().resolve()
    args.v20_submission = args.v20_submission.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.history_dir.is_dir():
        raise SystemExit(f"history directory not found: {args.history_dir}")
    if not args.v20_submission.is_file():
        raise SystemExit(f"v20 submission not found: {args.v20_submission}")

    paths = _history_paths(args.history_dir)
    train_paths, validation_paths = _split_histories(paths, args.validation_fraction, args.split_seed)
    device = choose_device(args.device)
    print(f"device={device} precision=fp16", flush=True)
    print(f"v20 histories={len(paths)} train/validation={len(train_paths)}/{len(validation_paths)}", flush=True)

    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    sample_rng = random.Random(args.training_seed)

    model = SellActorCritic(STATE_DIM, args.hidden).to(device).half()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-4)
    args.parent_v20_submission_sha256 = _sha256(args.v20_submission)

    pf = preflight(paths[0], model, device, args.v20_submission, args)
    print(f"[preflight] {pf['episode']}: recorded=OK v20=OK sell-replacement=OK", flush=True)
    if args.preflight_only:
        print(json.dumps(pf, indent=2))
        return 0

    out = args.output_dir
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (out / "history_split.json").write_text(json.dumps({
        "history_dir": str(args.history_dir),
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "training": [p.name for p in train_paths],
        "validation": [p.name for p in validation_paths],
    }, indent=2) + "\n", encoding="utf-8")
    (out / "preflight.json").write_text(json.dumps(pf, indent=2) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps({
        **vars(args),
        "algorithm": CHECKPOINT_ALGORITHM,
        "precision": "fp16 parameters/activations/logits/values/returns/advantages/losses/gradients",
        "optimizer": "Adam(fp16 parameter states; eps=1e-4)",
        "control_scope": "SELL orders only; all other v20 policy logic frozen",
        "training_protocol": "static recorded-opponent replay from game_history/v20",
        "primary_reward": "terminal win/loss plus bounded terminal margin bonus",
        "secondary_reward": "tiny sale-price percentile shaping and overflow-risk penalty",
        "worker_telemetry": "v21-equivalent production/transport/delivery metrics; diagnostics only, never PPO reward",
        "reserve_rule": "WHEAT max(4, live+2), FERTILIZER 4 from day 10; both released only for final 5 turns",
        "capacity_rule": 100,
        "state_dim": STATE_DIM,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "product_feature_names": PRODUCT_FEATURE_NAMES,
        "products": PRODUCTS,
        "action_fractions": ACTION_FRACTIONS,
    }, indent=2, default=str) + "\n", encoding="utf-8")

    start_update = 0
    if args.resume is not None:
        resume = args.resume.expanduser().resolve()
        if not resume.is_file():
            raise SystemExit(f"resume checkpoint not found: {resume}")
        start_update, _ = _load_checkpoint(resume, model, optimizer, device)

    started = time.monotonic()
    initial_validation = evaluate_histories(
        validation_paths, model, device, args.v20_submission, args, "initial_validation"
    )
    write_jsonl(
        out / "validation.jsonl",
        {"update": start_update - 1, **{k: v for k, v in initial_validation.items() if k != "rows"}},
    )

    for update in range(start_update, start_update + args.updates):
        if (time.monotonic() - started) / 3600.0 >= args.max_training_hours:
            _save_checkpoint(checkpoints / "timeout.pt", model, optimizer, update - 1, args)
            return 0

        episode_rows = []
        all_steps = []
        all_returns = []
        attempts = 0
        while len(episode_rows) < args.episodes_per_update:
            attempts += 1
            if attempts > args.episodes_per_update * 3:
                raise RuntimeError("too many failed static training episodes")
            path = sample_rng.choice(train_paths)
            result, steps, returns = run_static_episode(
                path, model, device, args.v20_submission, args,
                deterministic=False, collect=True,
            )
            write_jsonl(out / "episodes.jsonl", {"update": update, **result})
            if not result["ok"] or not steps:
                continue
            episode_rows.append(result)
            all_steps.extend(steps)
            all_returns.append(returns)

        returns = np.concatenate(all_returns).astype(np.float16, copy=False)
        ppo_stats = ppo_update(model, optimizer, device, all_steps, returns, args)
        margins = np.asarray([row["margin"] for row in episode_rows], dtype=np.float64)
        metrics = {
            "update": update,
            "episodes": len(episode_rows),
            "sell_decisions": len(all_steps),
            "wins": int((margins > 0).sum()),
            "ties": int((margins == 0).sum()),
            "losses": int((margins < 0).sum()),
            "training_win_rate": float((margins > 0).mean()),
            "mean_margin": float(margins.mean()),
            "mean_produced_value": float(np.mean([r["produced_value"] for r in episode_rows])),
            "mean_delivered_value": float(np.mean([r["delivered_value"] for r in episode_rows])),
            "mean_transport_progress_value": float(np.mean([r["transport_progress_value"] for r in episode_rows])),
            "mean_quoted_sale_value": float(np.mean([r["quoted_sale_value"] for r in episode_rows])),
            "mean_expected_overflow_units": float(np.mean([r["expected_overflow_units"] for r in episode_rows])),
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
            **ppo_stats,
        }
        write_jsonl(out / "metrics.jsonl", metrics)
        print(
            f"[update {update:04d}] W/T/L={metrics['wins']}/{metrics['ties']}/{metrics['losses']} "
            f"win_rate={metrics['training_win_rate']:.3f} margin={metrics['mean_margin']:.1f} "
            f"decisions={metrics['sell_decisions']} produced={metrics['mean_produced_value']:.1f} "
            f"delivered={metrics['mean_delivered_value']:.1f} sale_value={metrics['mean_quoted_sale_value']:.1f} "
            f"loss={metrics['loss']}", flush=True,
        )

        _save_checkpoint(checkpoints / "latest.pt", model, optimizer, update, args)
        if args.checkpoint_every_updates > 0 and (update + 1) % args.checkpoint_every_updates == 0:
            _save_checkpoint(checkpoints / f"update_{update:04d}.pt", model, optimizer, update, args)

        if args.validate_every_updates > 0 and (update + 1) % args.validate_every_updates == 0:
            validation = evaluate_histories(
                validation_paths, model, device, args.v20_submission, args,
                f"validation_{update:04d}",
            )
            write_jsonl(
                out / "validation.jsonl",
                {"update": update, **{k: v for k, v in validation.items() if k != "rows"}},
            )
            (out / "validation_latest.json").write_text(
                json.dumps(validation, indent=2, default=str) + "\n", encoding="utf-8"
            )
            if validation["games_valid"] > 0 and validation["win_rate"] >= args.target_win_rate:
                _save_checkpoint(checkpoints / "target.pt", model, optimizer, update, args)
                (out / "TARGET_REACHED.json").write_text(
                    json.dumps(validation, indent=2, default=str) + "\n", encoding="utf-8"
                )
                return 0

        if (time.monotonic() - started) / 3600.0 >= args.max_training_hours:
            _save_checkpoint(checkpoints / "timeout.pt", model, optimizer, update, args)
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
