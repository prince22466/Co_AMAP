#!/usr/bin/env python3
"""Repeatable, engine-verified diagnostics for the production-signals agent family.

Example:
 python local_arena/static_reply/diagnose.py --histories path/a.json path/b.json \
   --agent local_arena/static_reply/v20_3.py --output local_arena/static_reply/diagnostics/run

Only diagnostic wrappers are installed; the agent source is never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from reply_template import load_agent, load_engine, make_environment, state_core


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v), separators=(",", ":"))
                             if isinstance(v, (dict, list, tuple)) else clean(v)
                             for k, v in row.items()})


def owned(private):
    result = Counter(private["shed"])
    for inv in private["inventories"]:
        result.update(inv)
    return result


def assets(farm):
    return {(x, y): t for y, row in enumerate(farm["tiles"]) for x, t in enumerate(row)
            if isinstance(t, dict) and ("animal" in t or "crop" in t)}


def identity(tile):
    if not isinstance(tile, dict):
        return None
    return (tile.get("animal", tile.get("crop")), tile.get("placed_day", tile.get("planted_day")))


def quantity(orders, op, item):
    return sum(o[2] for o in orders if len(o) >= 3 and o[:2] == [op, item])


class EngineTrace:
    """Observe actual engine commits, worker effects, and the pre-market inventory.

    All wrappers call original functions exactly once and are restored in finally.
    Full state parity verifies that instrumentation did not change recorded play.
    """
    def __init__(self, env, agent_globals):
        self.g = env.interpreter.__globals__
        self.a = agent_globals
        self.original = {}
        self.step = 0
        self.ledger = []
        self.work = []
        self.losses = []
        self.sales_audit = []
        self.enddays = []
        self.farm_ids = {}
        self.private_ids = {}
        self.obs = None
        self.market = None

    def install(self):
        for name, handler in (("_apply_unit_action", self.unit), ("_process_market", self.market_phase),
                              ("_commit_unit", self.commit), ("_do_hire", self.hire),
                              ("_do_buy_land", self.land), ("_end_of_day", self.endday)):
            self.original[name] = self.g[name]
            self.g[name] = handler

    def restore(self):
        self.g.update(self.original)

    def unit(self, farm, private, idx, action, *args, **kwargs):
        # Engine copies state before stepping, so map seats by private.player traversal.
        if id(farm) not in self.farm_ids:
            self.farm_ids[id(farm)] = len(self.farm_ids)
        seat = self.farm_ids[id(farm)]
        self.private_ids[id(private)] = seat
        pos = self.g["_farmer_position"](farm, idx)
        before_owned = owned(private)
        before_tile = copy.deepcopy(farm["tiles"][pos[1]][pos[0]]) if pos else None
        before_inv = dict(private["inventories"][idx]) if idx < len(private["inventories"]) else {}
        self.original["_apply_unit_action"](farm, private, idx, action, *args, **kwargs)
        if not pos:
            return
        after_tile = farm["tiles"][pos[1]][pos[0]]
        after_inv = private["inventories"][idx] if idx < len(private["inventories"]) else {}
        op = action[0] if action else "PASS"
        item = (before_tile or {}).get("crop", (before_tile or {}).get("animal")) if isinstance(before_tile, dict) else None
        changes = {k: after_inv.get(k, 0) - before_inv.get(k, 0) for k in set(before_inv) | set(after_inv)
                   if after_inv.get(k, 0) != before_inv.get(k, 0)}
        if op in ("HARVEST", "FEED", "FERTILIZE", "PLANT", "PLACE", "DIG", "WATER", "CARE", "COLLECT_FERTILIZER", "DROP"):
            self.work.append(dict(step=self.step, seat=seat, worker=idx, x=pos[0], y=pos[1], op=op,
                                  producer=item, action=action, inventory_delta=changes,
                                  tile_changed=before_tile != after_tile, before=before_tile, after=copy.deepcopy(after_tile)))
        if op == "DROP":
            lost = before_owned - owned(private)
            for item, n in lost.items():
                self.losses.append(dict(step=self.step, seat=seat, kind="drop_overflow", item=item, units=n))

    def market_phase(self, state, env):
        self.market = state[0].observation.market
        for seat, entry in enumerate(state):
            farm = state[0].observation.farms[seat]
            self.farm_ids[id(farm)] = seat
            private = entry.observation.private
            obs = copy.deepcopy(self.obs[seat])
            actions = entry.action if isinstance(entry.action, dict) else {}
            actual_workers = [actions.get("farmer", ["PASS"]), *actions.get("hands", [])]
            predicted, predicted_total = self.a["post_action_market_inventory"](obs, actual_workers)
            total = owned(private)
            live = sum("animal" in t for t in assets(farm).values())
            reserve_wheat = max(live, 4 if live < 4 else 8 if live < 8 else 14)
            day = self.step // 24
            reserve_fert = 0 if day < 10 or day == 29 else 4
            for product in self.a["SELLABLE_PRODUCTS"]:
                available = private["shed"].get(product, 0)
                reserve = reserve_wheat if product == "WHEAT" else reserve_fert if product == "FERTILIZER" else 0
                eligible = min(available, max(0, total.get(product, 0) - reserve))
                requested = quantity(actions.get("market", [])[:10], "SELL", product)
                if available or eligible or requested or predicted.get(product, 0) != available:
                    self.sales_audit.append(dict(step=self.step, seat=seat, product=product,
                        available=available, total=total.get(product, 0), reserve=reserve,
                        eligible=eligible, requested=requested, unsent=max(0, eligible-requested),
                        predicted=predicted.get(product, 0), prediction_error=predicted.get(product, 0)-available,
                        carried=total.get(product, 0)-available))
        return self.original["_process_market"](state, env)

    def commit(self, op, item, price, farm, private, market, *args, **kwargs):
        ok = self.original["_commit_unit"](op, item, price, farm, private, market, *args, **kwargs)
        if ok:
            self.ledger.append(dict(step=self.step, seat=self.farm_ids[id(farm)], op=op, item=item,
                                    units=1, cash=price if op == "SELL" else -price, price=price))
        return ok

    def other_cash(self, name, farm, *args, **kwargs):
        before = farm["money"]
        result = self.original[name](farm, *args, **kwargs)
        if farm["money"] != before:
            self.ledger.append(dict(step=self.step, seat=self.farm_ids[id(farm)], op="HIRE" if name == "_do_hire" else "BUY_LAND",
                                    item="", units=1, cash=farm["money"]-before, price=before-farm["money"]))
        return result

    def hire(self, farm, *args, **kwargs):
        return self.other_cash("_do_hire", farm, *args, **kwargs)

    def land(self, farm, *args, **kwargs):
        return self.other_cash("_do_buy_land", farm, *args, **kwargs)

    def endday(self, state, env, day):
        before_assets = [copy.deepcopy(assets(f)) for f in state[0].observation.farms]
        before_owned = [owned(s.observation.private) for s in state]
        self.original["_end_of_day"](state, env, day)
        for seat, farm in enumerate(state[0].observation.farms):
            after_assets = assets(farm)
            total = owned(state[seat].observation.private)
            for item, n in (before_owned[seat] - total).items():
                self.losses.append(dict(step=self.step, seat=seat, kind="endday_overflow", item=item, units=n))
            for pos, tile in before_assets[seat].items():
                if identity(tile) != identity(after_assets.get(pos)):
                    kind = "animal_escape" if "animal" in tile else "crop_died"
                    self.losses.append(dict(step=self.step, seat=seat, kind=kind,
                        item=tile.get("animal", tile.get("crop")), units=1, x=pos[0], y=pos[1], tile=tile))
            herd = [t for t in before_assets[seat].values() if "animal" in t]
            self.enddays.append(dict(step=self.step, seat=seat, day=day, live=len(herd),
                unfed=sum(not t["fed_today"] for t in herd), wheat=before_owned[seat].get("WHEAT", 0),
                seeds=dict(state[seat].observation.private["seeds"]),
                assets=dict(Counter(t.get("animal", t.get("crop")) for t in after_assets.values()))))


def gate_rows(g, obs, signals, action):
    """Explain exact procurement preconditions; earlier orders use agent's cash estimate."""
    farm = obs["farms"][obs["player"]]
    private = obs["private"]
    stock = owned(private)
    crop_slots = max(0, sum(t is None and (x, y) not in g["ANIMAL_POINTS"]
        for y, row in enumerate(farm["tiles"]) for x, t in enumerate(row)) - sum(private["seeds"].values()))
    animal_slots = max(0, min(g["HERD_LIMIT"], sum(g["tile"](farm, p) != "LOCKED" for p in g["ANIMAL_POINTS"]))
        - sum("animal" in t for t in assets(farm).values()) - sum(stock[a] for a in g["ANIMALS"]))
    cash = farm["money"]
    orders = action.get("market", [])
    prior_slots = 0
    hires = farm["hires_today"]
    fib = [1, 1]
    for _ in range(20):
        fib.append(sum(fib[-2:]))
    for order in orders:
        if order[0] in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"):
            continue
        prior_slots += 1
        if order[0] == "SELL":
            cash += order[2] * max(1, obs["market"]["prices"][order[1]] * .8)
        elif order[0] == "HIRE":
            cash -= fib[hires]
            hires += 1
        elif order[0] == "BUY_LAND":
            cash -= {1: 1000, 2: 2000, 3: 4000}[len(farm["unlocked_quadrants"])]
    rows = []
    for rank, signal in enumerate(signals):
        producer = signal["producer"]
        is_crop = producer in g["CROPS"]
        cost = g["CROPS"][producer][0] if is_crop else g["ANIMALS"][producer][0]
        space = crop_slots if is_crop else animal_slots
        gap = signal["producer_equivalent_gap"]
        needed = math.floor(gap) if math.isfinite(gap) else 0
        affordable = max(0, math.floor((cash-g["PROCUREMENT_CASH_RESERVE"])/cost))
        reasons = []
        if obs["day"] == 0: reasons.append("fixed_opening")
        if not signal["actionable"]: reasons.append("cannot_mature")
        if obs["day"] > g["LATEST_BUY_DAY"][producer]: reasons.append("buy_deadline")
        if math.isfinite(gap) and needed < 1: reasons.append("gap_below_one")
        if space < 1: reasons.append("no_deployment_space")
        if affordable < 1: reasons.append("cash_reserve")
        if prior_slots >= 10: reasons.append("order_limit")
        purchased = quantity(orders, "BUY_SEED" if is_crop else "BUY_ANIMAL", producer)
        rows.append(dict(rank=rank, producer=producer, product=signal["product"], score=signal["score"],
            demand=signal["hard_demand"], capacity=signal["capacity"], gap=gap, needed=needed,
            space=space, affordable=affordable, bought=purchased, reasons=reasons))
        if purchased:
            cash -= purchased*cost
            prior_slots += 1
            if is_crop: crop_slots -= purchased
            else: animal_slots -= purchased
    return rows


def event_probes(g, obs, previous, action, signals):
    if previous is None:
        return []
    events = []
    old_shops = Counter(previous["town"]["unlocked_shops"])
    new_shops = Counter(obs["town"]["unlocked_shops"]) - old_shops
    for shop, count in new_shops.items():
        for instance in range(count):
            altered = copy.deepcopy(obs)
            altered["town"]["unlocked_shops"].remove(shop)
            events.append(("shop_open", shop, list(g["SHOPS"][shop]), altered, None))
    opponent = 1-obs["player"]
    old = assets(previous["farms"][opponent])
    for pos, tile in assets(obs["farms"][opponent]).items():
        if identity(tile) == identity(old.get(pos)):
            continue
        producer = tile.get("animal", tile.get("crop"))
        product = g["ANIMALS"][producer][1] if producer in g["ANIMALS"] else producer
        altered = copy.deepcopy(obs)
        altered["farms"][opponent]["tiles"][pos[1]][pos[0]] = None
        events.append(("opponent_capacity", producer, [product, "WHEAT"] if "animal" in tile else [product], altered, pos))
    output = []
    actual = {s["product"]: s for s in signals}
    workers = [action["farmer"], *action["hands"]]
    for kind, name, products, altered, pos in events:
        alt_signals = g["production_signals"](altered)
        alt = {s["product"]: s for s in alt_signals}
        alt_orders = g["market_orders"](altered, alt_signals, workers)
        output.append(dict(step=obs["day"]*24+obs["hour"], kind=kind, name=name, position=pos,
            products=products, actual_orders=action["market"], without_event_orders=alt_orders,
            changes=[dict(product=p, demand_delta=actual[p]["hard_demand"]-alt[p]["hard_demand"],
                capacity_delta=actual[p]["capacity"]-alt[p]["capacity"],
                gap_before=alt[p]["producer_equivalent_gap"], gap_after=actual[p]["producer_equivalent_gap"],
                score_before=alt[p]["score"], score_after=actual[p]["score"],
                buy_before=quantity(alt_orders, "BUY_SEED" if p in g["CROPS"] else "BUY_ANIMAL", g["PRODUCT_PRODUCER"][p]),
                buy_after=quantity(action["market"], "BUY_SEED" if p in g["CROPS"] else "BUY_ANIMAL", g["PRODUCT_PRODUCER"][p]))
                for p in dict.fromkeys(products)], gates=gate_rows(g, obs, signals, action)))
    return output


def install_variant(g, name):
    """Bounded diagnostic interventions; never edit the production agent."""
    original_market = g["market_orders"]
    original_units = g["unit_actions"]
    original_signals = g["production_signals"]
    if name in ("ignore_opponent_capacity", "no_repeat_capacity"):
        if name == "no_repeat_capacity":
            def capacity(t, day, step):
                return float(sum(n for age, n in g["CROPS"][t["crop"]][2]
                    if t["planted_day"]+age < 30 and (t["planted_day"]+age)*24 > step))
            g["_current_crop_remaining_capacity"] = capacity
        else:
            def signals(obs):
                changed = copy.deepcopy(obs)
                changed["farms"][1-obs["player"]]["tiles"] = [[None for _ in row] for row in obs["farms"][1-obs["player"]]["tiles"]]
                result = original_signals(changed)
                # Preserve animal feed demand; intervene on capacity only.
                demands = g["_remaining_hard_demand"](obs)
                for s in result:
                    s["hard_demand"] = demands[s["product"]]
                    s["gap"] = max(0, s["hard_demand"]-s["capacity"])
                    s["producer_equivalent_gap"] = s["gap"]/s["new_producer_capacity"] if s["new_producer_capacity"] else (math.inf if s["gap"] else 0)
                    s["score"] = 1-math.exp(-s["producer_equivalent_gap"])
                return sorted(result, key=lambda s: (-s["score"], s["product"]))
            g["production_signals"] = signals
    if name in ("feed_buffer", "fert_no_churn", "feed_and_fert"):
        def market(obs, signals, actions):
            orders = original_market(obs, signals, actions)
            if name in ("fert_no_churn", "feed_and_fert"):
                # Diagnostic: no fertilizer purchase; animal collection still available.
                orders = [o for o in orders if o[:2] != ["BUY_PRODUCT", "FERTILIZER"]]
            if name in ("feed_buffer", "feed_and_fert") and 0 < obs["day"] < 29:
                live = sum("animal" in t for t in assets(obs["farms"][obs["player"]]).values())
                _, total = g["post_action_market_inventory"](obs, actions)
                need = max(0, 2*live-total.get("WHEAT", 0))
                price = obs["market"]["prices"]["WHEAT"]
                n = min(need, int(max(0, obs["farms"][obs["player"]]["money"]-100)//(price+3)))
                if n:
                    # Prioritize short-term feed over expansion; selling retained first.
                    at = next((i for i, o in enumerate(orders) if o[0] != "SELL"), len(orders))
                    orders.insert(at, ["BUY_PRODUCT", "WHEAT", n])
            return orders[:10]
        g["market_orders"] = market
    if name == "deliver_products":
        def units(obs, animal, crops):
            actions = original_units(obs, animal, crops)
            farm = obs["farms"][obs["player"]]
            positions = [farm["farmer"], *farm["hands"]]
            for i, inv in enumerate(obs["private"]["inventories"]):
                if any(inv.get(a, 0) for a in g["ANIMALS"]): continue
                goods = [c for c in g["SELLABLE_PRODUCTS"] if c not in ("WHEAT", "FERTILIZER") and inv.get(c, 0)]
                if not goods: continue
                p = tuple(positions[i])
                actions[i] = ["PLACE", max(goods, key=lambda c: inv[c]*obs["market"]["prices"][c]), inv[max(goods, key=lambda c: inv[c]*obs["market"]["prices"][c])]] if p in g["SHED"] else g["move"](p, g["nearest_shed"](p))
            return actions
        g["unit_actions"] = units


def run_case(engine, history_path, agent_path, output, seat_override=None, variant="baseline"):
    history = json.loads(history_path.read_text(encoding="utf-8"))
    config = history.get("configuration", {})
    expected_config = {"turnsPerDay": 24, "episodeSteps": 720, "boardSize": 10,
                       "shedCapacity": 100, "maxMarketOrdersPerTurn": 10,
                       "townShopSellInterval": 4, "townCenterSellInterval": 24,
                       "farmHandCostMult": 1}
    for key, expected in expected_config.items():
        if config.get(key, expected) != expected:
            raise ValueError(f"v20 diagnostic adapter requires {key}={expected}; got {config[key]}")
    if len(history.get("steps", [])) != 720 or any(len(row) != 2 for row in history["steps"]):
        raise ValueError("expected a complete 720-state, two-player default-season replay")
    metadata = history.get("info", {}).get("static_replay", {})
    seat = seat_override if seat_override is not None else metadata.get("candidate_seat")
    if seat not in (0, 1):
        raise ValueError("candidate seat missing; specify --seat; never infer from a loss")
    expected_hash = metadata.get("agent_sha256")
    if variant == "baseline" and expected_hash and expected_hash != digest(agent_path):
        raise ValueError("agent hash differs from recording; use the exact archived agent")
    env = make_environment(engine, history)
    if state_core(env.state) != state_core(history["steps"][0]):
        raise ValueError("initial engine state differs from source")
    getter = getattr(env, "_Environment__get_shared_state")
    turns, events = [], []
    previous = None
    with load_agent(agent_path) as (agent, with_config):
        if with_config:
            raise ValueError("diagnostic adapter currently supports agent(obs)")
        g = agent.__globals__
        if variant != "baseline": install_variant(g, variant)
        trace = EngineTrace(env, g)
        trace.install()
        try:
            for index in range(1, len(history["steps"])):
                if index % 240 == 0:
                    print(f"  decision {index-1}/718", flush=True)
                obs = copy.deepcopy(getter(seat).observation)
                trace.step = obs["day"]*24+obs["hour"]
                trace.farm_ids = {}
                trace.private_ids = {}
                trace.obs = [copy.deepcopy(getter(p).observation) for p in range(2)]
                action = agent(obs)
                recorded = history["steps"][index][seat]["action"]
                if variant == "baseline" and action != recorded:
                    raise ValueError(f"agent-action mismatch at decision {trace.step} / history row {index}")
                if variant == "baseline":
                    signals = g["production_signals"](obs)
                    animal = g["animal_plan"](obs, signals)
                    crops = g["crop_plan"](obs, signals)
                    turns.append(dict(step=trace.step, history_action_index=index, day=obs["day"], hour=obs["hour"],
                        money=obs["farms"][seat]["money"], signals=signals,
                        animal_plan={str(k): v for k, v in animal.items()}, crop_plan={str(k): v for k, v in crops.items()},
                        seeds=obs["private"]["seeds"], owned=owned(obs["private"]),
                        own_assets=dict(Counter(t.get("animal", t.get("crop")) for t in assets(obs["farms"][seat]).values())),
                        opponent_assets=dict(Counter(t.get("animal", t.get("crop")) for t in assets(obs["farms"][1-seat]).values())),
                        prices=obs["market"]["prices"], action=action, gates=gate_rows(g, obs, signals, action)))
                    events.extend(event_probes(g, obs, previous, action, signals))
                actions = [copy.deepcopy(s["action"]) for s in history["steps"][index]]
                actions[seat] = action
                env.step(actions)
                if variant == "baseline" and state_core(env.state) != state_core(history["steps"][index]):
                    raise ValueError(f"engine parity mismatch at history row {index}")
                previous = obs
        finally:
            trace.restore()
    rewards = [float(s.reward) for s in env.state]
    cash_rows = []
    for player in range(2):
        grouped = defaultdict(lambda: [0, 0])
        for row in trace.ledger:
            if row["seat"] == player:
                grouped[row["op"], row["item"]][0] += row["units"]
                grouped[row["op"], row["item"]][1] += row["cash"]
        for (op, item), (units, cash) in grouped.items():
            cash_rows.append(dict(seat=player, op=op, item=item, units=units, cash=cash))
        start = history["steps"][0][0]["observation"]["farms"][player]["money"]
        assert start + sum(r["cash"] for r in cash_rows if r["seat"] == player) == rewards[player], "cash ledger fails reconciliation"
    audit = [r for r in trace.sales_audit if r["seat"] == seat]
    losses = [r for r in trace.losses if r["seat"] == seat]
    summary = dict(episode=history_path.stem, variant=variant, seat=seat, rewards=rewards,
        candidate_reward=rewards[seat], margin=rewards[seat]-rewards[1-seat], decisions=len(history["steps"])-1,
        action_parity="exact" if variant == "baseline" else "intervention",
        state_parity="exact" if variant == "baseline" else "not_applicable",
        cash_reconciled=True, sell_unsent_units=sum(r["unsent"] for r in audit),
        sale_prediction_errors=sum(r["prediction_error"] != 0 for r in audit),
        losses=dict(Counter(r["kind"] for r in losses)),
        terminal_private=copy.deepcopy(getter(seat).observation.private), cash=cash_rows)
    destination = output / history_path.stem / variant
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "summary.json", summary)
    write_csv(destination / "cash.csv", cash_rows)
    write_csv(destination / "ledger.csv", trace.ledger)
    write_csv(destination / "losses.csv", trace.losses)
    write_csv(destination / "selling.csv", trace.sales_audit)
    write_csv(destination / "enddays.csv", trace.enddays)
    write_json(destination / "worker_effects.json", trace.work)
    if variant == "baseline":
        for event in events:
            event["responses"] = []
            for product in dict.fromkeys(event["products"]):
                producer = g["PRODUCT_PRODUCER"][product]
                buys = [r for r in trace.ledger if r["seat"] == seat and r["op"] in ("BUY_SEED", "BUY_ANIMAL")
                        and r["item"] == producer and r["step"] >= event["step"]]
                event["responses"].append(dict(product=product, first_buy_step=min((r["step"] for r in buys), default=None),
                    units_24=sum(r["units"] for r in buys if r["step"] < event["step"]+24),
                    units_72=sum(r["units"] for r in buys if r["step"] < event["step"]+72)))
        write_json(destination / "turns.json", turns)
        write_json(destination / "events.json", events)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--histories", nargs="+", type=Path, required=True)
    parser.add_argument("--agent", type=Path, default=Path(__file__).with_name("v20_3.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1))
    parser.add_argument("--experiments", nargs="*", choices=("feed_buffer", "fert_no_churn", "feed_and_fert", "ignore_opponent_capacity", "no_repeat_capacity", "deliver_products"), default=[])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty; keep prior runs for comparison")
    args.output.mkdir(parents=True, exist_ok=True)
    engine = load_engine()
    manifest = dict(engine_version=engine.__version__, agent=str(args.agent.resolve()), agent_sha256=digest(args.agent),
        sources=[dict(path=str(p.resolve()), sha256=digest(p)) for p in args.histories],
        command=sys.argv, assumptions=["24 turns/day; 30 days; v20 production adapter", "event probes hold worker actions fixed",
            "static opponents do not adapt; intervention gains are not additive", "719 executed actions; final row is day 29 hour 23, no action there"], games=[])
    for path in args.histories:
        for variant in ["baseline", *args.experiments]:
            print(f"{path.stem} {variant}: running", flush=True)
            result = run_case(engine, path, args.agent.resolve(), args.output, args.seat, variant)
            manifest["games"].append(result)
            write_json(args.output / "manifest.json", manifest)
            print(f"  score={result['candidate_reward']:.0f}, margin={result['margin']:.0f}, losses={result['losses']}, unsent={result['sell_unsent_units']}", flush=True)
    write_csv(args.output / "comparison.csv", [{k: r[k] for k in ("episode", "variant", "candidate_reward", "margin", "sell_unsent_units", "losses")} for r in manifest["games"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
