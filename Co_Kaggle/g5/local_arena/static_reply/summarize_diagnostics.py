#!/usr/bin/env python3
"""Build repeatable Markdown evidence reports from diagnose.py output (no dependencies)."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path

from diagnose import write_csv, write_json, identity

ANIMALS = {"COW": (400, "MILK", 8), "SHEEP": (500, "WOOL", 6), "GOOSE": (300, "EGG", 4)}
SEEDS = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clock(step):
    return "never" if step is None else f"D{int(step)//24} H{int(step)%24} (t={step})"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |" for row in rows]
    return "\n".join(out)


def collect(base):
    summary = json.loads((base / "summary.json").read_text())
    seat = summary["seat"]
    work = [r for r in json.loads((base / "worker_effects.json").read_text()) if r["seat"] == seat]
    ledger = [r for r in csv_rows(base / "ledger.csv") if int(r["seat"]) == seat]
    for r in ledger:
        for key in ("step", "units", "price"):
            r[key] = int(float(r[key]))
        r["cash"] = float(r["cash"])
    losses = [r for r in csv_rows(base / "losses.csv") if int(r["seat"]) == seat]
    selling = [r for r in csv_rows(base / "selling.csv") if int(r["seat"]) == seat]
    lifetimes = {}
    for row in work:
        before, after = row["before"], row["after"]
        if identity(after) and identity(after)[0] and identity(before) != identity(after):
            producer, day = identity(after)
            key = (row["x"], row["y"], producer, day)
            lifetimes[key] = dict(x=row["x"], y=row["y"], producer=producer, day=day, placed_step=row["step"],
                harvested_units=0, fertilizer_collected=0, fate="present_at_end", end_step=None)
        if identity(before) and identity(before)[0]:
            producer, day = identity(before)
            key = (row["x"], row["y"], producer, day)
            life = lifetimes.get(key)
            if not life: continue
            if row["op"] == "HARVEST":
                product = ANIMALS[producer][1] if producer in ANIMALS else producer
                life["harvested_units"] += max(0, row["inventory_delta"].get(product, 0))
            if row["op"] == "COLLECT_FERTILIZER":
                life["fertilizer_collected"] += max(0, row["inventory_delta"].get("FERTILIZER", 0))
            if identity(before) != identity(after):
                life.update(fate=row["op"].lower(), end_step=row["step"])
    for row in losses:
        if row["kind"] not in ("animal_escape", "crop_died"): continue
        t = json.loads(row["tile"])
        producer, day = identity(t)
        key = (int(row["x"]), int(row["y"]), producer, day)
        if key in lifetimes:
            lifetimes[key].update(fate=row["kind"], end_step=int(row["step"]))
    # Fulfilled procurement -> successful deployment FIFO. This is a lag attribution,
    # not physical identity of fungible seeds/animals. Actions precede market fills.
    queues = defaultdict(deque)
    lags = []
    work_by_step, ledger_by_step = defaultdict(list), defaultdict(list)
    for r in work: work_by_step[r["step"]].append(r)
    for r in ledger: ledger_by_step[r["step"]].append(r)
    for step in range(summary["decisions"]):
        for r in work_by_step[step]:
            if identity(r["after"]) and identity(r["after"])[0] and identity(r["before"]) != identity(r["after"]):
                producer = identity(r["after"])[0]
                if not queues[producer]:
                    raise ValueError(f"deployment has no purchased producer: {producer} at {step}")
                bought = queues[producer].popleft()
                lags.append(dict(producer=producer, bought_step=bought, deployed_step=step, lag=step-bought,
                                 x=r["x"], y=r["y"]))
        for r in ledger_by_step[step]:
            if r["op"] in ("BUY_SEED", "BUY_ANIMAL"):
                queues[r["item"]].extend([step]*r["units"])
    # Trace fertilizer accounting with FIFO attribution of fungible inventory.
    lots = deque()
    fert = Counter()
    for step in range(summary["decisions"]):
        for r in work_by_step[step]:
            delta = r["inventory_delta"].get("FERTILIZER", 0)
            if r["op"] == "COLLECT_FERTILIZER" and delta > 0:
                lots.extend([("collected", 0, step)]*delta)
                fert["collected"] += delta
            elif r["op"] == "FERTILIZE" and delta < 0:
                for _ in range(-delta):
                    source, cost, bought_step = lots.popleft()
                    fert["consumed"] += 1
                    fert["consumed_purchase_cost"] += cost
        for r in ledger_by_step[step]:
            if r["item"] != "FERTILIZER": continue
            if r["op"] == "BUY_PRODUCT":
                lots.append(("purchased", r["price"], step))
                fert["bought"] += 1
                fert["buy_cost"] += r["price"]
            elif r["op"] == "SELL":
                source, cost, bought_step = lots.popleft()
                fert["sold"] += 1
                fert["sales_cash"] += r["price"]
                if source == "purchased":
                    fert["purchased_resold"] += 1
                    fert["purchased_resale_pnl"] += r["price"]-cost
                    if step == bought_step: fert["same_turn_resold"] += 1
                else:
                    fert["collected_sales_cash"] += r["price"]
        for r in losses:
            if int(r["step"]) == step and r["item"] == "FERTILIZER" and "overflow" in r["kind"]:
                for _ in range(int(r["units"])): lots.popleft()
                fert["overflow"] += int(r["units"])
    fert["remaining"] = len(lots)
    # End-of-day/worker overflow ordering is only unambiguous when absent; disclose.
    fert["fifo_has_overflow_caveat"] = bool(fert["overflow"])
    assert fert["bought"]+fert["collected"] == fert["sold"]+fert["consumed"]+fert["remaining"]+fert["overflow"]
    # Harvest-to-sale age for non-reserve produce; transfers do not change ownership.
    product_lots = defaultdict(deque)
    delivery_rows = []
    nonreserve = set(SEEDS) | {a[1] for a in ANIMALS.values()}
    nonreserve.discard("WHEAT")
    for step in range(summary["decisions"]):
        for r in work_by_step[step]:
            if r["op"] == "HARVEST":
                for product, n in r["inventory_delta"].items():
                    if product in nonreserve and n > 0:
                        product_lots[product].extend([step]*n)
        for r in losses:
            if int(r["step"]) == step and r["kind"] == "drop_overflow" and r["item"] in nonreserve:
                for _ in range(int(r["units"])): product_lots[r["item"]].popleft()
        for r in ledger_by_step[step]:
            if r["op"] == "SELL" and r["item"] in nonreserve:
                harvested = product_lots[r["item"]].popleft()
                delivery_rows.append(dict(product=r["item"], harvested_step=harvested, sold_step=step,
                    lag=step-harvested, price=r["price"], crosses_day=step//24 > harvested//24))
        for r in losses:
            if int(r["step"]) == step and r["kind"] == "endday_overflow" and r["item"] in nonreserve:
                for _ in range(int(r["units"])): product_lots[r["item"]].popleft()
    delivery_summary = []
    for product in sorted(nonreserve):
        rows = [r for r in delivery_rows if r["product"] == product]
        if rows:
            delivery_summary.append(dict(product=product, units=len(rows),
                mean_lag=round(sum(r["lag"] for r in rows)/len(rows), 2), max_lag=max(r["lag"] for r in rows),
                crossed_day=sum(r["crosses_day"] for r in rows),
                remaining_harvested_unsold=len(product_lots[product])))
    immediate = Counter()
    for r in selling:
        n = int(r["unsent"])
        if n:
            immediate[r["product"]+"_unit_turns"] += n
            immediate[r["product"]+"_decisions"] += 1
    producer_rows = []
    for producer in [*SEEDS, *ANIMALS]:
        lives = [r for r in lifetimes.values() if r["producer"] == producer]
        bought = sum(r["units"] for r in ledger if r["item"] == producer and r["op"] in ("BUY_SEED", "BUY_ANIMAL"))
        product = ANIMALS[producer][1] if producer in ANIMALS else producer
        sold = [r for r in ledger if r["op"] == "SELL" and r["item"] == product]
        delay = [r["lag"] for r in lags if r["producer"] == producer]
        producer_rows.append(dict(producer=producer, bought=bought, deployed=len(lives),
            median_deploy_lag=sorted(delay)[len(delay)//2] if delay else None, max_deploy_lag=max(delay, default=None),
            died=sum(r["fate"] in ("animal_escape", "crop_died") for r in lives),
            zero_harvest=sum(r["harvested_units"] == 0 for r in lives),
            harvested=sum(r["harvested_units"] for r in lives), sold=sum(r["units"] for r in sold),
            sales_cash=sum(r["cash"] for r in sold), purchase_cost=bought*(ANIMALS[producer][0] if producer in ANIMALS else SEEDS[producer])))
    result = dict(summary=summary, producers=producer_rows, lifetimes=list(lifetimes.values()), deployment_lags=lags,
        fertilizer=dict(fert), immediate_selling=dict(immediate), delivery=delivery_summary,
        delivery_lags=delivery_rows,
        escaped_acquisition_cost=sum(ANIMALS[r["item"]][0]*int(r["units"]) for r in losses if r["kind"] == "animal_escape"))
    return result


def render(output):
    manifest = json.loads((output / "manifest.json").read_text())
    episodes = list(dict.fromkeys(r["episode"] for r in manifest["games"]))
    lines = ["# Production policy diagnostic evidence", "",
        "Generated from exact agent-action reconstruction and instrumented engine replay. All times use zero-based days/hours. "
        "Decision t uses history `steps[t]`; its chosen action and resulting state are in `steps[t+1]`. "
        "These recordings contain 720 states and 719 executed decisions (t=0..718).",
        "", "## Validation and outcome", "",
        table(["Episode", "Our score", "Opponent score", "Margin", "Action/state parity", "Cash reconciles"],
            [[r["episode"], r["candidate_reward"], r["rewards"][1-r["seat"]], r["margin"],
              r["action_parity"]+" / "+r["state_parity"], r["cash_reconciled"]] for r in manifest["games"] if r["variant"] == "baseline"])]
    all_metrics = {}
    for episode in episodes:
        base = output / episode / "baseline"
        result = collect(base)
        all_metrics[episode] = result
        write_json(base / "derived.json", result)
        write_csv(base / "lifetimes.csv", result["lifetimes"])
        write_csv(base / "deployment_lags.csv", result["deployment_lags"])
        write_csv(base / "producers.csv", result["producers"])
        write_csv(base / "delivery_lags.csv", result["delivery_lags"])
        turns = json.loads((base / "turns.json").read_text())
        events = json.loads((base / "events.json").read_text())
        summary = result["summary"]
        lines += ["", f"## Episode {episode}", "", "### Producer purchases, deployment, and outcomes", "",
            table(["Producer", "Bought", "Deployed", "Lag median/max turns", "Escaped/died", "Zero-harvest assets", "Harvested", "Sold", "Sales coins", "Producer cost"],
                [[r["producer"], r["bought"], r["deployed"], f"{r['median_deploy_lag']}/{r['max_deploy_lag']}", r["died"],
                  r["zero_harvest"], r["harvested"], r["sold"], r["sales_cash"], r["purchase_cost"]] for r in result["producers"]]), "",
            "Deployment lag matches fulfilled purchases to successful placements/plantings FIFO. Zero-harvest includes terminal immature crops; it does not automatically mean a planner bug. "
            "Sales minus producer cost excludes shared labor, fertilizer, land, and feed. Wheat sales also depend on feed consumption.", "",
            f"Escaped animals have **{result['escaped_acquisition_cost']:,.0f} coins** of acquisition cost. This is lost asset cost, not an additive estimate of the final-score improvement from preventing escapes.",
            "", "### Shop openings and subsequent buying", "",
            "Each row removes one instance of the new shop from the same current observation. All other state and chosen worker actions are fixed. "
            "The before/after gap is in producer equivalents. Future purchase windows describe observed response, not an isolated causal effect of that shop. "
            "`None` means no producible remaining capacity with positive demand; it is not zero shortage.", ""]
        shop_rows = []
        for event in events:
            if event["kind"] != "shop_open": continue
            for change, response in zip(event["changes"], event["responses"]):
                gate = next(r for r in event["gates"] if r["product"] == change["product"])
                fmt = lambda x: "None" if x is None else f"{x:.2f}"
                shop_rows.append([clock(event["step"]), event["name"], change["product"],
                    f"{fmt(change['gap_before'])} -> {fmt(change['gap_after'])}", gate["bought"],
                    ", ".join(gate["reasons"]) or "eligible", clock(response["first_buy_step"]), response["units_24"], response["units_72"]])
        lines += [table(["Open", "Shop", "Product", "Gap without -> with", "Buy now", "Gates now", "Next buy", "Bought in 24t", "Bought in 72t"], shop_rows),
            "", "### Opponent supply additions", ""]
        opponent_events = [e for e in events if e["kind"] == "opponent_capacity"]
        changed = [e for e in opponent_events if e["actual_orders"] != e["without_event_orders"]]
        rows = []
        for producer, n in Counter(e["name"] for e in opponent_events).items():
            selected = [e for e in opponent_events if e["name"] == producer]
            suppressed = sum(c["buy_before"] > c["buy_after"] for e in selected for c in e["changes"])
            increased = sum(c["buy_before"] < c["buy_after"] for e in selected for c in e["changes"])
            rows.append([producer, n, suppressed, increased])
        lines += [table(["Opponent producer", "Additions", "Same-product/feed buy reductions", "Same-product/feed buy increases"], rows), "",
            f"{len(changed)} of {len(opponent_events)} individual event removals change the immediate market-order list. "
            "Counts are per producer tile, including replanting; simultaneous tile additions are individually removed, so effects are not additive. "
            "No change in buying does not imply no signal response: cash, floor(gap), deadlines, land, and order limits can dominate.", ""]
        for event in changed[:20]:
            lines += [f"- {clock(event['step'])}, new opponent {event['name']} at {event['position']}: "
                f"orders without event `{event['without_event_orders']}`; actual `{event['actual_orders']}`."]
        lines += ["", "### Immediate selling and fertilizer", "",
            f"Post-worker shed prediction mismatches: **{summary['sale_prediction_errors']}**. "
            f"Eligible but unrequested stock totals **{summary['sell_unsent_units']} unit-turns** (repeat exposure may count the same item more than once, not destroyed units). "
            f"Breakdown: `{result['immediate_selling']}`.", "",
            "Eligibility is based on actual post-worker stock, retaining the code's wheat/fertilizer reserve thresholds. "
            "Workers collecting fertilizer or harvesting wheat increase total stock before the market; the agent's predictor does not add those gains to reserve totals. "
            "This can under-sell reserved products even while predicting shed contents correctly. Carried products outside the shed cannot be sold directly.", "",
            table(["Product", "Units sold", "Mean harvest-to-sale turns", "Max turns", "Crossed a day boundary", "Harvested left unsold"],
                [[r["product"], r["units"], r["mean_lag"], r["max_lag"], r["crossed_day"], r["remaining_harvested_unsold"]] for r in result["delivery"]), "",
            "These FIFO ages include transport and waiting for automatic end-of-day deposits; they do not prove a sellable-shed backlog. "
            "The delivery intervention measures the score effect including extra worker travel.", "",
            table(["Fertilizer measure", "Value"], result["fertilizer"].items()), "",
            "Fertilizer resale P&L uses FIFO assignment across owned stock, collected output, applications, and actual executed market units. "
            "High gross purchase spend is not itself a loss: resale cash must be included. This allocation is accounting, not a causal intervention.",
            "", "### Largest cash contributions to the score gap", ""]
        by_category = defaultdict(lambda: [0, 0])
        for r in summary["cash"]:
            by_category[(r["op"], r["item"])][r["seat"]] += r["cash"]
        differences = [(op, item, cash[seat := summary["seat"]], cash[1-seat], cash[seat]-cash[1-seat]) for (op, item), cash in by_category.items()]
        differences.sort(key=lambda x: x[-1])
        lines += [table(["Operation", "Product", "Our cash", "Opponent cash", "Our minus opponent"], differences), "",
            "All categories sum exactly to the score margin; both players start with the same cash. "
            "Separate fertilizer buys and sells offset heavily and should be interpreted together. Revenue gaps are accounting explanations, not proof that one subsystem caused the full gap."]
    baseline = {r["episode"]: r for r in manifest["games"] if r["variant"] == "baseline"}
    lines += ["", "## Full-season interventions", "",
        "Each experiment starts a fresh environment and fresh agent globals, changes only its documented wrapper, "
        "and replays the same opponent commands. Prices, legality, weeds, purchases, and future candidate decisions evolve with the changed state. "
        "The opponent does not adapt; these are two selected loss cases, not generalization evidence. Improvements are not additive.", "",
        table(["Episode", "Intervention", "Our score", "Change vs baseline", "Margin", "Margin change", "Escapes"],
            [[r["episode"], r["variant"], r["candidate_reward"], r["candidate_reward"]-baseline[r["episode"]]["candidate_reward"],
              r["margin"], r["margin"]-baseline[r["episode"]]["margin"], r["losses"].get("animal_escape", 0)] for r in manifest["games"] if r["variant"] != "baseline"]),
        "", "Interventions: `feed_buffer` buys toward two days of wheat for currently placed livestock, after sales and before other spending, from day 1 through 28; "
        "`fert_no_churn` disables all fertilizer buying (also removes useful applications, so this is broader than pure churn removal); "
        "`feed_and_fert` combines those; `ignore_opponent_capacity` removes opponent producer output from signals while preserving feed demand; "
        "`no_repeat_capacity` removes assumed future replanting of existing crops; `deliver_products` forces workers carrying non-reserve produce to return and deposit it.",
        "", "## Evidence and reproducibility", "",
        "- `manifest.json`: exact input/agent hashes, engine version, invocation, parity and scores.",
        "- `<episode>/baseline/turns.json`: each pre-action observation summary, complete signals, plans, procurement gates and recorded action index.",
        "- `<episode>/baseline/events.json`: every shop/opponent event, same-state removal probes and follow-up buying windows.",
        "- `<episode>/<variant>/ledger.csv`: each successfully executed market unit and its actual price; includes both players.",
        "- `selling.csv`: actual post-worker availability, reserves, requested sells, eligible unsent stock and prediction errors.",
        "- `losses.csv`, `enddays.csv`, `worker_effects.json`: engine-observed escapes, drought, overflow, feeding and worker effects.",
        "- `lifetimes.csv`, `deployment_lags.csv`, `producers.csv`, `derived.json`: derived deployment, asset and accounting evidence.",
        "", "The adapter is deliberately specific to the v20 production API and default 24-turn/30-day game. "
        "An archived-agent hash or replay mismatch stops the analysis rather than silently attributing current-code behavior to an old recording."]
    (output / "evidence_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(output / "derived_metrics.json", all_metrics)
    print(output / "evidence_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    render(parser.parse_args().output)
