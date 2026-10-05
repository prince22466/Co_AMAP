# Investigation log: 111572223 and 111574474

## Scope

Diagnose the two v20_3 loss histories:

- `runs/v20_3/20260926T222057_211234Z/histories/111572223.json`
- `runs/v20_3/20260926T222057_211234Z/histories/111574474.json`

Primary attribution targets:

- `production_signals`
- `animal_plan`
- `crop_plan`
- `market_orders`

Questions to answer:

1. When a shop opens, does the policy generate the corresponding production signal and buy the appropriate animal/seed?
2. When the opponent gains production capacity, how do our signals and purchases change?
3. Is the intended "sell immediately" policy actually carried out after harvest/collection and delivery?
4. Which subsystem responses work as intended, and which responses correlate with the largest economic losses?
5. Separate gross trading activity from true loss, especially fertilizer churn.

## Existing diagnostics examined

The current diagnostic stack is already designed for this investigation:

- `DIAGNOSTICS.md`
- `diagnose.py`
- `summarize_diagnostics.py`
- `test_diagnose.py`

Important existing capabilities verified:

- Exact sequential reconstruction of `production_signals`, `animal_plan`, `crop_plan`, and `market_orders`.
- Action/state parity checks against recorded histories.
- Engine instrumentation for successful market transactions, worker effects, animal escapes, crop deaths, overflow, and end-of-day feeding.
- Shop-opening event probes by removing one new shop from the same observation and recomputing signals/orders.
- Opponent-capacity event probes by removing one new opponent producer from the same observation.
- Follow-up purchase windows after events.
- Procurement gate explanations such as cash, capacity, deadline, order-slot, and fractional-gap blocking.
- Post-worker sell audit that compares actually sellable shed inventory with submitted sell orders.
- Fertilizer FIFO accounting so gross fertilizer purchases are not incorrectly reported as pure loss.
- Producer purchase/deployment/lifecycle accounting and deployment lag.
- Full-season interventions such as `feed_buffer`, `fert_no_churn`, `ignore_opponent_capacity`, `no_repeat_capacity`, and `deliver_products`.

## Behaviors already established

These are observations already established for the selected games and should be treated as starting evidence, not re-discovered assumptions:

- All four opening animals disappear before day 2.
- No wheat purchases are recorded before those early animal losses.
- The agent repeatedly buys and sells fertilizer at very high volume.
- Across the examined losses, 16 animals are lost with 7,600 coins of acquisition cost.
- The first Farmers Market opening does trigger the corresponding seed purchases one turn later.
- Later shop-triggered purchases are often blocked by cash or available deployment space.
- Fertilizer purchases and fertilizer sales substantially offset each other in cash, so gross fertilizer spend is not a valid loss metric.

## Current interpretation

### Likely major loss: animal lifecycle / feed procurement

The strongest currently observed failure is not simply "animal_plan bought the wrong animal."

The early lifecycle is:

1. animals exist,
2. feed requirement appears,
3. no wheat procurement is recorded,
4. animals disappear before day 2.

This points toward a gap between production planning and short-horizon survival procurement. The diagnostic `feed_buffer` intervention is specifically intended to quantify that effect.

The 7,600-coin acquisition cost of escaped animals is an accounting loss of assets, but must not be treated as the exact counterfactual score gain from fixing feeding.

### Shop response is partially working

At least one important desired behavior is confirmed:

- Farmers Market opening -> corresponding seed signal/purchase response -> seed purchases one turn later.

Therefore, shop-opening detection and producer mapping are not globally broken.

Later failures must be separated into:

- no signal generated,
- signal generated but `floor(gap) == 0`,
- cash reserve blocks order,
- no crop/animal slot,
- purchase deadline reached,
- earlier orders consume the 10 market slots,
- purchase succeeds but deployment is delayed or fails.

### Opponent capacity requires causal event-level analysis

The existing event probe is appropriate:

- compare the actual observation with a counterfactual where one newly placed opponent producer is removed,
- recompute signals and orders,
- record whether same-product/feed buying is suppressed or increased.

This is needed because "opponent gains capacity" can reduce our product shortage while an opponent animal simultaneously increases public wheat/feed demand.

Immediate market orders may remain unchanged even when the signal changes because another gate dominates.

### Fertilizer churn is suspicious but gross spend is not the loss

The large fertilizer buy/sell volume must be analyzed net of resale.

Required decomposition:

- purchased fertilizer later resold,
- purchased fertilizer consumed,
- fertilizer collected from animals,
- fertilizer remaining,
- fertilizer overflow/destruction,
- actual FIFO resale P&L.

Only the net economic effect plus secondary effects such as market-slot occupation and cash timing should be treated as harmful.

### "Sell immediately" must be tested after worker actions

The correct audit point is post-worker, pre-market state.

For each sellable product:

- actual quantity in shed,
- total owned,
- policy reserve,
- eligible quantity,
- submitted sell quantity,
- prediction error,
- stock carried by workers,
- eligible but unsent quantity.

Repeated unsent unit-turns must not be summed as if each observation were a separate destroyed unit.

Harvest-to-sale lag should be reported separately from immediately sellable shed backlog because workers can carry harvested output until later deposit.

## Tooling notes / limitations found

The GitHub connector can fetch the diagnostic source files, but the two very large history JSON blobs currently return an empty content payload through `fetch_file` even though their blob SHAs are visible.

Therefore the next episode-specific pass should use one of:

1. the repository checked out locally and run the existing replay diagnostics;
2. already-generated diagnostic evidence committed into the repo;
3. a GitHub-accessible smaller derived artifact generated from those histories.

Do not infer turn-level behavior from the empty connector payload.

## Next concrete analysis

Run:

```powershell
python local_arena/static_reply/diagnose.py --agent local_arena/static_reply/v20_3.py --histories local_arena/static_reply/runs/v20_3/20260926T222057_211234Z/histories/111572223.json local_arena/static_reply/runs/v20_3/20260926T222057_211234Z/histories/111574474.json --output local_arena/static_reply/diagnostics/two_loss_investigation --experiments feed_buffer fert_no_churn feed_and_fert ignore_opponent_capacity no_repeat_capacity deliver_products

python local_arena/static_reply/summarize_diagnostics.py local_arena/static_reply/diagnostics/two_loss_investigation

python -m unittest discover -s local_arena/static_reply -p test_diagnose.py -v
```

Then report per episode:

### A. Shop openings

For every opening:

- shop,
- product,
- signal gap before/after opening,
- corresponding seed/animal producer,
- whether an immediate buy was submitted,
- exact blocking gates,
- first later successful buy,
- units bought within 24 and 72 turns,
- deployment lag,
- eventual production and sales.

### B. Opponent production additions

For every newly added opponent crop/animal:

- product capacity delta,
- wheat/feed demand delta where relevant,
- our signal change,
- our same-product buy change,
- our feed buy change,
- whether a different procurement gate masked the signal change.

Aggregate separately by producer type.

### C. Animal plan

For each purchased/starting animal:

- placement step,
- care/feed actions,
- feed availability,
- end-of-day fed status,
- production collected,
- escape step,
- acquisition cost,
- deployment lag for purchased animals.

This should establish whether the dominant failure is feed procurement, labor scheduling, placement, or inappropriate buying.

### D. Crop plan

For each purchased seed:

- shop trigger or other signal source,
- purchase step,
- plant step,
- deployment lag,
- water/fertilizer care,
- harvest count,
- zero-harvest terminal/dead crops,
- sale cash.

Separate planning failures from purchases that were valid but arrived too late.

### E. Market orders

Classify every failed desired purchase by:

- cash,
- reserve,
- space,
- fractional producer-equivalent gap,
- deadline,
- market order limit,
- no actionable demand.

For selling:

- eligible shed stock vs submitted sells,
- prediction errors,
- carried inventory,
- harvest-to-sale latency,
- terminal leftovers.

### F. Economic attribution

Do not simply add causal estimates.

Use:

- exact cash-category reconciliation,
- escaped animal acquisition cost,
- fertilizer FIFO P&L,
- full-season intervention score/margin deltas.

Intervention effects interact and must not be summed.

## Resume point

Resume from the episode-specific replay outputs, not from broad hypotheses.

The first comparisons to inspect are:

1. baseline vs `feed_buffer`,
2. baseline vs `fert_no_churn`,
3. baseline vs `ignore_opponent_capacity`,
4. shop event rows in `events.json`,
5. post-worker `selling.csv`,
6. animal escape rows in `losses.csv` joined with `enddays.csv` and `worker_effects.json`.

The main unresolved question is no longer whether the agent has obvious bad outcomes; it is which policy stage creates each one and the size of each effect after controlling for cash, space, order limits, deployment, and opponent supply.
