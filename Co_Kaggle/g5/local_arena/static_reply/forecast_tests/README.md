# v20_1 forecast event tests

This directory tests `v20_1.forecast()` as a black box. The production forecast
function is not modified or instrumented.

The first suite uses one real replay:

`game_history/v20/111548564.json`

For every event, the harness evaluates two observations at the exact same game time:

1. the real observation after the event;
2. a counterfactual copy with only that event reverted.

The difference between the two forecast outputs isolates the behavior caused by the
event while holding market inventory, clock, worker positions, and every unrelated
observation field constant.

The suite keeps an independent test specification for the season clock, shop demand,
shed geometry, COW production, and crop schedules. Expected values do not call the
agent's `dist()`, `nearest_shed()`, `SHOPS`, `ANIMALS`, or `CROPS` values when
computing event deltas. This prevents a production bug from being copied into the
expected result.

Every forecast invocation also validates the output contract:

- `net_flow[product]` has exactly 720 turn entries;
- `projected[product]` has exactly 31 absolute-day entries;
- product-key sets match live market inventory;
- daily projected deltas equal cumulative turn-level `net_flow` deltas at day
  boundaries;
- the fixed replay's `day/hour` clock matches history step 0..719 exactly.

## Current event tests

### SHOP_OPEN

Detects newly unlocked shop instances from consecutive replay observations.

Until the next shop-unlock boundary, the suite requires:

- each added shop product to reduce `net_flow` by its exact demand quantity;
- the change to occur only on the 4-turn shop-consumption ticks;
- unrelated products and non-shop ticks to have zero delta.

The report also prints the resulting daily `projected` inventory deltas.

### OPP_COW_APPEAR

Detects clean opponent COW placements where the previous tile had no crop or animal.

The suite requires:

- +3 MILK on every modeled future well-cared cow production event, shifted by the
  current forecast's efficient shed-delivery ETA;
- -1 WHEAT on every future daily herd-demand tick;
- zero `net_flow` delta for unrelated products/turns.

The report prints the MILK supply turns, WHEAT demand turns, and daily projected MILK
inventory delta.


### OPP_CROP_APPEAR

Detects clean opponent crop placements where the previous tile had no crop or animal.

The suite requires:

- visible `yield_units`, if already present in the observation, to enter supply at the
  current harvest/transport ETA;
- remaining production of the currently visible crop to enter only at its scheduled
  production day plus efficient shed-delivery ETA;
- one-time crops to subtract already-visible held yield from their later scheduled total;
- ongoing TOMATO/STRAWBERRY production to use the forecast's current 1/2-unit
  well-cared/fertilization rule;
- zero `net_flow` delta for unrelated products/turns.

### OPP_YIELD_CHANGE

Detects TOMATO/STRAWBERRY `yield_units` changes while crop identity and planted day
remain unchanged.

The suite isolates only the held-yield field and requires:

- a yield increase to add exactly that many units at the current visible-yield arrival ETA;
- a harvest/disappearance to remove exactly that many forecast units from that ETA;
- future recurring crop production to remain unchanged;
- zero delta for unrelated products/turns.

In the reviewed replay run, all four event suites plus the fixture/contract test passed:
shop opens, clean COW placements, clean crop placements, ongoing-crop yield changes,
and clock/output/projection invariants.

## Run

From `Co_Kaggle/g5`:

```bash
python local_arena/static_reply/forecast_tests/test_v20_1_history_events.py -v
```

The next event families should use the same same-time-counterfactual pattern:

- opponent/own crop placement;
- visible yield increase and harvest disappearance;
- own shed and carried-inventory changes;
- opponent worker movement when visible yield exists;
- fertilization-state changes;
- new sheep/goose placement;
- day, center-consumption, and future-shop boundaries.
