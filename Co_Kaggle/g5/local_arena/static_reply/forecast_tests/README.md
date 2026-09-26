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
