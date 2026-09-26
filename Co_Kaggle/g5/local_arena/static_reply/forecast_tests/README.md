# v20_1 forecast event tests

This directory tests `v20_1.forecast()` as a black box. The production forecast
function is not modified or instrumented.

## Replay fixtures

The suite uses five real v20 histories:

- `111548564.json`
- `111549675.json`
- `111551968.json`
- `111553265.json`
- `111587965.json`

The fifth history was selected specifically to add real opponent TOMATO behavior; the
first four plus the original history did not contain opponent TOMATO events.

For every detected event, the harness evaluates two observations at the exact same game
time:

1. the real observation after the event;
2. a counterfactual copy with only that event field removed/reverted.

The output delta therefore isolates the forecast response to that event while market
inventory, time, worker positions, and unrelated observation state remain fixed.

## Independent specification

Expected values do not reuse production forecast helpers for the quantities under test.
The suite keeps independent definitions for:

- 24 turns/day and the 720-turn season;
- shop demand, shop cadence, unlock cadence, and the 8-shop cap;
- shed geometry and Manhattan transport distance;
- COW/SHEEP/GOOSE product, first-production day, interval, and well-cared output;
- WHEAT/CARROT/TOMATO/STRAWBERRY/MELON production schedules.

This avoids a tautological test where a bug in `forecast()` is copied into the expected
answer.

Every forecast invocation also verifies:

- product-key sets match live market inventory;
- `net_flow[product]` has exactly 720 turn entries;
- `projected[product]` has exactly 31 absolute-day entries;
- each replay's `day/hour` clock matches history steps 0..719;
- daily projected deltas equal cumulative turn-level `net_flow` deltas at day
  boundaries.

## Event coverage

### SHOP_OPEN — all shop types

Every real shop opening in all five histories is isolated.

The suite checks the exact full-season demand delta:

- newly known shop demand appears on each 4-turn shop tick;
- unrelated products and non-shop turns remain unchanged;
- after future unlock boundaries, the reduction in unknown future-shop slots is also
  included in the expected delta.

Latest run: **40 shop events**, covering all eight shop types.

### OPP_ANIMAL_APPEAR — all animal types

Clean opponent placements are detected where the previous tile had no crop or animal.

For COW, SHEEP, and GOOSE the suite checks:

- visible held yield at its estimated harvest/transport arrival turn;
- all future well-cared production events;
- exactly one additional WHEAT demand unit on each future daily herd tick;
- zero flow delta for unrelated products/turns.

Latest run: **85 placements**:
`COW=38, SHEEP=30, GOOSE=17`.

### OPP_ANIMAL_YIELD_CHANGE — all animal types

For an existing animal with unchanged identity/placed day, every `yield_units` change
is isolated.

The suite requires the exact yield delta to move at the current visible-yield arrival
ETA while recurring future animal production remains unchanged.

Latest run: **1,415 yield events**:
`COW=688, SHEEP=382, GOOSE=345`.

### OPP_CROP_APPEAR — all crop types

Clean opponent crop placements are tested for WHEAT, CARROT, TOMATO, STRAWBERRY, and
MELON.

The suite checks:

- visible held yield at harvest/transport ETA;
- remaining production of the currently visible crop only;
- one-time crops subtract currently held yield from the later scheduled total;
- ongoing TOMATO/STRAWBERRY use the current 1/2-unit fertilization rule;
- no replacement crop is invented;
- unrelated product/turn flow remains zero.

Latest run: **1,190 placements**:
`WHEAT=783, CARROT=176, TOMATO=10, STRAWBERRY=164, MELON=57`.

### OPP_CROP_YIELD_CHANGE — all crop types

For an unchanged crop identity/planted day, every `yield_units` transition is isolated.

The suite checks:

- visible-yield increase/disappearance at its estimated arrival ETA;
- ongoing crop future production remains unchanged;
- for one-time crops, the future scheduled remainder changes inversely with already
  visible held yield.

Latest run: **3,687 yield events**:
`WHEAT=1727, CARROT=302, TOMATO=80, STRAWBERRY=1293, MELON=285`.

## Latest validation

The locked five-history suite passed:

```text
Ran 6 tests in 91.220s
OK
```

This validates the behavior of the current forecast model. It does not prove that model
assumptions such as the 80% opponent hidden-shed prior are empirically optimal; those
belong in a separate forecast-accuracy/backtesting layer.

## Run

From `Co_Kaggle/g5`:

```bash
python local_arena/static_reply/forecast_tests/test_v20_1_history_events.py -v
```

CI runs on relevant pull requests, relevant pushes to `main`, and manual dispatch, with
sparse checkout limited to `v20_1.py`, the tests, and the five replay fixtures.
