# Kaggriculture Quick Rules for the v23 Research Agent

Use this file as the **first-pass strategic reference**.

Interpret labels carefully:
- **RULE** = guaranteed engine mechanic.
- **DERIVED FACT** = follows directly from rules.
- **POLICY HYPOTHESIS** = plausible strategy to test with static replay; not guaranteed optimal.

For edge cases or implementation detail, consult the other files in `competition_material/`.

---

## 1. Turn structure and action capacity

**RULE**
- Each farmer/hand performs at most **1 worker action per turn**.
- Different workers act independently in the same turn.
- Market orders are a separate action channel.
- Up to **10 market orders per player per turn**.
- One order may contain a quantity, e.g. `SELL MILK 20` is one order.
- `HIRE` and `BUY_LAND` each consume one market-order slot.

**DERIVED FACT**
Worker parallelism and market actions can happen in the same turn.

**POLICY HYPOTHESIS**
Treat the 10 market-order slots as a scarce per-turn budget only when demand approaches the cap. Sell, buy, hire, and land orders compete for these slots.

---

## 2. Crop timing and production

Crop ages are **calendar-day based**, not exact elapsed-turn timers.

| Crop | Seed cost | First harvest / production age | Production | Max useful yield |
|---|---:|---:|---|---|
| Wheat | 10 | age 2 | one-time | up to 4; 6 with fertilizer |
| Carrot | 20 | age 2 | one-time | up to 3; 4 with fertilizer |
| Melon | 80 | age 10 | one-time | up to 6 |
| Tomato | 50 | ages 8, 9, 10, 11 | ongoing | 1/event; 2 if watered + fertilized |
| Strawberry | 100 | ages 10, 12, 14, 16 | ongoing | 1/event; 2 if watered + fertilized |

**RULE**
- Planting consumes a worker action.
- A crop planted late in a day reaches future ages sooner than `age × 24` elapsed turns.
- Water on the planting day. Otherwise a crop can become a weed at the first day boundary.
- Two consecutive unwatered days kill a surviving crop.
- Early harvest ends wheat/carrot/melon and may sacrifice future yield.
- Tomato/strawberry store at most 4 unharvested units.
- One fertilizer application consumes 1 fertilizer and lasts 3 calendar days.

**POLICY HYPOTHESIS**
Late-game seed purchases should be gated by **time-to-realizable-sale**, not just seed cost. For example, a new melon near the end of a 30-day season is usually uneconomic because first harvest age is 10 days plus setup/harvest/delivery time.

---

## 3. Animal timing and production

Timing starts from **placement on the correct structure**, not purchase.

| Product | Animal | Purchase price | First production age | Repeat |
|---|---|---:|---:|---:|
| Eggs | Goose | 300 | 4 days | daily |
| Milk | Cow | 400 | 8 days | every 2 days |
| Wool | Sheep | 500 | 6 days | every 3 days |
| Fertilizer | any animal | — | next day boundary | daily availability |

With daily feeding/care and no storage waste:
- Goose first batch can reach 4, then 2/event.
- Cow first batch can reach 6, then 3/event.
- Sheep first batch can reach 6, then 4/event.
- Without care bonuses, animal production is 1 unit/event.

**RULE**
Animal setup path:
`BUY_ANIMAL → shed → PICKUP → travel → matching structure → PLACE`.

Also:
- Building coop/pasture costs a worker action but no money/material.
- Feed consumes wheat.
- CARE is a separate worker action.
- Two consecutive unfed days cause escape.
- Fertilizer collection is a separate worker action.
- Missed fertilizer collections do not accumulate.

**POLICY HYPOTHESIS**
Gate animal purchases by remaining season horizon and setup distance. A cow bought very late may never reach first milk production, and actual cutoff can be earlier because purchase, pickup, travel, structure setup, and placement all consume time.

---

## 4. Feeding versus production

**RULE**
- Survival requires avoiding 2 consecutive unfed days.
- Daily feeding is not strictly required for survival.
- CARE bonus accumulates only on days the animal is both fed and cared for.
- On an unfed production day, production falls to base output and accumulated CARE bonus is cleared.

**POLICY HYPOTHESIS**
Feeding frequency is a controllable tradeoff:
- less feeding → lower wheat/labor cost;
- more feeding/care → higher production from existing animals.

Test this as an economic decision, not a survival rule.

---

## 5. Harvest-to-sale pipeline

**RULE**

`HARVEST / COLLECT → worker inventory → travel → DROP at shed → SELL`

- Every movement step costs one worker action.
- DROP costs a worker action.
- Worker actions occur before market orders.
- Therefore goods dropped into the shed can be sold in the **same turn**.
- Carried inventory auto-drops at day end if capacity allows.
- Overflow is discarded.

**DERIVED FACT**
Time-to-cash includes setup + growth + harvest + transport + delivery + optional holding.

**POLICY HYPOTHESIS**
Evaluate production using **time-to-realizable-cash**, not raw yield alone.

Travel is not inherently wasteful; it is worthwhile when future production/delivery value exceeds worker-turn opportunity cost.

---

## 6. Shed and inventory

**RULE**
- Shed capacity: **100 non-seed items**.
- Seeds are separate.
- Anything beyond capacity is discarded permanently.
- Selling requires goods to be in the shed.

**POLICY HYPOTHESIS**
When expected incoming inventory would overflow the shed, free capacity before overflow.

Choose what to sell using **expected future value**, not only current price:
- current price,
- expected demand,
- likely appreciation/depreciation,
- production scarcity,
- time remaining.

---

## 7. Production economics

There is no direct engine fee for WATER, CARE, HARVEST, transport, or SELL. Their cost is worker time and any hiring expense.

Approximate direct-cost framework:

| Product | Direct cost components |
|---|---|
| Crops | seed + fertilizer consumed |
| Eggs | allocated goose cost + wheat feed |
| Milk | allocated cow cost + wheat feed |
| Wool | allocated sheep cost + wheat feed |
| Fertilizer | allocated animal purchase/feed cost as joint product |

Then add:
- worker hiring,
- land cost,
- opportunity cost of consumed wheat/fertilizer,
- travel/action capacity.

**DERIVED FACT**
Homegrown wheat and fertilizer are not economically free because using them sacrifices possible sale value.

**POLICY HYPOTHESIS**
Compare strategies using expected net terminal money, not gross production value.

---

## 8. Hiring

**RULE**
- Hired workers disappear at day end.
- Hires can happen during the day.
- New hires can act starting the **next turn**.
- Hiring uses market orders.
- Daily marginal hire costs follow:
  `1, 1, 2, 3, 5, 8, 13, 21, ...`
- The sequence resets every day.

**DERIVED FACT**
The hiring-price formula is deterministic, but marginal hire cost is not constant.

**POLICY HYPOTHESIS**
Hiring earlier in the day increases the number of productive turns available per hired worker. Evaluate a hire by expected incremental value versus its Fibonacci marginal cost.

---

## 9. What is observable

### Public opponent information
- money,
- farmer/worker positions and worker count,
- hires today,
- unlocked land,
- visible crops,
- visible animals,
- farm layout.

### Hidden opponent information
- shed contents,
- seed inventory,
- worker-carried inventory,
- submitted orders,
- intended future actions.

### Shared market/town information
- current product prices,
- market inventory quantities,
- currently unlocked shops, including duplicates,
- deterministic product demand of each shop type.

**POLICY HYPOTHESIS**
Use public farm state to estimate opponent future supply, but keep uncertainty around private inventory and intended actions.

---

## 10. Town Center baseline demand

**RULE**
- Town Center consumes **1 unit of every non-fertilizer product every 24 turns** by default.
- It operates throughout the season.
- It is independent of random shop unlocks.
- Fertilizer is excluded.
- Consumption removes units from shared market inventory.
- Players do not sell directly to Town Center.

Affected products:
`WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL`.

**DERIVED FACT**
There is predictable baseline demand even with zero unlocked shops.

**POLICY HYPOTHESIS**
Town Center demand creates recurring upward price pressure relative to no demand, but it does **not** create a guaranteed price floor. The engine's actual minimum sale price remains $1.

---

## 11. Shops and predictable demand

**RULE**
- Shops consume goods from the shared market every 4 turns by default = 6 times/day.
- One random shop opens every 3 days.
- Maximum 8 shop instances.
- Unlocks are with replacement, so duplicate shop types are possible.
- Shops never buy directly from players.

Per consumption tick:

| Shop | Demand |
|---|---|
| Bakery | 1 wheat + 1 egg |
| Pizza shop | 1 milk + 1 tomato + 1 wheat |
| Brunch spot | 1 egg + 1 wheat + 1 strawberry |
| Yarn store | 2 wool |
| Ice cream shop | 1 strawberry + 1 milk + 1 wheat |
| Pet café | 2 carrots |
| Smoothie shop | 1 strawberry + 1 milk |
| Farmers Market | 1 wheat + 1 carrot + 1 tomato + 1 strawberry |

**DERIVED FACT**
Given the current unlocked-shop multiset, near-term shop consumption is directly calculable. Only future unlock identities are uncertain.

**POLICY HYPOTHESIS**
Use shop composition to forecast product-specific demand and test shop-aware hold/sell/production policies.

---

## 12. Fertilizer economics

**RULE**
- Animals create fertilizer availability.
- Workers must collect it.
- Fertilizer can be sold or used on crops.
- Fertilizer occupies shed capacity.
- Fertilizer can also be bought through the shared market.

**POLICY HYPOTHESIS**
If either player has many animals relative to crop use, fertilizer supply may rise. If excess fertilizer is sold, market inventory can rise and the buy price can fall.

This may make **buying fertilizer** cheaper. Home-produced fertilizer still has opportunity cost.

---

## 13. Fixed and deterministic prices

**RULE**
Animal purchase prices are constant:
- Goose: 300
- Cow: 400
- Sheep: 500

Hiring follows a deterministic daily-reset Fibonacci schedule.

**DERIVED FACT**
Animal prices are constant. Hiring prices are deterministic but increase with the number hired that day.

---

## 14. Market price and sell timing

**RULE**
- Players trade only with the shared market.
- Sell prices depend on shared market inventory.
- Town consumption and player buys reduce market inventory.
- Player sells increase market inventory.
- The engine price floor is **$1**.

**POLICY HYPOTHESIS**
Selling everything immediately is not always optimal.

Candidate signals for hold/sell decisions:
- current price,
- price trend,
- current market inventory,
- Town Center demand,
- unlocked-shop demand,
- opponent visible future supply,
- own expected future supply,
- shed pressure,
- remaining season horizon.

This should be tested by static replay against v20 loss cases.

---

## 15. Worker routing

**RULE**
Approximate worker-action costs:

| Activity | Worker actions |
|---|---:|
| move one tile | 1 |
| plant | 1 |
| water | 1 |
| feed | 1 |
| harvest | 1 |
| care | 1 |
| collect fertilizer | 1 |
| drop/deliver at shed | 1 |

**DERIVED FACT**
Worker time is a major scarce resource even when an action has zero direct cash cost.

**POLICY HYPOTHESIS**
Prefer routes that chain productive work:
`move → operate → move → deliver`.

Minimize movement that does not improve future production, care, harvesting, delivery, or strategic positioning.

---

# Agent reasoning template

For each proposed improvement, reason in this order:

1. **RULE** — which engine mechanic matters?
2. **STATE SIGNAL** — what observable variable reveals the situation?
3. **ECONOMIC EFFECT** — how does it change expected terminal money?
4. **POLICY HYPOTHESIS** — what decision should change?
5. **STATIC REPLAY TEST** — which v20 loss cases can falsify it?
6. **METRICS** — compare:
   - candidate margin,
   - margin improvement vs recorded v20,
   - repaired losses,
   - worsened cases,
   - action divergences.

Do not promote a policy because it sounds reasonable. Promote it only when static-replay evidence supports it without unacceptable regressions.
