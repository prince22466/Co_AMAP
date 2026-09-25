# v20 Model Structure and Runtime Workflow

Purpose: let the v23 research agent understand `kaggriculture-sub_v20.ipynb` quickly without reading the full notebook first.

## One-sentence summary

v20 is a **hybrid rule-based planner + learned residual Q ranker**:

```text
observation
  -> forecast future supply/demand
  -> plan animals
  -> plan crops
  -> generate feasible worker tasks
  -> rank worker-task candidates with:
       normalized learned tree prior + learned neural residual Q
  -> build worker actions
  -> build rule-based market orders
  -> return Kaggriculture action dict
```

The learned model does **not** control the whole agent. It mainly changes which feasible worker/task assignment is selected.

---

# 1. Top-level runtime

Entrypoint:

```python
agent(obs)
```

High-level flow:

```text
agent(obs)
  |
  +-- classify opponent style
  |
  +-- forecast(obs)
  |     -> projected market inventory
  |     -> estimated demand
  |
  +-- animal_plan(obs, projected)
  |     -> desired animal placement/type by tile
  |
  +-- crop_plan(obs, projected, animal_plan)
  |     -> desired crop by tile
  |
  +-- unit_actions(obs, animal_plan, crop_plan)
  |     -> farmer + hand actions
  |     -> learned Q ranker acts here
  |
  +-- market_orders(obs, animal_plan, crop_plan, unit_actions)
        -> SELL / HIRE / BUY / BUY_LAND orders

return {
  "farmer": actions[0],
  "hands": actions[1:],
  "market": market_orders(...)
}
```

---

# 2. What is learned vs rule-based

## Learned part

Worker-task selection has two learned components: an inherited ensemble of 16 decision trees (`_tree_0` through `_tree_15`, summed by `learned_task_score`) and a residual action-value network. The prior is therefore learned, although its inputs include hand-designed task weights and features.

Conceptually:

```text
Q(candidate)
  = normalized learned_task_score(candidate)
  + neural residual(state, candidate)
```

The notebook describes this as a residual Double-DQN checkpoint from update 9.

Embedded model:

```text
input -> 64 -> 64 -> 1
```

The checked-in v20 model labels its embedded weights as update 9 and runs without PyTorch at submission time. Source inspection verifies the embedded architecture and inference, but does not independently authenticate the checkpoint's training history.

The learned ranker is used only after the agent has already generated feasible worker/task candidates.

## Rule-based / heuristic parts

These remain hand-designed:

- market price approximation
- future supply/demand forecast
- crop choice
- animal choice
- feasible task generation
- resource constraints
- path distance / movement
- wheat/fertilizer reserves
- selling logic
- hiring logic
- land buying
- seed buying
- animal buying
- fertilizer buying
- opening build order
- opponent-style classification

Important implication for v23:

> A failure can come from the planner, candidate generation, Q ranking, or market logic. Do not assume the neural ranker is the main bottleneck.

---

# 3. Forecast layer

Function:

```python
forecast(obs)
```

Inputs used:

- current shared market inventory
- unlocked shops
- both public farms
- visible crops
- visible animals
- own private inventory

It builds:

```text
supply[product][future_day]
demand[product]
projected[product][future_day]
```

Important behavior:

- Town-center baseline demand is represented as baseline demand for every non-fertilizer product.
- Current unlocked shops add deterministic product demand.
- Future unknown shops are represented by expected average demand.
- Visible crops and animals from both farms contribute expected future supply.
- Own stored/carried inventory contributes current supply.
- Animal count increases expected wheat demand.

This forecast feeds both crop and animal planning.

Potential v23 research area:
- forecast bias,
- opponent supply estimation,
- future shop expectation,
- treatment of own stored inventory,
- price impact estimation.

---

# 4. Animal planner

Function:

```python
animal_plan(obs, projected)
```

Responsibilities:

1. Preserve animals already placed on predefined `ROUTES` tiles (not arbitrary tiles elsewhere).
2. Allocate animals already owned but not placed.
3. Consider adding new animals to predefined animal tiles.
4. Score candidate animal types by projected economics.

Animal economics include approximately:

```text
future product revenue
+ fertilizer value
- wheat feed cost
- labor proxy
- animal purchase price
```

Important hard-coded behavior:

- predefined animal-route tiles
- herd limit
- new-animal planning mainly during an early/mid-game window
- minimum profitability threshold

Potential v23 research area:
- animal purchase cutoff,
- herd size,
- species mix,
- feed/fertilizer economics,
- placement layout,
- transport cost.

---

# 5. Crop planner

Function:

```python
crop_plan(obs, projected, animal)
```

Responsibilities:

1. Enumerate usable crop tiles outside the fixed `ANIMAL_POINTS` set.
2. Preserve crops already growing on the enumerated crop tiles.
3. Use a fixed cheap opening layout early.
4. Later score crop candidates using projected prices and production timing.
5. Reserve future projected supply after assigning a crop.

Despite its signature, `crop_plan(obs, projected, animal)` never reads `animal`. It excludes every predefined animal tile, including tiles absent from the current animal plan. `animal_plan` also modifies a local copy of the forecast, so its additions are not passed back through `projected` to the crop planner.

Approximate crop score:

```text
expected revenue
- seed cost
- estimated fertilizer cost

then normalized partly by production duration
and modified by crop-specific heuristic multipliers
```

Notable hard-coded preferences exist for:

- wheat
- strawberry
- tomato
- early-game wheat
- owned seeds

Potential v23 research area:
- crop mix,
- late-game cutoffs,
- yield timing,
- fertilizer ROI,
- crop-specific multipliers,
- production-duration normalization.

---

# 6. Worker-task generation

Function:

```python
unit_actions(obs, animal, crops)
```

This is the most important bridge between the planners and the learned model.

It produces actions through both direct rules and ranked tasks, including:

- move toward planned animal/crop locations
- build coop/pasture
- place animals
- plant
- water
- fertilize
- feed
- care
- harvest
- collect fertilizer
- pick up wheat/fertilizer
- drop carried goods

Only entries in the ranked `tasks` list have this structure (directly assigned actions do not):

```text
(target_position, operation, heuristic_weight, required_resource)
```

Workers with already-forced actions are removed first.
Remaining workers and tasks form feasible candidate pairs.

The distinction matters: delivery moves and DROP/PLACE, carried-animal transport/build/place/dig, shed pickups of new animals, and some immediate wheat/fertilizer pickups are assigned directly by `unit_actions` before Q ranking. They are not options selected by the residual network. Movement toward a ranked task is then produced by `move`; fallback DROP/PASS is assigned after ranking.

Feasibility checks include:

- required carried resource
- distance
- remaining turns in the day
- current position
- task/resource compatibility

Here, "feasible" means passing the policy's filters, not a guarantee of successful engine execution. These checks do not simulate all same-turn interactions. The ranked operation types are PLANT, DIG, HARVEST, WATER, FERTILIZE, FEED, CARE, COLLECT_FERTILIZER, and PICKUP.

---

# 7. Worker-task scoring

For every feasible:

```text
(worker_i, task_j)
```

v20 computes:

```text
task_features(...)
learned tree baseline = learned_task_score(features)
normalized prior = q_normalized_prior(all baselines)
global state = q_global_state(...)
state_bias = q_state_bias(global state)
candidate vector = q_norm_task(features)
neural residual = q_residual_score(state_bias, candidate)
```

Final candidate value:

```text
Q = normalized_prior + neural_residual
```

The highest-ranked pair is assigned.

Then:

- that worker is removed from the free-worker set;
- tasks targeting the same tile are removed;
- selection repeats until workers or tasks run out, or no pair passes the filters. There is no positive-score threshold and no explicit PASS candidate in the ranking.

Equal Q values are resolved by higher raw tree score, then shorter distance, lower worker index, and lower task index. Candidate normalization and the global-state contribution are recomputed after each assignment.

Unassigned workers:

- DROP if carrying goods while at the shed;
- otherwise PASS.

---

# 8. Learned model role

The neural network is deliberately **residual**.

That means:

```text
learned tree score
        |
        v
 normalized prior
        +
 learned correction
        |
        v
 final task ranking
```

So v20 inherits strong structural bias from the pre-existing task-scoring system.

This matters for diagnosis:

### If a useful action is never generated
The Q network cannot choose it.

### If a useful candidate exists but is ranked badly
The learned tree prior or residual Q may be responsible.

### If worker actions are good but money is still poor
The failure may instead be crop/animal planning or market logic.

---

# 9. Q evaluation optimization

v20 does not evaluate every residual candidate blindly.

It:

1. sorts candidates by the learned tree prior;
2. evaluates the current best;
3. uses a known maximum residual bound;
4. skips candidates whose best possible final Q cannot beat the current best.

Conceptually:

```text
if prior(candidate) + MAX_POSSIBLE_RESIDUAL < best_Q:
    stop evaluating lower-prior candidates
```

This is an exact pruning optimization, not a policy approximation.

---

# 10. Market-order logic

Function:

```python
market_orders(obs, animal, crops, actions)
```

This is entirely rule-based: it does not call the residual Q network. It does depend on the selected worker actions for same-turn inventory accounting, so Q choices can indirectly affect orders.

Major sequence:

### A. Account for same-turn DROP/PLACE
Goods expected to enter the shed from current worker actions are treated as sellable.

### B. SELL
v20 generally sells available goods immediately, subject to reserves.

Reserve rules include:

```text
WHEAT:
  reserve = max(4, live_animals + 2)
  except final day

FERTILIZER:
  reserve 4 on days 10–28
  reserve 0 on days 0–9 and day 29
```

### C. Opening
Day 0 / hour 0 uses a hard-coded opening purchase package including workers, wheat seeds, animals, and melon seeds.

### D. HIRE
Target hand count depends on:
- day,
- unlocked land,
- opponent style in some cases.

### E. BUY_PRODUCT WHEAT
Maintains wheat required for animals/reserve.

### F. BUY_LAND
Buys additional quadrants only within a defined day/cash window.

### G. BUY_ANIMAL
Buys toward the animal planner's desired population, with a fixed purchase deadline.

### H. BUY_SEED
Buys seeds required by the crop plan.

### I. BUY_PRODUCT FERTILIZER
Buys fertilizer when expected crop fertilizer value justifies maintaining a target inventory.

Finally:

```python
return orders[:10]
```

Potential v23 research area:
- immediate selling,
- order-slot competition,
- reserve sizing,
- hiring count,
- buying deadlines,
- market timing.

---

# 11. Opponent-style classifier

Global state:

```python
OPP_STYLE
```

Early-game public observations classify opponents into rough archetypes such as:

- TRADER
- V16
- NORMAL
- later trader subtypes

This classification affects some heuristics, including worker counts and selected early behavior.

Specifically, `V16` changes final-day harvest weights in `unit_actions`, final-day hand targets in `market_orders`, and a day-0 seed-sort condition. That seed-sort condition cannot affect the opening return at hour 0. Trader subtype labels have no downstream policy branch; the mix constants are unused, and the old opening worker-PASS override is disabled by `if False`.

It is not a learned classifier.

Potential v23 research area:
- whether these archetypes still help,
- false classification,
- replacing discrete style labels with continuous observable features.

---

# 12. Important hard-coded constants

Examples include:

```text
HERD_LIMIT
FERT_FACTOR
LABOR_COST
HERD_THRESHOLD

TRADER_MIX
NORMAL_MIX
V16_MIX

V16_FINAL_HANDS
V16_FINAL_BONUS
```

There are also:

- fixed crop/animal tables,
- fixed animal route tiles,
- crop-specific score multipliers,
- fixed date windows/deadlines,
- reserve thresholds.

The active constants are useful ablation targets because they affect behavior independently of the learned Q model. However, `TRADER_MIX`, `NORMAL_MIX`, and `V16_MIX` are defined but never read in this notebook: changing them has no effect. `V16_FINAL_HANDS` is used by `market_orders`, and `V16_FINAL_BONUS` by `unit_actions`.

---

# 13. Dependency graph

This is a simplified ranking path. Direct worker assignments bypass the ranker. `market_orders` also receives `obs`, `animal`, and `crops` directly; those edges are omitted below. The crop planner receives an unused `animal` argument and does not consume the animal planner's forecast adjustments.

```text
                         obs
                          |
          +---------------+----------------+
          |                                |
          v                                v
   opponent classifier                forecast
                                           |
                                  +--------+--------+
                                  |                 |
                                  v                 v
                            animal_plan        crop_plan
                                  \                 /
                                   \               /
                                    v             v
                                     unit_actions
                                          |
                              feasible worker-task pairs
                                          |
                    +---------------------+--------------------+
                    |                                          |
                    v                                          v
          learned_task_score prior                    residual Q MLP
                    |                                          |
                    +---------------------+--------------------+
                                          |
                                          v
                                 worker-task ranking
                                          |
                                          v
                                   worker actions
                                          |
                                          +------+
                                                 |
                                                 v
                                           market_orders
                                                 |
                                                 v
                                      final action dictionary
```

---

# 14. Where v23 should look first when a loss occurs

Use this diagnostic order.

## A. Did the economic plan make sense?

Inspect:

- crop mix,
- animal mix,
- purchase timing,
- expected demand,
- expected prices.

Functions:
- `forecast`
- `animal_plan`
- `crop_plan`

## B. Were the right tasks generated?

Inspect the feasible task list in `unit_actions`.

If an important action never exists as a candidate, changing Q weights cannot fix it.

## C. Was the right task ranked incorrectly?

Compare:

```text
learned tree prior
neural residual
final Q
selected candidate
```

This isolates learned-ranking failures.

## D. Was execution/logistics inefficient?

Inspect:

- travel,
- feeding,
- watering,
- harvest timing,
- pickup/drop,
- shed overflow,
- idle workers.

## E. Was market behavior wrong?

Inspect:

- sell timing,
- reserves,
- order-slot pressure,
- hiring,
- land,
- wheat/fertilizer purchases,
- late animal/seed purchases.

---

# 15. Highest-leverage mutation surfaces for v23

Prefer isolated changes.

### Planner layer
- demand forecast
- supply forecast
- crop scoring
- animal scoring
- late-game production cutoff

### Logistics layer
- task generation
- routing
- resource pickup
- harvest timing
- worker allocation

### Learned-ranking layer
- state features
- task features
- learned-tree-prior weight
- residual network / Q objective

### Market layer
- sell/hold
- wheat reserve
- fertilizer reserve
- hiring
- buying deadlines
- land timing

Do not change several layers simultaneously unless testing an explicit interaction.

---

# 16. What v23 should not assume

Do **not** assume:

- v20 is primarily a neural policy;
- the Q model can select arbitrary actions;
- a Q failure explains every loss;
- the forecast is accurate;
- immediate selling is optimal;
- fixed crop/animal deadlines are optimal;
- more production means more terminal money;
- static replay proves performance against an adaptive opponent.

---

# 17. Minimal mental model

If the research agent remembers only this:

```text
v20 = PLAN -> GENERATE TASKS -> RANK TASKS -> MARKET

PLAN
  forecast + crop planner + animal planner

GENERATE TASKS
  rule-based feasible worker actions

RANK TASKS
  learned tree prior + learned residual Q

MARKET
  mostly rule-based selling/buying/hiring/land logic
```

For a v20 loss, first identify **which of those four stages produced the bad decision**, then make the smallest change and verify it with static replay.

---

# 18. Difference from v19

The most important architectural change is narrow but significant:

```text
v19:
planner + feasible candidate generation
        -> learned_task_score(features)
        -> greedy worker-task ranking

v20:
same planner + same feasible candidate generation
        -> normalized v19 learned_task_score prior
        + learned residual Q(state, task)
        -> worker-task ranking
```

So v20 is **not a full redesign of v19**. It deliberately keeps the v19 planning/candidate constraints and changes the final worker-task ranking stage.

## What stayed essentially the same

v20 preserves the main v19 structure around:
- `forecast(obs)`
- animal planning
- crop planning
- fixed farm-layout assumptions / animal routes
- task generation and feasibility checks
- market-order policy
- opponent-style heuristics
- hard-coded economic constants and timing windows

This means many v19 behavioral biases remain reachable in v20.

If v19 never generates a useful task, v20's residual Q cannot invent it.

## What changed

### v19 ranking

v19 uses `learned_task_score(task_features)` as the final learned tree score for feasible worker-task assignments.

Conceptually:

```text
candidate -> task features -> learned_task_score -> greedy choice
```

### v20 ranking

v20 keeps that v19 score as a **prior**, normalizes it across the current candidate set, then adds a learned state-dependent neural correction:

```text
candidate
   |
   +-- task_features -----------------> v19 learned_task_score
   |                                      |
   |                                      v
   |                                normalized prior
   |
   +-- task features + global state -> residual Q network
                                          |
                                          v
                         normalized prior + residual
                                          |
                                          v
                                      final choice
```

This lets v20 change worker-task preferences according to broader game state while retaining v19's strong prior.

## Why residual instead of replacing v19 directly

The design preserves a known-feasible baseline and learns only a correction.

Practical consequence:

```text
v19 knowledge = prior
v20 learning  = correction
```

The residual can:
- promote a task v19 undervalues;
- demote a task v19 overvalues;
- make ranking depend on global state.

But it still operates only over v19-generated feasible candidates.

## New state awareness in v20

v19's final ranking is driven primarily by task features through `learned_task_score`.

v20 adds a global-state representation through `q_global_state(...)`, combined with normalized task features before residual scoring. v19 was not state-blind: its task features already include day, hour, worker count, and local inventory/tile information. The new 17-feature vector adds broader context such as both players' money, hand/land counts, market prices, and candidate/free-worker counts.

Therefore two similar worker-task candidates can receive different corrections depending on broader game state.

## Embedded neural model

v20 adds embedded residual-network weights to the submission.

Architecture documented by the v20 checkpoint:

```text
38 -> 64 -> 64 -> 1
```

The weights are embedded in `main.py`, decoded with Python standard-library code, so Kaggle runtime still requires no PyTorch/NumPy/filesystem checkpoint.

v19 does not have this residual MLP stage.

## New candidate-set normalization

v20 converts the v19 candidate scores into a normalized prior before adding the residual.

That means the network learns a correction relative to the alternatives available **at that decision point**, rather than simply replacing the absolute v19 score.

## New exact Q pruning

v20 also adds a runtime optimization around residual evaluation.

Candidates are ordered by v19 prior. Once:

```text
prior(candidate) + maximum_possible_residual < current_best_Q
```

lower candidates can be skipped safely.

This pruning exists because v20 has a bounded residual neural score; it is not part of the v19 greedy ranking.

## Training relationship

The v20 notebook identifies the selected model as residual Double-DQN update 9.

Its reported selection evidence was evaluated on v19 loss-case seeds from both seats. That historical result is useful for understanding why the checkpoint was chosen, but v23 should evaluate new changes against the current `loss_games_v20` corpus.

The notebook reports 21/30 wins (70.0%) and mean margin +1322.47. These are notebook-reported results, not independently reproduced by this source review.

## Diagnostic implication for v23

When comparing a v20 loss to v19-style behavior, ask:

1. **Planner/candidate failure?**
   - Both v19 and v20 inherit it.
   - Fix planner or task generation.

2. **v19 prior failure corrected by v20?**
   - Residual Q is doing useful work.
   - Preserve or strengthen the correction.

3. **v19 prior was good but v20 residual changed it badly?**
   - Investigate Q features/weights/ranking.

4. **Worker ranking is fine but result still loses?**
   - Look at market logic, production mix, timing, logistics, reserves, and selling.

Compactly:

```text
v19 = PLAN -> CANDIDATES -> PRIOR RANK

v20 = PLAN -> CANDIDATES -> PRIOR RANK + STATE-DEPENDENT RESIDUAL Q
```

This is the central v19 -> v20 difference.

## Source-review verification

The v20 notebook here and `submission_nb/kaggriculture-sub_v20.ipynb` were byte-identical when reviewed. Comparing parsed function definitions against v19 showed that `unit_actions` was the only existing function changed; six Q helper functions were added. This verifies the architectural comparison above, not the reported win rate.

A second source audit also confirmed that existing module-level assignment values were unchanged from v19, and loaded the embedded inference code to verify 16 tree functions, 21 task scales, and 6,721 network parameters. The stored network parameters are float32 bytes decoded into Python floats; inference is ordinary Python floating-point arithmetic, not FP16 tensor inference. These checks establish runtime structure, not training provenance or playing strength.

# 19. Component-to-function map

All names below refer to the notebook's embedded `main.py`. Several components are blocks within a larger function rather than standalone functions.

| Component | Corresponding functions / code | Role and boundary |
|---|---|---|
| Entrypoint and orchestration | `agent` | Classifies opponent, calls planners, assembles worker and market actions. |
| Opponent classification | `agent`, global `OPP_STYLE` | Rule-based early-game classification; no separate classifier function. |
| Economic constants and production schedules | `CROPS`, `ANIMALS`, `SHOPS`, `HERD_LIMIT`, `FERT_FACTOR`, `LABOR_COST`, `HERD_THRESHOLD` | Tables/constants used by forecast, planners, and execution. |
| Price approximation | `price` | Estimates product prices from projected market inventory. |
| Inventory aggregation | `totals` | Sums shed and worker inventory; seeds remain separate. |
| Tile access and layout | `tile`, `ROUTES`, `ANIMAL_POINTS`, `SHED` | Fixed spatial assumptions shared across planning and execution. |
| Distance and movement | `dist`, `move`, `nearest_shed` | Manhattan distance, one-step movement, and nearest shed tile. |
| Supply/demand forecast | `forecast` | Uses both farms and own inventory to return projected market inventory and demand. |
| Animal allocation and economics | `animal_plan` | Keeps route animals, allocates owned stock, evaluates purchases using a local forecast copy. |
| Crop allocation and economics | `crop_plan` | Preserves crops, applies opening layout, scores future production; ignores its `animal` argument. |
| Fertilizer benefit | `fert_value` | Used by `unit_actions` for fertilizer tasks and `market_orders` for purchase targets. |
| Forced worker actions | Initial blocks of `unit_actions` | Delivery, animal setup, and immediate resource pickups bypass learned ranking. |
| Task generation and feasibility | `unit_actions` | Creates remaining tasks and filters worker-task pairs by resources and time/distance. |
| Task features | `task_features` | Produces 21 features, including heuristic task weight. |
| Learned tree prior | `learned_task_score`, `_tree_0` through `_tree_15`, `_TREE_FUNCTIONS` | Sums 16 tree predictions inherited from v19. |
| Prior normalization | `q_normalized_prior`, `_q_clip` | Subtracts candidate-set maximum, divides by standard deviation floored at 1, clips to [-5, 5]. |
| Task-input normalization | `q_norm_task`, `_Q_TASK_SCALES`, `_q_clip` | Scales and clips the 21 task features. |
| Global-state features | `q_global_state`, `_q_clip` | Produces 17 state features, including candidate and free-worker counts. |
| Embedded network loading | Module-level `_Q_WEIGHTS_B64`, `_Q_ALL`, `_Q_W*`, `_Q_B*` | Decodes 6,721 float32 parameters using `base64` and `struct`; no external checkpoint needed at runtime. |
| Shared first-layer computation | `q_state_bias` | Computes the state contribution once per assignment decision. |
| Neural residual inference | `q_residual_score` | 38 → 64 → 64 → 1 MLP with tanh hidden layers and linear output. |
| Greedy assignment and exact pruning | Ranking loop in `unit_actions`, `_Q_RESIDUAL_MAX` | Adds normalized prior and residual, prunes by upper bound, removes selected worker and same-tile tasks. |
| Fallback worker actions | Final block of `unit_actions` | Assigns DROP at shed when carrying inventory, otherwise PASS. |
| Same-turn inventory accounting | Initial block of `market_orders` | Accounts for worker DROP and eligible PLACE actions before forming sales. |
| Selling and reserves | SELL loop in `market_orders` | Immediate sales subject to wheat/fertilizer reserves. |
| Opening market package | Day-0/hour-0 branch in `market_orders` | Returns the fixed ten-order opening package. |
| Hiring and purchases | Remaining blocks of `market_orders` | HIRE, wheat, land, animals, seeds, fertilizer; final order cap of ten. |
| Inactive mix settings | `TRADER_MIX`, `NORMAL_MIX`, `V16_MIX` | Defined only; no runtime consumer. |

