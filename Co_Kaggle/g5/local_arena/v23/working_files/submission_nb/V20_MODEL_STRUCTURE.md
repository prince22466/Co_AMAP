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
       normalized heuristic prior + learned neural residual Q
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

The learned component is a residual action-value network used inside worker-task selection.

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

The checked-in v20 model uses the embedded update-9 weights and runs without PyTorch at submission time.

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

1. Preserve animals already placed.
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

1. Enumerate usable crop tiles not reserved for animals.
2. Preserve crops already growing.
3. Use a fixed cheap opening layout early.
4. Later score crop candidates using projected prices and production timing.
5. Reserve future projected supply after assigning a crop.

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

It generates tasks such as:

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

Each task has roughly:

```text
(target_position, operation, heuristic_weight, required_resource)
```

Workers with already-forced actions are removed first.
Remaining workers and tasks form feasible candidate pairs.

Feasibility checks include:

- required carried resource
- distance
- remaining turns in the day
- current position
- task/resource compatibility

---

# 7. Worker-task scoring

For every feasible:

```text
(worker_i, task_j)
```

v20 computes:

```text
task_features(...)
heuristic baseline = learned_task_score(features)
normalized prior = q_normalized_prior(all baselines)
global state = q_global_state(...)
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
- selection repeats until no useful worker/task pairs remain.

Unassigned workers:

- DROP if carrying goods while at the shed;
- otherwise PASS.

---

# 8. Learned model role

The neural network is deliberately **residual**.

That means:

```text
heuristic/tree score
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
The heuristic prior or residual Q may be responsible.

### If worker actions are good but money is still poor
The failure may instead be crop/animal planning or market logic.

---

# 9. Q evaluation optimization

v20 does not evaluate every residual candidate blindly.

It:

1. sorts candidates by the heuristic prior;
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

This is mostly rule-based and independent of the residual Q network.

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
  reserve 4 during much of the game
  released near end
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

These are high-value ablation targets because they affect behavior independently of the learned Q model.

---

# 13. Dependency graph

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
heuristic prior
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
- heuristic-prior weight
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
  heuristic prior + learned residual Q

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

v19 uses `learned_task_score(task_features)` as the final learned/heuristic score for feasible worker-task assignments.

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

v20 adds a global-state representation through `q_global_state(...)`, combined with normalized task features before residual scoring.

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

