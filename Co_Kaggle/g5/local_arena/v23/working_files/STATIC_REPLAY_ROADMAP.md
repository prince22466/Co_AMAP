# Static Replay Roadmap for v23

Purpose: help the v23 research agent understand how `example_train_v21_static_history.py` performs static replay without reading the entire training script.

## One-sentence summary

Static replay means:

```text
load a recorded v20 loss game
-> assume the lower-reward seat is v20 because this corpus contains v20 losses
-> create a fresh environment from the history's saved name/configuration/info
-> verify that replaying both recorded action streams exactly reproduces the saved history
-> replace only the inferred v20 seat with a candidate policy
-> keep submitting the opponent's recorded action commands verbatim
-> step the new counterfactual environment forward
-> compare the candidate's final reward/margin against the original v20 result
```

The opponent is **not adaptive**. This is a counterfactual replay of one historical trajectory.

---

# 1. Core static-replay execution path

The minimum conceptual path is:

```text
history JSON
   |
   v
_load_history(path)
   |
   +--> _saved_final_rewards(history)
   |
   +--> _infer_v20_seat(history)
   |
   v
_environment_from_history(history)
   |
   v
for each replay step:
   |
   +--> _agent_observation(env, candidate_seat)
   |
   +--> candidate_action = candidate(obs)
   |
   +--> _recorded_step_actions(history, replay_step)
   |
   +--> opponent_action = recorded_actions[opponent_seat]
   |
   +--> env.step([candidate_action, opponent_action])
   |
   v
final states
   |
   +--> rewards
   +--> margin
   +--> WIN / LOSS / TIE
```

For v23, this is the important part. Most of the rest of the file is v21 training machinery.

---

# 2. The single most important function

## `run_static_episode(...)`

This is the v21 training script's reference implementation for one **Q-controller candidate** vs recorded-opponent replay.

For a generic notebook/Python policy replacement, `_run_submission_replacement(...)` from the imported evaluation helper is actually the simpler pattern.

High-level logic:

```python
history = _load_history(history_path)

candidate_seat = _infer_v20_seat(history)
opponent_seat = 1 - candidate_seat

original_rewards = _saved_final_rewards(history)

env = _environment_from_history(history)

for replay_step in range(1, len(history["steps"])):
    obs = _agent_observation(env, candidate_seat)

    candidate_action = controller(obs)

    recorded_actions = _recorded_step_actions(history, replay_step)
    opponent_action = recorded_actions[opponent_seat]

    actions = [None, None]
    actions[candidate_seat] = candidate_action
    actions[opponent_seat] = opponent_action

    env.step(actions)

final_rewards = ...
margin = candidate_reward - opponent_reward
```

That loop is the essence of static replay.

---

# 3. Minimal helper set that matters to v23

Several replay helpers used by this example are **imported**, not defined in this file. The current example retains legacy dependencies on sibling v20/v21 modules. Under the v23-only filesystem design, treat this script as a reference implementation and copy/reimplement the required replay helpers locally rather than depending on those external paths.


The v23 agent should care primarily about these helpers.

## History / seat helpers

### `_load_history(path)`

Loads one replay JSON and validates that it contains replay steps.

### `_saved_final_rewards(history)`

Reads the original terminal rewards from the recorded game.

### `_infer_v20_seat(history)`

Infers the candidate seat by selecting the lower-reward seat.

Current logic:

```text
if reward[0] < reward[1]:
    candidate/v20 seat = 0
else:
    candidate/v20 seat = 1
```

Important: the function does **not** inspect metadata proving that this seat ran v20. It relies on the dataset contract that `loss_games_v20` contains games lost by v20. Tied histories are rejected because this convention cannot identify a unique losing seat.

---

## Environment reconstruction

### `_environment_from_history(history)`

Creates a **fresh Kaggriculture environment** using the history's saved game name, `configuration`, and `info`.

It does **not** restore an arbitrary serialized simulator state from a later turn. Replay starts from the newly created episode's initial state and advances by actions.

This is why `recorded_action_parity(history)` is mandatory: it proves that, under the current engine, recreating the episode and replaying the recorded actions reproduces the saved observations/rewards/statuses exactly.

---

## Observation extraction

### `_agent_observation(env, candidate_seat)`

Extracts the observation exactly as the replacement candidate should see it.

Use the candidate seat, not a generic full environment state.

---

## Recorded-action extraction

### `_recorded_step_actions(history, replay_step)`

Returns the two players' actions recorded for that historical step.

Static replay uses:

```python
opponent_action = recorded_actions[opponent_seat]
```

The candidate's original v20 action is ignored except for parity/debug comparisons.

---

## Replay validation

### `recorded_action_parity(history)`

Checks that replaying the recorded action stream reproduces the historical game exactly.

This is a trust gate.

If parity fails, do not use that replay as evidence.

---

# 4. Candidate replacement logic

The recorded loss contains two historical players:

```text
seat 0
seat 1
```

The replay identifies the losing v20 seat:

```text
candidate_seat = losing v20 seat
opponent_seat = other seat
```

Then each turn becomes:

```text
candidate seat:
    NEW candidate action

opponent seat:
    ORIGINAL recorded action
```

So only one side changes.

This isolates the effect of replacing v20 under the opponent behavior that actually occurred.

---

# 5. What remains fixed

During one static replay, keep these inputs fixed:

- saved episode name/configuration/info;
- inferred candidate/opponent seats;
- the opponent's recorded **action commands** at each replay step;
- game rules / engine version;
- replay length.

The counterfactual **state trajectory is not fixed**. Once the candidate takes a different action, later observations, rewards, inventories, prices, positions, etc. may diverge from the recorded history.

The opponent still receives no new decision-making: the harness continues submitting its historical action commands. Those commands may have different effects—or even become invalid/no-ops—in the changed counterfactual state.

---

# 6. What static replay does NOT mean

Do not interpret static replay as:

- a rematch;
- self-play;
- an adaptive opponent evaluation;
- proof of Kaggle leaderboard improvement;
- proof that the opponent would make the same decisions after the candidate changes the game state.

The opponent action **commands** are replayed verbatim even after the candidate changes the world state. The opponent is never re-queried for a state-appropriate response.

Therefore a recorded command may be strategically nonsensical or have a different effect in the counterfactual state. That is the main limitation.

---

# 7. Why static replay is still useful

It is useful for answering:

```text
"Given the opponent behavior that beat v20,
would this candidate have handled that trajectory better?"
```

This is excellent for:

- repairing known failure modes;
- testing isolated policy changes;
- comparing candidate-vs-v20 margin;
- finding the first action divergence;
- checking whether a hypothesis improves the known loss corpus.

---

# 8. Preflight / parity gate

Before trusting candidate results, validate the replay machinery.

The reference script's `_preflight(...)` checks:

1. v20 model format;
2. recorded-action replay parity;
3. checked-in v20 submission parity.

For v23, the key principle is:

```text
recorded history replay
   must reproduce
recorded final result
```

and ideally:

```text
running original v20 as replacement
   must reproduce
the original v20 actions and rewards
```

If not, the static-replay harness is wrong or incompatible.

---

# 9. Useful v23 baseline test

Before testing a modified candidate, run the checked-in original v20 notebook through the generic submission-replacement path.

Expected:

- zero action divergences from the recorded v20 action stream;
- terminal statuses `DONE/DONE`;
- exactly the recorded terminal rewards;
- therefore the same result and margin.

The reference `_preflight(...)` performs this v20-submission parity check on one selected history, in addition to full recorded-action parity. For a new v23 harness, it is safer to establish parity across every history before treating the whole corpus as valid evidence.

---

# 10. Action-divergence tracking

The reference script optionally compares:

```text
candidate_action
vs
recorded_v20_action
```

and records:

- total action divergences;
- first divergent replay step;
- day;
- hour;
- recorded v20 action;
- candidate action.

This is extremely useful for v23.

When a candidate improves or worsens a game, inspect the **first divergence** first.

That often identifies the causal policy change much faster than reading the entire replay.

---

# 11. Final result metrics

The core result from static replay is:

```text
candidate_margin
  = candidate_reward - opponent_reward
```

For v23 comparison also compute:

```text
original_v20_margin
margin_improvement
  = candidate_margin - original_v20_margin
```

Useful corpus metrics:

- wins;
- ties;
- losses;
- repaired losses;
- repair rate;
- margin-improved cases;
- margin-worsened cases;
- mean candidate margin;
- mean margin improvement;
- worst candidate margin;
- first action divergence.

Do not judge only by win count.

---

# 12. Functions related to training, not required for basic v23 replay

The example file contains many functions/classes that exist because v21 was training a Q model.

For simple candidate static replay, the v23 agent usually does **not** need to understand:

- `ReplayBuffer`
- `Transition`
- `DecisionRecord`
- `q_update_v21(...)`
- `build_worker_credit_transitions(...)`
- `_sample_stratified_replay(...)`
- target-network synchronization
- optimizer setup
- epsilon schedules
- checkpoint saving/loading
- FP16/FP8 quantization
- replay warmup
- SGD loop

These are training infrastructure, not the core replay protocol.

---

# 13. Functions/classes useful only if v23 studies worker-Q training

The following matter only when researching learned worker allocation:

- `TreeFreeQSelector`
- `TreeFreeQController`
- `DenseWorkerRewardTracker`
- `build_worker_credit_transitions`
- `q_update_v21`

They add:

- candidate feature extraction;
- Q-based worker-task selection;
- exploration;
- dense production/transport/delivery reward;
- per-worker credit assignment;
- replay-buffer training.

Do not confuse these with static replay itself.

---

# 14. Reward tracker is optional for policy evaluation

The reference `run_static_episode` also computes:

- produced value;
- transport progress value;
- delivered value;
- worker reward value.

These were designed for v21 training diagnostics.

For v23 candidate evaluation, terminal money/margin remains the primary game objective.

Use the dense metrics only to explain *why* a candidate changed performance.

Example:

```text
candidate margin improved
+ delivered value improved
+ produced value unchanged

=> likely logistics/delivery improvement rather than production increase
```

---

# 15. `evaluate_histories(...)`

This is a batch wrapper around `run_static_episode`.

Conceptually:

```python
for history in histories:
    result = run_static_episode(
        history,
        deterministic=True,
        collect=False,
    )

aggregate:
    W / T / L / errors
    mean_margin
    mean_produced_value
    mean_delivered_value
```

For v23, an equivalent batch evaluator is useful, but training state/model mode is not inherently required.

---

# 16. `_history_paths(...)` and `_split_histories(...)`

### `_history_paths(history_dir)`

Simply lists replay JSON files.

Useful.

### `_split_histories(...)`

Creates train/validation splits.

This was needed for v21 learning.

For v23 rule/policy research, use it only if the agent starts tuning repeatedly on a subset and needs held-out replay cases.

Otherwise, evaluate the whole v20 loss corpus after a candidate survives small-stage tests.

---

# 17. Recommended v23 evaluation funnel

To save compute and API budget:

```text
candidate
   |
   v
syntax / import / agent-contract check
   |
   v
original-v20 parity sanity check
   |
   v
1 representative loss
   |
   v
3 diverse losses
   |
   v
all v20 loss cases
```

Promote only if evidence survives each stage.

---

# 18. Recommended v23 static-replay API

The research agent should think in terms of:

```text
static_replay_candidate(
    candidate,
    episodes,
    max_episodes
)
```

Internally that should perform:

```text
for each selected history:
    load history
    check parity
    infer losing v20 seat
    rebuild environment
    load candidate agent
    replay recorded opponent
    run to terminal
    compare candidate margin to original v20 margin
    return structured result
```

The research agent should not need to call low-level helpers directly.

---

# 19. Minimal pseudocode for a generic v23 replay

```python
def replay(history_path, candidate_agent):
    history = load_history(history_path)

    assert recorded_action_parity(history)

    original_rewards = saved_final_rewards(history)

    candidate_seat = infer_losing_v20_seat(history)
    opponent_seat = 1 - candidate_seat

    env = environment_from_history(history)

    divergences = []

    for step in range(1, len(history["steps"])):
        obs = agent_observation(env, candidate_seat)

        new_action = candidate_agent(obs)

        recorded = recorded_step_actions(history, step)
        opponent_action = recorded[opponent_seat]
        old_v20_action = recorded[candidate_seat]

        if new_action != old_v20_action:
            divergences.append(step)

        actions = [None, None]
        actions[candidate_seat] = new_action
        actions[opponent_seat] = opponent_action

        env.step(actions)

    rewards = final_rewards(env.steps[-1])

    return {
        "candidate_margin":
            rewards[candidate_seat] - rewards[opponent_seat],
        "original_v20_margin":
            original_rewards[candidate_seat] -
            original_rewards[opponent_seat],
        "first_divergence":
            divergences[0] if divergences else None,
    }
```

That is the static replay mechanism v23 should remember.

---

# 20. Diagnostic workflow after a replay

When a candidate changes a result:

```text
1. find first action divergence
2. inspect state at that turn
3. identify changed policy rule/module
4. follow downstream state differences
5. compare terminal margin
6. check other loss cases for regressions
```

Do not inspect all 720 turns first.

Start at the first causal divergence.

---

# 21. Most important distinction for the agent

```text
STATIC REPLAY ENGINE
    history reconstruction
    + recorded opponent actions
    + candidate replacement
    + final comparison

TRAINING SYSTEM
    Q-network
    + dense reward
    + replay buffer
    + optimizer
    + checkpoints
```

For v23's auto-research loop, the first block is mandatory.

The second block is optional and should only be used when the research hypothesis specifically requires learning.

---

# 22. Minimal mental model

If the agent remembers only this:

```text
STATIC REPLAY =
REBUILD OLD LOSS
+ REPLACE V20
+ FREEZE OPPONENT ACTIONS
+ RUN FORWARD
+ COMPARE MARGIN
```

And always require parity before trusting the result.
