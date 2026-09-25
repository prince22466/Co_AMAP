# v23 research program

## Objective

v23 is a low-cost autonomous research system for improving the Kaggriculture v20 policy using only local evidence, deterministic analysis, and static counterfactual replay.

The scientific loop is:

```text
v20 source + 25 recorded v20 loss histories
        ↓
deterministic analysis
        ↓
Performance Analyst
        ↓
10 falsifiable improvement ideas
        ↓
Experiment Engineer
        ↓
candidate code
        ↓
isolated static replay
        ↓
durable evidence + game records
        ↓
Performance Analyst reviews the completed batch
        ↓
next 10 ideas
        ↓
repeat until the durable goal is reached
```

v23 starts from the v20 submission notebook, the v20 loss corpus, competition material, and locally generated v23 evidence.

## Current system architecture

```text
                         Python supervisor
              ┌────────────────┼────────────────┐
              │                │                │
              │                │                │
       context manager      durable state     budget/goal control
              │                │                │
              │                │                │
              ▼                ▼                ▼
    Performance Analyst   experiments.sqlite3  progress logs
    stronger reasoning          │
    read-only                   │
              │                 │
              │ 10 ideas        │
              ▼                 │
       research_ideas           │
              │                 │
              ▼                 │
    Experiment Engineer ────────┘
    cheaper execution model
              │
              │ candidate code
              ▼
      isolated replay process
              │
              │ per-game metrics + traces
              ▼
        replay evidence
              │
              └──────────────► next Analyst review
```

The supervisor, not either LLM role, owns orchestration, persistence, budgets, stopping conditions, context construction, idea assignment, and progress snapshots.

## Role contracts

### Performance Analyst

The Performance Analyst is read-only. It may use a stronger reasoning model than the Experiment Engineer.

Responsibilities:
- diagnose why v20 and recent candidates lose;
- reason across the whole policy as a coupled system;
- use deterministic analysis tools before reading raw histories;
- synthesize prior negative evidence;
- generate exactly 10 structurally distinct falsifiable ideas per batch;
- review the completed 10-idea batch before creating the next batch.

It must reason about interactions among components such as:
- `crop_plan`;
- `animal_plan`;
- worker/task ranking;
- movement and logistics;
- worker inventory;
- shed capacity;
- `market_orders`;
- hiring and purchases;
- land allocation;
- cash and market prices;
- future production.

At least 3 of every 10 ideas must be multi-component hypotheses involving 2 or more components.

A multi-component experiment is valid when the coordinated changes implement one falsifiable interaction hypothesis. It is not permission to change unrelated variables at once.

Example:

```text
animal_plan increases egg production
        +
worker policy collects faster
        +
market_orders sells immediately
        ↓
inventory/market pressure
        ↓
lower realized prices
        ↓
high production but worse final cash
```

### Experiment Engineer

The Experiment Engineer receives exactly one assigned idea at a time.

Responsibilities:
- implement only that assigned hypothesis;
- preserve coordinated changes when the hypothesis is multi-component;
- create or resume the linked experiment;
- run the smallest useful static-replay screen;
- expand replay only when the idea's promotion rule is satisfied;
- verify the canonical lineage with `idea_dossier`;
- close the experiment as `SUPPORTED`, `REJECTED`, `UNRESOLVED`, or `ERROR`.

The Engineer must not skip ahead to another queued idea or invent an untracked experiment.

### Python supervisor

The supervisor owns:
- analyst review cadence;
- 10-idea queue management;
- resuming unfinished ideas before assigning new ones;
- role-specific context packs;
- SQLite persistence;
- candidate-version binding;
- replay subprocess execution;
- goal checks;
- stagnation/progress signals;
- API budget enforcement;
- stop conditions;
- progress snapshots.

A model final answer is only a cycle boundary. It does not terminate an unmet durable goal.

## Ten-idea research batches

The durable unit of research is an idea batch.

```text
Analyst review
    ↓
ideas 1..10 persisted in SQLite
    ↓
Engineer tests idea 1
    ↓
Engineer tests idea 2
    ↓
...
    ↓
Engineer tests idea 10
    ↓
batch exhausted
    ↓
Analyst receives canonical results/lineage
    ↓
next 10 ideas
```

Each idea records:
- `idea_id`;
- title;
- hypothesis;
- causal layer;
- affected components;
- interaction hypothesis;
- predicted system effect;
- rationale;
- smallest test;
- promotion rule;
- status;
- linked experiment and conclusion.

If an Engineer cycle ends before an experiment is finished, the next cycle resumes the existing `RUNNING` idea and experiment before assigning another pending idea.

## Durable scientific lineage

Evidence must remain unambiguous.

```text
strategy_review
    ↓
idea_id
    ↓
experiment_id
    ↓
candidate path
candidate SHA-256
    ↓
replay_call_id
    ↓
episode
    ↓
game_record_path
    ↓
metrics / conclusion
```

A single experiment is bound to one candidate path and SHA-256. If the same path changes content during the experiment, replay is rejected rather than silently mixing code versions.

Per-game replay traces live under deterministic paths below:

```text
workspace/replay_records/<idea_id>/<experiment_id>/<replay_call_id>/<episode>.json
```

Both roles can inspect the same canonical evidence through `idea_dossier(idea_id)`.

## Context engineering

v23 does not rely on an ever-growing conversation transcript.

The controller builds bounded role-specific context packs.

### Analyst context pack

Contains:
- durable goal and current progress;
- current batch evidence;
- component/component-combination summaries;
- compact canonical lineage;
- context-selection policy.

Excluded by default:
- full raw history JSON;
- full replay traces;
- duplicate staged replay rows;
- old conversational chatter.

The Analyst drills down only when needed with deterministic tools.

### Engineer context pack

Contains:
- exactly one assigned idea;
- hypothesis and affected components;
- system interaction and prediction;
- rationale;
- smallest test;
- promotion rule;
- resume state / existing experiment ID;
- compact lineage for that idea;
- compact project progress.

Context sections are independently bounded so an oversized lineage section cannot destroy the rest of the structured context.

## Deterministic analysis toolkit

The Performance Analyst has read-only local Python tools. They compress evidence; they do not call the model themselves.

Core tools:
- `analyze_history_game(episode)`: summarize one recorded v20 loss and identify high-activity windows;
- `analyze_history_window(episode, start_turn, end_turn)`: detailed turn-by-turn inspection;
- `analyze_cash_flow(episode)`: observable cash-like state changes;
- `analyze_inventory_flow(episode)`: inventory/capacity/resource trajectories;
- `analyze_worker_utilization(episode)`: action utilization by transport/crop/animal/idle/admin categories;
- `cluster_loss_histories()`: deterministic behavioral-signature grouping of the loss corpus;
- `analyze_experiments(review_id=None)`: canonical experiment summaries;
- `analyze_component_effects(review_id=None)`: descriptive effects by component and component combination;
- `compare_candidate_v20(idea_id, episode)`: candidate/v20 action divergence plus measured replay outcome;
- `evaluate_hypothesis_evidence(idea_id)`: compare the original hypothesis with measured replay evidence;
- `idea_dossier(idea_id)`: canonical idea/code/replay/game-record lineage;
- `research_progress()`: compact current research state.

These tools are evidence compressors, not causal oracles. Historical correlations and component matrices are descriptive. Causal claims require controlled static replay.

Staged replay evidence is deduplicated by episode for analysis, using the latest valid replay for each episode so 1→5→25 expansion does not double-count the same game.

## Static replay contract

The corpus contract is that v20 is the losing side in `working_files/loss_games_v20`.

Static replay means:

```text
load recorded v20 loss
    ↓
recreate same initial configuration
    ↓
replace only the v20 side with candidate actions
    ↓
replay recorded opponent commands verbatim
    ↓
evolve the counterfactual state
    ↓
compare candidate outcome with original v20 outcome
```

Opponent commands are fixed; the resulting trajectory is not. After the candidate diverges, some recorded opponent commands may become invalid or become no-ops. The opponent does not adapt.

A malformed candidate, crash, precision violation, or interface violation invalidates the replay.

## Replay funnel

Replay is local and API-free, so evaluation should be much cheaper than additional reasoning calls.

Typical funnel:

```text
1 representative case
    ↓ promising
~5 representative/stratified cases
    ↓ promising
all 25 loss histories
    ↓
durable goal check
```

Representative cases may be selected using deterministic loss-history clustering. Repeated evaluation of the same episode is deduplicated in the analysis layer.

## Durable goal

For the current project target:

```text
wins >= 13
min_games_total = 25
```

A subset result such as 13/13 cannot satisfy the final goal.

Only structured replay metrics can mark a goal reached.

The controller stops only when:
- the durable goal is reached;
- the configured per-run/project API budget is exhausted;
- a genuine runtime/environment blocker is recorded.

Rejected experiments, batch completion, or model prose are not stop conditions.

`--max-turns` limits one Agents SDK cycle, not the full research horizon.

## Progress and observability

Low-level event/audit data is written through `RunLog` under:

```text
workspace/runs/<timestamp-pid>/
    config.json
    events.jsonl
    final.md
```

The controller also maintains compact operational progress:

```text
workspace/progress_latest.json
workspace/progress.jsonl
```

`progress_latest.json` is written atomically and represents the latest canonical status.

`progress.jsonl` is append-only and records progress after every autonomous cycle and at run completion.

Snapshots include:
- durable goal;
- batch counts;
- current work item;
- best current-batch result;
- experiment/replay counts;
- stagnation signal;
- latest review metadata;
- conservative cost.

## Durable storage

`workspace/experiments.sqlite3` stores scientific memory:
- runs;
- strategy reviews;
- research ideas;
- experiments;
- replay rows;
- goals.

`workspace/agent_sessions.sqlite3` stores Agents SDK conversational/tool session history.

Scientific memory and conversational memory are intentionally separate.

## Agent/replay environment isolation

OpenAI orchestration and Kaggriculture replay run in separate interpreters.

```text
.venv-agent/
    openai
    openai-agents
    agent/runtime.py
        │
        │ subprocess + JSON
        ▼
.venv-replay/
    kaggle-environments
    torch
    numpy
    replay/runner.py
```

Rules:
- the agent environment does not depend on Kaggle Environments;
- replay code does not call OpenAI APIs;
- the replay subprocess receives no API credentials;
- the runtime rejects using the same interpreter for both roles;
- generic `run_python` cannot execute model-written replay candidates;
- candidate policies execute only through `static_replay_candidate`.

## Execution contract

With `--allow-exec`, a successful research cycle must perform measured work unless budget or a genuine blocker prevents it.

The Engineer should:
1. receive one assigned idea;
2. create or resume its experiment;
3. create/select candidate code;
4. run static replay;
5. inspect measured evidence;
6. verify lineage;
7. finish the experiment.

A run with execution enabled but no valid replay evidence, or with an unfinished experiment at normal completion, is incomplete.

## Precision contract

Candidate learned floating-point computation uses FP16.

Required:
- model weights: FP16;
- floating activations/tensors: FP16;
- NumPy floating candidate arrays: FP16.

Integer IDs, indices, coordinates, counters, shapes, booleans, enums, and environment schema values keep their required types.

Static replay performs source/runtime checks and rejects obvious wider floating candidate state such as FP32/FP64/BF16 unless a narrowly scoped backend requirement is explicitly handled.

## Cost policy

The execution model and analyst model are independently configurable.

Recommended pattern:

```text
Experiment Engineer:
    cheaper model
    low reasoning

Performance Analyst:
    stronger reasoning model
    medium/high reasoning
    invoked only at batch-review boundaries
```

Python, not the LLM, performs replay, aggregation, persistence, progress tracking, and most analysis.

The project maintains a persistent API-usage ledger and applies conservative cost accounting. API keys must never be written to logs, databases, candidate subprocess arguments, or replay environment variables.

## Filesystem boundary

v23 is self-contained.

Readable research inputs:

```text
working_files/
    competition_material/
    loss_games_v20/
    submission_nb/
    reference/
```

Generated state:

```text
workspace/
    candidates/
    replay_records/
    runs/
    experiments.sqlite3
    agent_sessions.sqlite3
    progress_latest.json
    progress.jsonl
```

Tools resolve paths against the v23 root and reject traversal outside it.

## Research standard

Every useful experiment should leave enough durable evidence to answer:
1. What was observed?
2. What hypothesis was tested?
3. Which components were changed?
4. Which exact candidate code version ran?
5. Which games were evaluated?
6. What changed in the measured metrics?
7. Was the hypothesis supported, rejected, or unresolved?
8. What did this teach the next Analyst review?

Negative results are first-class evidence.
