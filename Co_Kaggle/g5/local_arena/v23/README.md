# v23: low-cost autonomous Kaggriculture research system

v23 is an autonomous research harness for improving the v20 Kaggriculture policy. It is not tied to a specific RL algorithm.

The system uses a stronger read-only **Performance Analyst** to diagnose failures and create research ideas, a cheaper **Experiment Engineer** to implement/test those ideas, and a deterministic Python supervisor to control context, replay, persistence, budgets, and stopping conditions.

## Architecture

```text
                 ┌──────────────────────────┐
                 │     Python supervisor    │
                 │ context / queue / budget │
                 │ goals / persistence      │
                 └─────────────┬────────────┘
                               │
                  bounded Analyst context
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Performance Analyst    │
                 │ stronger reasoning model │
                 │ read-only                │
                 └─────────────┬────────────┘
                               │
                     exactly 10 ideas
                               │
                               ▼
                 ┌──────────────────────────┐
                 │     research_ideas       │
                 │   experiments.sqlite3    │
                 └─────────────┬────────────┘
                               │
                   one assigned idea/cycle
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Experiment Engineer    │
                 │ cheaper execution model  │
                 └─────────────┬────────────┘
                               │
                        candidate code
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ isolated static replay   │
                 │ .venv-replay             │
                 └─────────────┬────────────┘
                               │
                   metrics + game records
                               │
                               ▼
                      durable evidence
                               │
                               └────► next Analyst review
```

The current durable target is:

```text
wins >= 13
min_games_total = 25
```

so a subset such as 13/13 cannot prematurely satisfy the goal.

## Research cycle

The Analyst generates exactly 10 structurally distinct ideas. At least 3 must be multi-component system hypotheses.

The Engineer then consumes the batch one idea at a time:

```text
Analyst review
    ↓
10 ideas persisted
    ↓
idea 1 → code → replay → result
idea 2 → code → replay → result
...
idea 10 → code → replay → result
    ↓
Analyst reviews the completed batch
    ↓
next 10 ideas
```

If an Engineer cycle ends before finishing an experiment, the controller resumes that same `RUNNING` idea/experiment before assigning a new one.

The loop stops only when:
- the durable replay goal is reached;
- API budget is exhausted;
- a genuine runtime/environment blocker is reported.

A rejected candidate or normal model final answer does not stop the loop.

## Project layout

```text
v23/
├── research_agent.py
├── PROGRAM.md
├── README.md
├── requirements-agent.txt
├── requirements-replay.txt
├── setup_envs.py
├── agent/
│   ├── runtime.py       # supervisor + Agents SDK orchestration
│   ├── support.py       # filesystem, logs, budget utilities
│   ├── analysis.py      # deterministic performance-analysis toolkit
│   └── context.py       # role-specific context + progress snapshots
├── replay/
│   ├── core.py
│   └── runner.py        # isolated candidate/static-replay worker
├── tests/
│   ├── smoke_runtime.py
│   └── smoke_fp16.py
├── working_files/
│   ├── competition_material/
│   ├── loss_games_v20/
│   ├── submission_nb/
│   └── reference/
└── workspace/           # generated, gitignored
```

## Context engineering

The agents do not receive the full research history every cycle.

### Performance Analyst context

The Analyst gets a bounded pack containing:
- durable goal/current progress;
- current batch evidence;
- component-effect summaries;
- compact canonical lineage.

Raw histories and full replay traces are excluded by default. The Analyst uses deterministic tools to drill down only when needed.

### Experiment Engineer context

The Engineer gets:
- one assigned `idea_id`;
- hypothesis;
- affected components;
- interaction hypothesis;
- predicted system effect;
- rationale;
- smallest test;
- promotion rule;
- existing experiment ID if resuming;
- compact lineage;
- compact progress.

This prevents the execution model from being distracted by unrelated ideas.

## Performance Analyst tools

The Analyst is read-only and has deterministic local analysis tools including:

```text
research_progress()
analyze_history_game(episode)
analyze_history_window(episode, start_turn, end_turn)
analyze_cash_flow(episode)
analyze_inventory_flow(episode)
analyze_worker_utilization(episode)
cluster_loss_histories()
analyze_experiments(review_id=None)
analyze_component_effects(review_id=None)
compare_candidate_v20(idea_id, episode)
evaluate_hypothesis_evidence(idea_id)
idea_dossier(idea_id)
```

Python performs the bookkeeping and aggregation. The reasoning model interprets the compact evidence.

These tools are descriptive evidence compressors; causal claims still require controlled static replay.

## Canonical lineage

Every tested idea has an explicit evidence chain:

```text
idea_id
  ↓
experiment_id
  ↓
candidate path + SHA-256
  ↓
replay_call_id
  ↓
episode
  ↓
game_record_path
  ↓
metrics / conclusion
```

Candidate content is immutable within one experiment. Reusing the same path with different bytes is rejected.

Per-game replay records are written below:

```text
workspace/replay_records/<idea_id>/<experiment_id>/<replay_call_id>/<episode>.json
```

## Static replay semantics

For each recorded v20 loss:

```text
same initial configuration
    ↓
replace the losing v20 side with candidate actions
    ↓
replay the recorded opponent commands verbatim
    ↓
evolve the new counterfactual state
    ↓
compare candidate result with original v20 result
```

The opponent commands are fixed, but the trajectory is not. Once the candidate diverges, later recorded opponent commands can become invalid/no-ops; the opponent does not adapt.

Replay is local and API-free.

Typical evaluation funnel:

```text
1 case
  ↓ promising
~5 representative/stratified cases
  ↓ promising
all 25 histories
```

The analysis layer counts the latest valid replay per episode so staged 1→5→25 evaluation does not double-weight repeated games.

## Progress monitoring

The low-level audit log remains under:

```text
workspace/runs/<timestamp-pid>/
    config.json
    events.jsonl
    final.md
```

The compact operational status is:

```text
workspace/progress_latest.json
workspace/progress.jsonl
```

`progress_latest.json` is atomically replaced with the current canonical snapshot.

`progress.jsonl` is append-only and gets a snapshot after every autonomous cycle and at final completion.

Useful commands on Vertex:

```bash
cat workspace/progress_latest.json
tail -f workspace/progress.jsonl
```

Snapshots include goal status, current batch counts, current idea/experiment, best batch result, replay/experiment totals, stagnation signal, latest Analyst review, and conservative cost.

## Persistent state

Scientific memory:

```text
workspace/experiments.sqlite3
```

contains:
- runs;
- goals;
- strategy reviews;
- research ideas;
- experiments;
- replay rows.

Agents SDK session history:

```text
workspace/agent_sessions.sqlite3
```

Conversation/tool history and scientific evidence are intentionally separate.

## Environment isolation

v23 uses separate Python environments:

```text
.venv-agent/
    openai
    openai-agents
    agent runtime
        │
        │ subprocess + JSON
        ▼
.venv-replay/
    kaggle-environments
    torch
    numpy
    replay worker
```

Create both:

```bash
python setup_envs.py
```

Smoke tests:

```bash
.venv-agent/bin/python tests/smoke_runtime.py
.venv-replay/bin/python tests/smoke_fp16.py
```

Expected:

```text
v23 infrastructure smoke test: OK
v23 FP16 enforcement smoke test: OK
```

## Recommended run

A strong reasoning model can be reserved for the Analyst while the cheaper model handles implementation/execution:

```bash
.venv-agent/bin/python research_agent.py \
  --model gpt-6-luna \
  --reasoning-effort low \
  --analyst-model gpt-6-sol \
  --analyst-reasoning-effort medium \
  --allow-exec \
  --task "Reach at least 13 wins across all 25 v20 loss histories."
```

The model names above are the model strings configured by this project.

The API key can be provided with `--api-key` or `OPENAI_API_KEY`. Do not commit it. CLI arguments may appear in shell history/process listings.

Replay subprocesses strip secret/token/password/credential environment variables.

## Budget behavior

Python tracks API usage in the persistent project ledger. The Analyst and Engineer can use different model prices.

The key cost principle is:

```text
expensive reasoning
    only at high-value Analyst review boundaries

cheap execution model
    for candidate implementation

local Python
    for replay, aggregation, context construction,
    analysis tools, persistence, and progress tracking
```

This keeps API calls concentrated on hypothesis quality rather than mechanical work.

## FP16 candidate contract

Candidate learned floating-point computation is FP16 by default and enforced in replay.

The replay process checks candidate source/runtime state for obvious wider floating dtypes and sets PyTorch default floating dtype to FP16 before candidate import.

Integer IDs, coordinates, counters, indices, shapes, booleans, enums, and environment-required schema values remain in their required types.

## Filesystem boundary

v23 is self-contained. Tools resolve paths relative to `local_arena/v23` and reject traversal outside that root.

Research inputs:

```text
working_files/submission_nb/kaggriculture-sub_v20.ipynb
working_files/loss_games_v20/*.json
working_files/competition_material/*
working_files/reference/*
```

Generated artifacts stay under `workspace/`.

For behavioral and scientific invariants, see [PROGRAM.md](PROGRAM.md).
