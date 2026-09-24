# v23: low-cost autonomous research agent v2

v23 starts as a **research harness**, not another RL algorithm.

The agent starts directly from `working_files/submission_nb/kaggriculture-sub_v20.ipynb` and `working_files/loss_games_v20`. It diagnoses v20, proposes a candidate, and evaluates that candidate by replacing the losing v20 seat while replaying the recorded opponent actions verbatim. Python owns execution, filesystem boundaries, and API-budget enforcement.

## Agent loop

```text
OBSERVE local evidence
    -> HYPOTHESIZE one bottleneck
    -> DESIGN smallest falsification experiment
    -> EXECUTE existing repo code (only with --allow-exec)
    -> ANALYZE before/after evidence
    -> RECORD result + next action
```

Available model tools are intentionally small: bounded tree listing, narrow text reads, local search, JSONL metric reduction, a dedicated `static_replay_candidate` evaluator, opt-in execution of existing Python files, and writes only under `local_arena/v23/workspace/`.

Generic `run_python` still refuses model-written workspace files and strips `OPENAI_API_KEY` from child processes. Candidate code is executable only through the dedicated static-replay path.

## GCP Vertex AI Workbench

From `Co_Kaggle/g5`:

```bash
python -m pip install -r local_arena/v23/requirements.txt
export OPENAI_API_KEY="..."
```

Do not commit the key. For a persistent Workbench deployment, inject it through your normal secret-management mechanism instead of storing it in a notebook.

### No-API smoke test

After installing requirements, validate SDK imports plus the local run/experiment/replay/goal SQLite lifecycle without consuming API credit:

```bash
python local_arena/v23/smoke_test_v2.py
```

Expected output:

```text
v23 infrastructure smoke test: OK
```


### First read-only run

```bash
python local_arena/v23/research_agent.py \
  --task "Inspect the v20 notebook and its recorded loss cases. Diagnose one concrete failure mechanism and design the smallest candidate change to test it with static replay."
```

### Permit existing experiments

```bash
python local_arena/v23/research_agent.py \
  --allow-exec \
  --task "Start from v20 and its loss cases. Build or select one candidate agent, evaluate it with static replacement replay on the smallest useful subset, then report repaired, improved, and worsened cases."
```

## Budget controls

Defaults:

```text
model                  gpt-6-luna
reasoning effort       low
project ledger cap     $5.00
per-run cap            $0.25
max API turns          12
max output / turn      2500 tokens
```

The project cap intentionally leaves roughly $1 of a $6 credit balance outside this autonomous agent. Usage is written to `local_arena/v23/.agent_usage.json`.

The v2 estimate distinguishes uncached input, cached input, cache-write tokens, and output. Cache reads use 0.10× input price, cache writes use 1.25× input price, then a 10% safety multiplier is applied. Known GPT-6 Standard-tier prices are built in; an unknown model requires explicit CLI token prices.

The ledger only tracks this program. It cannot know API spend made by other programs or projects.

Each run creates `workspace/runs/<UTC timestamp>-<pid>/config.json`, `events.jsonl`, and `final.md`.

## v2 infrastructure

v2 uses the OpenAI Agents SDK over the Responses API. It keeps one research agent and adds:
- persistent conversation memory with `SQLiteSession`;
- built-in SDK tracing for model/tool spans;
- durable research memory in `workspace/experiments.sqlite3`;
- run, experiment, and replay-case timing;
- structured experiment IDs and hypothesis status;
- explicit numeric goal tracking;
- request/input/output/total token accounting;
- cached-input, cache-write, and reasoning-token accounting when returned by the API;
- conservative local cost accounting that still prices all input as uncached;
- persistent project summaries through the `project_status` tool.

The durable hierarchy is:

```text
PROJECT
  -> RUN
      -> EXPERIMENT
          -> STATIC REPLAY CALL
              -> REPLAY CASE
```

This makes time-to-goal, experiments-to-goal, replay-cases-to-goal, API requests, tokens, cache utilization, and cost queryable instead of inferred from prose.

The Agents SDK tracing exporter is enabled by default, but trace payloads are configured with `trace_include_sensitive_data=False`. Use `--disable-tracing` to disable tracing entirely.

Conversation history is persisted in `workspace/agent_sessions.sqlite3`. By default only the most recent 80 session items are retrieved for a run; change this with `--session-history-limit`.

The model path explicitly enables OpenAI implicit prompt caching with a 30-minute TTL. Stable instructions/tool definitions and repeated session prefixes can therefore be reused by the API. `cached_tokens` and `cache_write_tokens` are recorded separately and included in the budget estimate using their distinct Standard-tier multipliers.

Negative experiment conclusions are stored in the experiment DB so the agent can avoid re-testing rejected ideas.


## Self-contained v23 boundary

The agent's readable/executable root is `local_arena/v23` itself. Tool paths are resolved relative to this directory and rejected if they escape it.

Research inputs are under `working_files/`:
- `submission_nb/kaggriculture-sub_v20.ipynb` — primary baseline
- `submission_nb/kaggriculture-sub_v19.ipynb` — optional reference baseline
- `loss_games_v20/*.json` — failure corpus
- `competition_material/*` — rules/domain material
- `example_train_v21_static_history.py` — local static-replay implementation reference

Runtime replay now uses the self-contained `static_replay.py` helper inside v23. `working_files/example_train_v21_static_history.py` remains reference-only; its legacy sibling imports are never runtime dependencies.

Generated research artifacts remain confined to `workspace/`.


## Execution and FP16 enforcement

Execution-enabled runs are expected to do work, not only produce plans. When `--allow-exec` is present, a successful run must execute at least one static replay. If the agent exits normally without any replay call, the runtime marks the run `INCOMPLETE`.

Candidate floating-point computation follows the FP16 contract:
- PyTorch default floating dtype is set to `torch.float16` in the isolated replay process before candidate import;
- candidate source is audited for obvious explicit wider floating dtypes such as FP32, FP64, double, and BF16;
- loaded global PyTorch tensors/modules and NumPy floating arrays are audited for non-FP16 state;
- persistent non-FP16 floating runtime state created during replay causes the candidate replay to fail.

Integer indices, IDs, coordinates, counters, shapes, booleans, action schemas, and environment-required non-floating values remain in their required types. Backend operations that genuinely cannot execute in FP16 are exceptional and must be narrowly scoped and documented.
