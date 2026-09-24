# v23: low-cost autonomous research agent

v23 starts as a **research harness**, not another RL algorithm.

The v1 agent starts directly from `working_files/submission_nb/kaggriculture-sub_v20.ipynb` and `working_files/loss_games_v20`. It diagnoses v20, proposes a candidate, and evaluates that candidate by replacing the losing v20 seat while replaying the recorded opponent actions verbatim. Python owns execution, filesystem boundaries, and API-budget enforcement.

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

v1 never executes a Python file written by the model and removes `OPENAI_API_KEY` from child-process environments.

## GCP Vertex AI Workbench

From `Co_Kaggle/g5`:

```bash
python -m pip install -r local_arena/v23/requirements.txt
export OPENAI_API_KEY="..."
```

Do not commit the key. For a persistent Workbench deployment, inject it through your normal secret-management mechanism instead of storing it in a notebook.

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

The estimate conservatively prices all input tokens at the uncached rate. Known GPT-6 prices are built in; an unknown model requires explicit CLI token prices so the budget guard cannot silently undercount.

The ledger only tracks this program. It cannot know API spend made by other programs or projects.

Each run creates `workspace/runs/<UTC timestamp>-<pid>/config.json`, `events.jsonl`, and `final.md`.

## v1 boundaries

v1 intentionally does not modify v20/v21/v22, run model-written code, use external web search, call an expensive fallback model automatically, launch broad training before inspecting metrics, or claim hidden/Kaggle improvement without evidence.

After reviewing v1 behavior, the natural v2 is a controlled candidate-patch + local A/B evaluation workflow.


## Self-contained v23 boundary

The agent's readable/executable root is `local_arena/v23` itself. Tool paths are resolved relative to this directory and rejected if they escape it.

Research inputs are under `working_files/`:
- `submission_nb/kaggriculture-sub_v20.ipynb` — primary baseline
- `submission_nb/kaggriculture-sub_v19.ipynb` — optional reference baseline
- `loss_games_v20/*.json` — failure corpus
- `competition_material/*` — rules/domain material
- `example_train_v21_static_history.py` — local static-replay implementation reference

The static replay tool loads helpers from the copied `working_files/example_train_v21_static_history.py`; it no longer imports from `v20_rl`, `v21_rl`, `v22_rl`, `game_history`, or any other sibling directory.

Generated research artifacts remain confined to `workspace/`.
