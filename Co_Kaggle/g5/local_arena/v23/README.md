# v23: low-cost autonomous research agent

v23 starts as a **research harness**, not another RL algorithm.

The v1 agent lets an OpenAI model inspect the local Kaggriculture codebase, histories, and metrics, choose a small falsifiable experiment, optionally run existing Python experiments, and record what it learned. Python owns execution, filesystem boundaries, and API-budget enforcement.

## Agent loop

```text
OBSERVE local evidence
    -> HYPOTHESIZE one bottleneck
    -> DESIGN smallest falsification experiment
    -> EXECUTE existing repo code (only with --allow-exec)
    -> ANALYZE before/after evidence
    -> RECORD result + next action
```

Available model tools are intentionally small: bounded tree listing, narrow text reads, local search, JSONL metric reduction, opt-in execution of existing Python files, and writes only under `local_arena/v23/workspace/`.

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
  --task "Inspect v21 worker Q-learning and v22 selling PPO evidence. Identify the single highest-information next v23 experiment. Do not run training yet."
```

### Permit existing experiments

```bash
python local_arena/v23/research_agent.py \
  --allow-exec \
  --task "Inspect existing v21/v22 metrics first. Run only the smallest existing evaluation needed to test the strongest hypothesis, then report the result."
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
