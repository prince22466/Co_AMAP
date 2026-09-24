# v23 research program

## Objective

Build an autonomous, evidence-driven research loop for improving the Kaggriculture agent while keeping experimentation reproducible and OpenAI API spend extremely low.

v23 should first understand the existing chain:

```text
v20 notebook + working_files/loss_games_v20
    -> diagnose a concrete failure
    -> candidate replacement policy
    -> static replay with recorded opponent actions
    -> compare candidate vs recorded v20 margins
    -> next hypothesis
```

v23 is an orchestration/research layer built directly on v20. v21/v22 are not research inputs; only their generic replay implementation patterns may be reused where useful.

## Research contract

Every useful v23 run should answer:

1. What was observed?
2. What single falsifiable hypothesis was selected?
3. Why was this experiment higher-information than alternatives?
4. What exact command/configuration was executed, if any?
5. What changed in the measured metrics?
6. Was the hypothesis supported, rejected, or unresolved?
7. What is the smallest justified next experiment?

Negative results are first-class results.

## Experimental rules

- Keep v20/v21/v22 source and checkpoints immutable.
- Inspect existing metrics before new training.
- Prefer evaluation/ablation before adding model complexity.
- Change one conceptual variable at a time unless an interaction is the hypothesis.
- Keep train/validation separation explicit.
- Distinguish static-history counterfactual performance from live/adaptive-opponent performance.
- Preserve commands, seeds, configs, and metric paths in the run record.
- Never infer hidden-score improvement from local proxy metrics alone.
- Avoid external/public solution lookup in v23; use our own code, histories, and measurements.

## Cost policy

The agent model is selected at startup with `--model`. The OpenAI API key may be supplied with `--api-key` or, as a fallback, `OPENAI_API_KEY`. The key itself must never be written to run logs, config files, experiment databases, or replay-process arguments.

Available OpenAI API credit is approximately $6. v23 uses `gpt-6-luna` with low reasoning effort and a persistent local ledger. The default project ceiling is $5.00 with a $0.25 per-run ceiling. There is no automatic escalation to Sol/Astra.

## Success criteria

A bounded run should locate relevant prior evidence without dumping the repository into context, state a concrete hypothesis, identify or execute a small experiment, separate evidence from conjecture, write a concise research record, and stay well below budget.

Candidate changes are accepted only through measured static-replay evidence and recorded experiment conclusions.


## Filesystem boundary

v23 is self-contained. The research agent must not inspect or depend on sibling directories. Its complete input surface is `working_files/`, and its generated research surface is `workspace/`. All tool paths are resolved against the v23 directory and traversal outside that root is rejected.


## v2 agent infrastructure contract

The production research harness should make progress measurable. Every run records:
- wall-clock duration;
- model request count;
- input/output/total tokens;
- cached-input and cache-write tokens when available;
- reasoning tokens when available;
- conservative API-cost estimate;
- experiment count;
- replay call count and replay-case count.

Every experiment has a durable ID and records:
- hypothesis;
- candidate and optional parent candidate;
- start/end time;
- static replay cases;
- mean/best margin improvement;
- wins/losses/regressions;
- conclusion status: SUPPORTED, REJECTED, UNRESOLVED, or ERROR.

Explicit numeric goals are stored separately. When a replay summary satisfies the active goal (and any regression guard), the system records the exact run and experiment that first reached it. This supports direct calculation of time-to-goal, experiments-to-goal, replay-cases-to-goal, tokens-to-goal, and cost-to-goal.

Conversation memory and research memory are separate:
- OpenAI Agents SDK SQLiteSession stores conversational/tool context across invocations.
- experiments.sqlite3 stores compact scientific evidence and negative results so the model does not need to replay the entire conversation history.

Tracing is provided by the OpenAI Agents SDK. Domain-specific research semantics remain in the local SQLite database.




## Agent/replay isolation contract

OpenAI orchestration and Kaggriculture replay are independent systems.

- The **agent environment** contains the OpenAI SDK and OpenAI Agents SDK. It performs reasoning, planning, candidate generation, experiment selection, memory, tracing, and result analysis.
- The **replay environment** contains Kaggle Environments, PyTorch, and NumPy. It reconstructs games, runs candidate actions, replays recorded opponent commands, and emits structured metrics.
- v23 replay code must not import or call OpenAI APIs. Kaggle may carry unrelated transitive OpenAI/LiteLLM packages internally; those are not part of the replay interface.
- The agent environment must not install, import, or depend on Kaggle Environments.
- Communication across the boundary is subprocess arguments plus JSON output.
- The agent runtime must use a replay interpreter different from its own interpreter.
- Kaggle/LiteLLM dependency constraints must never determine the OpenAI Agents SDK version.

This separation is architectural, not optional dependency management.

## Autonomous goal loop

v23 follows a cheap Karpathy-style autoresearch loop: the model chooses hypotheses and candidate changes, while local Python owns execution, replay, scoring, persistence, and termination.

For an execution-enabled run with a durable numeric goal, model-generated final prose is only a cycle boundary. After every Agents SDK cycle, Python checks the durable goal and resumes the same SQLite session when the goal remains unmet.

The controller stops only when:
- measured replay evidence marks the durable goal reached;
- the configured per-run or project API budget is exhausted;
- a concrete runtime/environment blocker is recorded through `report_blocker`.

A rejected candidate, completed experiment, next-step recommendation, or ordinary final answer is not a stop condition.

`--max-turns` limits one Agents SDK cycle, not the total autonomous research horizon.

## Mandatory execution contract

When the agent is launched with `--allow-exec`, planning-only completion is not acceptable.

An execution-enabled run MUST, unless blocked by a concrete runtime error or exhausted budget:
1. inspect enough evidence to select one falsifiable hypothesis;
2. create or select a concrete candidate policy;
3. call `start_experiment`;
4. execute at least one `static_replay_candidate` evaluation;
5. analyze the measured result;
6. call `finish_experiment` with SUPPORTED, REJECTED, UNRESOLVED, or ERROR.

The agent must not stop after proposing code, describing an experiment, or writing a plan when execution is enabled. It must perform the experiment and record evidence.

If execution is blocked, the final record must name the exact blocker and the failed tool/step. A run that has execution enabled but performs zero static replay calls is incomplete.

## FP16 precision contract

FP16 is the default and required floating-point precision for v23 candidate models and numerical policy computation.

Required:
- neural-network/model floating weights: `float16`;
- floating activations/tensors created by candidate code: `float16`;
- NumPy floating arrays created for candidate computation: `float16`;
- PyTorch default floating dtype is set to `torch.float16` in the isolated replay child before candidate loading;
- checkpoints or learned floating parameters created by v23 experiments must be stored in FP16 unless an external format cannot represent FP16.

Not converted to FP16:
- integer action codes, IDs, coordinates, counters, indices, shapes, lengths, seeds, and enum values;
- boolean masks and flags;
- Kaggriculture environment observations/actions whose schema requires Python integers, booleans, strings, or other non-floating types;
Disallowed in candidate floating-point computation:
- explicit `float32`, `float64`, `double`, or `bfloat16` model/tensor dtypes;
- silent promotion of model weights or activations to wider floating precision;
- widening a candidate computation merely because a backend operation is inconvenient in FP16.

If an algorithm cannot run under this contract on the available backend, change the algorithm rather than silently widening precision.

Static replay performs a source-level precision audit before executing a candidate and rejects obvious explicit wider floating-point dtypes.
