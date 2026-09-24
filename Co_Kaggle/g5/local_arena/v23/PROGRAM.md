# v23 research program

## Objective

Build an autonomous, evidence-driven research loop for improving the Kaggriculture agent while keeping experimentation reproducible and OpenAI API spend extremely low.

v23 should first understand the existing chain:

```text
v20 notebook + game_history/v20 losses
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
- Avoid external/public solution lookup in v1; use our own code, histories, and measurements.

## Cost policy

Available OpenAI API credit is approximately $6. v1 uses `gpt-6-luna` with low reasoning effort and a persistent local ledger. The default project ceiling is $5.00 with a $0.25 per-run ceiling. There is no automatic escalation to Sol/Astra.

## v1 success criteria

A bounded run should locate relevant prior evidence without dumping the repository into context, state a concrete hypothesis, identify or execute a small experiment, separate evidence from conjecture, write a concise research record, and stay well below budget.

Only after that behavior is reliable should v23 gain source-patching and automatic A/B candidate promotion.


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
