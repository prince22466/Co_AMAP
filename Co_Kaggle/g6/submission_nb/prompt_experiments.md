# Prompt experiments

## Observation and candidate

Recorded on 2026-10-09 from the user's report:

| Version | Changes | Kaggle score observations | Task diagnostics |
|---|---|---|---|
| Submission 1 | Original main-coder/optional-analyzer candidate, no adapters | 0.10, 0.06 | Not supplied |
| Submission 2 | Two instruction files revised; configuration unchanged | Not evaluated | Not evaluated |

Submission 1's local original archive SHA256 is `242c31b9c1c87f51ba124bd129ed05a2a7002b0f93808ade92e9a930c28fe4aa`. This identifies the preserved local artifact; the two Kaggle executions have not been independently inspected.

Two aggregate scores cannot distinguish incorrect edits from empty patches, timers, tool-call failures, or task setup problems. They also do not quantify evaluation variance without task counts, shared task identities, and execution details. Do not treat 0.06 as proof of a regression when the reported candidate was unchanged.

## Experiment design

Keep the two roles, base model, tools, sampling, budgets, and input tasks fixed. Change the instruction text only. The notebook checks this configuration equivalence before model startup.

1. Open `evaluate_submission_2.ipynb` with the competition data, model, wheelhouse, and GPU resources attached.
2. Smoke test both `baseline` and `candidate` on the same two complete tasks. This checks execution rather than general performance.
3. Set explicit `TASK_IDS` covering several repositories and task types. Keep these IDs across prompt revisions and a separate set for validation. Compare both versions on each task, preferably in repeated runs with reversed variant order.
4. Read `results/<variant>/report/diagnostics.md`, JSON/CSV, the run manifests, patches, and verification output. Compare task outcomes directly, not only aggregate scores.
5. Promote based on stronger task resolution within the runtime envelope. Check whether gains depend on one repository, incomplete data, or inconsistent runtime/configuration before drawing a broad conclusion.

## Hypotheses to inspect

Whitespace-delimited prompt word counts: main 768 to 653; analyzer 267 to 195. These measure text length, not tokenizer usage or realized inference time.

| Hypothesis | Observable evidence |
|---|---|
| Fewer premature completions | More nonempty patches; submission calls and final stop evidence where available. |
| More complete implementation | Improved verification on tasks with multiple requested behaviors; inspect exact diffs. |
| Less repeated exploration | Search/read frequencies and time to a confirmed edit when supported by traces. |
| Less helper overhead | Delegation count, helper trace content, request/token metrics when recorded, and task duration. |
| Better budget use | Fewer explicit exhaustion stops; agent time remains distinct from total task time. |
| Better failure handling | Test output distinguishes assertions, collection/dependency errors, and timeouts; no unsupported success claims. |

Prompt targets such as short reports and limited helper calls may be ignored. Only the runtime's enforced budgets are hard constraints. Unknown or unsupported trace fields remain null; do not infer a causal improvement from shorter prompts alone.

The comparison uses the starter-compatible subprocess backend by default. Scoring Docker parity, full task coverage, and total inference allowance checks remain necessary before final promotion. There are no new measured performance claims in this record.

## Design references

The local harness guide establishes final-response termination, protected-file resets, tool argument signatures, and free status/submission calls. The prompt revision follows those interfaces. Google documents Gemma's native function-call format; the harness handles its formatting and execution, so the prompts request actual tool use rather than introducing a custom protocol. [Google Gemma 4 function-calling guide](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4).
