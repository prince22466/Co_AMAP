# Prompt experiments

## Observation and candidate

Recorded on 2026-10-09 from the user's report:

| Version | Changes | Kaggle score observations | Task diagnostics |
|---|---|---|---|
| Submission 1 | Original main-coder/optional-analyzer candidate, no adapters | 0.10, 0.06 | Not supplied |
| Submission 2 | Two instruction files revised; configuration unchanged | 0.10, reported in submission_records.md on 2026-10-11 | Not supplied |

Update on 2026-10-11: the current [submission records](../submission_records.md) also report the starter at 0.03. Submission 2 matches Submission 1's best reported score; these runs do not establish an improvement or explain the two Submission 1 outcomes. Its earlier unmeasured status is superseded by the reported public score, while task-level diagnostics remain unavailable.

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

The comparison uses the starter-compatible subprocess backend by default. Scoring Docker parity, full task coverage, and total inference allowance checks remain necessary before final promotion. There are no new local GPU benchmark results in this record.

## Submission 3: reviewer experiment

Prepared on 2026-10-11. [Submission 3](submission_3/README.md) preserves Submission 2's model, sampling, analyzer and total budget, and adds an optional patch reviewer with only `read_file` access. The coder supplies the issue, acceptance conditions, diff and observed verification results, retains all edits/tests/submission, and skips review unless resources remain for repairs. Review thresholds and response/read limits are prompt targets rather than enforced helper quotas.

`evaluate_submission_3.ipynb` compares coder-only, coder plus analyzer, and coder plus analyzer plus reviewer on the same tasks under the same Submission 3 coder prompt. The two-agent control is derived from Submission 3, not the original Submission 2 prompt. This isolates helper availability; it does not isolate prompt changes between releases. Reports distinguish analyzer and reviewer delegation when supported traces exist; missing reviewer usage remains unknown. Inspect findings and resulting repairs to assess benefit instead of treating invocation as success.

At preparation time Submission 3 had no GPU benchmark or reported public score. On 2026-10-12 the user reported **0.0** for `artifacts/submission_3.zip`, with no Kaggle error message. No task-level logs were supplied. A local recheck found an intact ZIP matching the normalized source; the cause of the score remains unknown. Earlier unmeasured statements in candidate notes are superseded by this report.

## Submission 4: locator and executable verification

Prepared on 2026-10-12 in response to Submission 3's zero. [Submission 4](submission_4/README.md) replaces the graph-based analyzer with a shell-based repository locator and replaces the read-only patch reviewer with a test verifier that can execute focused checks. The coder retains source edits, repairs and submission, with six direct tools and shorter instructions. Helpers have only shell/read tools, no full conversation history, and source-preservation instructions; shell access is not technically read-only.

The model, coder sampling, helper sampling and four-minute/40-call/80-turn task allowance remain those of Submission 2. Optional helper calls are gated by prompt budget thresholds, limited to one invocation each by instruction, and skipped when direct action is clearer. A helper's failed or inconclusive response does not block patch capture.

`evaluate_submission_4.ipynb` compares coder only, locator only, verifier only and both helpers under one shared coder prompt. It can isolate helper-availability effects within Submission 4. This release changes roles, tools and prompts together, so a score change against historical candidates would not identify a single causal intervention.

Local tests validate configuration, controls, package closure and evidence-based locator/verifier counts. At preparation time no real Gemma inference, official compiler run or public score was available for Submission 4. On 2026-10-13 the user reported **0.0**, without additional log information. The zero-score cause remains unknown. [The investigation](zero_score_investigation.md) records the archive checks, runtime-source findings and logging limits.

## Submission 5: command/grading timeout and thinking

Prepared on 2026-10-13 from Submission 4. Only four packaged files change: `eval_config.yaml` sets timeout 300 instead of 60, and the three sampling files set only `thinking_config.include_thoughts: false`, omitting numeric thinking budgets. The inspected compiler bridge turns thinking off for this setting; actual behavior must be confirmed in the scoring runtime. No `thinking_budget: 0` or unsupported reasoning-level setting is introduced.

The coder, repository locator and test verifier retain their exact declarations, tool access and prompts. The model, output ceilings, temperature, top-p, four-minute agent allowance, 40 counted calls and 80 turns remain unchanged. The inspected harness shares the command timeout with grading, so raising it allows longer grading checks while the agent session retains its original total allowance and short-check prompt instructions.

`evaluate_submission_5.ipynb` embeds Submission 4 and Submission 5. It compares `baseline` (60 seconds/thinking on), `timeout_only` (300/on), `thinking_off_only` (60/off), and `candidate` (300/off). Every variant retains all three agents and receives the same task IDs. Preflight permits only the intended runtime changes, and the candidate is the default release. These controls can separate the two effects; the combined release's public score alone cannot do so.

[Submission 5](submission_5/README.md) was unmeasured at preparation time. The records inspected on 2026-10-14 report **0.0**; no task-level scoring artifacts identify the cause. Local configuration, control-generation and artifact checks do not establish official compilation, effective thinking behavior, coding-task resolution or total inference time.

## Submission 6: model substitution only

Prepared on 2026-10-14 after reading the [organizer's new-model announcement supplied locally](../competition_material/Update_new_models.md). The smallest allowed variant is `gemma-4-e2b-it-qat-w4a16-ct`. [Submission 6](submission_6/README.md) copies Submission 2 and changes only the main coder and analyzer model declarations from 31B to E2B. All other packaged files, including prompts, generation settings, thinking configuration and evaluation budgets, are preserved byte-for-byte. No adapters are included.

The local validator now accepts the four listed variants and rejects mixing them within a submission, including unused standalone and inline agents. The generated notebook selects the candidate's declared model and matching weight path, uses the supplied updated starter's installation cell for new variants, and registers only the served model. The 12B startup override documented in that starter is also supported by the shared generator.

`evaluate_submission_6.ipynb` compares coder-only and optional-analyzer controls, both E2B. Comparing 31B Submission 2 with E2B Submission 6 requires separate model sessions on identical task IDs. The existing prompt/runtime comparison modes reject cross-model variants because one server cannot serve the unmounted second model.

Submission 6 has no score or real-model benchmark. Local tests check allowed-model policy, mixing rejection, exact model-only package scope, E2B weight selection and stubbed server routing. They do not establish updated compiler acceptance, effective generation controls, coding-task quality or inference speed.

## Design references

The local harness guide establishes final-response termination, protected-file resets, tool argument signatures, and free status/submission calls. The prompt revision follows those interfaces. Google documents Gemma's native function-call format; the harness handles its formatting and execution, so the prompts request actual tool use rather than introducing a custom protocol. [Google Gemma 4 function-calling guide](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4).
