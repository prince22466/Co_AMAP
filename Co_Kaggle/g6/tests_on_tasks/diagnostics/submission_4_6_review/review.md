# Submission 4/6 diagnostics and the Submission 7 revision

Reviewed on 2026-10-18. Sources are the supplied [Submission 4 archive](../submission-4-zip_First30Tasks.zip) and [Submission 6 archive](../submission-6-zip_First30Tasks.zip). Both archive audits report zero integrity errors. [analyze.py](analyze.py) reproduces [summary.json](summary.json), [task_evidence.json](task_evidence.json) and [task_comparison.csv](task_comparison.csv) directly from those archives without executing any submitted code.

## What can be compared

Both runs contain identical task descriptions, repository commits and task IDs, with the same runtime versions and effective evaluator limits: 40 counted calls, 80 turns, 60 seconds per command and 300 seconds for final grading. **Both disable the agent session timer.** Their compaction interval is 15 and model context is 32,768 tokens. These are local diagnostic outcomes, not the recorded public leaderboard scores.

Submission 4 uses 31B and three configured agents. Submission 6 uses E2B and two configured agents, with different prompts/tools. Their differences do not isolate model size or architecture.

| Observation | Submission 4 | Submission 6 |
| --- | ---: | ---: |
| Reported resolved tasks | 4/30 | 1/30 |
| Resolved tasks with a nonempty patch | 3 | 1 |
| Empty patches | 2 | 7 |
| Explicit context-window errors | 2 | 6 |
| Tasks reaching 40 counted calls | 10 | 0 |
| Helper invocations | 1 locator, 0 verifier | 23 analyzer calls across 11 tasks |
| Tool argument-validation error responses | 21 | 6 |
| Workspace file-tool path rejections | 32 | 0 |
| Edit responses reporting unmatched old text | 1 | 40 |
| Median reported task duration | 340.31 s | 96.63 s |

Reported task duration includes setup and grading; it is not an agent-only timer measurement. The empty-patch pass is `fastapi_14851`; retain the recorded resolved outcome, but do not credit it as evidence of an implemented fix. No rerun with an empty-patch baseline was performed here.

The audits flag live/ATIF token-total differences and missing usage on rejected model requests. This review does not add those token sources or use aggregate token totals to infer context length. The direct context exceptions and per-response output usage are the relevant evidence.

## Actionable findings and changes

1. **Output reservation can reject a request before the model generates anything.** Submission 4 fails on `requests_7502` and `rich_4006` with a 16,384-token output reservation and at least 16,385 input tokens against a 32,768-token limit. Submission 6 records six such errors. Submission 4's largest observed response is 3,976 output tokens, with no observed length finish; two rejected requests lack usage. Submission 7 now reserves 8,192 output tokens, nominally leaving 24,576 for input instead of 16,384. This adds headroom; it does not guarantee that all future requests fit or responses finish. Thinking remains enabled and its declared budget stays 4,096. Check future finish reasons and context errors.

2. **Scratch-test instructions contradicted workspace file-tool capabilities.** Submission 4 makes 32 rejected `write_file` calls for paths outside `/workspace`, then often recreates those scratch scripts in the repository. Fourteen returned patches contain paths named like reproduction/check scripts; this filename heuristic is separate from grading failure attribution. The revised coder explicitly uses shell/stdin for reproductions and shell creation for necessary `/tmp` files. It never asks `write_file` to write outside the workspace, and checks that tests import the actual working-tree source.

3. **Repeated malformed tool calls consume time without progress.** `fastapi_15661` repeats the same rejected write without a `filepath` argument 11 times. `requests_7315` repeats a malformed edit six times and embeds other fields inside read paths. Submission 6 also records 40 unmatched-edit responses. The revised coder puts paths in separate native fields, avoids double encoding source text, rebuilds rejected calls instead of repeating them, and switches to a small assertion-guarded shell edit after two failures. It also treats no-match searches as search results and confirms test paths before execution.

4. **Configured helpers were either unused or expensive.** Submission 4 never invokes its verifier. Its only locator call, on `rich_4079`, spends 163.23 seconds and makes 26 repository-tool calls despite its prompt's small target. The final patch is empty. Nested calls are counted by helper author within the observed delegation interval, because parent call IDs are not consistently populated. Submission 6 invokes its analyzer nine times on `rich_4075`, also showing that prompt-only call limits can be ignored. Submission 7's two helpers now declare `tools: []`: the analyst reviews supplied evidence, and the tester proposes a short check that the coder executes. This removes their declared ability to enter repository-tool loops. It does not enforce at-most-once invocation by the coder or eliminate inference cost. The tradeoff is dependence on the coder supplying adequate excerpts; helpers must report missing evidence instead of guessing.

5. **Some patches implement one layer or break existing runtime behavior.** `fastapi_14978` reaches grading without the required constructor support across API layers. `rich_4070` removes runtime imports and grading reports undefined names. `fastapi_15588` again fails six field-validation message checks. The revised prompts explicitly track wrappers/constructors/consumers, runtime uses of moved imports, normal controls and exact contracts supported by visible evidence. No task IDs, benchmark-specific expected strings or grading-patch answers enter the submission prompts.

6. **Submission must remain the last operation.** The traces contain repeated `submit_patch` calls and work after an earlier capture. The revised coder explicitly ends with a brief response after successful capture and performs no further edits or submissions. All reproduction cleanup and inspection precede capture.

## Environment and measurement limitations

Submission 4's grading output includes missing `inline_snapshot` on five tasks and recursive `httpbin`/`httpbin_secure` fixtures on four Requests tasks. Submission 6 reports those patterns on two and three tasks respectively. These are observed setup/collection problems, not proof that the submitted implementations are otherwise correct; implementation defects can coexist with them. In particular, a missing requested project module is different from a missing third-party test dependency.

The candidate is instructed to use an available isolated behavioral check and report incomplete coverage rather than editing test infrastructure or installing dependencies. A reliable architecture comparison also needs an evaluator environment that can execute the affected tests. This revision does not modify the notebook environment or private scoring behavior, nor promise to fix those grading issues through an agent prompt.

Submission 7 keeps the four-minute, 40-call, 80-turn task allowance and 60-second command timeout. Unlimited diagnostic timers do not establish performance under that four-minute limit. Its new helpers still use the largest permitted 31B variant, with unchanged sampling. The only sampling change is the coder's output ceiling. Thus the revised release combines prompt, helper-permission and output-reservation changes; any later improvement cannot automatically be attributed to one component.

## Next evaluation

Use the existing notebook with 31B weights and point its submission directory at the rebuilt candidate. Set `EVAL_OVERRIDES['max_time_minutes'] = INHERIT` for release-budget comparisons. Use the same final-verification timeout, instrumentation and compaction settings for every variant; inspect saved effective settings. The official scoring runtime remains authoritative.

Compare original Submission 4, original Submission 4 with only the coder output ceiling reduced to 8,192, and revised Submission 7. The intermediate control separates the context-reservation change from the remaining revision. The original Submission 7 archive and manifest are preserved here as `pre_revision_submission_7.zip` and `pre_revision_submission_7.manifest.json`; they can also be compared directly. These controls have not been run or newly packaged.

Start with the frozen smoke/pilot task sets. Confirm that a real helper returns, the coder executes the proposed check, and a nonempty intended patch reaches grading. Count actual checks and outcomes, not proposed commands. Inspect context failures, rejected file writes, repeated invalid calls, scratch files in patches, helper cost and paired newly solved/lost tasks. Keep failed environment checks separate from executed behavioral assertions.

Four original validation IDs appear in these diagnostic runs: `fastapi_14978`, `requests_7328`, `requests_7427`, and `rich_4075`. They are now feedback data. [next_evaluation_sets.json](next_evaluation_sets.json) retains the existing development sets and supplies 35 remaining validation tasks after excluding all observed repository/commit groups. The original inventory/split is unchanged. Do not reuse the four exposed tasks as untouched validation.

## Local verification

The source configuration is locally validated and the ZIP is rebuilt with an updated hash manifest. Checks verify three 31B agents, both helpers having no declared tools, the 8,192-token coder ceiling, preserved thinking/helper sampling/task budgets, dependency closure and ZIP/source byte equality. The existing packaging/diagnostic unit suite reports **22 passed, one platform-dependent skip** (23 run). One stale fixture path caused by the move to `tests_on_tasks/tools` was corrected to the actual `submission_nb/submission_4` location. The rebuilt ZIP contains ten files and is 7,278 bytes; its SHA-256 is recorded in `submission_nb/artifacts/submission_7.manifest.json`.

No new Gemma inference or Kaggle score was obtained. The revision is an evidence-based candidate, not a demonstrated score improvement.
