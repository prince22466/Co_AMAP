# Submission 7 performance review and Submission 8

Reviewed on 2026-10-18. Source: [Submission 7 diagnostic archive](../submission-7-zip_First30Tasks.zip), compared with [Submission 4](../submission-4-zip_First30Tasks.zip). The new [archive audit](audit/archive_audit.md) reports zero integrity errors. [analyze.py](analyze.py) reproduces [summary.json](summary.json), [task_evidence.json](task_evidence.json) and [paired_tasks.csv](paired_tasks.csv), reusing the previous archive reader. All ten local Submission 7 package files match the evaluated configuration hashes.

## Outcome: operational improvements, no resolution gain

Both runs use the same 30 task descriptions and commits, runtime versions, model variant and evaluator limits. They disable the task timer, retain 40 counted calls and 80 turns, and allow 60 seconds per command and 300 seconds for final grading. These diagnostic results are separate from public leaderboard scores: [submission_records.md](../../../submission_records.md) records 0.13 for Submission 7 and 0.12 for Submission 4.

| Observation | Submission 4 | Submission 7 |
| --- | ---: | ---: |
| Reported resolved | 4/30 (13.3%) | 4/30 (13.3%) |
| Resolved with a nonempty patch | 3 | 3 |
| Context-window exceptions | 2 | 0 |
| Rejected workspace-tool writes outside the workspace | 32 | 0 |
| Returned patches with scratch-like filenames | 14 | 6 |
| Empty patches | 2 | 3 |
| Argument-validation failures | 21 | 33 |
| Tasks reaching 40 counted calls | 10 | 12 |
| Median reported task duration | 340.31 s | 340.39 s |

Submission 7 newly resolves `fastapi_14794` but loses `fastapi_14873`. Both pass `rich_4077`, `fastapi_14786`, and the empty-patch case `fastapi_14851`. The aggregate tie hides this task swap. The empty-patch result remains a reported pass, not evidence of an implemented fix. Scratch-like filenames are a heuristic, not an attribution of grading failure.

The lower output reservation and revised scratch guidance are consistent with the corresponding failures disappearing in this run. Multiple changes and one sample per candidate prevent isolated causal attribution. They did not raise the resolved count. Reported task duration includes setup/grading; it is not the release's four-minute agent allowance.

## Failures that inform the next candidate

### 1. Prompt instructions did not stop malformed edits

Submission 7 records 32 `edit_file` calls missing mandatory fields and one malformed `write_file`. There are also ten unmatched edits. The 33 argument-validation failures are concentrated in four tasks: `fastapi_14962` (9), `fastapi_14964` (8), `rich_4070` (15), and `requests_7502` (1). The model repeatedly emits the same damaged multi-field arguments despite the recovery instructions. `fastapi_14962` finishes with an empty patch.

By comparison, all 453 observed `run_command` calls supplied their required argument. Shell execution still fails for other reasons: syntax/quoting errors, assertions, missing imports, no-match searches and timeouts. This is evidence for simplifying the tool-call schema, not for treating shell commands as infallible.

**Submission 8 change:** only `run_command`, `get_status` and `submit_patch` are exposed as direct coder tools. Reading, editing and checks use shell commands. The prompt supplies a Python stdin editing pattern that requires a unique exact match and compiles Python content before writing. This structurally removes the multi-field edit/write tool interface instead of relying on another retry instruction. Short scripts and ordinary quoting remain necessary.

### 2. Verification work sometimes replaced implementation

On the newly lost `fastapi_14873`, the agent spends its allowance on reads and a reproduction with guessed lifecycle APIs. The intended source edit is rejected after call exhaustion; the returned patch contains only `repro.py`. `rich_4076` likewise returns only a reproduction, and `rich_4075` ends with no patch. Nonempty patch length therefore overstates implementation progress.

**Submission 8 change:** an explicit five-stage work order favors a supported implementation after a few relevant searches, uses confirmed APIs/test paths, limits repeated attempts to make a reproduction run, and requires final inspection for implementation rather than scratch-only changes. Scratch checks use stdin or `/tmp`. These priorities remain prompt policies, not enforced stage timers.

### 3. The helpers still did not provide useful coverage

Only one contract-analyst invocation occurs across all 30 tasks, taking 23.47 seconds; the counterexample tester is never called. The single analyst response labels running four existing scripts as “Explicit” requirements although the supplied question merely listed available scripts and described a sparse issue. That is unsupported inference. The agent also invokes `get_status` only once, after a patch was already submitted, so existing budget/routing instructions were not reliably followed.

**Submission 8 change:** the analyst must quote a short original-issue phrase for every explicit requirement and otherwise label the point inferred or unknown. It must challenge unsupported premises rather than convert filenames or customary workflows into a specification. The unused test designer is replaced with a reviewer of the actual diff, original issue and observed verification results. Review becomes the normal pre-submission checkpoint when there is a patch, at least six calls remain, and any known timer is at least 60 seconds. An absent timer uses the call threshold alone. Both helpers retain no tools. Routing is still a prompt policy: actual invocation must be checked in the next run.

### 4. Passing a narrow reproduction does not cover the contract

`requests_7205` initially tests an installed Requests copy; after correcting its source import path, its reproduction passes, but final grading still rejects a missed credential case. This was not an ignored final local assertion: the last local reproduction passed and its coverage was incomplete. `fastapi_14978` still omits constructor support at another API layer. These examples support review of the behavior checked, not just the command's exit code.

**Submission 8 change:** the reviewer examines missing API layers, boundary/default cases, runtime imports, implementation-only coverage and the evidence for each expectation. Its opinion cannot establish test success. It may propose one short additional check, explicitly unrun; the coder remains responsible for execution and repairs. No benchmark-specific expected answer enters the prompts.

### 5. A smaller reservation still permits long output

There is one `MAX_TOKENS` response on `fastapi_15280`: 8,192 output tokens and 235.69 seconds of observed request time. It occurs in the timer-disabled run, and is an obvious risk under a four-minute task limit. Submission 8 retains the existing 8,192 ceiling and thinking settings to avoid changing sampling alongside the tool-interface experiment. It asks for small replacements, short reasoning and no whole-file reconstruction from partial reads. This does not impose a hard per-request latency bound; finish reasons and elapsed time need measurement.

## What remains outside this agent revision

Five Submission 7 tasks report missing `inline_snapshot` during grading and four report recursive `httpbin` fixtures. These patterns are the same as the earlier run. Implementation defects can coexist with those environment errors; do not reclassify those tasks as correct or blame every failure on the patch. The agent must not rewrite protected test infrastructure to obtain a pass.

A separate evaluator-environment repair and scoring-runtime parity check would make future comparisons more informative. No notebook environment, grading tests or dependencies are changed here. No claim is made that Submission 8 fixes those environment problems.

## Next experiment

The release candidate is [submission_8](../../../submission_nb/submission_8/README.md): still three 31B agents, unchanged four-minute/40-call/80-turn task limits, unchanged coder/helper sampling, no adapters. Changes concern the direct tool interface, prompts and review role. The command-oriented design has a precedent in the official [mini-SWE-agent configuration](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/default.yaml); the candidate keeps the competition's native function calls and explicit `submit_patch` protocol rather than adopting that framework's runtime or completion token.

1. Compare Submission 7 and Submission 8 on identical fixed tasks with the same environment. Set the notebook timer override to `INHERIT` for release-budget comparisons; both supplied diagnostic runs used `None`.
2. Smoke-test real shell edits, helper returns and explicit patch capture before a larger run. Record actual reviewer calls and subsequent repairs, not merely configured agent counts.
3. If promising, compare Submission 8 with its reviewer connection removed while retaining its coder prompt/sampling/budgets. The prompt has an unavailable-helper fallback. This control tests reviewer availability; it is not built or evaluated here.
4. Use the previously saved [remaining validation split](../submission_4_6_review/next_evaluation_sets.json). The new run exposes the same 30 tasks, so the existing 35-task remaining validation set needs no further exclusions.

Compare paired solved/lost tasks, malformed-call categories, context failures, scratch-only patches, reviewer utilization, remaining-budget failures and total generation time. Do not infer improvement from a handful of cleaned-up tool errors or the unadjusted 13.3% diagnostic rate.

## Local verification status

Local validation passed: three 31B agents, ten packaged files, isolated helpers, and byte-identical sampling/task budgets compared with Submission 7. ZIP integrity and exact source/archive byte equality passed. The Python edit-pattern body passed four fixture checks: a multiline replacement containing quotes, and rejection without changing the file for missing matches, ambiguous matches and invalid Python. The existing packaging/diagnostic suite passed 22 tests with one platform-dependent skip. These checks do not execute Gemma or establish Linux shell/runtime parity.

[prepare_submission_8.py](prepare_submission_8.py) reproduces the candidate checks and ZIP build. Results and archive/file hashes are recorded in the [Submission 8 manifest](../../../submission_nb/artifacts/submission_8.manifest.json).

Submission 8 has no measured task-resolution result or public score yet. Submission 7 and its archive remain unchanged.
