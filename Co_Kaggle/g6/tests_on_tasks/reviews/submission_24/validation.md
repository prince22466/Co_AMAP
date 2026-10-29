# Submission 24: evidence, design, and validation

## Observed starting point

The two archived **S1** runs contain the same 15 IDs. Do not confuse them with
S21's earlier first-15 run or Kaggle leaderboard scores.

| Metric | S21 S1 | S22 S1 |
|---|---:|---:|
| Resolved | 2/15 | 2/15 |
| Empty patches | 4 | 5 |
| Total recorded evaluation duration | 34.92 min | 30.12 min |
| Optional skill calls | 0 | 0 |

S21 solved `rich_4077`, `rich_3905`; S22 solved `rich_3518`, `rich_3454`.
[baseline_evidence.json](baseline_evidence.json) records source archive hashes,
per-task outcomes, recorded tool-call types, versions and effective budgets.
Recorded calls include failed/budget-denied attempts; they need not equal the
harness's charged tool count. These are historical observations, not S24 results.

## Trace-driven choices

| Observed failure | S24 intervention | Remaining limitation |
|---|---|---|
| S22 `rich_4077`: printed tool syntax, zero tools | Short schema-oriented prompts, fresh downstream stage, bounded thinking | No custom parser repair is allowed; E4B must still produce a real call |
| S22 `rich_3905`: repeated first-page reads, budget exhaustion | Dedicated symbol/caller search, explicit 150-line cap, short state handoffs | Per-stage call limits remain instructions |
| S22 `rich_3882`: repeated unsuccessful replacements | Focused editor, two-attempt rule, native command fallback, independent feedback | The model must follow the recovery rule |
| Incorrect/incomplete nonempty patches | Reviewer compares actual diff to task and checks behavior; second edit round | Semantic judgment still depends on E4B and test relevance |
| Optional skills never invoked | One simple no-argument audit assigned to the reviewer | Invocation remains model-selected; native fallback is retained |
| Premature submission | `submit_patch` registered only on finalizer | Global exhaustion can still prevent reaching it; harness fallback remains |
| Fresh stages could lose task hints | Use the actual harness state key `{hints?}` | Hints are absent if the task has none |

This is a single selected architecture, not an empirically winning search over
architectures. Parallel editors were not selected because the checkout is shared.
Custom callbacks were not selected because official `swegemma 0.2.11`
`harness/agent_runner.py` calls `compile_submission` without a callback registry.
Mandatory skill-only editing was rejected based on the earlier S17/S20 failures.

## Compliance and runtime evidence

- Every LLM stage declares `gemma-4-e4b-it-qat-w4a16-ct`; no adapters or weights.
- Only supported `SequentialAgent`, `LoopAgent`, `LlmAgent`, native harness tools
  and the sandboxed ADK skill executor are used.
- Official wheelhouse packages were retrieved from Kaggle and SHA-256 checked:
  `adk-submission 0.2.13`, `adk-eval-core 0.1.0`, `swegemma 0.2.11`,
  `google-adk 1.36.1`, `google-genai 2.11.0`. Hashes are in the runtime installer.
- `Update_new_models.md` in competition materials supplies the newer E4B
  allowance; the older HARNESS_README's 31B-only model section is superseded.
- The archive excludes task data, reference answers, tests, development reports
  and evaluation tooling. The model sees the normal task/hints and checkout.
- No scoring/evaluator code, benchmark tests, or previous submissions are modified.

The official compiler/Runner tests actually execute the archive against temporary
Git repositories. The synthetic model deliberately proposes invalid Python;
the real skill catches it, ADK state carries the finding to the next editor
round, the repaired code passes the same independent probe, and one native
submission preserves the diff. Other cases cover empty output and a locator
that emits plain text, plus previous-tool-context removal and harness hint retention.

`runtime_smoke.json` is **scripted mechanics only**, with package hash and runtime
versions. The unit/runtime suite contains 17 passing tests after PR review.
The new CI workflow repeats them and verifies the committed ZIP is reproducible.
This does not test E4B's token parser, generated repairs, success rate, latency
on GPU, or the competition-wide 12-hour wall-clock budget.

## PR review corrections

Review of commit `86a913a` reproduced two evaluation-runner defects against the
official runtime: it omitted the archived S1 cache/compaction configuration, and
an official `ContainerSetupError` escaped before the failed task and run summary
were saved. The runner now records and applies the same context settings, retains
each failed outcome with its traceback, continues the remaining tasks, and blocks
candidate inference when the baseline preflight fails. Candidate errors are listed
in each repeat's summary. Regression tests fail against the original code and pass
after correction.

The review also fixed the runner's hard-coded unauthenticated model check / `EMPTY`
inference key: it now honors notebook v11's `LOCAL_API_KEY` for both requests without
recording it in the manifest. Nonfinite time limits are rejected. Tests cover both
repeat seeds, baseline-only execution without a server request, complete results
after an injected failure, and cancellation propagation with sandbox cleanup.
The graph smoke now uses the same ADK App cache/compaction configuration as v11.
The submission package bytes and agent architecture are unchanged by these fixes.

## Acceptance still requiring GPU execution

The package retains 40 charged calls, 48 LLM turns and a 4-minute scored-task
budget. Historical S1 notebook runs disabled that timer. Compare under the same
effective configuration and record package/runtime/server identity.

Run the five-task pilot, then both full S1 repeats and additional S2 tasks. Report
all outcomes, paired gains/losses, empty patches, malformed calls, stage reach,
audit usage, relevant final checks, time and tokens. The comparison tool rejects
partial/duplicate task sets and missing outcomes. Its >=6/15 development target
is deliberately stricter than obtaining two different wins again.

**Not measured here:** real E4B S24 results, superiority over S21/S22, and hidden
competition performance. No GPU/model weights are available in this workspace.
