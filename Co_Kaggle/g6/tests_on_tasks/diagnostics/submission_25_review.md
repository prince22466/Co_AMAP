# Submission 25 Review: Why It Regressed and What Must Not Recur

## 1. Overall result

**Submission 25 was a severe regression, not an improvement over Submission 24.** It replaced S24's native source-editing workflow with one E4B agent operating a custom `run_skill_script` workbench. The workbench passed synthetic/runtime tests, but the real Gemma 4 E4B agent repeatedly called it incorrectly and usually could not inspect or modify the repository.

All development results below come from **the same selected 15 S1 tasks**. Kaggle public score is separate from these local task-resolution results.

| Metric | S21 | S22 | S24 | **S25** |
|---|---:|---:|---:|---:|
| S1 tasks resolved | 2/15 | 2/15 | 1/15 | **0/15** |
| Nonempty patches | 11/15 | 10/15 | 12/15 | **1/15** |
| Empty patches | 4/15 | 5/15 | 3/15 | **14/15** |
| S1 recorded duration | 34.9 min | 30.1 min | 49.6 min | **15.5 min** |
| Aggregate observed tokens | 3.28M | 3.67M | 3.34M | **4.10M** |

The [submission records](../../submission_records.md) show **S25 Kaggle public score = 0.00**; S24 was not submitted to Kaggle. Durations and observed tokens are evaluator trace aggregates (including repeated context), **not** GPU billing, unique generated tokens, or evidence of efficient solving. The shorter duration coincided with widespread unsuccessful or empty-patch termination; it does not demonstrate faster task solving.

S25 lost S24's `rich_4077` success. Its only nonempty patch (`rich_4075`, 861 bytes) failed evaluation and was recovered as an **unsubmitted working-tree fallback**, not a successful explicit submission. The printed S25 trace shows only **four explicit `submit_patch` calls**, all returning `patch_size: 0`; **no nonempty patch was explicitly submitted** in this S1 evaluation.

## 2. Confirmed root causes

| Failure | Evidence from real S25 traces | Root cause / lesson |
|---|---|---|
| **Native tools replaced by a fragile wrapper** | Hundreds of `run_skill_script` attempts; visible `INVALID_ARGUMENTS` and `SCRIPT_NOT_FOUND` responses | A complicated nested tool interface was introduced without proving that E4B could use it reliably. S21/S24 native `read_file`, `edit_file`, `write_file`, `run_command` should not have been removed without real-model parity evidence. |
| **Repository path confused with skill script path** | `rich_4077`: calls used `file_path="rich/file_proxy.py"` or `"/workspace"`, although the only valid script was `scripts/work.py`; task ended with zero patch | The same word “path” served different layers. The model repeatedly tried to execute source files as scripts and lost a previously solved case. |
| **Missing script arguments** | `fastapi_14258`: repeated `args={"op":"read","path":"fastapi/routing.py"}` without the required outer `file_path`; no patch | The tool could not even begin the intended read. |
| **Silent wrong-operation fallback** | `requests_7205`: calls omitted `args.op` while supplying `args.query="netrc"`; the script returned `status` help rather than searching; 40 charged calls exhausted | `work.py` defaults missing `--op` to `status`. A superficially successful tool response concealed a no-op and permitted repetition. |
| **No outer-call recovery** | Repeated invalid calls persisted until turn/tool limits; `rich_4077` hit turn limit with only one charged tool call | Protections *inside* `work.py` cannot intercept invalid `run_skill_script` calls rejected **before** that script runs. Prompt rules such as “never repeat” are not enforcement. |
| **Over-restrictive editing scope** | In `rich_4075`, attempts to edit `pyproject.toml` and documentation were rejected; the agent later changed `rich/errors.py`. The actual requirement is **uncertain** because the S1 task list and trace describe different issues. | S25 allowed edits only to existing tracked, non-protected `.py` implementation files. Regardless of the task-label discrepancy, this blocks legitimate packaging, metadata, configuration, documentation, and new-file repairs. |
| **Verification tests overinterpreted** | PR #166 passed 27 tests and official-runtime scripted smoke tests; S25 still resolved 0/15 with real E4B | Scripted model calls prove tool mechanics, **not** that E4B will construct, correct, or interpret those calls, or solve real tasks. |
| **Pilot promotion was not substantiated** | S25 README specified a five-task pilot, but the checked-in PR #166 evidence contains **no passing real-E4B pilot**. Full S1 diagnostics and a Kaggle score were subsequently recorded; the S1 result failed the documented targets. | An unverified README gate is not a release gate. The precise order of any private pilot and Kaggle runs cannot be reconstructed from the checked-in evidence; promotion must require a recorded passing pilot. |

The printed S25 log contains approximately **670 real `run_skill_script` attempt lines** (plus duplicated diagnostic display excerpts), with at least **366 visible `INVALID_ARGUMENTS`** and **54 `SCRIPT_NOT_FOUND`** responses. These are diagnostic occurrence counts, not distinct tasks or charged-tool counts; the harness can reject some calls before charging a tool. The dominant problem was **model–tool incompatibility**, prior to patch correctness.

`rich_4075` is also a task-provenance warning: [`selected_tasks_s1.txt`](selected_tasks_s1.txt) labels it **Console empty output**, whereas the S25 agent trace discusses **dropping Python 3.8 support**. The earlier [S21–22 review](submission_21_22_review.md) already flagged this discrepancy. Until the original task record and evaluator are checked, do **not** treat either title or the agent's proposed fix as ground truth. The `.py`-only edit restriction is still independently established by `work.py`.

The `ensurepip unavailable` warning appears in multiple submissions and is **not supported as the primary cause** of S25's collapse. Likewise, zero thinking budget, 4,096 maximum output tokens, and E4B's underlying code-reasoning limits may contribute, but **this run does not isolate their causal effect**.

## 3. Why the earlier design review was wrong

The S25 code and synthetic tests did implement many intended mechanisms: numbered edits, syntax rejection, saved checks, rollback, deterministic packaging, and native `submit_patch`. Those tests validated **behavior conditional on receiving correctly structured calls**.

In [`smoke_submission25.py`](../tools/smoke_submission25.py), a scripted model supplies a deliberately malformed call **followed by a separately scripted correct call**. That is not evidence that Gemma itself can notice an error and repair the next call. The reported `malformed_dispatch_recovered=true` condition was therefore too easy to interpret as model recovery.

Moreover, duplicate-edit detection and safeguards were mostly located inside the script; they had no opportunity to stop malformed outer tool calls. The S25 PR [validation report](../reviews/submission_25/validation.md) explicitly warned that real E4B performance was unmeasured, but the absence of a real-model gate made the implementation appear more ready than it was.

**Review error:** I gave too much weight to passing software tests and an architecturally coherent design, and too little weight to *the actual model's tool-use ability*. A “works in CI” claim was not a “works as a Gemma agent” claim. The earlier warning did not substitute for empirical validation.

## 4. Non-reintroduction rules for every future submission

These are **regression requirements**, not optional prompt suggestions. A future change that violates them must be identified as an experiment and must not replace the working baseline without matched evidence.

1. **Never remove a functioning native edit/read/command/submission path solely to centralize safeguards.** Prefer competition-permitted native `read_file`, `edit_file`, `write_file`, `run_command`, `submit_patch` for core repository work. Optional skills should complement, not gate, those capabilities until real E4B calls prove parity.
2. **Never require E4B to navigate extra tool-schema layers without a real-model dispatch test.** Distinguish script paths from source paths unambiguously. Verify correct fields and successful read/edit calls *from the model*, not from hand-authored tool invocations.
3. **Never silently treat a malformed operation as success.** Missing/unknown `op` or required arguments must fail explicitly. A successful status/help response must not substitute for the search/read/edit requested.
4. **Never let identical unsuccessful calls consume the full budget.** Detect and bound repeated invalid outer calls, unchanged queries, failed edits, and empty checks *at the layer that sees them*, using mechanisms permitted by the official runtime. After a small bounded retry count, switch strategy or stop; reserve budget for patch creation and `submit_patch`.
5. **Never forbid legitimate repair files by blanket extension/path restrictions.** Permit task-required tracked/untracked additions and modifications to source, packaging metadata, config, and documentation while preventing benchmark/evaluator tampering. Verify the final Git patch includes all intended files.
6. **Never equate a syntactically valid patch or generic green test with a solved task.** Confirm the task ID against its original problem description before designing the fix, especially when task lists and traces disagree. Check the requested behavior using relevant baseline/candidate assertions when feasible; distinguish blocked environment tests, zero-collected tests, and irrelevant checks from evidence of correctness. Recheck the *final* patch.
7. **Never count scripted-model, unit, packaging, or official-runtime smoke tests as E4B task-performance evidence.** Keep them in CI, but label their scope. Run actual E4B tool dispatch, edit, and submission pilots before any broad evaluation or improvement claim.
8. **Never promote a candidate after a failing pilot.** Make the pilot an actual release decision with inspected trace/error counts and required regression cases; a README checklist alone is insufficient.
9. **Never infer improvement from shorter runtime, more tool calls, or more guardrails.** Report resolution rate, empty patches, valid tool-call rate, repeated-error loops, patch relevance, and observed resource use together.
10. **Never change several major axes and then assign a single causal explanation.** S25 changed agent topology, editing interface, permissible file scope, generation settings, and verification behavior. Future iterations should use matched, preferably repeated, ablations to distinguish harness, prompt, model, and resource effects.

These rules reinforce the earlier [S17–19 review](submission_17_18_19_review.md) and [S21–22 review](submission_21_22_review.md): S17 had already exposed skill-path failures, while S20 had already shown the danger of a mandatory skill-edit gateway. **S25 reintroduced both classes of failure.**

## 5. Required release gates for a successor

**Gate A — CPU/official-runtime mechanics (necessary but insufficient).**
- Package compiles against the official runtime and passes schema/sandbox/patch tests.
- Missing required tool arguments and missing `op` are rejected explicitly; no silent status/no-op.
- The known working native read/edit/write/command/submit path remains callable **unless** a replacement demonstrates equivalent real-E4B task and tool-use reliability under matched conditions. Legitimate `.toml`, documentation, configuration, and new-file repairs are not arbitrarily blocked.
- Failure handling is exercised at the *outer* tool-dispatch boundary, not only inside an optional script.

**Gate B — Actual E4B pilot (mandatory before full S1 or Kaggle).**
- For an E4B successor, use the **predeclared five-task pilot** from S25: `rich_4077`, `rich_3905`, `rich_3518`, `rich_3454`, `rich_3882`. Require **at least 3/5 resolved**, retention of `rich_4077`, **no more than one empty patch**, and **zero agent execution errors in every repeat** before full S1 or Kaggle. These are development release criteria, not hidden-score predictions.
- Inspect **every model-emitted call and result**. Require demonstrated source read, a real edit, meaningful verification where supported, and nonempty patch submission. Record malformed outer calls, valid operation dispatches, repeated no-op responses, and charged-versus-rejected tool calls; passing scripted-model tests do not count.
- **Block promotion** on sustained malformed-wrapper loops, repeated silent status calls, or inability to produce useful patches. Compare against the native-tool control under matching model/runtime/budget conditions.
- Where feasible, repeat nondeterministic evaluations; investigate any loss of previously solved cases rather than hiding it in an aggregate score.

**Gate C — Matched task-solving evidence (before an improvement claim).**
- Run the same 15 S1 tasks, same model and evaluation limits, alongside retained S21/S22/S24 controls where feasible.
- Report task-by-task resolved IDs, empty patches, tool-call failures, execution-limit failures, time/tokens, and actual diffs. A single S1 run is development evidence, not statistical proof of general improvement.
- Keep a separate **unused public-task validation set**, where available, so repeated optimization on the S1 tasks cannot masquerade as generalization. Do not modify reference grading tests to manufacture passes.
- Do not call a candidate “better” unless it shows real task-resolution gains and acceptable non-regression under replicated tests. Only then spend resources on a broader Kaggle run.

## 6. Conclusion

**S25 failed primarily because its custom workbench was not reliably usable by the target E4B model, and its fail-safes did not control the outer invocation failures.** Restricting legitimate edit targets compounded the problem; the successful scripted tests gave an incomplete picture of readiness.

The lasting engineering lesson is: **design the harness around observed model behavior, preserve proven tool access, enforce failure recovery where failures actually occur, and promote only on real-model evidence.** Do not reintroduce S17/S20/S25-style mandatory wrapper bottlenecks.

## Sources

- [S25 real E4B task trace](submission-25-zip_s1TasksPrintouts.txt.txt) and [S25 task archive](submission-25-zip_s1Tasks.zip)
- [S24 task trace](submission-24-zip_s1TasksPrintouts.txt) and [S21/S22 review](submission_21_22_review.md)
- [S25 workbench](../../submission_nb/submission_25/skills/repair/scripts/work.py), [agent instruction](../../submission_nb/submission_25/instruction.md), and [README / pilot criteria](../../submission_nb/submission_25/README.md)
- [S25 scripted-model smoke](../tools/smoke_submission25.py), [validation scope](../reviews/submission_25/validation.md), and [PR #166](https://github.com/prince22466/Co_AMAP/pull/166)
