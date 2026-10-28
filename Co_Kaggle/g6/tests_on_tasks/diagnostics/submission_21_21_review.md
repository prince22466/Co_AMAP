# Submission 21–22 Review: Progress and Remaining Problems

## 1. Overall result

**Submission 21 restored a workable agent architecture after Submission 20's tool-access problems. Submission 22 added recovery and verification mechanisms, but these did not produce a measurable improvement in task resolution.**

Both versions use Gemma 4 E4B and were evaluated on the same 15 selected S1 tasks.

| Metric | Submission 21 | Submission 22 |
|---|---:|---:|
| Tasks resolved | 2/15 | 2/15 |
| Nonempty patches | 11/15 | 10/15 |
| Empty patches | 4/15 | 5/15 |
| Total evaluation duration | 34.9 min | 30.1 min |
| Reported token usage | 3.28M | 3.67M |
| Optional skill calls observed | 0 | 0 |

**Conclusion:** Submission 22 did not outperform Submission 21 in resolution rate or patch-production reliability.

## 2. Progress made

### Submission 21 — Restoring basic functionality

- Restored native `read_file`, `edit_file`, `write_file`, `run_command`, and `submit_patch` access.
- Avoided Submission 20's mandatory skill-edit gateway, which had prevented effective patch creation.
- Successfully resolved `rich_4077` and `rich_3905`.
- Established a useful baseline for investigating agent-level failures.

**Achievement:** The agent could reliably access its native tools and sometimes produce correct fixes. The primary bottleneck shifted from tool availability toward tool-use quality and task-solving behavior.

### Submission 22 — Improving the architecture

- Added bounded edit-recovery logic for unsuccessful exact-string replacements.
- Added structural auditing, patch inspection, and empty-patch detection.
- Added candidate behavioral checks, import provenance checks, and changed-code execution checks.
- Expanded CPU regression coverage to 24 passing tests.
- Successfully resolved two previously failed tasks: `rich_3518` and `rich_3454`.
- Reduced total evaluation duration by approximately 14%.

**Achievement:** The supporting code and recovery mechanisms became more robust, and two additional tasks were shown to be solvable.

**Limitation:** The agent did not invoke the optional skills during the S1 evaluation. Therefore, the added architecture was not demonstrated to improve actual agent behavior.

## 3. Problems that remain

| Problem | Evidence | Likely cause / category |
|---|---|---|
| Malformed tool calls | `rich_4077`: zero actual tool calls, despite text resembling tool invocations | Model–tool interface reliability |
| Repeated failed edits | `rich_3882`: 18 edit attempts, 17 visible errors | Model behavior + insufficient runtime enforcement |
| Repetitive file exploration | `rich_3905`: repeated reads, 40-call exhaustion, empty patch | Planning and context management |
| Empty submissions | S22 returned five empty patches | Patch preservation and workflow reliability |
| Weak behavioral verification | Existing tests were often run without clear evidence that they tested the requested change | Verification design + model judgment |
| Unused architectural safeguards | Zero `run_skill_script` calls observed in S22 | Optional mechanisms not adopted by agent |
| Environment uncertainty | Recurring `ensurepip` warnings and ambiguous command-error output | Evaluation environment / observability |
| Incorrect or incomplete fixes | Most nonempty patches still failed evaluation | Model capability, task understanding, or verification failure |

The S1 logs do not show the earlier `inline_snapshot` or recursive `httpbin` fixture errors, but this does not establish that all evaluation environments are clean.

There is also a task-description discrepancy for `rich_4075`: the selected-task document and the agent trace describe different issues. Its original task record should be checked.

## 4. Regressions from Submission 21 to 22

**Previously solved tasks that failed in Submission 22:**
- `rich_4077` — zero valid tool calls.
- `rich_3905` — repeated exploration and tool-budget exhaustion.

**Newly solved tasks in Submission 22:**
- `rich_3518` — table column highlighting.
- `rich_3454` — hyperlink highlighting.

There were no tasks solved by both versions.

This suggests significant sensitivity to agent behavior across runs. A single evaluation per version is insufficient to determine whether the architectural changes provide stable improvements.

## 5. Agent architecture vs. model capability

**Agent-system problems we can address through engineering:**

- Tool-call validation and malformed-call recovery.
- Limits on repeated reads and identical failed edits.
- Tool-budget reservation for editing, testing, and submission.
- Automatic final-patch existence checks.
- More reliable baseline-versus-candidate behavioral verification.
- Better recording of environment failures and evaluator results.

**Potential model capability limitations:**

- Difficulty locating the correct implementation in large repositories.
- Incorrect or incomplete interpretation of task requirements.
- Repeated use of unsuccessful editing strategies.
- Failure to construct meaningful behavioral tests.
- Inability to finish complex tasks within the available context and tool-call budget.

The logs demonstrate these behaviors but do not yet establish how much each category contributes to failed resolutions.

## 6. Recommendation for Submission 23

**Priority 1 — Tool-call reliability:** Prevent malformed tool-call responses and address zero-tool-call failures before introducing more complex agent orchestration.

**Priority 2 — Enforced workflow control:** Use supported ADK mechanisms to limit repetitive searches, repeated failed edits, and premature submissions. Keep native editing available to avoid reintroducing Submission 20's failures.

**Priority 3 — Reliable verification:** Require meaningful evidence that the requested behavior changed, the final patch exists, and checks are relevant. Distinguish failures caused by the environment from failures caused by the patch.

Evaluate these changes on the same selected tasks, with repeat runs where feasible, and compare against both Submission 21 and Submission 22.

## 7. Final conclusion

**Submission 21 established a usable native-tool baseline. Submission 22 improved the supporting code but did not improve measured task-solving performance.**

The key lesson is that **implementing safeguards is not the same as enforcing them during agent execution**.

The next improvement should prioritize reliable tool invocation, bounded execution, and meaningful behavioral verification before adding further skills or agent complexity.
