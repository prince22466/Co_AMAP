# Evidence for submission 9

Source snapshot: `cbe742951dcc089b56ba9331f64b3d64647ae595` (2026-10-20). The [public-score record](../../../submission_records.md) is authoritative for reported leaderboard scores. Counts below are recomputed from committed diagnostic ZIPs, not the absent prior reviews linked by older READMEs.

## Baseline selection

| Version | Public score | Diagnostic resolved | Missing-argument errors | Tasks at 40 counted calls |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.12 | 4/30 | 21 | 10/30 |
| 6 | 0.03 | 1/30 | 6 | 0/30 |
| 7 | **0.13** | 4/30 | **33** | **12/30** |
| 8 | 0.12 | 0/7 | 0 | 3/7 |
| 9 | Not measured | Not run | Not measured | Not measured |

Version 8 has only seven diagnostic tasks; its 0/7 is not comparable to a full 30-task rate. Both versions 7 and 8 fail on those seven recorded tasks. Public and local scores are different measurements.

Version 7's reported passes are `rich_4077`, `fastapi_14851`, `fastapi_14794` and `fastapi_14786`. `fastapi_14851` has an empty patch: there are **three nonempty-patch passes**, not evidence of four implemented fixes. Keep all four as regression cases and report the empty-patch case separately.

All historical runs disable the task timer and use swegemma 0.2.11, adk-submission 0.2.13, google-adk 1.36.1 and vLLM 0.19.1. Recorded task durations are not isolated generation time and cannot directly validate the packaged four-minute timer. Five version-7 tasks have missing `inline_snapshot`; four have recursive fixture errors. These do not exclude simultaneous implementation defects.

## Trace evidence and interventions

Paths are relative to the root inside [the submission-7 diagnostic ZIP](../submission-7-zip_First30Tasks.zip). Task-specific evidence stays outside the candidate archive.

| Observation | Interpretation | Skill response and limit |
| --- | --- | --- |
| `report/*_tools.json`: 33 missing-field rejections: 15 on `rich_4070`, 9 on `fastapi_14962`, 8 on `fastapi_14964`, 1 on `requests_7502`. | Baseline prompt guidance did not prevent repeated malformed calls. | Short, explicit recovery and guarded shell fallback; scripts reduce repeated generated inspection/check/audit payloads. No new run proves the coder follows the guidance. |
| `report/rich_4075_tools.json`: searches consume 40 calls; the implementation edit is rejected for exhausted budget; final patch is empty. | Navigation can consume the entire editing allowance. | Bounded definition/range inspection and earlier decision checkpoint. The script still needs a candidate source file. |
| `report/rich_4076_tools.json`: repeated print-based checks near the call limit. | Plausible output is weak verification. | Assert exact boundary output; label print-only checks. Assertions may still encode incorrect expectations. |
| `report/requests_7315_tools.json`: checks import installed requests; a later PYTHONPATH change finally tests workspace/src. | Experiments can test the wrong code. | Workspace-first imports, explicit named-module origin check, fresh process. Transitive origins are not exhaustively verified. |
| Same trace: a caught failure prints a message but exits 0; a repository test is edited to match the candidate. | Exit 0 and changed expected answers create false confidence. | Count executed inline assertions and caught failures; distinguish uncalled/print-only/early-exit checks; audit protected changes. Coder must still act on the findings. |
| `test_outputs/fastapi_14978_returned.log`: APIRouter rejects a new keyword; the coder's reproduction exercised a lower-level Body option. | A private-path pass can miss the public contract. | Map public entry, forwarding, consumer, default and override. Sparse issues still leave uncertainty. |
| `test_outputs/fastapi_15030_returned.log`: missing public export alongside missing external dependencies. | Collection errors need individual classification. | Inspect exports; distinguish an omitted implementation from an environment failure. |
| `test_outputs/fastapi_15588_returned.log`: six failures are error-message regex mismatches. Test and candidate both reject multiline values during construction. | This is observable wording mismatch, **not a validation-stage mismatch**. | Inspect validator/message conventions and preserve evidence-backed wording. No task-specific expected string is shipped; unspecified strings remain unknowable from a generic skill. |
| One analyst invocation, zero test-designer invocations and one get_status call across 30 tasks. | Merely adding another helper or instruction may have no effect. | Keep helpers unchanged; explicitly route to one skill and measure actual activation. Activation remains model-driven. |

The final audit is advisory. It flags structural problems without changing files, staging, submitting or altering the harness. It cannot judge semantic correctness. Existing native tools remain available.

## Scope and rules

The source retains `skills/repo_navigation/{SKILL.md,scripts/,resources/}` as requested. The builder packages it as `skills/repo-navigation`, adjusts `agent.yaml`, and generates `references/` aliases from the canonical resource markdown. The manifest name matches the packaged directory. The model, tools, helper roles, sampling and budgets remain as in submission 7. Scripts run inside the provided sandbox, sharing its budget. No custom host Python entry point is added.

Rule sources: [Description](../../../competition_material/Description.md), [Evaluation](../../../competition_material/Evaluation.md), [HARNESS_README](../../../competition_material/competition_data/HARNESS_README.md). Runtime verification uses official packages from [wheelhouse version 31](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse/versions/31). The installed ADK loader rejects snake_case names by default and ignores `resources/`; compilation and real resource loading confirm that the build conversions resolve both issues. See also [ADK skill tools](https://github.com/google/adk-python/blob/v1.36.1/src/google/adk/tools/skill_toolset.py) and [skill loading](https://github.com/google/adk-python/blob/v1.36.1/src/google/adk/skills/_utils.py).

The deterministic release contains 20 active files, including three generated reference aliases. Task IDs, diagnostic logs, grading outputs, task data, reference patches and expected grading answers remain outside the ZIP. The evidence generator reads archived observations, not tasks.jsonl's reference `patch` or `test_patch` fields.

## Validation and measurement

Twenty-eight CPU-only synthetic tests and eight official-component smoke checks pass. The [runtime report](runtime_smoke.json) ties the results to the ZIP hash and dependency versions. Compilation, all three resource loads and all three scripts run through the real ADK tools and adk-eval-core subprocess executor. The model/native tools are test doubles and the workspace is synthetic; this is not a full Kaggle environment or an inference/grading run. Gemma performance and competition budget accounting remain unmeasured. The [candidate README](../../../submission_nb/submission_9/README.md) contains reproduction commands and the remaining model/evaluation gates.

The review corrected false passes for uncalled/caught/optimized assertions, output overflow from long definition names, missed untracked whitespace/non-Python conflicts/dangling symlinks, and the audit counting its own ADK executor artifacts as a patch. New regression checks cover these cases. Assertions in imported or dynamically generated code are not counted, and an assertion that encodes the wrong requirement can still pass.

Freeze the ZIP hashes, check loading/execution on one development task, then compare 7 and 9 on identical 30-task runs with the packaged timer. Report paired wins/losses and environment failures separately. Measure skill invocation, rejected calls, repeated reads, wrong import origins, protected mutations, patch emptiness and total generation/setup cost. Then evaluate a frozen unseen split excluding earlier exposed base-commit groups. Small-task overhead is a regression risk; skill steps should replace equivalent native operations.

Recompute metrics from repository root:

```bash
python -B Co_Kaggle/g6/tests_on_tasks/tools/review_submission9_evidence.py --output Co_Kaggle/g6/tests_on_tasks/diagnostics/submission_9_review/evidence.json
```

[evidence.json](evidence.json) records archive hashes, run settings, tool counts, task-level error attribution and environment signals. Nothing in this PR establishes that submission 9 outperforms submission 7.
