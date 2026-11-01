# Submission 29: evidence-first native repair with release safety gates

**Unscored experiment** for Gemma E4B. Do not describe S29 as better than S27
until actual matched E4B evaluations demonstrate more resolved issues and
retention of known successes.

## Hypothesis and isolation

S28 resolved **1/15** versus S27's **2/15** on the same S1 set. S28 had
fewer reads and fewer tool-budget exhaustions, but **93 observed failed edits**
versus S27's **78**, and a catastrophic FastAPI patch of 1,123,099 characters
(20,487 lines changed). S28 lost the previously solved `rich_3454`.

S29 restores a conservative, evidence-first workflow and removes the dangerous
shell-edit fallback. Its only runtime difference from S27/S28 is
`instruction.md`. Agent model, generation settings, six native tool schemas,
and evaluation limits are byte-identical to S27/S28. There is **no mandatory
skill wrapper, custom tool, secondary agent, LoRA, or altered benchmark**.

## Mechanisms

| Failure | S29 intervention | Enforcement scope |
| --- | --- | --- |
| Wrong nearby code target (S28 `rich_4077`) | Inspect issue-named implementation and caller; define observable behavior | Agent instruction only |
| Malformed native `edit_file` (S28 `rich_3454`) | Explicit required arguments, small uniquely anchored replacement | Agent instruction; schema still validated by harness |
| Repeated failed anchors (S28 `rich_3718`) | One new verified anchor after failure; change strategy after second | Agent instruction; post-run trace gate |
| Runaway shell insertion (S28 `fastapi_14258`) | Prohibit source mutation via `sed -i`, scripted writes and redirects; inspect numstat | Agent instruction; deterministic post-run trace gate |
| Green but irrelevant tests | Require relevant observable behavior / negative case | Agent instruction; manual review of real diff/check |
| Excess patch size or lost known successes | Reject candidate promotion | Deterministic post-run gate, NOT in-agent interception |

The official declarative configuration does not provide an interception hook
for every built-in native tool call. S29 therefore **does not claim to enforce**
these constraints at edit or submit time. Adding a mandatory custom workbench
would repeat S25's empirically failed interface.

## Files and execution

- `submission_nb/artifacts/submission_29.zip`: uploadable agent config.
- `tests_on_tasks/tools/build_submission29.py`: reproducible ZIP and manifest.
- `tests_on_tasks/tools/test_submission29.py`: packaging and gate tests (not model tests).
- `tests_on_tasks/tools/gate_submission29.py`: evaluate printed S1 traces
  and reject known unsafe patterns; printed traces omit exact tool arguments.
- `tests_on_tasks/tools/run_submission29_eval.py`: matched official-runtime
  pilot/S1 runner; never automatically promotes a submission.

From `Co_Kaggle/g6/tests_on_tasks/tools`:
```sh
python -m unittest test_submission29.py
python build_submission29.py
python gate_submission29.py \
  ../diagnostics/submission-27-zip_s1TasksPrintouts.txt.txt \
  /path/to/submission-29-zip_s1TasksPrintouts.txt.txt --stage s1
```

## Release procedure (manual approval remains required)

First run the official environment preflight. Test **two real E4B pilot
repeats** on the predeclared tasks `rich_4077`, `rich_3905`, `rich_3518`,
`rich_3454`, `rich_3882` (same model and resource limits as S27).
Require >=3/5 resolved **per repeat**, retain both `rich_3905` and
`rich_3454`, at most one empty patch, and no agent errors. Inspect full
tool-call arguments, patch diffs and relevant tests manually. Reject promotion
on unsafe shell editing, excessive patch size, or sustained repeated errors.

Only after passing the pilot, evaluate the matched **15-task S1** and compare
against S27. An S1 candidate must resolve >=3/15, retain both known wins,
show no catastrophic patch, and be independently reviewed. These thresholds
are engineering release criteria, not guarantees of hidden Kaggle performance.
If model or sandbox access is unavailable, mark the performance gate
**NOT RUN**, never PASS. A single matched test is directional evidence only.

Do not edit `gemma-4-developer-agent-testing_v9.ipynb` or grading tests.
