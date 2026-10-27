# Submission 21 — recover native editing; keep checks advisory

Gemma E4B single agent. This candidate is derived from Submission 19, **not**
Submission 20's mandatory candidate skill gateway.

## Evidence and design

On the same first 15 tasks, S19 produced 12 nonempty patches but solved 0;
S20 produced 0 nonempty patches and solved 0. S20 repeatedly used nonexistent
skills and tools, and never invoked its behavioral check or publish operations.
This makes patch production the immediate gating problem.

- Native get_status, read_file, edit_file, write_file, run_command and submit_patch
  are restored in agent.yaml. No special editing wrapper, candidate checkout,
  separate finalizer, or additional model.
- S19's **optional read-only** patch-audit skill is retained, with stronger
  protected-path coverage from S20 (case-insensitive test directories,
  .pytest.ini and _swegemma_stubs.py). Generated root-level .adk_exec_*
  scripts are intentionally excluded as harness scratch files, not protected;
  the audit returns an explicit ignored_runtime_scratch count. It detects syntax,
  removed top-level public definitions, changed protected paths and empty diffs.
- S20's baseline-fail / candidate-pass insight is translated into a model-guided
  **same-check before/after** behavioral workflow using native run_command.
  This does not add a hard publication gate. Native edits and submission can still
  bypass audits/checks, so no mandatory enforcement is claimed.
- E4B is told to stop after two failed edit attempts, avoid repetitive wrappers,
  reserve a finishing budget, check checkout import provenance, and report environment
  failures honestly. These are prompt guidance, **not enforced retry quotas**.
- A standalone official-evaluator pilot checks five fixed tasks only after all
  baselines are inspected for dependency/collection blockers. It writes full
  per-task diagnostics and never auto-starts a larger evaluation.

## What S21 does NOT claim

This is not an automatic safety/security boundary and cannot prevent native
submission of a syntax-broken patch; the agent must heed audit findings.
It does not guarantee baseline-fail/fixed-pass checks, faithful task coverage,
model tool-call competence or more solved tasks. Real E4B inference and leaderboard
improvement remain **unmeasured** until the pilot runs. It does not modify the
existing testing notebook or prior submission records.

## Packaging & validation

From repo root:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission21.py
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission21.py -v
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission21_pilot.py -v
# Optional pinned official runtime (not installed on generic CI):
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission21_runtime.py -v
```

The builder produces submission_nb/artifacts/submission_21.zip and a manifest.
CI uploads these two files as a GitHub Actions artifact. Runtime versions used
in earlier smoke tests: adk-submission 0.2.13, adk-eval-core 0.1.0,
swegemma 0.2.11, google-adk 1.36.1, google-genai 2.11.0.

## Five-task real-model pilot (instead of another 15-task run)

Use the official task snapshots, wheels and a running E4B OpenAI-compatible
endpoint. Baseline checks stop if dependencies or fixtures are broken. Next,
the pinned official ADK runtime must pass the packaged scripted audit/repair
smoke **without using E4B**. Before any of the five tasks are run, a
**real E4B synthetic native-tool smoke** must
read, edit, verify the same observed behavior, and submit a patch without a
skill dispatcher. Failed smoke stops the pilot; scripted smoke is not sufficient.

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/run_submission21_pilot.py \
  --tasks /path/to/tasks.jsonl --snapshots /path/to/snapshots \
  --wheels /path/to/wheels --model-url http://localhost:8000/v1 \
  --output /path/to/submission21-pilot
```

The fixed IDs are fastapi_15588, fastapi_14986, requests_7427,
requests_7433 and fastapi_15023. The automated pilot only marks
`candidate_for_review` if all five return nonempty patches, report
successful *post-edit focused checks followed by submission* in their
traces, have no recorded tool-dispatch failures, and at least one
task resolves. These heuristics do **not** establish semantic relevance.
Unknown trace layouts fail closed instead of inventing check results.

The first pilot always sets `ready_for_15_tasks: false`.
After independently reviewing all five diffs and behavioral checks,
create a JSON file with `tasks` containing all five `instance_id`
values and boolean `patch_relevant`, `behavior_check_relevant`,
and `final_diff_reviewed` for each. Mark true ONLY after direct
review. Re-evaluate the saved results without any GPU access:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/run_submission21_pilot.py \
  --review-existing --output /path/to/submission21-pilot \
  --review-file /path/to/manual_review.json
```

This produces `manual_review_result.json`. Until independent
verification succeeds the promotion flag stays false. The criterion
is an experimental pilot decision, not a Kaggle score or an
unbypassable in-agent guard.
