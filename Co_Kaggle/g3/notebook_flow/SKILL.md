# Notebook Iteration Flow for NeuroGolf Submissions

Use this skill when improving Kaggle NeuroGolf notebooks, especially solver-family notebooks that build one ONNX model per task. The goal is fast, disciplined notebook iteration: copy a known baseline, make one controlled modelling change, verify locally as much as possible, run on Kaggle, and record the score.

## Core Rules

- Keep task scope explicit. Do not mix task subtypes unless the submission record says the run is a hybrid.
- Competition requirement is one `taskNNN.onnx` model per submitted task.
- Always copy the best valid notebook for the current task scope before changing it.
- Change as few cells as possible, usually the trainer/model-selection cell and sometimes the dry-run/build cells.
- Every notebook must generate `/kaggle/working/submission.zip`.
- Track the exact notebook filename, score, modelling approach, task set, model sizes, memory profile, and visible validation in `submission_records`.
- If a model has visible accuracy `1.0` but score drops, treat it as hidden-generalization failure, not an export failure.

## Standard Notebook Workflow

1. Pick the current best in-scope notebook from the submission record.
2. Copy it to the next versioned notebook filename.
3. Update `MODEL_VERSION`.
4. Make one modelling change.
5. Run dry fitting without saving models.
6. Verify selected task IDs and trainer choices.
7. Verify visible examples by simulation or ONNX validation.
8. Build the submission zip.
9. On Kaggle, run all cells from top to bottom.
10. Download the executed notebook and record score/profile.

## Required Cells

Every competition notebook should contain:

- inline helper functions, not imports from local repo files that Kaggle cannot upload
- dependency setup for `onnx` and `onnxruntime`
- explicit `FAMILY`, `MODEL_VERSION`, `DATA_DIR`, `OUT_DIR`
- task selection cell
- trainer/model-definition cell
- dry-run selector report
- build cell using `build_family_submission`
- validation/profile cell reporting:
  - `task_id`
  - trainer/model type
  - file size
  - params
  - nodes/op counts
  - static memory
  - runtime memory
  - train/test/arc-gen/visible accuracy
- final copy to `/kaggle/working/submission.zip`

## Dry-Run Cell Safety

Dry-run and build cells should be self-contained enough to run even if a display/selection cell was skipped. If they rely on variables like `task_ids`, `submission_task_ids`, or `EXTRA_TASK_IDS`, define fallback logic inside the cell.

Example pattern:

```python
if 'task_ids' not in globals():
    task_map = load_task_type_map()
    family_df = task_map[task_map['primary_family'] == FAMILY].copy()
    local_3x3_df = family_df[family_df['candidate_flags'].fillna('').str.contains('local_3x3_consistent')].copy()
    task_ids = local_3x3_df['task_id'].tolist()
```

## Local Verification

Local environment may not have `onnx` or `onnxruntime`. If so, still verify:

- notebook JSON loads
- all code cells compile
- dry-run selector executes without ONNX export
- pure NumPy simulation of the model rule matches all visible examples
- selected task IDs match the intended scope
- no stale task files are included in `OUT_DIR`

Do not assume local success means Kaggle export success. Kaggle-executed notebook profile is the source of truth for ONNX runtime memory and visible accuracy.

## Local_3x3 Case Study

Task subtype: `fill_enclosed_regions / local_3x3`

Strict task set:

```text
task015
task081
task095
task220
task230
task258
task331
task352
```

Score history:

```text
v1   47.92
v2   48.74
v3   74.32
v5   91.51
v6   91.51
v7   93.39
v8   93.64
v9   107.26
v11  108.49
v12  108.74
v13  108.49
v15  91.51
v17  13.46
v21  109.55
v22  109.69
v24  96.06
v25  96.06
v26  109.69
v27  111.27
v28  114.29
v29  114.97
v30  101.34
v31  115.20
v34  115.20
```

Best strict-scope baseline so far:

```text
05-fill-additive-marking-local-3x3-31.ipynb
score: 115.20
```

## Local_3x3 Successful Models

Current strong model assignment:

```text
task015 -> semantic_stamp_cnn
task081 -> l_corner_fill_cnn, lite identity path
task095 -> semantic_stamp_cnn
task220 -> semantic_stamp_cnn
task230 -> task230_direct5x5_cnn
task258 -> semantic_pair_gap_cnn
task331 -> semantic_stamp_cnn
task352 -> semantic_stamp_cnn
```

Useful semantic patterns:

- `semantic_stamp_cnn`: one Conv, background-gated stamp around seed colors.
- `semantic_pair_gap_cnn`: one Conv for horizontal `1 0 1 -> 1 2 1`.
- `task230_direct5x5_cnn`: direct 5x5 linear Conv for 2x2 block marker.
- `l_corner_fill_cnn`: detects 2x2 blocks with exactly three `8` cells and fills the missing corner.

## Lessons From Failed or Neutral Runs

- Smaller kernel is not automatically better. The 2x2 idea did not get selected because 3x3/semantic models explained the data better.
- Float16 model internals are acceptable when graph I/O stays float32. Pure float16 I/O failed submission.
- Direct fitted models can overfit hidden tests even with visible accuracy `1.0`.
- `task081` direct 7x7 Conv scored badly despite perfect visible validation. Keep task081 on the semantic l-corner rule.
- `task331` compact seed-stamp attempts produced visible accuracy failures. Keep task331 on the safe semantic directional stamp unless ONNX validation proves otherwise.
- Priority suppression for overlapping semantic stamps was neutral: safe, but did not improve score.
- Adding tasks from another subtype can push score but is out of scope for a strict subtype record.

## Good Improvement Pattern

Prefer semantic compression over blind fitting:

1. Inspect task grids.
2. Name the rule in human terms.
3. Simulate the rule on all train/test/arc-gen examples.
4. Export the smallest ONNX graph that preserves that rule.
5. Compare Kaggle score.

Good examples:

- v27: semantic models for `task015` and `task220`, score improved.
- v28: semantic models for `task095`, `task258`, `task331`, `task352`, score improved.
- v29: direct 5x5 model for `task230`, score improved.
- v31: l-corner lite architecture for `task081`, score improved slightly.

## Red Flags

Avoid or treat with suspicion:

- changing hidden semantics for a task that already has a good semantic rule
- direct high-capacity fitted Conv for tiny visible datasets
- adding extra tasks when the record is meant to track one subtype
- relying on a variable from an optional display cell
- visible accuracy `1.0` as the only success criterion
- notebook versions whose filename and record scope disagree

## When Score Drops

If score drops:

1. Check visible accuracy from the downloaded Kaggle notebook.
2. If visible failed, fix ONNX export or model logic.
3. If visible is `1.0`, classify as hidden-generalization failure.
4. Revert to last best in-scope notebook.
5. Preserve the failed notebook as evidence, but do not build from it.

Known examples:

- v24/v25: `task331` compact export visibly failed, score dropped to `96.06`.
- v30: `task081` direct 7x7 visibly passed but hidden-generalization failed, score dropped to `101.34`.

## Submission Record Discipline

For each effective submission, record:

- Kaggle submission name/version
- notebook filename
- score
- task scope
- model assignment per task
- model params and memory
- train/test/arc-gen/visible accuracy
- what changed from previous version
- interpretation: memory win, hidden-generalization win/loss, export failure, or neutral

## Transfer to Other Task Types

This workflow transfers directly to other subtype notebooks:

- define strict task set first
- build one model per task
- make dry-run selector transparent
- use semantic models before fitted models
- only use fitted direct Conv after quantized visible validation
- keep fallback models safe
- inspect hidden-generalization failures by comparing score movement, not only visible accuracy
- do not mix task subtypes unless the notebook and record explicitly say it is a hybrid run

