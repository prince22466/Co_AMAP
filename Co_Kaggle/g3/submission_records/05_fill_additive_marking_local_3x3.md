# 05 Fill Additive Marking - `local_3x3` v1 Submission Record

This file records submissions for only the `local_3x3` subtype of `fill_enclosed_regions`.

Notebook: `submission_nbs/05_fill_additive_marking.ipynb`  
Task tree: `task_groups/05_fill_additive_marking_task_tree.md`  
Subtype: `local_3x3`  
Model version: `fill-additive-v0.1`  
Submission file expected by Kaggle: `/kaggle/working/submission.zip`

## Scope

Task selection rule:

```text
primary_family == "fill_enclosed_regions"
candidate_flags contains "local_3x3_consistent"
```

Selected task files:

| Task | Task File | Model Type | Unique 3x3 Rules | Estimated Params |
| --- | --- | --- | ---: | ---: |
| `task015` | `competition_material/taskfiles/task015.json` | `symbolic_3x3_patch_cnn` | 458 | 46,268 |
| `task081` | `competition_material/taskfiles/task081.json` | `symbolic_3x3_patch_cnn` | 267 | 26,977 |
| `task095` | `competition_material/taskfiles/task095.json` | `symbolic_3x3_patch_cnn` | 42 | 4,252 |
| `task220` | `competition_material/taskfiles/task220.json` | `symbolic_3x3_patch_cnn` | 92 | 9,302 |
| `task230` | `competition_material/taskfiles/task230.json` | `symbolic_3x3_patch_cnn` | 46 | 4,656 |
| `task258` | `competition_material/taskfiles/task258.json` | `symbolic_3x3_patch_cnn` | 209 | 21,119 |
| `task331` | `competition_material/taskfiles/task331.json` | `symbolic_3x3_patch_cnn` | 82 | 8,292 |
| `task352` | `competition_material/taskfiles/task352.json` | `symbolic_3x3_patch_cnn` | 675 | 68,185 |

## Model Type

Architecture:

```text
Conv(3x3 exact patch detectors)
-> Relu
-> Conv(1x1 detector-to-color logits)
```

Training method:

```text
symbolic rule extraction, one model per task
output color at cell = f(3x3 input patch around that cell)
```

ONNX ops:

```text
Conv, Relu, Conv
```

Competition shape contract:

```text
input:  [1, 10, 30, 30]
output: [1, 10, 30, 30]
positive logit means selected color channel
negative logits mean no selected channel
zero-hot output outside the task output grid
```

## Submission Summary

| Date | Submission ID | Submission Filename | Tasks Submitted | Model Type | Public LB Score | Estimated Local Points | Decision |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| YYYY-MM-DD | pending | `/kaggle/working/submission.zip` | 8 | `symbolic_3x3_patch_cnn` | TBD | up to `8 * 25 = 200` | Submit after Kaggle validation/profile cells pass |

## Model Size Summary

These are pre-export parameter estimates from the symbolic rule tables. Fill ONNX file sizes from the notebook profile after Kaggle build.

| Metric | Min | Avg | Max |
| --- | ---: | ---: | ---: |
| Unique 3x3 rules | 42 | 233.875 | 675 |
| Estimated params | 4,252 | 23,631.375 | 68,185 |
| ONNX file size bytes | TBD | TBD | TBD |
| Static memory bytes | TBD | TBD | TBD |
| Runtime memory bytes | TBD | TBD | TBD |
| Estimated competition cost | TBD | TBD | TBD |

## Visible Performance

Expected visible performance before ONNX runtime validation:

| Split | Exact Match | Total Examples | Notes |
| --- | ---: | ---: | --- |
| Train | expected 22 | 22 | All selected tasks have zero 3x3 rule conflicts. |
| Test | expected 8 | 8 | Visible test examples are included in rule fitting. |
| ARC-GEN | expected 2094 | 2094 | All selected tasks have zero 3x3 rule conflicts. |
| Visible all | expected 2124 | 2124 | Must confirm with ONNX runtime validation cell after export. |

After running the notebook profile cell, record actual ONNX validation here:

| Task | Train Right / Total | Test Right / Total | ARC-GEN Right / Total | Visible Right / Total | File Size Bytes | Params | Runtime Memory Bytes | Static Memory Bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task015` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `task081` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `task095` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `task220` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `task230` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `task258` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `task331` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `task352` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Kaggle Result

Fill this after each effective submission.

```text
Submission ID:
Submission filename:
Notebook output zip:
Public LB score:
Public LB delta:
Private score, if known:
Validation errors:
Rejected reason, if any:
```

## Notes

- This record is only for the `local_3x3` subtype.
- The notebook should emit only 8 ONNX files for this run.
- No identity fallback models should be included in this submission.
- If `submission.zip` contains more than the 8 selected task files, rerun the build cell after the stale-model cleanup step.
