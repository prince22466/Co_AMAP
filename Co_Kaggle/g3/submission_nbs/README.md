# Submission Notebooks

This folder contains one starter notebook per task family from `task_groups`.

Each notebook follows the same workflow:

1. Load task ids from `task_groups/task_type_groups.json`.
2. Inspect task metadata from `task_groups/task_type_map.csv`.
3. Define `train_family_task(task)`.
4. Generate one ONNX model per task for that family.
5. Build a `submission.zip` from the generated models.

## Files

| Notebook | Family |
| --- | --- |
| `01_identity_noop.ipynb` | `identity_noop` |
| `02_global_color_remap.ipynb` | `global_color_remap` |
| `03_same_shape_local_rule.ipynb` | `same_shape_local_rule` |
| `04_mask_object_selection.ipynb` | `mask_object_selection` |
| `05_fill_additive_marking.ipynb` | `fill_enclosed_regions` |
| `06_expansion_tiling.ipynb` | `expansion_tiling` |
| `07_cropping_extraction.ipynb` | `cropping_extraction` |
| `08_geometric_transform.ipynb` | `geometric_transform` |
| `09_pattern_completion.ipynb` | `pattern_completion` |
| `10_counting_relational.ipynb` | `counting_relational` |
| `11_composite_unknown.ipynb` | `composite_or_unknown` |

## Shared Helpers

`neurogolf_nb_common.py` contains shared functions for:

- loading task files and task groups
- converting grids to `[1, 10, 30, 30]` tensors
- inferring global color mappings
- building simple ONNX models
- validating visible examples with `onnxruntime`
- creating `submission.zip`

## Current Solver Coverage

Implemented starter model builders:

- identity / no-op
- global color remapping via 1x1 convolution

Other notebooks include the correct family-specific scaffolding and explicit `train_family_task` placeholders. They are ready for implementing the next solver generators without duplicating submission boilerplate.

The local environment used to create these files did not have `onnx` installed, so ONNX export was not executed locally.
