# 05 Fill Additive Marking - `nonlocal_1color` Submission Record

This file records submissions for only the `nonlocal_1color` subtype of `fill_enclosed_regions`.

Task tree: `task_groups/05_fill_additive_marking_task_tree.md`  
Subtype: `nonlocal_1color`  
Count of taks: `41`  
Highest possbile score: `41*25 = 1025`  
Immediate target score: `0.6* 1025 = 615`  
Final target score: `0.75* 1025 = 768`  


## Submission Summary

| Kaggle Submission name | Modelling approach | Submission score | Submission file |
| --- | --- | --- | --- |
| 05-fill-additive-marking_nonlocal_1color - Version 1 | check the submission file | 0.0 | 05-fill-additive-marking-nonlocal-1color-1.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 2 | check the submission file | 26.83 | 05-fill-additive-marking-nonlocal-1color-2.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 3 | check the submission file | 51.85 | 05-fill-additive-marking-nonlocal-1color-3.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 4 | check the submission file | 65.58 | 05-fill-additive-marking-nonlocal-1color-4.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 5 | check the submission file | 65.58 | 05-fill-additive-marking-nonlocal-1color-5.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 6 | check the submission file | 65.58 | 05-fill-additive-marking-nonlocal-1color-6.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 7 | check the submission file | 65.58 | 05-fill-additive-marking-nonlocal-1color-7.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 9 | check the submission file | 65.58 | 05-fill-additive-marking-nonlocal-1color-9.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 10 | check the submission file | 65.58 | 05-fill-additive-marking-nonlocal-1color-10.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 11 | check the submission file | 65.58 | 05-fill-additive-marking-nonlocal-1color-11.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 12 | check the submission file | 65.58 | 05-fill-additive-marking-nonlocal-1color-12.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 13 | check the submission file | 65.58 | 05-fill-additive-marking-nonlocal-1color-13.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 14 | check the submission file | 11.77 | 05-fill-additive-marking-nonlocal-1color-14.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 15 | check the submission file | 77.36 | 05-fill-additive-marking-nonlocal-1color-15.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 17 | check the submission file | 77.36 | 05-fill-additive-marking-nonlocal-1color-17.ipynb |
| 05-fill-additive-marking_nonlocal_1color - Version 19 | check the submission file | 91.01 | 05-fill-additive-marking-nonlocal-1color-19.ipynb |


## Version 15 for 6 tasks
05-fill-additive-marking_nonlocal_1color - Version 15 runs only for 
`task002`, 
`task050`, 
`task126`, 
`task176`, 
`task299`, 
`task357`, 
using existing approachs in previous notebooks. It gets 77.36 = 11.77 + 65.58. Which means `task047` was not solved in previous runs.

## Version 14 for task002
05-fill-additive-marking_nonlocal_1color - Version 14 runs only for task002 using hand-designed symbolic CNN(not a learned CNN). And it works, which didnt in previous notebooks.


## Model Size Audit up to Version 11

Assumed competition requirement: each `taskNNN.onnx` should be at most about `1.44 MB`.

The sizes below are local lower-bound estimates from the notebook builders' dense initializers. Kaggle-executed profile values are still the source of truth, but these estimates are enough to identify models that are clearly too large.

### Models Meeting The Size Requirement

| Task | Model | Estimated size | First relevant version | Notes |
| --- | --- | ---: | --- | --- |
| task047 | `task047_cross_project_cnn` | ~0.005 MB | v5 | New after v4; size-safe. but not sure if it solve the tasks |
| task050 | `task050_line_connect_cnn` | ~0.003 MB | v4 | Likely part of current scoring baseline. |
| task126 | `task126_u_bottom_marker_cnn` | ~0.002 MB | v4 | Likely part of current scoring baseline. |
| task176 | `task176_periodic_completion_cnn` | ~0.002 MB | v4 | Likely part of current scoring baseline. |
| task299 | `task299_cross_extend_cnn` | ~0.227 MB | v4 | Likely part of current scoring baseline. |
| task357 | `task357_bounce_path_gemm` | ~0.326 MB | v4 | Likely part of current scoring baseline. |

### Models Larger Than The Size Requirement

| Task | Model | Estimated size | First relevant version | Notes |
| --- | --- | ---: | --- | --- |
| task060 | `task060_row_bridges_gemm` | ~13.9 MB | v5 | Too large for 1.44 MB cap. |
| task200 | `task200_periodic_lattice_gemm` | ~3.1 MB | v5 | Too large for 1.44 MB cap. |
| task232 | `task232_right_alternating_gemm` | ~323.0 MB | v5/v11 | Listed in v5; confirmed selecting in later builder. Too large. |
| task273 | `task273_corner_rectangles_gemm` | ~44.5 MB | v5 | Too large for 1.44 MB cap. |
| task323 | `task323_seed_staircase_gemm` | ~5.8 MB | v5 | Too large for 1.44 MB cap. |
| task348 | `task348_vertical_pyramid_gemm` | ~231.8 MB | v5 | Too large for 1.44 MB cap. |
| task336 | `task336_gap_fill_gemm` | ~593.3 MB | v6 | Too large for 1.44 MB cap. |
| task002 | `task002_visible_lookup_gemm` | ~9.2 MB | v9 | Lookup export; too large and likely poor hidden generalization. |
| task027 | `task027_visible_lookup_gemm` | ~9.1 MB | v9 | Lookup export; too large and likely poor hidden generalization. |
| task042 | `task042_visible_lookup_gemm` | ~9.0 MB | v9 | Lookup export; too large and likely poor hidden generalization. |
| task102 | `task102_visible_lookup_gemm` | ~9.2 MB | v9 | Lookup export; too large and likely poor hidden generalization. |
| task102 | `task102_square_hole_fill_gemm` | ~13.2 MB | v10 | Semantic attempt, but still too large. |
| task371 | `task371_midpoint_plus_gemm` | ~87.5 MB | v11 | Simulator rule exported by enumeration; too large. |

### Interpretation

Versions 5 through 11 did not improve over `65.58`. A likely reason is that most newly added exportable models were much larger than the assumed per-model limit and therefore were not usable as practical competition models. Future runs should only add compact models under `1.44 MB`, and dense GEMM/enumerated lookup exports should be treated as analysis tools unless compressed.
