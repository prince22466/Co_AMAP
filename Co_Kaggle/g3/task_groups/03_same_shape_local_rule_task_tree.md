# Same-Shape Local Rule Task Profile

Source: `task_groups/task_type_map.csv`

## Task Type

| Field | Value |
| --- | --- |
| Primary family | `same_shape_local_rule` |
| Task count | `27` |
| Common shape relation | `same_shape_variable_size` |
| Common behavior | Output has the same shape as input and each output cell is fully or nearly determined by its local 3x3 input neighborhood. |
| Current notebook | `(not yet assigned)` |

## Flag-Based Subtypes

| Subtype Flags | Count | Solver Meaning |
| --- | ---: | --- |
| `local_3x3_consistent` | 12 | Local same-shape rule using only existing input colors. |
| <code>local_3x3_consistent&#124;new_output_colors</code> | 15 | Local same-shape rule that emits new output colors. |

## Practical Solver Subtypes

This split is mechanical and solver-oriented:

- `exact_local` means the 3x3 profile has zero conflicting input-patch labels.
- `near_local` means the local score is at least 0.995 but a few conflicting patch labels remain.
- `reuse_colors` means all output colors occur in the input; `new_colors` means at least one does not.

| Practical Subtype | Count | Characteristics | Current Solver Status |
| --- | ---: | --- | --- |
| `exact_local_reuse_colors` | 9 | Zero-conflict local rule using existing colors. | Best candidate for exact symbolic 3x3 patch lookup. |
| `exact_local_new_colors` | 12 | Zero-conflict local rule that emits new colors. | Candidate for symbolic 3x3 inference with complete output channels. |
| `near_local_reuse_colors` | 3 | Near-consistent local rule using existing colors. | Needs conflict resolution, larger context, or fallback. |
| `near_local_new_colors` | 3 | Near-consistent local rule that emits new colors. | Needs conflict resolution and complete color encoding. |

## Practical Subtype Metadata

### `exact_local_reuse_colors`

| Metric | Value |
| --- | --- |
| Count | `9` |
| New color count modes | `[(0, 9)]` |
| Local 3x3 score min/avg | `1.0000` / `1.0000` |
| Local 3x3 conflicts min/avg | `0` / `0.0000` |
| Added nonzero cells avg | `264.6667` |
| Changed cells avg | `5555.2222` |
| Input nonzero preserved ratio avg | `0.5992` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `task004` | `competition_material/taskfiles/task004.json` | `local_3x3_consistent` | `[]` | `9x8:8;15x9:7;16x12:7;16x14:7;8x10:7` | 1.0000 | 0 | 2252 | 4504 | 0.6649 |
| `task053` | `competition_material/taskfiles/task053.json` | `local_3x3_consistent` | `[]` | `3x3:60` | 1.0000 | 0 | 130 | 260 | 0.1613 |
| `task073` | `competition_material/taskfiles/task073.json` | `local_3x3_consistent` | `[]` | `5x5:15` | 1.0000 | 0 | 0 | 48 | 0.6098 |
| `task097` | `competition_material/taskfiles/task097.json` | `local_3x3_consistent` | `[]` | `17x14:5;13x10:5;20x7:4;15x8:4;19x6:4` | 1.0000 | 0 | 0 | 1976 | 0.5287 |
| `task098` | `competition_material/taskfiles/task098.json` | `local_3x3_consistent` | `[]` | `10x10:9;12x14:8;20x20:8;17x17:8;8x6:8` | 1.0000 | 0 | 0 | 4975 | 0.7096 |
| `task192` | `competition_material/taskfiles/task192.json` | `local_3x3_consistent` | `[]` | `20x20:26;20x19:22;18x20:17;19x20:17;19x19:16` | 1.0000 | 0 | 0 | 3852 | 0.8715 |
| `task193` | `competition_material/taskfiles/task193.json` | `local_3x3_consistent` | `[]` | `8x8:25;9x9:25;18x18:22;7x7:21;15x15:20` | 1.0000 | 0 | 0 | 1797 | 0.8298 |
| `task222` | `competition_material/taskfiles/task222.json` | `local_3x3_consistent` | `[]` | `16x16:266` | 1.0000 | 0 | 0 | 31971 | 0.0990 |
| `task293` | `competition_material/taskfiles/task293.json` | `local_3x3_consistent` | `[]` | `13x9:7;6x5:7;13x11:6;5x7:6;13x7:6` | 1.0000 | 0 | 0 | 614 | 0.9180 |

### `exact_local_new_colors`

| Metric | Value |
| --- | --- |
| Count | `12` |
| New color count modes | `[(1, 9), (3, 1), (4, 2)]` |
| Local 3x3 score min/avg | `1.0000` / `1.0000` |
| Local 3x3 conflicts min/avg | `0` / `0.0000` |
| Added nonzero cells avg | `2507.7500` |
| Changed cells avg | `4975.1667` |
| Input nonzero preserved ratio avg | `0.3465` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `task120` | `competition_material/taskfiles/task120.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[8]` | `15x13:28;14x12:24;12x14:23;12x12:23;14x14:20` | 1.0000 | 0 | 0 | 4929 | 0.6937 |
| `task127` | `competition_material/taskfiles/task127.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[6,7,8,9]` | `7x11:202;3x11:65` | 1.0000 | 0 | 11256 | 12663 | 0.7816 |
| `task147` | `competition_material/taskfiles/task147.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[8]` | `5x3:24;3x3:22;6x3:20;6x5:20;5x5:19` | 1.0000 | 0 | 0 | 2381 | 0.1145 |
| `task151` | `competition_material/taskfiles/task151.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[4]` | `8x8:38;5x5:36;6x6:34;4x4:33;9x9:30` | 1.0000 | 0 | 1064 | 2128 | 0.7242 |
| `task261` | `competition_material/taskfiles/task261.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[2]` | `7x7:72;5x5:68;6x6:62;4x4:44;3x3:19` | 1.0000 | 0 | 744 | 1834 | 0.0000 |
| `task266` | `competition_material/taskfiles/task266.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[3,6,7,8]` | `3x5:20` | 1.0000 | 0 | 45 | 65 | 0.0000 |
| `task272` | `competition_material/taskfiles/task272.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[1]` | `5x4:44;5x5:34;4x5:33;5x3:29;3x4:28` | 1.0000 | 0 | 0 | 477 | 0.7743 |
| `task282` | `competition_material/taskfiles/task282.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[1]` | `9x9:265` | 1.0000 | 0 | 7384 | 8307 | 0.0000 |
| `task283` | `competition_material/taskfiles/task283.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[1,2,4]` | `10x10:265` | 1.0000 | 0 | 0 | 10369 | 0.0000 |
| `task294` | `competition_material/taskfiles/task294.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[2]` | `10x10:265` | 1.0000 | 0 | 0 | 4689 | 0.6486 |
| `task317` | `competition_material/taskfiles/task317.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[1]` | `9x9:265` | 1.0000 | 0 | 9600 | 10800 | 0.0000 |
| `task344` | `competition_material/taskfiles/task344.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[8]` | `9x7:11;6x10:11;8x9:10;10x10:9;8x7:9` | 1.0000 | 0 | 0 | 1060 | 0.4211 |

### `near_local_reuse_colors`

| Metric | Value |
| --- | --- |
| Count | `3` |
| New color count modes | `[(0, 3)]` |
| Local 3x3 score min/avg | `0.9979` / `0.9986` |
| Local 3x3 conflicts min/avg | `1` / `2.6667` |
| Added nonzero cells avg | `50.0000` |
| Changed cells avg | `3797.0000` |
| Input nonzero preserved ratio avg | `0.4645` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `task129` | `competition_material/taskfiles/task129.json` | `local_3x3_consistent` | `[]` | `3x3:265` | 0.9981 | 1 | 150 | 1590 | 0.3287 |
| `task329` | `competition_material/taskfiles/task329.json` | `local_3x3_consistent` | `[]` | `3x3:80;7x7:74;5x5:59;9x9:53` | 0.9979 | 5 | 0 | 4261 | 0.1557 |
| `task359` | `competition_material/taskfiles/task359.json` | `local_3x3_consistent` | `[]` | `15x13:9;15x15:8;16x15:8;13x15:8;13x14:7` | 0.9999 | 2 | 0 | 5540 | 0.9090 |

### `near_local_new_colors`

| Metric | Value |
| --- | --- |
| Count | `3` |
| New color count modes | `[(1, 3)]` |
| Local 3x3 score min/avg | `0.9970` / `0.9980` |
| Local 3x3 conflicts min/avg | `3` / `19.0000` |
| Added nonzero cells avg | `0.0000` |
| Changed cells avg | `3772.0000` |
| Input nonzero preserved ratio avg | `0.6069` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `task070` | `competition_material/taskfiles/task070.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[3]` | `17x17:266` | 0.9988 | 20 | 0 | 3486 | 0.8942 |
| `task077` | `competition_material/taskfiles/task077.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[4]` | `17x18:21;15x15:19;20x19:19;15x16:18;20x20:18` | 0.9981 | 34 | 0 | 3365 | 0.9265 |
| `task389` | `competition_material/taskfiles/task389.json` | <code>local_3x3_consistent&#124;new_output_colors</code> | `[0]` | `5x5:94;3x3:91;4x4:81` | 0.9970 | 3 | 0 | 4465 | 0.0000 |
