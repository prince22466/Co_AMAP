# Expansion / Tiling Task Profile

Source: `task_groups/task_type_map.csv`

## Task Type

| Field | Value |
| --- | --- |
| Primary family | `expansion_tiling` |
| Task count | `35` |
| Common shape relation | `expands` |
| Common behavior | Output is larger than input, usually through fixed or variable repetition, tiling, scaling, padding, or shape-derived canvas expansion. |
| Current notebook | `(not yet assigned)` |

## Flag-Based Subtypes

| Subtype Flags | Count | Solver Meaning |
| --- | ---: | --- |
| `shape_expands` | 30 | Expansion using only colors already present in the input. |
| <code>shape_expands&#124;new_output_colors</code> | 5 | Expansion that also introduces one or more new output colors. |

## Practical Solver Subtypes

This split is mechanical and solver-oriented:

- `fixed_uniform_integer_scale` means both output axes use the same fixed integer multiplier.
- `fixed_axial_or_anisotropic_scale` means each axis has a fixed integer multiplier, but the multipliers differ or one axis is unchanged.
- `variable_integer_scale` means all axis ratios are integers but at least one multiplier varies by example.
- `shape_derived_expansion` covers non-integer or size-dependent axis ratios, including padding, fixed-canvas, square-canvas, and formula-derived growth.

| Practical Subtype | Count | Characteristics | Current Solver Status |
| --- | ---: | --- | --- |
| `fixed_uniform_integer_scale` | 16 | Fixed equal integer replication on height and width. | Candidate for nearest-neighbor cell scaling or block replication. |
| `fixed_axial_or_anisotropic_scale` | 8 | Fixed integer replication with unequal axis factors. | Candidate for row/column repetition or separable axial tiling. |
| `variable_integer_scale` | 6 | Integer expansion factors vary by example. | Unhandled; requires multiplier inference before rendering. |
| `shape_derived_expansion` | 5 | Dimensions follow padding, fixed-canvas, square-canvas, or other non-factor rules. | Unhandled; requires output-shape inference and placement logic. |

## Practical Subtype Metadata

### `fixed_uniform_integer_scale`

| Metric | Value |
| --- | --- |
| Count | `16` |
| New color count modes | `[(0, 13), (1, 3)]` |
| Height ratio modes | `[(2, 2926), (3, 1072)]` |
| Width ratio modes | `[(2, 2926), (3, 1072)]` |
| Output/input area ratio min/avg | `4.0000 / 5.3407` |

| Task | Task File | Flags | New Colors | Input Shapes | Output Shapes | Height Ratio Modes | Width Ratio Modes | Area Ratio Min/Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `task001` | `competition_material/taskfiles/task001.json` | `shape_expands` | `[]` | `3x3:268` | `9x9:268` | `[(3, 268)]` | `[(3, 268)]` | 9.0000 / 9.0000 |
| `task019` | `competition_material/taskfiles/task019.json` | <code>shape_expands&#124;new_output_colors</code> | `[8]` | `6x4:20;4x2:15;6x3:15;3x4:13;6x5:13` | `12x8:20;8x4:15;12x6:15;6x8:13;12x10:13` | `[(2, 267)]` | `[(2, 267)]` | 4.0000 / 4.0000 |
| `task083` | `competition_material/taskfiles/task083.json` | `shape_expands` | `[]` | `3x4:266` | `6x8:266` | `[(2, 266)]` | `[(2, 266)]` | 4.0000 / 4.0000 |
| `task104` | `competition_material/taskfiles/task104.json` | `shape_expands` | `[]` | `3x3:7` | `9x9:7` | `[(3, 7)]` | `[(3, 7)]` | 9.0000 / 9.0000 |
| `task106` | `competition_material/taskfiles/task106.json` | `shape_expands` | `[]` | `3x3:134;2x2:132` | `6x6:134;4x4:132` | `[(2, 266)]` | `[(2, 266)]` | 4.0000 / 4.0000 |
| `task108` | `competition_material/taskfiles/task108.json` | `shape_expands` | `[]` | `10x10:266` | `20x20:266` | `[(2, 266)]` | `[(2, 266)]` | 4.0000 / 4.0000 |
| `task123` | `competition_material/taskfiles/task123.json` | `shape_expands` | `[]` | `5x5:265` | `10x10:265` | `[(2, 265)]` | `[(2, 265)]` | 4.0000 / 4.0000 |
| `task142` | `competition_material/taskfiles/task142.json` | `shape_expands` | `[]` | `3x3:266` | `6x6:266` | `[(2, 266)]` | `[(2, 266)]` | 4.0000 / 4.0000 |
| `task152` | `competition_material/taskfiles/task152.json` | `shape_expands` | `[]` | `3x3:267` | `6x6:267` | `[(2, 267)]` | `[(2, 267)]` | 4.0000 / 4.0000 |
| `task194` | `competition_material/taskfiles/task194.json` | `shape_expands` | `[]` | `3x3:266` | `6x6:266` | `[(2, 266)]` | `[(2, 266)]` | 4.0000 / 4.0000 |
| `task223` | `competition_material/taskfiles/task223.json` | `shape_expands` | `[]` | `3x3:265` | `9x9:265` | `[(3, 265)]` | `[(3, 265)]` | 9.0000 / 9.0000 |
| `task304` | `competition_material/taskfiles/task304.json` | <code>shape_expands&#124;new_output_colors</code> | `[0]` | `3x3:266` | `9x9:266` | `[(3, 266)]` | `[(3, 266)]` | 9.0000 / 9.0000 |
| `task307` | `competition_material/taskfiles/task307.json` | `shape_expands` | `[]` | `4x4:84;3x3:64;5x5:63;2x2:55` | `8x8:84;6x6:64;10x10:63;4x4:55` | `[(2, 266)]` | `[(2, 266)]` | 4.0000 / 4.0000 |
| `task315` | `competition_material/taskfiles/task315.json` | `shape_expands` | `[]` | `3x3:266` | `9x9:266` | `[(3, 266)]` | `[(3, 266)]` | 9.0000 / 9.0000 |
| `task327` | `competition_material/taskfiles/task327.json` | `shape_expands` | `[]` | `3x3:265` | `6x6:265` | `[(2, 265)]` | `[(2, 265)]` | 4.0000 / 4.0000 |
| `task388` | `competition_material/taskfiles/task388.json` | <code>shape_expands&#124;new_output_colors</code> | `[8]` | `4x4:61;6x6:58;3x3:54;2x2:47;5x5:46` | `8x8:61;12x12:58;6x6:54;4x4:47;10x10:46` | `[(2, 266)]` | `[(2, 266)]` | 4.0000 / 4.0000 |

### `fixed_axial_or_anisotropic_scale`

| Metric | Value |
| --- | --- |
| Count | `8` |
| New color count modes | `[(0, 8)]` |
| Height ratio modes | `[(1, 1064), (2, 800), (3, 266)]` |
| Width ratio modes | `[(1, 800), (2, 1330)]` |
| Output/input area ratio min/avg | `2.0000 / 2.4995` |

| Task | Task File | Flags | New Colors | Input Shapes | Output Shapes | Height Ratio Modes | Width Ratio Modes | Area Ratio Min/Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `task116` | `competition_material/taskfiles/task116.json` | `shape_expands` | `[]` | `3x4:267` | `6x4:267` | `[(2, 267)]` | `[(1, 267)]` | 2.0000 / 2.0000 |
| `task164` | `competition_material/taskfiles/task164.json` | `shape_expands` | `[]` | `3x3:267` | `3x6:267` | `[(1, 267)]` | `[(2, 267)]` | 2.0000 / 2.0000 |
| `task172` | `competition_material/taskfiles/task172.json` | `shape_expands` | `[]` | `3x3:267` | `6x3:267` | `[(2, 267)]` | `[(1, 267)]` | 2.0000 / 2.0000 |
| `task210` | `competition_material/taskfiles/task210.json` | `shape_expands` | `[]` | `3x3:266` | `6x3:266` | `[(2, 266)]` | `[(1, 266)]` | 2.0000 / 2.0000 |
| `task211` | `competition_material/taskfiles/task211.json` | `shape_expands` | `[]` | `3x2:266` | `9x4:266` | `[(3, 266)]` | `[(2, 266)]` | 6.0000 / 6.0000 |
| `task231` | `competition_material/taskfiles/task231.json` | `shape_expands` | `[]` | `5x6:63;5x8:56;5x10:53;5x7:51;5x9:43` | `5x12:63;5x16:56;5x20:53;5x14:51;5x18:43` | `[(1, 266)]` | `[(2, 266)]` | 2.0000 / 2.0000 |
| `task249` | `competition_material/taskfiles/task249.json` | `shape_expands` | `[]` | `5x4:39;4x3:36;5x3:32;3x5:31;4x5:29` | `5x8:39;4x6:36;5x6:32;3x10:31;4x10:29` | `[(1, 265)]` | `[(2, 265)]` | 2.0000 / 2.0000 |
| `task311` | `competition_material/taskfiles/task311.json` | `shape_expands` | `[]` | `3x3:266` | `3x6:266` | `[(1, 266)]` | `[(2, 266)]` | 2.0000 / 2.0000 |

### `variable_integer_scale`

| Metric | Value |
| --- | --- |
| Count | `6` |
| New color count modes | `[(0, 5), (1, 1)]` |
| Height ratio modes | `[(2, 146), (3, 286), (4, 272), (5, 274), (6, 145), (7, 134), (8, 78), (9, 38), (10, 54), (15, 64), (20, 54), (25, 58)]` |
| Width ratio modes | `[(1, 306), (2, 200), (3, 318), (4, 295), (5, 257), (6, 103), (7, 90), (8, 34)]` |
| Output/input area ratio min/avg | `3.0000 / 24.5627` |

| Task | Task File | Flags | New Colors | Input Shapes | Output Shapes | Height Ratio Modes | Width Ratio Modes | Area Ratio Min/Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `task107` | `competition_material/taskfiles/task107.json` | <code>shape_expands&#124;new_output_colors</code> | `[2]` | `5x5:266` | `20x20:103;25x25:67;15x15:61;10x10:18;30x30:17` | `[(2, 18), (3, 61), (4, 103), (5, 67), (6, 17)]` | `[(2, 18), (3, 61), (4, 103), (5, 67), (6, 17)]` | 4.0000 / 17.1278 |
| `task221` | `competition_material/taskfiles/task221.json` | `shape_expands` | `[]` | `3x3:267` | `15x15:59;21x21:56;9x9:55;18x18:51;12x12:46` | `[(3, 55), (4, 46), (5, 59), (6, 51), (7, 56)]` | `[(3, 55), (4, 46), (5, 59), (6, 51), (7, 56)]` | 9.0000 / 27.2884 |
| `task269` | `competition_material/taskfiles/task269.json` | `shape_expands` | `[]` | `3x3:266` | `15x15:52;6x6:44;9x9:40;18x18:35;24x24:34` | `[(2, 44), (3, 40), (4, 27), (5, 52), (6, 35), (7, 34), (8, 34)]` | `[(2, 44), (3, 40), (4, 27), (5, 52), (6, 35), (7, 34), (8, 34)]` | 4.0000 / 27.7068 |
| `task289` | `competition_material/taskfiles/task289.json` | `shape_expands` | `[]` | `3x3:268` | `9x9:98;6x6:84;12x12:65;15x15:21` | `[(2, 84), (3, 98), (4, 65), (5, 21)]` | `[(2, 84), (3, 98), (4, 65), (5, 21)]` | 4.0000 / 10.3843 |
| `task295` | `competition_material/taskfiles/task295.json` | `shape_expands` | `[]` | `1x14:44;1x16:44;1x12:42;1x18:38;1x10:37` | `7x14:44;8x16:44;6x12:42;9x18:38;5x10:37` | `[(3, 32), (4, 31), (5, 37), (6, 42), (7, 44), (8, 44), (9, 38)]` | `[(1, 268)]` | 3.0000 / 6.1903 |
| `task398` | `competition_material/taskfiles/task398.json` | `shape_expands` | `[]` | `1x5:268` | `15x15:64;25x25:58;10x10:54;20x20:54;5x5:38` | `[(5, 38), (10, 54), (15, 64), (20, 54), (25, 58)]` | `[(1, 38), (2, 54), (3, 64), (4, 54), (5, 58)]` | 5.0000 / 58.6567 |

### `shape_derived_expansion`

| Metric | Value |
| --- | --- |
| Count | `5` |
| New color count modes | `[(0, 4), (1, 1)]` |
| Height ratio modes | `[(5/4, 78), (10/7, 83), (3/2, 320), (5/3, 207), (2, 233), (3, 93), (13/4, 10), (17/5, 10), (7/2, 9), (4, 60)]` |
| Width ratio modes | `[(1, 571), (3/2, 83), (5/3, 130), (2, 196), (3, 55), (4, 68)]` |
| Output/input area ratio min/avg | `1.2500 / 3.1443` |

| Task | Task File | Flags | New Colors | Input Shapes | Output Shapes | Height Ratio Modes | Width Ratio Modes | Area Ratio Min/Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| `task003` | `competition_material/taskfiles/task003.json` | <code>shape_expands&#124;new_output_colors</code> | `[2]` | `6x3:265` | `9x3:265` | `[(3/2, 265)]` | `[(1, 265)]` | 1.5000 / 1.5000 |
| `task114` | `competition_material/taskfiles/task114.json` | `shape_expands` | `[]` | `3x2:73;2x3:65;3x3:65;2x2:63` | `5x4:73;4x5:65;5x5:65;4x4:63` | `[(5/3, 138), (2, 128)]` | `[(5/3, 130), (2, 136)]` | 2.7778 / 3.3555 |
| `task124` | `competition_material/taskfiles/task124.json` | `shape_expands` | `[]` | `7x10:83;8x10:78;6x10:69;5x10:37` | `10x10:267` | `[(5/4, 78), (10/7, 83), (5/3, 69), (2, 37)]` | `[(1, 267)]` | 1.2500 / 1.5171 |
| `task275` | `competition_material/taskfiles/task275.json` | `shape_expands` | `[]` | `3x6:83;8x4:68;4x8:60;6x3:55` | `9x9:138;16x16:128` | `[(3/2, 55), (2, 68), (3, 83), (4, 60)]` | `[(3/2, 83), (2, 60), (3, 55), (4, 68)]` | 4.5000 / 6.1842 |
| `task376` | `competition_material/taskfiles/task376.json` | `shape_expands` | `[]` | `3x17:10;4x17:10;5x17:10;6x17:9` | `9x17:10;13x17:10;17x17:10;21x17:9` | `[(3, 10), (13/4, 10), (17/5, 10), (7/2, 9)]` | `[(1, 39)]` | 3.0000 / 3.2821 |
