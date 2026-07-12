# Mask / Object Selection Task Profile

Source: `task_groups/task_type_map.csv`

## Task Type

| Field | Value |
| --- | --- |
| Primary family | `mask_object_selection` |
| Task count | `66` |
| Common shape relation | `same_shape_variable_size` |
| Common behavior | Same-size input/output where selected objects, colors, or cells are kept, removed, moved, copied, highlighted, or recoded. |
| Current notebook | `(not yet assigned)` |

## Flag-Based Subtypes

| Subtype Flags | Count | Solver Meaning |
| --- | ---: | --- |
| `same_shape_nonlocal_changes` | 37 | Same-shape non-local object/mask change using only colors already present in the input. |
| <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | 29 | Same-shape object/mask change that introduces one or more output colors not present in the input. |

## Practical Solver Subtypes

This split is mechanical and solver-oriented:

- `same_position` means no new nonzero cells are written into background; changed cells are cleared or recolored in place.
- `additive` means at least one background cell becomes nonzero, so the solver must place copied, moved, or highlighted cells.
- `new_color` means at least one output color does not appear in the input, so the rule includes recoding or marking.
- All tasks are same-shape, but none are strict zero-conflict local 3x3 rules in the current profile.

| Practical Subtype | Count | Characteristics | Current Solver Status |
| --- | ---: | --- | --- |
| `same_position_recolor_or_clear` | 12 | Same-size object/mask selection with no new nonzero cells and no new output colors; cells are cleared or recolored in place. | Candidate for object/color filtering and same-position recolor rules; no current generic solver assigned. |
| `move_copy_or_additive_highlight` | 25 | Same-size selection with output cells added into background using existing colors; often move/copy, symmetry completion, or highlight-by-copy behavior. | Unhandled; requires object localization plus target-position logic. |
| `new_color_same_position_recode` | 18 | Same-size selection that introduces new colors without adding new nonzero positions; objects or cells are recoded in place. | Unhandled; requires object selection plus learned/derived output color encoding. |
| `new_color_additive_or_mixed_highlight` | 11 | Same-size selection that introduces new colors and also writes into background; combines object selection with marking, movement, or copied highlights. | Unhandled; highest risk because it combines selection, new colors, and added cells. |

## Practical Subtype Metadata

### `same_position_recolor_or_clear`

| Metric | Value |
| --- | --- |
| Count | `12` |
| New color count modes | `[(0, 12)]` |
| Local 3x3 score min/avg | `0.6667` / `0.8331` |
| Local 3x3 conflicts min/avg | `191` / `1075.0000` |
| Added nonzero cells avg | `0.0000` |
| Changed cells avg | `10716.5000` |
| Cleared cells avg | `448.0000` |
| Recolored cells avg | `10268.5000` |
| Input nonzero preserved ratio avg | `0.2665` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Cleared Cells | Recolored Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task035` | `competition_material/taskfiles/task035.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:266` | 0.9355 | 387 | 0 | 1217 | 0 | 1217 | 0.6953 |
| `task040` | `competition_material/taskfiles/task040.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:266` | 0.9647 | 212 | 0 | 1637 | 0 | 1637 | 0.7647 |
| `task069` | `competition_material/taskfiles/task069.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:264` | 0.8840 | 696 | 0 | 5804 | 1451 | 4353 | 0.0000 |
| `task071` | `competition_material/taskfiles/task071.json` | `same_shape_nonlocal_changes` | `[]` | `16x16:265` | 0.9668 | 510 | 0 | 4260 | 2597 | 1663 | 0.5851 |
| `task203` | `competition_material/taskfiles/task203.json` | `same_shape_nonlocal_changes` | `[]` | `6x6:44;16x16:42;14x14:41;10x10:39;8x8:38` | 0.8037 | 1720 | 0 | 37784 | 0 | 37784 | 0.0864 |
| `task267` | `competition_material/taskfiles/task267.json` | `same_shape_nonlocal_changes` | `[]` | `7x7:264` | 0.9054 | 278 | 0 | 3793 | 264 | 3529 | 0.0000 |
| `task312` | `competition_material/taskfiles/task312.json` | `same_shape_nonlocal_changes` | `[]` | `12x12:265` | 0.7113 | 2494 | 0 | 12766 | 0 | 12766 | 0.1910 |
| `task313` | `competition_material/taskfiles/task313.json` | `same_shape_nonlocal_changes` | `[]` | `8x8:24;6x6:23;14x14:23;10x10:22;7x7:20` | 0.6759 | 3265 | 0 | 43463 | 0 | 43463 | 0.0000 |
| `task342` | `competition_material/taskfiles/task342.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:266` | 0.9682 | 191 | 0 | 2128 | 1064 | 1064 | 0.0000 |
| `task354` | `competition_material/taskfiles/task354.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:266` | 0.7518 | 1489 | 0 | 7961 | 0 | 7961 | 0.0911 |
| `task368` | `competition_material/taskfiles/task368.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:265` | 0.7637 | 1418 | 0 | 7335 | 0 | 7335 | 0.2843 |
| `task373` | `competition_material/taskfiles/task373.json` | `same_shape_nonlocal_changes` | `[]` | `2x6:75` | 0.6667 | 240 | 0 | 450 | 0 | 450 | 0.5000 |

### `move_copy_or_additive_highlight`

| Metric | Value |
| --- | --- |
| Count | `25` |
| New color count modes | `[(0, 25)]` |
| Local 3x3 score min/avg | `0.6939` / `0.9065` |
| Local 3x3 conflicts min/avg | `39` / `720.7200` |
| Added nonzero cells avg | `3110.1200` |
| Changed cells avg | `7015.6400` |
| Cleared cells avg | `2983.1600` |
| Recolored cells avg | `922.3600` |
| Input nonzero preserved ratio avg | `0.4028` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Cleared Cells | Recolored Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task008` | `competition_material/taskfiles/task008.json` | `same_shape_nonlocal_changes` | `[]` | `12x16:8;9x8:7;14x9:6;9x16:6;10x13:6` | 0.9438 | 493 | 1795 | 3590 | 1795 | 0 | 0.4276 |
| `task011` | `competition_material/taskfiles/task011.json` | `same_shape_nonlocal_changes` | `[]` | `11x11:267` | 0.9463 | 390 | 4375 | 15055 | 6511 | 4169 | 0.5238 |
| `task018` | `competition_material/taskfiles/task018.json` | `same_shape_nonlocal_changes` | `[]` | `24x24:10;23x23:9;23x22:8;24x23:7;20x22:7` | 0.9733 | 609 | 2325 | 5820 | 3495 | 0 | 0.2508 |
| `task030` | `competition_material/taskfiles/task030.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:145;5x10:121` | 0.8847 | 542 | 1583 | 3166 | 1583 | 0 | 0.5748 |
| `task032` | `competition_material/taskfiles/task032.json` | `same_shape_nonlocal_changes` | `[]` | `4x4:95;6x6:88;5x5:83` | 0.8796 | 183 | 1130 | 2260 | 1130 | 0 | 0.4400 |
| `task034` | `competition_material/taskfiles/task034.json` | `same_shape_nonlocal_changes` | `[]` | `9x9:267` | 0.8422 | 767 | 4445 | 4979 | 0 | 534 | 0.5000 |
| `task068` | `competition_material/taskfiles/task068.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:266` | 0.9733 | 160 | 1502 | 9256 | 7185 | 569 | 0.0400 |
| `task075` | `competition_material/taskfiles/task075.json` | `same_shape_nonlocal_changes` | `[]` | `9x13:265` | 0.6939 | 2149 | 9568 | 10764 | 0 | 1196 | 0.7995 |
| `task078` | `competition_material/taskfiles/task078.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:266` | 0.9225 | 465 | 2334 | 4668 | 2334 | 0 | 0.7123 |
| `task086` | `competition_material/taskfiles/task086.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:96;12x12:90;11x11:80` | 0.9338 | 479 | 6373 | 9983 | 0 | 3610 | 0.0000 |
| `task122` | `competition_material/taskfiles/task122.json` | `same_shape_nonlocal_changes` | `[]` | `7x7:14;8x8:14;11x11:10;9x9:9;8x9:8` | 0.9536 | 417 | 1330 | 2660 | 1330 | 0 | 0.6661 |
| `task128` | `competition_material/taskfiles/task128.json` | `same_shape_nonlocal_changes` | `[]` | `15x15:266` | 0.9065 | 1262 | 7681 | 15362 | 7681 | 0 | 0.0000 |
| `task154` | `competition_material/taskfiles/task154.json` | `same_shape_nonlocal_changes` | `[]` | `15x15:266` | 0.9434 | 764 | 3449 | 6898 | 3449 | 0 | 0.5519 |
| `task161` | `competition_material/taskfiles/task161.json` | `same_shape_nonlocal_changes` | `[]` | `18x19:7;11x15:7;15x17:6;18x21:6;10x17:6` | 0.8585 | 2563 | 7267 | 15115 | 7096 | 752 | 0.1194 |
| `task163` | `competition_material/taskfiles/task163.json` | `same_shape_nonlocal_changes` | `[]` | `11x11:267` | 0.9763 | 172 | 567 | 7725 | 6936 | 222 | 0.6007 |
| `task234` | `competition_material/taskfiles/task234.json` | `same_shape_nonlocal_changes` | `[]` | `20x16:11;15x20:7;16x17:7;18x16:7;17x16:7` | 0.8638 | 2251 | 3673 | 8367 | 4694 | 0 | 0.7011 |
| `task245` | `competition_material/taskfiles/task245.json` | `same_shape_nonlocal_changes` | `[]` | `9x7:24;7x7:21;10x7:20;10x9:20;7x8:20` | 0.8649 | 595 | 2015 | 4030 | 2015 | 0 | 0.4607 |
| `task250` | `competition_material/taskfiles/task250.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:265` | 0.9637 | 218 | 1591 | 3182 | 1591 | 0 | 0.3998 |
| `task260` | `competition_material/taskfiles/task260.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:266` | 0.9398 | 361 | 1654 | 3697 | 2043 | 0 | 0.4863 |
| `task270` | `competition_material/taskfiles/task270.json` | `same_shape_nonlocal_changes` | `[]` | `15x15:266` | 0.9832 | 227 | 1060 | 2122 | 1060 | 2 | 0.3338 |
| `task298` | `competition_material/taskfiles/task298.json` | `same_shape_nonlocal_changes` | `[]` | `6x6:137;8x8:130` | 0.8677 | 408 | 1216 | 13252 | 1400 | 10636 | 0.0000 |
| `task301` | `competition_material/taskfiles/task301.json` | `same_shape_nonlocal_changes` | `[]` | `10x7:16;10x8:15;4x3:15;7x4:14;5x5:12` | 0.8260 | 477 | 2568 | 6505 | 2568 | 1369 | 0.3308 |
| `task353` | `competition_material/taskfiles/task353.json` | `same_shape_nonlocal_changes` | `[]` | `9x7:17;11x11:14;5x4:13;6x5:13;10x8:13` | 0.9900 | 39 | 271 | 542 | 271 | 0 | 0.5000 |
| `task362` | `competition_material/taskfiles/task362.json` | `same_shape_nonlocal_changes` | `[]` | `10x10:267` | 0.7887 | 1268 | 4539 | 9509 | 4970 | 0 | 0.0970 |
| `task390` | `competition_material/taskfiles/task390.json` | `same_shape_nonlocal_changes` | `[]` | `15x15:266` | 0.9438 | 759 | 3442 | 6884 | 3442 | 0 | 0.5529 |

### `new_color_same_position_recode`

| Metric | Value |
| --- | --- |
| Count | `18` |
| New color count modes | `[(1, 6), (2, 8), (3, 3), (4, 1)]` |
| Local 3x3 score min/avg | `0.7997` / `0.9160` |
| Local 3x3 conflicts min/avg | `4` / `431.3333` |
| Added nonzero cells avg | `0.0000` |
| Changed cells avg | `4862.7222` |
| Cleared cells avg | `321.5556` |
| Recolored cells avg | `4541.1667` |
| Input nonzero preserved ratio avg | `0.2567` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Cleared Cells | Recolored Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task010` | `competition_material/taskfiles/task010.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[1, 2, 3, 4]` | `9x9:265` | 0.8626 | 668 | 0 | 5276 | 0 | 5276 | 0.0000 |
| `task023` | `competition_material/taskfiles/task023.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[2, 8]` | `9x9:52;8x10:47;8x9:46;9x11:44;8x11:41` | 0.9568 | 219 | 0 | 5263 | 0 | 5263 | 0.0000 |
| `task052` | `competition_material/taskfiles/task052.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[0, 5]` | `3x3:267` | 0.9889 | 6 | 0 | 2403 | 1200 | 1203 | 0.0000 |
| `task125` | `competition_material/taskfiles/task125.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[3, 4]` | `15x15:265` | 0.9824 | 238 | 0 | 18680 | 0 | 18680 | 0.6867 |
| `task156` | `competition_material/taskfiles/task156.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[1, 2]` | `10x10:265` | 0.9080 | 552 | 0 | 3220 | 0 | 3220 | 0.6975 |
| `task167` | `competition_material/taskfiles/task167.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[0, 5]` | `3x3:268` | 0.9519 | 26 | 0 | 2412 | 1608 | 804 | 0.0000 |
| `task169` | `competition_material/taskfiles/task169.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[1, 2, 3]` | `10x10:266` | 0.9567 | 260 | 0 | 3967 | 0 | 3967 | 0.0000 |
| `task196` | `competition_material/taskfiles/task196.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[3]` | `9x9:95;15x15:89;12x12:82` | 0.9551 | 401 | 0 | 3546 | 0 | 3546 | 0.5037 |
| `task229` | `competition_material/taskfiles/task229.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[5]` | `3x3:267` | 0.9926 | 4 | 0 | 1130 | 0 | 1130 | 0.5298 |
| `task252` | `competition_material/taskfiles/task252.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[4]` | `14x14:26;12x12:24;7x7:24;5x5:23;13x13:22` | 0.9583 | 211 | 0 | 1377 | 0 | 1377 | 0.5408 |
| `task254` | `competition_material/taskfiles/task254.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[1, 2]` | `9x9:265` | 0.8601 | 680 | 0 | 5598 | 2980 | 2618 | 0.0000 |
| `task277` | `competition_material/taskfiles/task277.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[1, 2]` | `10x10:266` | 0.8985 | 609 | 0 | 6188 | 0 | 6188 | 0.0000 |
| `task292` | `competition_material/taskfiles/task292.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[6]` | `3x11:4;3x10:3;3x13:3;3x14:3;3x17:3` | 0.7997 | 244 | 0 | 292 | 0 | 292 | 0.6404 |
| `task320` | `competition_material/taskfiles/task320.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[8]` | `11x9:113;10x9:66;9x9:41;8x9:20;7x9:17` | 0.9180 | 439 | 0 | 2972 | 0 | 2972 | 0.5365 |
| `task330` | `competition_material/taskfiles/task330.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[1, 2]` | `10x10:266` | 0.8695 | 783 | 0 | 6998 | 0 | 6998 | 0.0000 |
| `task332` | `competition_material/taskfiles/task332.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[3]` | `3x11:31;3x12:30;3x10:29;3x14:29;3x18:24` | 0.8180 | 482 | 0 | 2013 | 0 | 2013 | 0.4860 |
| `task364` | `competition_material/taskfiles/task364.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[1, 2, 6]` | `14x16:10;11x11:9;12x12:9;10x8:9;18x16:8` | 0.8978 | 1425 | 0 | 11815 | 0 | 11815 | 0.0000 |
| `task374` | `competition_material/taskfiles/task374.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[1, 2, 4]` | `10x10:267` | 0.9138 | 517 | 0 | 4379 | 0 | 4379 | 0.0000 |

### `new_color_additive_or_mixed_highlight`

| Metric | Value |
| --- | --- |
| Count | `11` |
| New color count modes | `[(1, 10), (3, 1)]` |
| Local 3x3 score min/avg | `0.4347` / `0.7559` |
| Local 3x3 conflicts min/avg | `23` / `1130.2727` |
| Added nonzero cells avg | `5369.3636` |
| Changed cells avg | `7549.6364` |
| Cleared cells avg | `1881.4545` |
| Recolored cells avg | `298.8182` |
| Input nonzero preserved ratio avg | `0.2826` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Cleared Cells | Recolored Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task058` | `competition_material/taskfiles/task058.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[3]` | `6x6:2;8x8:2;15x15:2;13x13:2;10x10:2` | 0.5705 | 1614 | 2144 | 2144 | 0 | 0 | 0.0000 |
| `task062` | `competition_material/taskfiles/task062.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[3]` | `10x10:267` | 0.9663 | 202 | 24661 | 25200 | 0 | 539 | 0.7357 |
| `task131` | `competition_material/taskfiles/task131.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[8]` | `5x16:27;18x5:26;4x18:25;16x5:24;4x17:24` | 0.4347 | 2639 | 3433 | 5692 | 2233 | 26 | 0.3479 |
| `task148` | `competition_material/taskfiles/task148.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[4]` | `17x8:13;19x8:12;24x9:11;23x8:9;19x11:9` | 0.8217 | 2128 | 9615 | 10364 | 0 | 749 | 0.7820 |
| `task157` | `competition_material/taskfiles/task157.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[1]` | `10x15:265` | 0.6739 | 2935 | 4736 | 9472 | 4736 | 0 | 0.6628 |
| `task160` | `competition_material/taskfiles/task160.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[2]` | `10x10:265` | 0.9923 | 46 | 334 | 3006 | 1336 | 1336 | 0.5800 |
| `task171` | `competition_material/taskfiles/task171.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[8]` | `3x3:2;4x3:2;5x4:2;5x6:2;7x6:2` | 0.5605 | 825 | 1052 | 1052 | 0 | 0 | 0.0000 |
| `task186` | `competition_material/taskfiles/task186.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[2]` | `3x3:267` | 0.7963 | 110 | 531 | 1400 | 531 | 338 | 0.0000 |
| `task199` | `competition_material/taskfiles/task199.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[4]` | `5x5:27;4x4:25;14x14:24;12x12:23;7x7:23` | 0.7700 | 1143 | 6594 | 6860 | 0 | 266 | 0.0000 |
| `task262` | `competition_material/taskfiles/task262.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[2, 3, 4]` | `3x3:11` | 0.7677 | 23 | 66 | 99 | 0 | 33 | 0.0000 |
| `task338` | `competition_material/taskfiles/task338.json` | <code>same_shape_nonlocal_changes&#124;new_output_colors</code> | `[3]` | `10x10:83;20x20:73;15x15:57;25x25:54` | 0.9605 | 768 | 5897 | 17757 | 11860 | 0 | 0.0000 |

