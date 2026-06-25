# Fill Enclosed Regions Task Profile

Source: `task_groups/task_type_map.csv`

## Task Type

| Field | Value |
| --- | --- |
| Primary family | `fill_enclosed_regions` |
| Task count | `59` |
| Common shape relation | `same_shape_variable_size` |
| Common behavior | Input is mostly preserved while new output colors are added as fill/marking cells. |
| Current notebook | `submission_nbs/05_fill_additive_marking.ipynb` |

## Flag-Based Subtypes

| Subtype Flags | Count | Solver Meaning |
| --- | ---: | --- |
| `adds_new_color_preserves_input|new_output_colors` | 51 | Non-local fill/additive marking; current 3x3 solver is insufficient. |
| `adds_new_color_preserves_input|local_3x3_consistent|new_output_colors` | 8 | Local 3x3 symbolic CNN candidate. |

## Practical Solver Subtypes

| Practical Subtype | Count | Characteristics | Current Solver Status |
| --- | ---: | --- | --- |
| `local_3x3` | 8 | Same-size additive marking with zero 3x3 conflicts; output cell is a deterministic function of local input patch. | Handled by `symbolic_3x3_patch_cnn`. |
| `nonlocal_1color` | 41 | Same-size additive marking with one new output color; requires non-local or object-level context. | Unhandled; currently identity fallback only. |
| `nonlocal_multicolor` | 10 | Same-size additive marking with multiple new output colors; likely needs object/color-specific rules. | Unhandled; currently identity fallback only. |

## Practical Subtype Metadata

### `local_3x3`

| Metric | Value |
| --- | --- |
| Count | `8` |
| New color count modes | `[(1, 4), (4, 2), (2, 1), (3, 1)]` |
| Local 3x3 score min/avg | `1.0000` / `1.0000` |
| Local 3x3 conflicts min/avg | `0` / `0.0000` |
| Added nonzero cells avg | `3404.8750` |
| Input nonzero preserved ratio avg | `1.0000` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `task015` | `competition_material/taskfiles/task015.json` | `adds_new_color_preserves_input|local_3x3_consistent|new_output_colors` | `[4,7]` | `9x9:265` | 1.0000 | 0 | 1776 | 1.0000 |
| `task081` | `competition_material/taskfiles/task081.json` | `adds_new_color_preserves_input|local_3x3_consistent|new_output_colors` | `[1]` | `7x7:264` | 1.0000 | 0 | 634 | 1.0000 |
| `task095` | `competition_material/taskfiles/task095.json` | `adds_new_color_preserves_input|local_3x3_consistent|new_output_colors` | `[1]` | `9x9:265` | 1.0000 | 0 | 7368 | 1.0000 |
| `task220` | `competition_material/taskfiles/task220.json` | `adds_new_color_preserves_input|local_3x3_consistent|new_output_colors` | `[1,4,6]` | `15x15:37;14x14:36;13x13:31;12x12:29;11x11:29` | 1.0000 | 0 | 4248 | 1.0000 |
| `task230` | `competition_material/taskfiles/task230.json` | `adds_new_color_preserves_input|local_3x3_consistent|new_output_colors` | `[1,2,3,4]` | `15x15:149;10x10:117` | 1.0000 | 0 | 3392 | 1.0000 |
| `task258` | `competition_material/taskfiles/task258.json` | `adds_new_color_preserves_input|local_3x3_consistent|new_output_colors` | `[2]` | `10x10:52;7x7:49;9x9:45;6x6:43;8x8:39` | 1.0000 | 0 | 1696 | 1.0000 |
| `task331` | `competition_material/taskfiles/task331.json` | `adds_new_color_preserves_input|local_3x3_consistent|new_output_colors` | `[2,6,7,8]` | `10x10:265` | 1.0000 | 0 | 4958 | 1.0000 |
| `task352` | `competition_material/taskfiles/task352.json` | `adds_new_color_preserves_input|local_3x3_consistent|new_output_colors` | `[1]` | `8x9:32;10x10:30;8x8:24;7x7:24;5x6:24` | 1.0000 | 0 | 3167 | 1.0000 |

### `nonlocal_1color`

| Metric | Value |
| --- | --- |
| Count | `41` |
| New color count modes | `[(1, 41)]` |
| Local 3x3 score min/avg | `0.3637` / `0.8665` |
| Local 3x3 conflicts min/avg | `104` / `1217.3415` |
| Added nonzero cells avg | `6207.8780` |
| Input nonzero preserved ratio avg | `1.0000` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `task002` | `competition_material/taskfiles/task002.json` | `adds_new_color_preserves_input|new_output_colors` | `[4]` | `20x20:75;19x19:40;18x18:39;16x16:21;15x15:20` | 0.9193 | 1421 | 9828 | 1.0000 |
| `task027` | `competition_material/taskfiles/task027.json` | `adds_new_color_preserves_input|new_output_colors` | `[2]` | `10x10:265` | 0.9515 | 291 | 1260 | 1.0000 |
| `task042` | `competition_material/taskfiles/task042.json` | `adds_new_color_preserves_input|new_output_colors` | `[8]` | `10x10:266` | 0.9487 | 308 | 1071 | 1.0000 |
| `task043` | `competition_material/taskfiles/task043.json` | `adds_new_color_preserves_input|new_output_colors` | `[2]` | `10x10:266` | 0.7992 | 1205 | 5392 | 1.0000 |
| `task047` | `competition_material/taskfiles/task047.json` | `adds_new_color_preserves_input|new_output_colors` | `[2]` | `9x9:265` | 0.7189 | 1366 | 7950 | 1.0000 |
| `task050` | `competition_material/taskfiles/task050.json` | `adds_new_color_preserves_input|new_output_colors` | `[3]` | `14x7:6;11x5:6;10x4:5;8x12:5;3x5:5` | 0.9230 | 365 | 1289 | 1.0000 |
| `task060` | `competition_material/taskfiles/task060.json` | `adds_new_color_preserves_input|new_output_colors` | `[5]` | `5x11:265` | 0.7921 | 686 | 4095 | 1.0000 |
| `task063` | `competition_material/taskfiles/task063.json` | `adds_new_color_preserves_input|new_output_colors` | `[3]` | `14x14:93;10x10:87;12x12:86` | 0.7773 | 2001 | 8094 | 1.0000 |
| `task090` | `competition_material/taskfiles/task090.json` | `adds_new_color_preserves_input|new_output_colors` | `[6]` | `4x21:13;2x20:11;4x29:11;4x24:10;4x22:10` | 0.9095 | 483 | 2984 | 1.0000 |
| `task102` | `competition_material/taskfiles/task102.json` | `adds_new_color_preserves_input|new_output_colors` | `[2]` | `12x12:267` | 0.9505 | 428 | 978 | 1.0000 |
| `task105` | `competition_material/taskfiles/task105.json` | `adds_new_color_preserves_input|new_output_colors` | `[2]` | `11x13:71;12x13:58;13x13:52;10x13:42;9x13:22` | 0.9757 | 219 | 2137 | 1.0000 |
| `task119` | `competition_material/taskfiles/task119.json` | `adds_new_color_preserves_input|new_output_colors` | `[3]` | `12x12:266` | 0.9289 | 614 | 2534 | 1.0000 |
| `task126` | `competition_material/taskfiles/task126.json` | `adds_new_color_preserves_input|new_output_colors` | `[4]` | `7x10:9;8x14:7;10x17:6;10x14:6;5x5:5` | 0.9664 | 186 | 770 | 1.0000 |
| `task139` | `competition_material/taskfiles/task139.json` | `adds_new_color_preserves_input|new_output_colors` | `[7]` | `9x9:265` | 0.9510 | 238 | 2332 | 1.0000 |
| `task162` | `competition_material/taskfiles/task162.json` | `adds_new_color_preserves_input|new_output_colors` | `[1]` | `20x20:266` | 0.9874 | 303 | 4842 | 1.0000 |
| `task166` | `competition_material/taskfiles/task166.json` | `adds_new_color_preserves_input|new_output_colors` | `[2]` | `10x10:17;11x11:14;9x11:11;8x10:11;11x9:11` | 0.9112 | 548 | 3514 | 1.0000 |
| `task176` | `competition_material/taskfiles/task176.json` | `adds_new_color_preserves_input|new_output_colors` | `[4]` | `3x10:2;3x15:2;3x18:2;3x25:2;3x22:1` | 0.7937 | 237 | 271 | 1.0000 |
| `task200` | `competition_material/taskfiles/task200.json` | `adds_new_color_preserves_input|new_output_colors` | `[5]` | `10x10:84` | 0.6653 | 2008 | 2679 | 1.0000 |
| `task219` | `competition_material/taskfiles/task219.json` | `adds_new_color_preserves_input|new_output_colors` | `[1]` | `15x10:265` | 0.8758 | 1118 | 4711 | 1.0000 |
| `task232` | `competition_material/taskfiles/task232.json` | `adds_new_color_preserves_input|new_output_colors` | `[5]` | `12x8:9;14x8:8;8x11:8;7x9:8;12x11:8` | 0.8753 | 830 | 4641 | 1.0000 |
| `task246` | `competition_material/taskfiles/task246.json` | `adds_new_color_preserves_input|new_output_colors` | `[8]` | `18x14:7;11x10:7;18x16:6;10x12:6;18x12:6` | 0.9632 | 494 | 2108 | 1.0000 |
| `task251` | `competition_material/taskfiles/task251.json` | `adds_new_color_preserves_input|new_output_colors` | `[1]` | `12x12:105;11x11:58;10x10:52;9x9:30;8x8:21` | 0.9350 | 459 | 2708 | 1.0000 |
| `task255` | `competition_material/taskfiles/task255.json` | `adds_new_color_preserves_input|new_output_colors` | `[3]` | `30x30:265` | 0.7521 | 13386 | 61977 | 1.0000 |
| `task265` | `competition_material/taskfiles/task265.json` | `adds_new_color_preserves_input|new_output_colors` | `[2]` | `18x18:266` | 0.9820 | 350 | 5405 | 1.0000 |
| `task273` | `competition_material/taskfiles/task273.json` | `adds_new_color_preserves_input|new_output_colors` | `[2]` | `10x10:266` | 0.9222 | 467 | 1889 | 1.0000 |
| `task278` | `competition_material/taskfiles/task278.json` | `adds_new_color_preserves_input|new_output_colors` | `[3]` | `17x15:24;15x16:21;18x15:20;18x17:20;15x15:20` | 0.9205 | 1307 | 9497 | 1.0000 |
| `task299` | `competition_material/taskfiles/task299.json` | `adds_new_color_preserves_input|new_output_colors` | `[4]` | `6x6:21` | 0.8188 | 137 | 147 | 1.0000 |
| `task303` | `competition_material/taskfiles/task303.json` | `adds_new_color_preserves_input|new_output_colors` | `[2]` | `27x22:4;18x15:4;15x21:3;20x15:3;24x11:3` | 0.9535 | 1120 | 16276 | 1.0000 |
| `task323` | `competition_material/taskfiles/task323.json` | `adds_new_color_preserves_input|new_output_colors` | `[5]` | `13x13:172` | 0.9111 | 901 | 2801 | 1.0000 |
| `task335` | `competition_material/taskfiles/task335.json` | `adds_new_color_preserves_input|new_output_colors` | `[4]` | `10x12:7;18x14:7;11x10:7;18x16:6;18x12:6` | 0.9629 | 490 | 2095 | 1.0000 |
| `task336` | `competition_material/taskfiles/task336.json` | `adds_new_color_preserves_input|new_output_colors` | `[8]` | `10x10:31` | 0.8768 | 382 | 557 | 1.0000 |
| `task341` | `competition_material/taskfiles/task341.json` | `adds_new_color_preserves_input|new_output_colors` | `[8]` | `10x10:266` | 0.8898 | 661 | 2679 | 1.0000 |
| `task348` | `competition_material/taskfiles/task348.json` | `adds_new_color_preserves_input|new_output_colors` | `[8]` | `10x10:13;9x8:13;9x7:12;10x7:12;9x10:12` | 0.3637 | 2150 | 4663 | 1.0000 |
| `task350` | `competition_material/taskfiles/task350.json` | `adds_new_color_preserves_input|new_output_colors` | `[8]` | `13x13:9;11x10:7;11x11:7;18x17:7;10x9:6` | 0.6855 | 4965 | 25568 | 1.0000 |
| `task357` | `competition_material/taskfiles/task357.json` | `adds_new_color_preserves_input|new_output_colors` | `[8]` | `10x2:2;10x3:2;10x4:2;10x5:2;10x10:1` | 0.8471 | 104 | 667 | 1.0000 |
| `task367` | `competition_material/taskfiles/task367.json` | `adds_new_color_preserves_input|new_output_colors` | `[4]` | `18x14:7;11x10:7;18x16:6;10x12:6;18x12:6` | 0.8766 | 1690 | 5319 | 1.0000 |
| `task371` | `competition_material/taskfiles/task371.json` | `adds_new_color_preserves_input|new_output_colors` | `[3]` | `10x14:25;10x10:24;12x14:23;14x12:22;14x10:20` | 0.9600 | 300 | 1330 | 1.0000 |
| `task381` | `competition_material/taskfiles/task381.json` | `adds_new_color_preserves_input|new_output_colors` | `[9]` | `10x10:265` | 0.8823 | 706 | 2457 | 1.0000 |
| `task387` | `competition_material/taskfiles/task387.json` | `adds_new_color_preserves_input|new_output_colors` | `[5]` | `18x16:18;16x14:16;14x14:15;18x15:15;17x14:14` | 0.8537 | 2274 | 11688 | 1.0000 |
| `task392` | `competition_material/taskfiles/task392.json` | `adds_new_color_preserves_input|new_output_colors` | `[5]` | `10x10:266` | 0.6218 | 2269 | 20246 | 1.0000 |
| `task397` | `competition_material/taskfiles/task397.json` | `adds_new_color_preserves_input|new_output_colors` | `[3]` | `10x10:266` | 0.9260 | 444 | 3074 | 1.0000 |

### `nonlocal_multicolor`

| Metric | Value |
| --- | --- |
| Count | `10` |
| New color count modes | `[(2, 6), (3, 3), (5, 1)]` |
| Local 3x3 score min/avg | `0.2952` / `0.7079` |
| Local 3x3 conflicts min/avg | `333` / `4668.1000` |
| Added nonzero cells avg | `35497.9000` |
| Input nonzero preserved ratio avg | `1.0000` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `task055` | `competition_material/taskfiles/task055.json` | `adds_new_color_preserves_input|new_output_colors` | `[1,2,3,4,6]` | `22x18:5;15x14:5;19x11:4;20x22:4;22x14:4` | 0.5096 | 10423 | 40288 | 1.0000 |
| `task145` | `competition_material/taskfiles/task145.json` | `adds_new_color_preserves_input|new_output_colors` | `[1,8]` | `18x14:7;11x10:7;18x16:6;10x12:6;18x12:6` | 0.4850 | 6984 | 29620 | 1.0000 |
| `task187` | `competition_material/taskfiles/task187.json` | `adds_new_color_preserves_input|new_output_colors` | `[2,3]` | `23x21:16;20x23:11;25x24:11;24x20:11;20x20:10` | 0.8734 | 3885 | 113973 | 1.0000 |
| `task198` | `competition_material/taskfiles/task198.json` | `adds_new_color_preserves_input|new_output_colors` | `[3,4]` | `29x29:66;23x23:64;19x19:51;27x27:33;17x17:31` | 0.6528 | 12125 | 108291 | 1.0000 |
| `task204` | `competition_material/taskfiles/task204.json` | `adds_new_color_preserves_input|new_output_colors` | `[2,7]` | `11x11:32;10x10:30;12x12:29;14x14:28;20x20:24` | 0.9159 | 1158 | 6016 | 1.0000 |
| `task226` | `competition_material/taskfiles/task226.json` | `adds_new_color_preserves_input|new_output_colors` | `[1,2,3]` | `10x10:133` | 0.7338 | 1597 | 1612 | 1.0000 |
| `task256` | `competition_material/taskfiles/task256.json` | `adds_new_color_preserves_input|new_output_colors` | `[1,3]` | `9x9:15;10x10:12;8x9:9;11x10:9;9x10:9` | 0.2952 | 3250 | 5890 | 1.0000 |
| `task302` | `competition_material/taskfiles/task302.json` | `adds_new_color_preserves_input|new_output_colors` | `[6,7,8]` | `12x12:266` | 0.9225 | 670 | 2957 | 1.0000 |
| `task349` | `competition_material/taskfiles/task349.json` | `adds_new_color_preserves_input|new_output_colors` | `[1,3]` | `10x10:60;20x20:59;30x30:55;15x15:51;25x25:42` | 0.7465 | 6256 | 39211 | 1.0000 |
| `task369` | `competition_material/taskfiles/task369.json` | `adds_new_color_preserves_input|new_output_colors` | `[1,2,3]` | `10x10:265` | 0.9445 | 333 | 7121 | 1.0000 |
