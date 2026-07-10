# Cropping / Extraction Task Profile

Source: `task_groups/task_type_map.csv`

## Task Type

| Field | Value |
| --- | --- |
| Primary family | `cropping_extraction` |
| Task count | `99` |
| Common shape relation | `shrinks` |
| Common behavior | Output is smaller than input; it is usually a selected object, bounding box, subgrid, compressed mask, or summary extracted from the input. |
| Current notebook | `(not yet assigned)` |

## Flag-Based Subtypes

| Subtype Flags | Count | Solver Meaning |
| --- | ---: | --- |
| <code>shape_shrinks&#124;new_output_colors</code> | 15 | Shrink/extraction with one or more output colors not present in the input; requires recoding or summary colors. |
| `shape_shrinks` | 84 | Pure shrink/extraction; output colors are all present in the input. |

## Practical Solver Subtypes

This split is mechanical and solver-oriented:

- `exact_subgrid` means every visible output occurs as a contiguous rectangle inside the input.
- `fixed_output` means the task has exactly one output shape across train, test, and arc-gen examples.
- `summary` means the output is smaller than the input but is not always a verbatim crop.
- `recode` means the shrink task introduces output colors not present in the input.

| Practical Subtype | Count | Characteristics | Current Solver Status |
| --- | ---: | --- | --- |
| `exact_subgrid_fixed_output` | 13 | Exact contiguous crop with one fixed output shape. Selection is the hard part; rendering is direct copy once the crop window is known. | Candidate for crop-window search or fixed-output object selector; no current generic solver assigned. |
| `exact_subgrid_variable_output` | 14 | Exact contiguous crop but output size varies by example. Requires dynamic boundary detection or object-box selection. | Candidate for object bounding-box extraction; difficult for static output handling. |
| `fixed_size_extraction_or_summary` | 27 | Fixed-size smaller output that is not always an exact crop. Often object selection, compression, voting, or relational summary into a stable canvas. | Unhandled; fixed canvas helps, but rule inference is task-specific. |
| `variable_size_extraction_or_summary` | 30 | Variable-size smaller output that is not always an exact crop. Likely needs object grouping, filtering, and dynamic output bounds. | Unhandled; highest dynamic-shape burden without exact copy evidence. |
| `shrunk_with_new_colors_or_recode` | 15 | Shrink relation plus new output colors. Output is extracted and recoded, classified, or summarized rather than copied verbatim. | Unhandled; requires extraction plus recolor/encoding logic. |

## Practical Subtype Metadata

### `exact_subgrid_fixed_output`

| Metric | Value |
| --- | --- |
| Count | `13` |
| New color count modes | `[(0, 13)]` |
| Output shape count min/avg/max | `1` / `1.0000` / `1` |
| Exact subgrid ratio min/avg/max | `1.0000` / `1.0000` / `1.0000` |
| Examples per task min/avg/max | `264` / `266.3077` / `270` |

| Task | Task File | Flags | New Colors | Input Shapes | Output Shapes | Output Shape Count | Exact Crop Matches | Exact Crop Ratio |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `task039` | `competition_material/taskfiles/task039.json` | `shape_shrinks` | `[]` | `10x10:264` | `3x3:264` | 1 | 264/264 | 1.0000 |
| `task048` | `competition_material/taskfiles/task048.json` | `shape_shrinks` | `[]` | `7x5:24;5x5:21;6x7:21;8x5:20;8x7:20` | `1x1:270` | 1 | 270/270 | 1.0000 |
| `task079` | `competition_material/taskfiles/task079.json` | `shape_shrinks` | `[]` | `14x14:266` | `3x3:266` | 1 | 266/266 | 1.0000 |
| `task111` | `competition_material/taskfiles/task111.json` | `shape_shrinks` | `[]` | `10x10:265` | `3x3:265` | 1 | 265/265 | 1.0000 |
| `task135` | `competition_material/taskfiles/task135.json` | `shape_shrinks` | `[]` | `9x9:266` | `3x3:266` | 1 | 266/266 | 1.0000 |
| `task146` | `competition_material/taskfiles/task146.json` | `shape_shrinks` | `[]` | `9x3:267` | `3x3:267` | 1 | 267/267 | 1.0000 |
| `task207` | `competition_material/taskfiles/task207.json` | `shape_shrinks` | `[]` | `5x5:265` | `2x2:265` | 1 | 265/265 | 1.0000 |
| `task263` | `competition_material/taskfiles/task263.json` | `shape_shrinks` | `[]` | `3x15:50;3x9:48;9x3:47;3x12:44;12x3:39` | `3x3:267` | 1 | 267/267 | 1.0000 |
| `task271` | `competition_material/taskfiles/task271.json` | `shape_shrinks` | `[]` | `9x9:267` | `3x3:267` | 1 | 267/267 | 1.0000 |
| `task291` | `competition_material/taskfiles/task291.json` | `shape_shrinks` | `[]` | `18x18:20;18x15:18;16x18:17;17x18:16;18x17:15` | `1x1:265` | 1 | 265/265 | 1.0000 |
| `task326` | `competition_material/taskfiles/task326.json` | `shape_shrinks` | `[]` | `4x8:42;12x6:37;8x4:35;12x12:34;8x8:32` | `2x2:266` | 1 | 266/266 | 1.0000 |
| `task346` | `competition_material/taskfiles/task346.json` | `shape_shrinks` | `[]` | `10x6:9;11x9:8;12x6:8;6x9:8;5x7:8` | `1x1:267` | 1 | 267/267 | 1.0000 |
| `task355` | `competition_material/taskfiles/task355.json` | `shape_shrinks` | `[]` | `16x17:10;16x14:10;16x15:9;14x16:8;15x16:8` | `1x1:267` | 1 | 267/267 | 1.0000 |

### `exact_subgrid_variable_output`

| Metric | Value |
| --- | --- |
| Count | `14` |
| New color count modes | `[(0, 14)]` |
| Output shape count min/avg/max | `4` / `45.9286` / `182` |
| Exact subgrid ratio min/avg/max | `1.0000` / `1.0000` / `1.0000` |
| Examples per task min/avg/max | `265` / `266.1429` / `268` |

| Task | Task File | Flags | New Colors | Input Shapes | Output Shapes | Output Shape Count | Exact Crop Matches | Exact Crop Ratio |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `task014` | `competition_material/taskfiles/task014.json` | `shape_shrinks` | `[]` | `15x17:7;16x15:7;23x19:6;23x21:6;23x17:6` | `7x5:13;7x6:10;6x5:10;5x7:9;8x5:9` | 87 | 266/266 | 1.0000 |
| `task029` | `competition_material/taskfiles/task029.json` | `shape_shrinks` | `[]` | `18x15:5;25x12:4;20x13:4;22x19:4;24x11:4` | `8x9:4;1x6:4;10x2:4;16x1:4;11x14:4` | 182 | 266/266 | 1.0000 |
| `task031` | `competition_material/taskfiles/task031.json` | `shape_shrinks` | `[]` | `10x12:95;12x12:89;11x12:82` | `4x5:43;5x5:35;5x4:33;4x6:25;4x4:22` | 23 | 266/266 | 1.0000 |
| `task036` | `competition_material/taskfiles/task036.json` | `shape_shrinks` | `[]` | `30x30:265` | `5x4:38;4x3:36;5x3:32;3x5:30;4x5:29` | 11 | 265/265 | 1.0000 |
| `task049` | `competition_material/taskfiles/task049.json` | `shape_shrinks` | `[]` | `11x10:7;18x14:6;17x14:6;18x12:6;15x11:5` | `3x2:25;2x2:24;2x3:20;2x4:16;3x3:15` | 57 | 268/268 | 1.0000 |
| `task065` | `competition_material/taskfiles/task065.json` | `shape_shrinks` | `[]` | `3x3:46;11x11:41;13x13:40;5x5:39;7x7:39` | `1x1:46;5x5:41;6x6:40;2x2:39;3x3:39` | 7 | 266/266 | 1.0000 |
| `task067` | `competition_material/taskfiles/task067.json` | `shape_shrinks` | `[]` | `2x6:80;4x12:74;3x9:58;5x15:54` | `2x2:80;4x4:74;3x3:58;5x5:54` | 4 | 266/266 | 1.0000 |
| `task091` | `competition_material/taskfiles/task091.json` | `shape_shrinks` | `[]` | `13x10:11;11x13:10;14x11:10;9x9:8;12x9:8` | `4x6:9;5x5:7;3x10:7;3x8:7;5x7:6` | 99 | 266/266 | 1.0000 |
| `task174` | `competition_material/taskfiles/task174.json` | `shape_shrinks` | `[]` | `10x10:266` | `2x3:42;2x5:34;2x2:31;3x4:30;2x4:28` | 16 | 266/266 | 1.0000 |
| `task188` | `competition_material/taskfiles/task188.json` | `shape_shrinks` | `[]` | `4x4:27;8x3:22;6x2:20;2x8:19;8x2:17` | `4x3:39;3x2:36;4x2:32;2x4:31;3x4:29` | 9 | 266/266 | 1.0000 |
| `task216` | `competition_material/taskfiles/task216.json` | `shape_shrinks` | `[]` | `20x20:266` | `5x16:8;6x10:7;7x8:6;5x12:6;8x4:6` | 124 | 266/266 | 1.0000 |
| `task300` | `competition_material/taskfiles/task300.json` | `shape_shrinks` | `[]` | `9x11:33;9x9:23;8x12:22;9x10:22;8x11:21` | `3x3:161;4x3:87;2x3:12;3x2:7` | 4 | 267/267 | 1.0000 |
| `task310` | `competition_material/taskfiles/task310.json` | `shape_shrinks` | `[]` | `21x21:31;24x24:30;22x22:29;20x20:28;28x28:25` | `7x7:80;5x5:68;6x6:67;8x8:51` | 4 | 266/266 | 1.0000 |
| `task365` | `competition_material/taskfiles/task365.json` | `shape_shrinks` | `[]` | `10x10:266` | `3x3:47;4x3:34;3x4:27;5x3:25;4x4:21` | 16 | 266/266 | 1.0000 |

### `fixed_size_extraction_or_summary`

| Metric | Value |
| --- | --- |
| Count | `27` |
| New color count modes | `[(0, 27)]` |
| Output shape count min/avg/max | `1` / `1.0000` / `1` |
| Exact subgrid ratio min/avg/max | `0.0000` / `0.0200` / `0.3759` |
| Examples per task min/avg/max | `46` / `258.0370` / `269` |

| Task | Task File | Flags | New Colors | Input Shapes | Output Shapes | Output Shape Count | Exact Crop Matches | Exact Crop Ratio |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `task022` | `competition_material/taskfiles/task022.json` | `shape_shrinks` | `[]` | `11x11:266` | `3x3:266` | 1 | 0/266 | 0.0000 |
| `task038` | `competition_material/taskfiles/task038.json` | `shape_shrinks` | `[]` | `9x9:266` | `1x5:266` | 1 | 100/266 | 0.3759 |
| `task056` | `competition_material/taskfiles/task056.json` | `shape_shrinks` | `[]` | `3x3:46` | `1x1:46` | 1 | 5/46 | 0.1087 |
| `task057` | `competition_material/taskfiles/task057.json` | `shape_shrinks` | `[]` | `8x8:265` | `3x6:265` | 1 | 0/265 | 0.0000 |
| `task100` | `competition_material/taskfiles/task100.json` | `shape_shrinks` | `[]` | `10x10:266` | `2x2:266` | 1 | 0/266 | 0.0000 |
| `task121` | `competition_material/taskfiles/task121.json` | `shape_shrinks` | `[]` | `13x13:266` | `3x3:266` | 1 | 0/266 | 0.0000 |
| `task130` | `competition_material/taskfiles/task130.json` | `shape_shrinks` | `[]` | `9x9:265` | `3x3:265` | 1 | 5/265 | 0.0189 |
| `task134` | `competition_material/taskfiles/task134.json` | `shape_shrinks` | `[]` | `28x24:7;21x20:7;28x26:6;20x22:6;28x22:6` | `3x3:266` | 1 | 0/266 | 0.0000 |
| `task153` | `competition_material/taskfiles/task153.json` | `shape_shrinks` | `[]` | `10x10:265` | `3x3:265` | 1 | 0/265 | 0.0000 |
| `task180` | `competition_material/taskfiles/task180.json` | `shape_shrinks` | `[]` | `8x8:268` | `4x4:268` | 1 | 0/268 | 0.0000 |
| `task185` | `competition_material/taskfiles/task185.json` | `shape_shrinks` | `[]` | `29x29:184;27x27:83` | `3x3:267` | 1 | 0/267 | 0.0000 |
| `task189` | `competition_material/taskfiles/task189.json` | `shape_shrinks` | `[]` | `9x9:266` | `6x6:266` | 1 | 0/266 | 0.0000 |
| `task195` | `competition_material/taskfiles/task195.json` | `shape_shrinks` | `[]` | `15x17:92;17x19:90;16x18:83` | `9x9:265` | 1 | 0/265 | 0.0000 |
| `task242` | `competition_material/taskfiles/task242.json` | `shape_shrinks` | `[]` | `16x16:266` | `3x3:266` | 1 | 5/266 | 0.0188 |
| `task253` | `competition_material/taskfiles/task253.json` | `shape_shrinks` | `[]` | `13x13:265` | `4x4:265` | 1 | 0/265 | 0.0000 |
| `task257` | `competition_material/taskfiles/task257.json` | `shape_shrinks` | `[]` | `9x9:269` | `4x4:269` | 1 | 0/269 | 0.0000 |
| `task264` | `competition_material/taskfiles/task264.json` | `shape_shrinks` | `[]` | `16x16:111;16x15:63;15x16:30;15x15:19;16x14:17` | `9x9:265` | 1 | 0/265 | 0.0000 |
| `task274` | `competition_material/taskfiles/task274.json` | `shape_shrinks` | `[]` | `11x8:12;9x8:11;10x8:11;9x10:10;10x9:10` | `3x3:269` | 1 | 0/269 | 0.0000 |
| `task296` | `competition_material/taskfiles/task296.json` | `shape_shrinks` | `[]` | `5x7:268` | `3x3:268` | 1 | 1/268 | 0.0037 |
| `task316` | `competition_material/taskfiles/task316.json` | `shape_shrinks` | `[]` | `10x10:266` | `3x3:266` | 1 | 0/266 | 0.0000 |
| `task321` | `competition_material/taskfiles/task321.json` | `shape_shrinks` | `[]` | `4x14:267` | `4x4:267` | 1 | 0/267 | 0.0000 |
| `task351` | `competition_material/taskfiles/task351.json` | `shape_shrinks` | `[]` | `16x16:265` | `5x5:265` | 1 | 0/265 | 0.0000 |
| `task360` | `competition_material/taskfiles/task360.json` | `shape_shrinks` | `[]` | `10x9:266` | `10x4:266` | 1 | 1/266 | 0.0038 |
| `task372` | `competition_material/taskfiles/task372.json` | `shape_shrinks` | `[]` | `11x11:266` | `5x11:266` | 1 | 1/266 | 0.0038 |
| `task391` | `competition_material/taskfiles/task391.json` | `shape_shrinks` | `[]` | `15x10:52;13x13:45;15x16:45;13x16:43;13x10:43` | `3x1:267` | 1 | 0/267 | 0.0000 |
| `task393` | `competition_material/taskfiles/task393.json` | `shape_shrinks` | `[]` | `12x12:265` | `3x1:265` | 1 | 0/265 | 0.0000 |
| `task400` | `competition_material/taskfiles/task400.json` | `shape_shrinks` | `[]` | `24x24:266` | `5x5:266` | 1 | 2/266 | 0.0075 |

### `variable_size_extraction_or_summary`

| Metric | Value |
| --- | --- |
| Count | `30` |
| New color count modes | `[(0, 30)]` |
| Output shape count min/avg/max | `2` / `21.7000` / `134` |
| Exact subgrid ratio min/avg/max | `0.0000` / `0.1214` / `0.9962` |
| Examples per task min/avg/max | `265` / `266.1333` / `269` |

| Task | Task File | Flags | New Colors | Input Shapes | Output Shapes | Output Shape Count | Exact Crop Matches | Exact Crop Ratio |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `task021` | `competition_material/taskfiles/task021.json` | `shape_shrinks` | `[]` | `21x11:3;15x15:2;37x34:2;14x27:2;23x20:2` | `2x6:13;6x3:13;4x2:12;7x2:12;2x2:10` | 36 | 260/266 | 0.9774 |
| `task046` | `competition_material/taskfiles/task046.json` | `shape_shrinks` | `[]` | `3x12:42;3x14:42;3x11:35;3x16:30;3x17:27` | `3x9:54;3x11:44;3x10:37;3x13:30;3x8:29` | 11 | 0/267 | 0.0000 |
| `task088` | `competition_material/taskfiles/task088.json` | `shape_shrinks` | `[]` | `17x9:5;12x18:4;10x12:4;10x9:4;9x8:4` | `3x5:9;8x4:9;4x4:8;10x4:8;4x7:8` | 63 | 0/267 | 0.0000 |
| `task096` | `competition_material/taskfiles/task096.json` | `shape_shrinks` | `[]` | `18x18:11;18x15:10;15x17:9;17x15:9;17x14:9` | `9x9:96;7x7:90;11x11:80` | 3 | 0/266 | 0.0000 |
| `task109` | `competition_material/taskfiles/task109.json` | `shape_shrinks` | `[]` | `7x7:79;11x11:75;9x9:58;13x13:54` | `6x6:79;10x10:75;8x8:58;12x12:54` | 4 | 0/266 | 0.0000 |
| `task115` | `competition_material/taskfiles/task115.json` | `shape_shrinks` | `[]` | `14x14:6;13x15:5;15x11:5;18x13:4;10x14:4` | `1x3:75;1x4:66;3x1:63;4x1:62` | 4 | 19/266 | 0.0714 |
| `task138` | `competition_material/taskfiles/task138.json` | `shape_shrinks` | `[]` | `19x19:11;15x15:10;11x11:10;20x19:10;23x22:9` | `8x8:11;10x10:9;8x7:7;7x7:7;8x9:6` | 108 | 6/266 | 0.0226 |
| `task159` | `competition_material/taskfiles/task159.json` | `shape_shrinks` | `[]` | `23x20:5;30x17:4;25x18:4;27x24:4;29x16:4` | `5x5:134;8x8:83;11x11:34;14x14:14` | 4 | 0/265 | 0.0000 |
| `task170` | `competition_material/taskfiles/task170.json` | `shape_shrinks` | `[]` | `26x22:10;27x25:8;22x21:8;24x21:8;23x21:7` | `3x3:237;4x4:29` | 2 | 0/266 | 0.0000 |
| `task177` | `competition_material/taskfiles/task177.json` | `shape_shrinks` | `[]` | `18x14:7;11x10:7;18x16:6;10x12:6;18x12:6` | `7x4:20;7x5:17;4x6:15;8x6:14;8x7:13` | 25 | 13/265 | 0.0491 |
| `task178` | `competition_material/taskfiles/task178.json` | `shape_shrinks` | `[]` | `7x1:9;10x3:7;3x9:7;1x6:7;6x1:7` | `1x5:53;4x1:51;3x1:49;1x4:41;5x1:38` | 6 | 41/268 | 0.1530 |
| `task183` | `competition_material/taskfiles/task183.json` | `shape_shrinks` | `[]` | `6x6:95;10x10:88;8x8:82` | `2x2:95;6x6:88;4x4:82` | 3 | 0/265 | 0.0000 |
| `task184` | `competition_material/taskfiles/task184.json` | `shape_shrinks` | `[]` | `25x31:3;26x29:3;21x22:3;14x17:3;26x23:3` | `3x2:73;2x2:65;3x3:65;2x3:63` | 4 | 0/266 | 0.0000 |
| `task201` | `competition_material/taskfiles/task201.json` | `shape_shrinks` | `[]` | `13x13:266` | `6x8:53;5x6:50;4x6:44;6x6:44;5x8:39` | 7 | 0/266 | 0.0000 |
| `task205` | `competition_material/taskfiles/task205.json` | `shape_shrinks` | `[]` | `23x20:5;30x17:4;25x18:4;27x24:4;29x16:4` | `9x6:20;7x9:19;8x10:16;7x7:14;10x9:13` | 25 | 0/266 | 0.0000 |
| `task209` | `competition_material/taskfiles/task209.json` | `shape_shrinks` | `[]` | `20x17:12;17x19:12;17x17:11;19x16:11;15x15:10` | `11x15:10;10x15:9;10x16:8;12x12:8;11x14:7` | 102 | 0/266 | 0.0000 |
| `task213` | `competition_material/taskfiles/task213.json` | `shape_shrinks` | `[]` | `20x20:12;11x11:9;9x9:9;21x21:8;18x20:8` | `5x5:60;4x4:55;6x6:51;3x3:50;7x7:49` | 5 | 0/265 | 0.0000 |
| `task218` | `competition_material/taskfiles/task218.json` | `shape_shrinks` | `[]` | `21x21:266` | `3x2:73;3x3:66;2x2:65;2x3:62` | 4 | 65/266 | 0.2444 |
| `task233` | `competition_material/taskfiles/task233.json` | `shape_shrinks` | `[]` | `21x23:7;24x19:5;24x22:5;27x25:4;23x25:4` | `8x16:5;19x17:5;20x17:5;18x15:5;12x17:5` | 134 | 0/266 | 0.0000 |
| `task238` | `competition_material/taskfiles/task238.json` | `shape_shrinks` | `[]` | `16x15:38;15x14:35;14x16:32;16x14:32;15x16:30` | `6x6:98;5x5:85;7x7:83` | 3 | 0/266 | 0.0000 |
| `task244` | `competition_material/taskfiles/task244.json` | `shape_shrinks` | `[]` | `11x11:53;14x14:41;8x8:40;23x23:34;15x15:34` | `3x3:137;4x4:129` | 2 | 0/266 | 0.0000 |
| `task247` | `competition_material/taskfiles/task247.json` | `shape_shrinks` | `[]` | `10x10:269` | `5x1:30;3x3:29;5x3:28;3x1:27;4x2:25` | 13 | 36/269 | 0.1338 |
| `task290` | `competition_material/taskfiles/task290.json` | `shape_shrinks` | `[]` | `15x12:13;12x11:12;14x11:12;14x12:11;14x13:10` | `5x5:72;4x4:69;6x6:67;3x3:58` | 4 | 0/266 | 0.0000 |
| `task308` | `competition_material/taskfiles/task308.json` | `shape_shrinks` | `[]` | `15x14:8;12x14:7;8x9:7;18x20:7;8x7:7` | `3x3:95;7x7:88;5x5:83` | 3 | 4/266 | 0.0150 |
| `task319` | `competition_material/taskfiles/task319.json` | `shape_shrinks` | `[]` | `19x18:16;19x16:16;15x15:16;17x19:15;19x17:14` | `4x3:53;3x3:49;3x4:45;5x3:29;4x4:25` | 9 | 251/267 | 0.9401 |
| `task325` | `competition_material/taskfiles/task325.json` | `shape_shrinks` | `[]` | `9x8:8;16x10:7;15x9:7;16x12:7;16x14:7` | `3x3:80;2x2:78;4x4:59;5x5:32;1x1:10` | 6 | 10/266 | 0.0376 |
| `task366` | `competition_material/taskfiles/task366.json` | `shape_shrinks` | `[]` | `22x13:10;30x17:9;12x28:9;24x14:8;22x11:8` | `12x14:17;11x13:15;15x17:12;11x11:12;10x10:11` | 30 | 0/266 | 0.0000 |
| `task377` | `competition_material/taskfiles/task377.json` | `shape_shrinks` | `[]` | `28x24:7;21x20:7;28x26:6;20x22:6;28x22:6` | `3x3:122;5x5:107;7x7:28;9x9:9` | 4 | 0/266 | 0.0000 |
| `task394` | `competition_material/taskfiles/task394.json` | `shape_shrinks` | `[]` | `4x4:81;6x6:73;5x5:57;7x7:55` | `2x2:154;1x1:81;3x3:31` | 3 | 265/266 | 0.9962 |
| `task396` | `competition_material/taskfiles/task396.json` | `shape_shrinks` | `[]` | `16x13:11;17x14:10;17x17:9;14x16:9;15x16:8` | `6x7:25;6x8:22;7x7:20;5x7:19;8x6:18` | 24 | 0/266 | 0.0000 |

### `shrunk_with_new_colors_or_recode`

| Metric | Value |
| --- | --- |
| Count | `15` |
| New color count modes | `[(1, 13), (2, 1), (4, 1)]` |
| Output shape count min/avg/max | `1` / `1.2000` / `4` |
| Exact subgrid ratio min/avg/max | `0.0000` / `0.0073` / `0.1090` |
| Examples per task min/avg/max | `69` / `251.4000` / `271` |

| Task | Task File | Flags | New Colors | Input Shapes | Output Shapes | Output Shape Count | Exact Crop Matches | Exact Crop Ratio |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `task006` | `competition_material/taskfiles/task006.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[2]` | `3x7:266` | `3x3:266` | 1 | 0/266 | 0.0000 |
| `task026` | `competition_material/taskfiles/task026.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[8]` | `5x7:268` | `5x3:268` | 1 | 0/268 | 0.0000 |
| `task072` | `competition_material/taskfiles/task072.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[3]` | `13x5:268` | `6x5:268` | 1 | 0/268 | 0.0000 |
| `task103` | `competition_material/taskfiles/task103.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[1, 7]` | `3x3:223` | `1x1:223` | 1 | 0/223 | 0.0000 |
| `task144` | `competition_material/taskfiles/task144.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[3]` | `9x4:267` | `4x4:267` | 1 | 0/267 | 0.0000 |
| `task149` | `competition_material/taskfiles/task149.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[1]` | `11x11:267` | `3x3:267` | 1 | 0/267 | 0.0000 |
| `task227` | `competition_material/taskfiles/task227.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[2]` | `8x4:267` | `4x4:267` | 1 | 0/267 | 0.0000 |
| `task235` | `competition_material/taskfiles/task235.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[2, 3, 4, 8]` | `4x14:69` | `3x3:69` | 1 | 0/69 | 0.0000 |
| `task236` | `competition_material/taskfiles/task236.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[3]` | `9x4:267` | `4x4:267` | 1 | 0/267 | 0.0000 |
| `task259` | `competition_material/taskfiles/task259.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[0]` | `7x6:39;6x5:35;5x7:32;7x5:32;6x7:29` | `3x2:72;2x3:69;3x3:67;2x2:58` | 4 | 29/266 | 0.1090 |
| `task318` | `competition_material/taskfiles/task318.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[3]` | `9x4:267` | `4x4:267` | 1 | 0/267 | 0.0000 |
| `task334` | `competition_material/taskfiles/task334.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[5]` | `5x5:271` | `3x3:271` | 1 | 0/271 | 0.0000 |
| `task347` | `competition_material/taskfiles/task347.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[6]` | `3x6:269` | `3x3:269` | 1 | 0/269 | 0.0000 |
| `task386` | `competition_material/taskfiles/task386.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[3]` | `4x7:268` | `4x3:268` | 1 | 0/268 | 0.0000 |
| `task395` | `competition_material/taskfiles/task395.json` | <code>shape_shrinks&#124;new_output_colors</code> | `[2]` | `6x3:268` | `3x3:268` | 1 | 0/268 | 0.0000 |

