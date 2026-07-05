# Pattern Continuation / Completion Task Profile

Source: `task_groups/task_type_map.csv`

## Task Type

| Field | Value |
| --- | --- |
| Primary family | `pattern_completion` |
| Task count | `99` |
| Common shape relation | `same_shape_variable_size` |
| Common behavior | Input is mostly preserved while missing pattern cells are completed, extended, or repaired. |
| Current notebook | `(not yet assigned)` |

## Flag-Based Subtypes

| Subtype Flags | Count | Solver Meaning |
| --- | ---: | --- |
| `mostly_preserves_input_adds_or_changes` | 96 | Mostly same-color continuation/completion; output modifies or adds cells while preserving at least 80% of input nonzero cells. |
| `mostly_preserves_input_adds_or_changes|new_output_colors` | 3 | Completion introduces one or more colors not present in the input; may overlap with additive marking behavior. |

## Practical Solver Subtypes

| Practical Subtype | Count | Characteristics | Current Solver Status |
| --- | ---: | --- | --- |
| `localish_additive_completion` | 15 | Near-local same-color completion; mostly explainable by small neighborhoods and only adds cells into background. | Candidate for widened symbolic/local completion solver; not covered by strict 3x3 consistency. |
| `localish_recolor_completion` | 12 | Near-local completion with some existing nonzero cells also changed; local solver may need overwrite handling. | Candidate for widened symbolic/local completion solver; not covered by strict 3x3 consistency. |
| `nonlocal_additive_completion` | 57 | Long-range or object-level completion; input foreground is preserved and missing cells are added. | Unhandled; likely identity fallback only. |
| `nonlocal_recolor_completion` | 15 | Long-range completion with existing nonzero cells changed; highest risk for generic additive solvers. | Unhandled; likely identity fallback only. |

## Practical Subtype Metadata

### `localish_additive_completion`

| Metric | Value |
| --- | --- |
| Count | `15` |
| New color count modes | `[(0, 15)]` |
| Local 3x3 score min/avg | `0.9535` / `0.9677` |
| Local 3x3 conflicts min/avg | `135` / `647.0667` |
| Added nonzero cells avg | `7485.2667` |
| Changed cells avg | `7485.2667` |
| Input nonzero preserved ratio avg | `1.0000` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `task017` | `competition_material/taskfiles/task017.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `21x21:266` | 0.9616 | 1016 | 14860 | 14860 | 1.0000 |
| `task020` | `competition_material/taskfiles/task020.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.9608 | 235 | 798 | 798 | 1.0000 |
| `task041` | `competition_material/taskfiles/task041.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.9597 | 242 | 2748 | 2748 | 1.0000 |
| `task051` | `competition_material/taskfiles/task051.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `14x18:8;19x15:7;15x12:6;11x17:5;18x16:5` | 0.9720 | 375 | 1490 | 1490 | 1.0000 |
| `task061` | `competition_material/taskfiles/task061.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `18x18:267` | 0.9665 | 651 | 11265 | 11265 | 1.0000 |
| `task110` | `competition_material/taskfiles/task110.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `29x29:266` | 0.9723 | 1400 | 15426 | 15426 | 1.0000 |
| `task112` | `competition_material/taskfiles/task112.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `27x22:4;18x15:4;15x21:3;20x15:3;25x12:3` | 0.9617 | 918 | 4275 | 4275 | 1.0000 |
| `task133` | `competition_material/taskfiles/task133.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `28x27:5;30x27:4;23x27:4;20x26:4;17x18:3` | 0.9607 | 1265 | 4392 | 4392 | 1.0000 |
| `task168` | `competition_material/taskfiles/task168.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:265` | 0.9535 | 279 | 1200 | 1200 | 1.0000 |
| `task173` | `competition_material/taskfiles/task173.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `25x12:6;25x23:5;25x25:4;25x16:4;24x23:4` | 0.9816 | 382 | 896 | 896 | 1.0000 |
| `task175` | `competition_material/taskfiles/task175.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `21x21:266` | 0.9676 | 856 | 13432 | 13432 | 1.0000 |
| `task243` | `competition_material/taskfiles/task243.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `12x12:46;16x16:42;13x13:39;17x17:39;14x14:38` | 0.9815 | 246 | 23239 | 23239 | 1.0000 |
| `task285` | `competition_material/taskfiles/task285.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `29x29:43;28x28:32;30x30:31;26x26:22;27x27:21` | 0.9662 | 1326 | 6791 | 6791 | 1.0000 |
| `task305` | `competition_material/taskfiles/task305.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `16x16:266` | 0.9753 | 380 | 10671 | 10671 | 1.0000 |
| `task378` | `competition_material/taskfiles/task378.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:49;11x11:45;9x9:41;12x12:35;7x7:35` | 0.9743 | 135 | 796 | 796 | 1.0000 |

### `localish_recolor_completion`

| Metric | Value |
| --- | --- |
| Count | `12` |
| New color count modes | `[(0, 11), (1, 1)]` |
| Local 3x3 score min/avg | `0.9505` / `0.9711` |
| Local 3x3 conflicts min/avg | `147` / `568.0833` |
| Added nonzero cells avg | `503.7500` |
| Changed cells avg | `4725.2500` |
| Input nonzero preserved ratio avg | `0.8977` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `task025` | `competition_material/taskfiles/task025.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `25x16:4;20x17:4;17x23:4;30x23:4;15x14:3` | 0.9886 | 307 | 1064 | 2534 | 0.9103 |
| `task064` | `competition_material/taskfiles/task064.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `24x17:5;19x21:4;22x9:4;18x20:4;16x13:4` | 0.9714 | 452 | 0 | 2206 | 0.9706 |
| `task074` | `competition_material/taskfiles/task074.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `30x30:267` | 0.9609 | 2114 | 0 | 21717 | 0.8707 |
| `task093` | `competition_material/taskfiles/task093.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `14x14:265` | 0.9589 | 483 | 1395 | 3187 | 0.8789 |
| `task118` | `competition_material/taskfiles/task118.json` | `mostly_preserves_input_adds_or_changes|new_output_colors` | `[8]` | `15x14:7;18x15:6;11x10:6;19x21:6;22x20:6` | 0.9851 | 273 | 0 | 2600 | 0.9445 |
| `task143` | `competition_material/taskfiles/task143.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.9600 | 240 | 0 | 928 | 0.8574 |
| `task158` | `competition_material/taskfiles/task158.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `26x25:19;25x24:17;22x23:15;24x23:15;21x22:13` | 0.9629 | 1005 | 0 | 4987 | 0.9595 |
| `task182` | `competition_material/taskfiles/task182.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `20x20:267` | 0.9709 | 699 | 0 | 2516 | 0.8465 |
| `task208` | `competition_material/taskfiles/task208.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `21x21:266` | 0.9944 | 147 | 509 | 4836 | 0.9566 |
| `task228` | `competition_material/taskfiles/task228.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.9645 | 213 | 1064 | 2128 | 0.8009 |
| `task287` | `competition_material/taskfiles/task287.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `16x16:267` | 0.9857 | 220 | 0 | 4486 | 0.9344 |
| `task340` | `competition_material/taskfiles/task340.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `18x14:7;11x10:7;18x16:6;10x12:6;18x12:6` | 0.9505 | 664 | 2013 | 4578 | 0.8423 |

### `nonlocal_additive_completion`

| Metric | Value |
| --- | --- |
| Count | `57` |
| New color count modes | `[(0, 57)]` |
| Local 3x3 score min/avg | `0.0531` / `0.7523` |
| Local 3x3 conflicts min/avg | `49` / `2112.5439` |
| Added nonzero cells avg | `7511.9123` |
| Changed cells avg | `7511.9123` |
| Input nonzero preserved ratio avg | `1.0000` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `task005` | `competition_material/taskfiles/task005.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `21x21:266` | 0.9262 | 1952 | 7751 | 7751 | 1.0000 |
| `task007` | `competition_material/taskfiles/task007.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `7x7:266` | 0.4714 | 1554 | 10051 | 10051 | 1.0000 |
| `task009` | `competition_material/taskfiles/task009.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `17x17:65;23x23:58;20x20:50;29x29:50;26x26:42` | 0.9397 | 1875 | 7312 | 7312 | 1.0000 |
| `task012` | `competition_material/taskfiles/task012.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `12x12:265` | 0.9444 | 480 | 6360 | 6360 | 1.0000 |
| `task013` | `competition_material/taskfiles/task013.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `11x24:7;21x10:6;10x22:6;11x30:5;22x8:5` | 0.7760 | 3023 | 12588 | 12588 | 1.0000 |
| `task024` | `competition_material/taskfiles/task024.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `13x7:7;14x10:7;7x6:7;7x14:6;11x8:6` | 0.7137 | 1936 | 10100 | 10100 | 1.0000 |
| `task028` | `competition_material/taskfiles/task028.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:265` | 0.2452 | 4529 | 13250 | 13250 | 1.0000 |
| `task033` | `competition_material/taskfiles/task033.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `17x17:265` | 0.8270 | 3000 | 11880 | 11880 | 1.0000 |
| `task037` | `competition_material/taskfiles/task037.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.9117 | 530 | 2252 | 2252 | 1.0000 |
| `task045` | `competition_material/taskfiles/task045.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:265` | 0.9085 | 549 | 2024 | 2024 | 1.0000 |
| `task066` | `competition_material/taskfiles/task066.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `11x11:31;14x14:29;12x12:29;10x10:28;19x19:25` | 0.9305 | 952 | 4119 | 4119 | 1.0000 |
| `task076` | `competition_material/taskfiles/task076.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `15x14:37;14x13:35;13x15:31;15x13:31;13x13:30` | 0.9381 | 723 | 2265 | 2265 | 1.0000 |
| `task080` | `competition_material/taskfiles/task080.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `29x29:89;24x24:53;27x27:48;26x26:41;31x31:35` | 0.8636 | 6139 | 18032 | 18032 | 1.0000 |
| `task082` | `competition_material/taskfiles/task082.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `6x9:32;6x15:28;6x7:26;6x13:26;6x14:26` | 0.7572 | 864 | 5512 | 5512 | 1.0000 |
| `task084` | `competition_material/taskfiles/task084.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `15x15:10;3x3:10;7x7:10;10x10:10;20x20:9` | 0.8666 | 1276 | 3824 | 3824 | 1.0000 |
| `task089` | `competition_material/taskfiles/task089.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `13x13:267` | 0.9250 | 760 | 2165 | 2165 | 1.0000 |
| `task092` | `competition_material/taskfiles/task092.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `30x20:39;20x10:36;30x10:32;10x30:31;20x30:29` | 0.9173 | 2002 | 8640 | 8640 | 1.0000 |
| `task099` | `competition_material/taskfiles/task099.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:265` | 0.9255 | 447 | 5169 | 5169 | 1.0000 |
| `task101` | `competition_material/taskfiles/task101.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `14x12:151;17x14:114;17x21:1` | 0.9254 | 876 | 3661 | 3661 | 1.0000 |
| `task113` | `competition_material/taskfiles/task113.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x6:38;10x3:37;10x2:34;10x4:33;10x10:28` | 0.6980 | 1045 | 4616 | 4616 | 1.0000 |
| `task117` | `competition_material/taskfiles/task117.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `12x12:81;14x14:75;13x13:56;15x15:53` | 0.8930 | 1175 | 4305 | 4305 | 1.0000 |
| `task132` | `competition_material/taskfiles/task132.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `13x7:7;14x10:7;7x6:7;7x14:6;11x8:6` | 0.7024 | 1988 | 8356 | 8356 | 1.0000 |
| `task136` | `competition_material/taskfiles/task136.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.3088 | 4147 | 1542 | 1542 | 1.0000 |
| `task137` | `competition_material/taskfiles/task137.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `21x21:31;22x22:29;24x24:28;20x20:28;28x28:25` | 0.7306 | 10054 | 47479 | 47479 | 1.0000 |
| `task141` | `competition_material/taskfiles/task141.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `13x13:38;7x7:35;15x15:32;9x9:32;21x21:30` | 0.0531 | 10336 | 4372 | 4372 | 1.0000 |
| `task165` | `competition_material/taskfiles/task165.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `20x20:265` | 0.9353 | 1554 | 6943 | 6943 | 1.0000 |
| `task181` | `competition_material/taskfiles/task181.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `6x9:266` | 0.5756 | 1375 | 1249 | 1249 | 1.0000 |
| `task190` | `competition_material/taskfiles/task190.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.9465 | 321 | 1142 | 1142 | 1.0000 |
| `task191` | `competition_material/taskfiles/task191.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `23x23:267` | 0.9266 | 2329 | 10689 | 10689 | 1.0000 |
| `task197` | `competition_material/taskfiles/task197.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x8:36;14x10:32;6x8:30;14x8:28;12x8:26` | 0.8380 | 930 | 4045 | 4045 | 1.0000 |
| `task212` | `competition_material/taskfiles/task212.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:265` | 0.8048 | 1171 | 4731 | 4731 | 1.0000 |
| `task214` | `competition_material/taskfiles/task214.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `3x11:266` | 0.5227 | 945 | 4788 | 4788 | 1.0000 |
| `task215` | `competition_material/taskfiles/task215.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x12:7;18x14:7;11x10:7;15x19:6;18x16:6` | 0.6012 | 5395 | 23884 | 23884 | 1.0000 |
| `task217` | `competition_material/taskfiles/task217.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `9x9:266` | 0.1996 | 3890 | 3868 | 3868 | 1.0000 |
| `task224` | `competition_material/taskfiles/task224.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `14x13:22;14x15:21;13x13:17;13x12:16;15x14:13` | 0.7911 | 2280 | 9008 | 9008 | 1.0000 |
| `task225` | `competition_material/taskfiles/task225.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `6x6:265` | 0.3829 | 1333 | 2914 | 2914 | 1.0000 |
| `task237` | `competition_material/taskfiles/task237.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `7x4:11;8x5:10;5x7:9;3x3:8;6x3:8` | 0.7873 | 421 | 2543 | 2543 | 1.0000 |
| `task240` | `competition_material/taskfiles/task240.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `19x19:266` | 0.9268 | 1586 | 6838 | 6838 | 1.0000 |
| `task248` | `competition_material/taskfiles/task248.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x2:2;10x3:2;10x4:2;10x5:2;10x10:1` | 0.8471 | 104 | 117 | 117 | 1.0000 |
| `task268` | `competition_material/taskfiles/task268.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `7x7:49;10x10:49;9x9:45;6x6:45;5x5:40` | 0.7101 | 1024 | 6543 | 6543 | 1.0000 |
| `task280` | `competition_material/taskfiles/task280.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `20x20:186;10x10:81` | 0.8116 | 3673 | 15904 | 15904 | 1.0000 |
| `task284` | `competition_material/taskfiles/task284.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `19x9:8;10x13:7;19x10:7;11x7:6;7x21:6` | 0.8394 | 1336 | 5760 | 5760 | 1.0000 |
| `task286` | `competition_material/taskfiles/task286.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `18x15:5;20x13:4;22x19:4;24x11:4;11x15:4` | 0.6170 | 7132 | 28692 | 28692 | 1.0000 |
| `task288` | `competition_material/taskfiles/task288.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `5x5:87;7x7:82;9x9:50;3x3:48` | 0.9149 | 214 | 1076 | 1076 | 1.0000 |
| `task297` | `competition_material/taskfiles/task297.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x4:64;14x6:58;12x5:54;8x3:49;6x2:40` | 0.2965 | 1708 | 10126 | 10126 | 1.0000 |
| `task306` | `competition_material/taskfiles/task306.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `19x9:95;19x29:88;19x19:82` | 0.8844 | 2459 | 8009 | 8009 | 1.0000 |
| `task322` | `competition_material/taskfiles/task322.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `3x3:266` | 0.9093 | 49 | 733 | 733 | 1.0000 |
| `task328` | `competition_material/taskfiles/task328.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `7x7:27;8x8:27;17x17:25;10x10:22;6x6:22` | 0.1471 | 7458 | 18670 | 18670 | 1.0000 |
| `task333` | `competition_material/taskfiles/task333.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:265` | 0.9452 | 329 | 1048 | 1048 | 1.0000 |
| `task343` | `competition_material/taskfiles/task343.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `5x15:266` | 0.8747 | 564 | 2865 | 2865 | 1.0000 |
| `task345` | `competition_material/taskfiles/task345.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:264` | 0.8028 | 1183 | 7422 | 7422 | 1.0000 |
| `task356` | `competition_material/taskfiles/task356.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.9120 | 528 | 1722 | 1722 | 1.0000 |
| `task358` | `competition_material/taskfiles/task358.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `12x11:22;15x14:17;13x12:16;20x19:16;10x10:16` | 0.9094 | 1298 | 6070 | 6070 | 1.0000 |
| `task361` | `competition_material/taskfiles/task361.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.9430 | 342 | 1980 | 1980 | 1.0000 |
| `task363` | `competition_material/taskfiles/task363.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:265` | 0.9243 | 454 | 1948 | 1948 | 1.0000 |
| `task382` | `competition_material/taskfiles/task382.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `12x18:7;18x11:5;12x14:5;18x14:5;15x19:5` | 0.7244 | 3696 | 16076 | 16076 | 1.0000 |
| `task385` | `competition_material/taskfiles/task385.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x4:265` | 0.5325 | 1122 | 3151 | 3151 | 1.0000 |

### `nonlocal_recolor_completion`

| Metric | Value |
| --- | --- |
| Count | `15` |
| New color count modes | `[(0, 13), (1, 2)]` |
| Local 3x3 score min/avg | `0.4976` / `0.8654` |
| Local 3x3 conflicts min/avg | `350` / `1645.3333` |
| Added nonzero cells avg | `1420.9333` |
| Changed cells avg | `6349.6667` |
| Input nonzero preserved ratio avg | `0.8696` |

| Task | Task File | Flags | New Colors | Shapes | Local 3x3 Score | Conflicts | Added Cells | Changed Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `task044` | `competition_material/taskfiles/task044.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `10x10:266` | 0.9088 | 547 | 2544 | 5088 | 0.8054 |
| `task054` | `competition_material/taskfiles/task054.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `30x30:266` | 0.9251 | 4043 | 0 | 20366 | 0.9149 |
| `task059` | `competition_material/taskfiles/task059.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `11x11:266` | 0.9021 | 711 | 3918 | 5981 | 0.8596 |
| `task085` | `competition_material/taskfiles/task085.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `16x30:22;12x20:21;9x20:18;16x20:17;15x20:16` | 0.9436 | 1066 | 0 | 5183 | 0.8454 |
| `task094` | `competition_material/taskfiles/task094.json` | `mostly_preserves_input_adds_or_changes|new_output_colors` | `[6]` | `15x15:265` | 0.8764 | 1669 | 0 | 10834 | 0.8183 |
| `task202` | `competition_material/taskfiles/task202.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `19x14:4;20x18:4;17x20:4;14x15:3;10x35:3` | 0.9367 | 1176 | 0 | 4637 | 0.9419 |
| `task206` | `competition_material/taskfiles/task206.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `12x9:13;11x8:12;11x9:11;9x8:11;8x7:10` | 0.9351 | 350 | 1113 | 1379 | 0.8383 |
| `task279` | `competition_material/taskfiles/task279.json` | `mostly_preserves_input_adds_or_changes|new_output_colors` | `[8]` | `16x15:31;16x16:28;15x15:19;15x16:19;16x14:14` | 0.9213 | 992 | 0 | 5539 | 0.8982 |
| `task281` | `competition_material/taskfiles/task281.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `13x12:41;13x11:33;12x11:31;11x13:30;11x11:29` | 0.8519 | 1283 | 4473 | 5283 | 0.8194 |
| `task314` | `competition_material/taskfiles/task314.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `8x8:266` | 0.8862 | 437 | 0 | 522 | 0.9455 |
| `task324` | `competition_material/taskfiles/task324.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `18x14:8;11x10:7;17x19:6;18x16:6;10x12:6` | 0.7225 | 3738 | 0 | 10867 | 0.8174 |
| `task370` | `competition_material/taskfiles/task370.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `12x18:7;11x18:6;14x20:6;11x15:5;12x15:5` | 0.9361 | 867 | 0 | 3213 | 0.9426 |
| `task375` | `competition_material/taskfiles/task375.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `5x5:10;7x7:10;11x11:10;13x13:9;9x9:9` | 0.4976 | 3132 | 0 | 1016 | 0.8355 |
| `task379` | `competition_material/taskfiles/task379.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `13x12:8;13x20:7;20x16:7;15x19:7;12x14:7` | 0.8713 | 2025 | 5483 | 6159 | 0.9032 |
| `task383` | `competition_material/taskfiles/task383.json` | `mostly_preserves_input_adds_or_changes` | `[]` | `20x20:7;21x19:7;19x15:7;16x18:6;19x16:6` | 0.8659 | 2644 | 3783 | 9178 | 0.8581 |
