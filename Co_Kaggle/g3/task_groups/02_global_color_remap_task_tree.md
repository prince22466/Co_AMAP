# Global Color Remapping Task Profile

Source: `task_groups/task_type_map.csv`

## Task Type

| Field | Value |
| --- | --- |
| Primary family | `global_color_remap` |
| Task count | `4` |
| Common shape relation | `same_shape_variable_size` |
| Common behavior | Geometry and cell positions are preserved while input colors are replaced through one globally consistent color mapping. |
| Current notebook | `(not yet assigned)` |

## Solver Profile

All four tasks use a consistent global input-color-to-output-color mapping with zero mapping conflicts. They do not require practical solver subtypes.

| Field | Value |
| --- | --- |
| Recommended solver | `global_color_remap` |
| ONNX approach | One-hot color encoding followed by a `1x1` convolution or direct channel lookup. |
| Output shape | Always identical to the input shape. |
| Mapping conflicts | `0` for all tasks. |
| Local 3x3 score min/avg | `1.0000` / `1.0000` |
| Local 3x3 conflicts min/avg | `0` / `0.0000` |
| New color count modes | `[(0, 2), (1, 2)]` |
| Changed cells avg | `2418.5000` |
| Input nonzero preserved ratio avg | `0.3917` |

## Task Metadata

| Task | Task File | Flags | New Colors | Shapes | Global Mapping | Mapping Conflicts | Local 3x3 Score | Local 3x3 Conflicts | Changed Cells | Preserve Ratio |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `task016` | `competition_material/taskfiles/task016.json` | <code>global_color_remap&#124;mapping_consistent&#124;local_3x3_consistent</code> | `[]` | `3x3:267` | `{"1":5,"2":6,"3":4,"4":3,"5":1,"6":2,"8":9,"9":8}` | 0 | 1.0000 | 0 | 2403 | 0.0000 |
| `task276` | `competition_material/taskfiles/task276.json` | <code>global_color_remap&#124;mapping_consistent&#124;local_3x3_consistent&#124;new_output_colors</code> | `[2]` | `6x5:38;5x4:35;6x4:33;4x6:31;5x6:29` | `{"6":2,"7":7}` | 0 | 1.0000 | 0 | 3321 | 0.5011 |
| `task309` | `competition_material/taskfiles/task309.json` | <code>global_color_remap&#124;mapping_consistent&#124;local_3x3_consistent&#124;new_output_colors</code> | `[5]` | `3x4:94;3x6:88;3x5:83` | `{"1":1,"7":5,"8":8}` | 0 | 1.0000 | 0 | 1314 | 0.6679 |
| `task337` | `competition_material/taskfiles/task337.json` | <code>global_color_remap&#124;mapping_consistent&#124;local_3x3_consistent</code> | `[]` | `3x3:96;5x5:88;4x4:82` | `{"1":1,"2":2,"3":3,"4":4,"5":8,"6":6,"7":7,"8":5,"9":9}` | 0 | 1.0000 | 0 | 2636 | 0.3976 |
