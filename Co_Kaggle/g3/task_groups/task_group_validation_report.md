# Task Group Validation Report

This report validates `task_groups` artifacts against the raw task JSON files.

## Artifact Consistency

- CSV rows: 400
- JSON rows: 400
- Grouped families: 11
- Issues: 0
- Warnings: 0

## Family Counts

| Family | Count | High | Medium | Low | Property check failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `identity_noop` | 0 | 0 | 0 | 0 | 0 |
| `global_color_remap` | 4 | 4 | 0 | 0 | 0 |
| `same_shape_local_rule` | 27 | 0 | 27 | 0 | 0 |
| `mask_object_selection` | 66 | 0 | 0 | 66 | 0 |
| `fill_enclosed_regions` | 59 | 0 | 59 | 0 | 0 |
| `expansion_tiling` | 35 | 0 | 35 | 0 | 0 |
| `cropping_extraction` | 99 | 0 | 99 | 0 | 0 |
| `geometric_transform` | 7 | 7 | 0 | 0 | 0 |
| `pattern_completion` | 99 | 0 | 0 | 99 | 0 |
| `counting_relational` | 0 | 0 | 0 | 0 | 0 |
| `composite_or_unknown` | 4 | 0 | 0 | 4 | 0 |

## Validation Interpretation

- The families are heuristic routing labels for solver development, not proven ARC semantic labels.
- Shape-driven families are strongly validated by raw grid dimensions.
- `global_color_remap` is strongly validated by consistent cellwise color mappings.
- Same-shape semantic families are weaker: they separate tasks by observable signals such as local consistency, input preservation, and added colors.
- `fill_enclosed_regions` should be read as an additive fill/marking family; some tasks may fill enclosed regions, while others preserve input cells and add new marked cells.
- `mask_object_selection`, `pattern_completion`, and `composite_or_unknown` should be manually reviewed before relying on a specialized solver.

## Sample Tasks By Family

- `identity_noop`: (none)
- `global_color_remap`: task016, task276, task309, task337
- `same_shape_local_rule`: task004, task053, task070, task073, task077, task097, task098, task120, task127, task129, task147, task151
- `mask_object_selection`: task008, task010, task011, task018, task023, task030, task032, task034, task035, task040, task052, task058
- `fill_enclosed_regions`: task002, task015, task027, task042, task043, task047, task050, task055, task060, task063, task081, task090
- `expansion_tiling`: task001, task003, task019, task083, task104, task106, task107, task108, task114, task116, task123, task124
- `cropping_extraction`: task006, task014, task021, task022, task026, task029, task031, task036, task038, task039, task046, task048
- `geometric_transform`: task087, task140, task150, task155, task179, task241, task380
- `pattern_completion`: task005, task007, task009, task012, task013, task017, task020, task024, task025, task028, task033, task037
- `counting_relational`: (none)
- `composite_or_unknown`: task239, task339, task384, task399
