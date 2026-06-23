# Task Type Profiles

This document profiles the main task families in the NeuroGolf task files and how they should be treated when building per-task ONNX models.

The families are **solver-routing labels**, not guaranteed ARC semantic truth. They are intended to answer: "Which solver generator should we try first for this task?" The generated maps in this folder use conservative measurable signals such as shape relation, consistent color mapping, local-patch consistency, input preservation, and newly introduced colors.

## Dataset Format

Each task is stored as one JSON file:

```text
competition_material/taskfiles/task001.json
...
competition_material/taskfiles/task400.json
```

Each file contains:

```json
{
  "train": [ {"input": [[...]], "output": [[...]]} ],
  "test": [ {"input": [[...]], "output": [[...]]} ],
  "arc-gen": [ {"input": [[...]], "output": [[...]]} ]
}
```

Each grid is a rectangular list of lists with integer colors `0..9`.

Before ONNX inference, each input grid is encoded as:

```text
[BATCH=1, CHANNELS=10, HEIGHT=30, WIDTH=30]
```

The encoding is one-hot by color channel. Pixels outside the original grid boundary are zero-hot across all channels.

## Observed Shape Profile

From the 400 task files:

```text
Tasks: 400
Examples per task: min 7, average about 254, max 273
Same-shape examples: 65,907
Expanding examples: 9,319
Shrinking examples: 26,492
```

Common shape pairs include:

```text
10x10 -> 10x10
3x3   -> 3x3
9x9   -> 9x9
15x15 -> 15x15
12x12 -> 12x12
3x3   -> 6x6
3x3   -> 9x9
10x10 -> 3x3
```

This suggests the first solver effort should prioritize same-shape transformations, then expansion and extraction families.

The generated map files are:

```text
task_groups/task_type_map.csv
task_groups/task_type_map.json
task_groups/task_type_groups.json
```

The map can be regenerated with:

```text
python task_groups/generate_task_type_map.py
```

The artifacts can be validated with:

```text
python task_groups/validate_task_groups.py
```

## Current Task Counts

These counts come from the current generated `task_type_map.csv` and are heuristic primary-family assignments.

| Family Label | Profile Section | Task Count |
| --- | --- | ---: |
| `identity_noop` | Identity / No-Op | 0 |
| `global_color_remap` | Global Color Remapping | 4 |
| `same_shape_local_rule` | Same-Shape Local Rules | 27 |
| `mask_object_selection` | Mask / Object Selection | 66 |
| `fill_enclosed_regions` | Fill / Additive Marking | 59 |
| `expansion_tiling` | Expansion / Tiling | 35 |
| `cropping_extraction` | Cropping / Extraction | 99 |
| `geometric_transform` | Geometric Transformations | 7 |
| `pattern_completion` | Pattern Continuation / Completion | 99 |
| `counting_relational` | Counting / Relational Tasks | 0 |
| `composite_or_unknown` | Composite Tasks | 4 |

Total mapped tasks: `400`.

## Group 1: Identity / No-Op

Current mapped tasks: `0`.

Characteristics:

- Input grid equals output grid.
- Same height and width.
- All colors preserved.

Input:

- Any rectangular grid up to `30x30`.
- Usually sparse or simple object layouts.

Output:

- Same shape as input.
- Same color values at every cell.

Useful checks:

- `input == output` for all visible examples.

ONNX approach:

- Identity graph or 1x1 channel-preserving convolution.

Risk:

- Low if all visible examples are exact identity.

## Group 2: Global Color Remapping

Current mapped tasks: `4`.

Characteristics:

- Same input/output shape.
- Geometry is unchanged.
- One or more colors are replaced consistently.

Input:

- Rectangular grid with colors `0..9`.
- Object positions stay fixed.

Output:

- Same size.
- Same nonzero pattern or same object layout.
- Color IDs changed by a global mapping, such as `1 -> 2`.

Useful checks:

- For every cell position, output color depends only on input color.
- Mapping is consistent across `train + test + arc-gen`.

ONNX approach:

- 1x1 convolution / channel remap.

Risk:

- Low when mapping is global.
- Higher if color change depends on object role or position.

## Group 3: Same-Shape Local Rules

Current mapped tasks: `27`.

Characteristics:

- Input and output have the same shape.
- Output cell depends on a small neighborhood around the input cell.
- Examples include edge marking, isolated-pixel removal, neighbor-based recoloring, or local hole detection.

Input:

- Usually same-size grids such as `10x10`, `15x15`, `20x20`, or smaller.
- Colors represent local shapes or masks.

Output:

- Same size.
- Some cells are modified based on nearby context.

Useful checks:

- Try to explain output from a `3x3`, `5x5`, or `7x7` neighborhood.
- Compare whether the same local patch always maps to the same output center.

ONNX approach:

- Small convolution kernels.
- Stacked conv layers for multi-step local propagation.

Risk:

- Medium. Some tasks look local but require connected-component or global object reasoning.

## Group 4: Mask / Object Selection

Current mapped tasks: `66`.

Characteristics:

- Same shape.
- Only selected objects or colors remain, move, or get highlighted.
- Other objects are removed or turned into background.

Input:

- Multiple objects, colors, or connected components.
- Same grid size across examples is common.

Output:

- Same size.
- One object or subset of cells is kept, deleted, or recolored.

Useful checks:

- Object selected by color, size, shape, position, or uniqueness.
- Background usually remains `0`.

ONNX approach:

- Simple cases: color masks and local filters.
- Hard cases: connected-component logic or global comparisons.

Risk:

- Medium to high. Private examples may include different object counts or arrangements.

## Group 5: Fill / Additive Marking

Current mapped tasks: `59`.

Characteristics:

- Same shape.
- Existing non-background input cells are usually preserved.
- Background or selected cells are filled, marked, or recolored with one or more new colors.
- Some tasks are true enclosed-region fills; others are broader additive marking tasks.

Example:

- `task002` appears to use color `3` as boundary and color `4` as fill, so it is a true enclosed-region example.

Input:

- One or more outlines, loops, masks, or foreground objects.
- Often color `0` background plus one or more foreground colors.

Output:

- Same size.
- Existing foreground is often preserved.
- New cells are added as fill, markings, or inferred labels.

Useful checks:

- Does the output add new colors while preserving original nonzero cells?
- Are boundary or foreground cells preserved?
- Is the added fill/mark color fixed or predictable across examples?

ONNX approach:

- Easy cases: local pattern fill.
- Hard cases: flood-fill-like global enclosure detection.

Risk:

- Medium to high. Full enclosure detection is global, and broader additive marking can hide object-level or relational rules.

## Group 6: Expansion / Tiling

Current mapped tasks: `35`.

Characteristics:

- Output is larger than input.
- Common examples include `3x3 -> 6x6` or `3x3 -> 9x9`.
- Input pattern is repeated, scaled, or used as a template.

Input:

- Often small, especially `3x3`.
- Colors may define a mask or pattern.

Output:

- Larger rectangular grid.
- Pattern is copied, scaled, or arranged into blocks.

Useful checks:

- Output dimensions are integer multiples of input dimensions.
- Each input cell may correspond to a block or subgrid in the output.
- `task001` is a clear `3x3 -> 9x9` expansion-style task.

ONNX approach:

- Fixed deconvolution / ConvTranspose.
- Static reshape-like graph if allowed by validator.
- Learned fixed kernels for known scale factors.

Risk:

- Low to medium when scale factor is fixed.
- Higher when expansion depends on object content or selected pattern.

## Group 7: Cropping / Extraction

Current mapped tasks: `99`.

Characteristics:

- Output is smaller than input.
- The model extracts an object, bounding box, subgrid, or selected region.

Input:

- Larger grid, often containing one or more objects.
- May include noise, distractors, frames, or repeated patterns.

Output:

- Smaller grid.
- Usually the selected object or compressed representation.

Useful checks:

- Output equals bounding box of a color/object.
- Output size varies by example or is fixed per task.
- Selection rule may depend on color, size, position, or uniqueness.

ONNX approach:

- Fixed crop is easy.
- Dynamic crop is hard under static-shape ONNX constraints.
- Some tasks may be solved by writing output into top-left of the fixed `30x30` tensor and zero-hotting the rest.

Risk:

- High. Object localization and variable output boundaries are difficult in compact ONNX.

## Group 8: Geometric Transformations

Current mapped tasks: `7`.

Characteristics:

- Output is rotated, mirrored, shifted, aligned, or translated.
- May be same-size or size-changing.

Input:

- One or more geometric objects.
- Often sparse colored shapes.

Output:

- Transformed version of the input object(s).

Useful checks:

- Compare input and output under rotations, flips, translations, or transpose.
- Check whether color is preserved.

ONNX approach:

- Fixed transforms can be implemented as static channel/spatial permutation.
- Local shift can be implemented by convolution kernels.

Risk:

- Low for fixed transform.
- Medium when the transform depends on object orientation or position.

## Group 9: Pattern Continuation / Completion

Current mapped tasks: `99`.

Characteristics:

- Missing lines, shapes, symmetries, or repetitions are completed.
- Output often has the same shape.

Input:

- Partial pattern.
- May include gaps, anchors, or examples of the intended repetition.

Output:

- Completed pattern.
- Existing cells are usually preserved, new cells are added.

Useful checks:

- Identify repeated rows, columns, diagonals, or mirrored objects.
- Look for missing symmetric counterpart.

ONNX approach:

- Simple fixed-period repeats can use convolution or static copy.
- Long-range completion may need multiple layers or task-specific construction.

Risk:

- Medium to high. Often requires global structure inference.

## Group 10: Counting / Relational Tasks

Current mapped tasks: `0`.

Characteristics:

- Output depends on object counts, sizes, frequencies, ordering, or comparisons.
- Often not solvable by simple local rules.

Input:

- Multiple objects or colors.
- May include distractors.

Output:

- Selected color/object, count-coded shape, or transformed object based on relation.

Useful checks:

- Does the answer depend on largest/smallest object?
- Does color frequency determine output?
- Does position relative to another object matter?

ONNX approach:

- Hard under compact static ONNX.
- May need ad hoc solvers for specific tasks.

Risk:

- High. Save these for after easier families are exhausted.

## Group 11: Composite Tasks

Current mapped tasks: `4`.

Characteristics:

- More than one operation is required.
- Examples: crop then recolor, fill then extract, select object then tile, rotate then overlay.

Input:

- Varies widely.

Output:

- Varies widely.

Useful checks:

- Break the transformation into ordered substeps.
- Check whether one of the simpler families explains part of the output.

ONNX approach:

- Compose multiple small graph blocks if cost remains acceptable.
- Use task-specific generator logic.

Risk:

- High. These are later-stage tasks for moving from the 50% goal toward 70%.

## Recommended Priority

For the immediate 50% score target:

```text
1. Identity / no-op
2. Global color remapping
3. Same-shape local rules
4. Simple expansion / tiling
5. Simple geometric transforms
6. Fill / enclosed regions
7. Mask / object selection
```

For the 70% target:

```text
8. Cropping / extraction
9. Pattern completion
10. Counting / relational tasks
11. Composite tasks
```

The current generated map may assign zero tasks to some conceptual families, such as `identity_noop` or `counting_relational`. That does not mean those families are impossible or irrelevant; it means the current heuristic did not identify them as the primary routing label for any of the 400 task files.

## Submission Strategy by Group

With an assumed limit of about 30 submissions per day:

- Submit coherent solver-family batches, not random task changes.
- Keep a registry of which task is assigned to which family.
- For each submission, record which families changed and which task models changed.
- Use local validation on `train + test + arc-gen` before submitting.
- Treat leaderboard deltas as evidence about private generalization, not just visible correctness.

## Suggested Registry Columns

Create a task registry with columns like:

```text
task_id
input_shapes
output_shapes
shape_relation
colors_in
colors_out
candidate_family
visible_pass
onnx_generated
estimated_cost
submitted
public_lb_delta
notes
```

This makes the path to 50% and then 70% measurable instead of relying on manual inspection.

For notebook use, prefer the JSON-list color columns in `task_type_map.csv`:

```text
input_color_list
output_color_list
new_output_color_list
```

Columns like `input_colors` are compact human-readable strings and may be parsed by pandas as numbers if they start with `0`.
