"""Generate heuristic task-family maps for NeuroGolf task JSON files.

The output is intentionally notebook-friendly:
  - task_type_map.csv: one row per task
  - task_type_map.json: same records as JSON objects

The classifier is conservative. It is meant to route tasks to candidate solver
families, not to prove the true ARC rule.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "competition_material" / "taskfiles"
OUT_DIR = ROOT / "task_groups"

FAMILIES = {
    "identity_noop": "Input equals output for all visible examples.",
    "global_color_remap": "Same shape; output color is a consistent function of input color.",
    "same_shape_local_rule": "Same shape; output appears explainable by local neighborhoods.",
    "mask_object_selection": "Same shape; cells/objects are kept, removed, or highlighted.",
    "fill_enclosed_regions": "Same shape; input is mostly preserved while new fill or marking cells are added.",
    "expansion_tiling": "Output is larger than input, often by fixed scale or tiling.",
    "cropping_extraction": "Output is smaller than input, often selected object/subgrid.",
    "geometric_transform": "Output matches a fixed rotate/flip/transpose/shift transform.",
    "pattern_completion": "Output preserves input and adds missing pattern cells.",
    "counting_relational": "Likely depends on counts, object sizes, or comparisons.",
    "composite_or_unknown": "No simple family detected or multiple families likely apply.",
}


def shape(grid):
    return len(grid), len(grid[0]) if grid else 0


def colors(grid):
    return set(cell for row in grid for cell in row)


def same_grid(a, b):
    return shape(a) == shape(b) and all(row_a == row_b for row_a, row_b in zip(a, b))


def flatten_pair(inp, out):
    for r, row in enumerate(inp):
        for c, value in enumerate(row):
            yield value, out[r][c]


def infer_global_mapping(examples):
    mapping = {}
    conflicts = 0
    seen = 0
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if shape(inp) != shape(out):
            return None, 1
        for ic, oc in flatten_pair(inp, out):
            seen += 1
            prev = mapping.get(ic)
            if prev is None:
                mapping[ic] = oc
            elif prev != oc:
                conflicts += 1
    if not seen:
        return None, 1
    return mapping, conflicts


def transform_equal(inp, out, transform):
    ih, iw = shape(inp)
    if transform == "rot90":
        pred = [[inp[ih - 1 - r][c] for r in range(ih)] for c in range(iw)]
    elif transform == "rot180":
        pred = [list(reversed(row)) for row in reversed(inp)]
    elif transform == "rot270":
        pred = [[inp[r][iw - 1 - c] for r in range(ih)] for c in range(iw)]
    elif transform == "flip_h":
        pred = [list(reversed(row)) for row in inp]
    elif transform == "flip_v":
        pred = list(reversed(inp))
    elif transform == "transpose":
        pred = [[inp[r][c] for r in range(ih)] for c in range(iw)]
    else:
        return False
    return same_grid(pred, out)


def fixed_geometric_transform(examples):
    transforms = ["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"]
    matches = []
    for name in transforms:
        if all(transform_equal(ex["input"], ex["output"], name) for ex in examples):
            matches.append(name)
    return matches


def local_rule_consistency(examples, radius=1, max_examples=60):
    """Return fraction of local patches that map consistently to output center."""
    patch_to_output = {}
    conflicts = 0
    total = 0
    used = 0
    for ex in examples[:max_examples]:
        inp, out = ex["input"], ex["output"]
        if shape(inp) != shape(out):
            continue
        used += 1
        h, w = shape(inp)
        padded = defaultdict(int)
        for r in range(h):
            for c in range(w):
                padded[(r, c)] = inp[r][c]
        for r in range(h):
            for c in range(w):
                patch = tuple(
                    padded[(rr, cc)]
                    for rr in range(r - radius, r + radius + 1)
                    for cc in range(c - radius, c + radius + 1)
                )
                oc = out[r][c]
                total += 1
                prev = patch_to_output.get(patch)
                if prev is None:
                    patch_to_output[patch] = oc
                elif prev != oc:
                    conflicts += 1
    if total == 0 or used == 0:
        return 0.0, conflicts, total
    return 1.0 - conflicts / total, conflicts, total


def input_preserved_ratio(examples):
    same_positions = 0
    input_nonzero = 0
    added_nonzero = 0
    changed = 0
    comparable = 0
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if shape(inp) != shape(out):
            continue
        comparable += 1
        h, w = shape(inp)
        for r in range(h):
            for c in range(w):
                ic, oc = inp[r][c], out[r][c]
                if ic != 0:
                    input_nonzero += 1
                    if ic == oc:
                        same_positions += 1
                if ic == 0 and oc != 0:
                    added_nonzero += 1
                if ic != oc:
                    changed += 1
    preserved = same_positions / input_nonzero if input_nonzero else 0.0
    return preserved, added_nonzero, changed, comparable


def shape_relation(input_shapes, output_shapes):
    if input_shapes == output_shapes and len(input_shapes) == 1:
        return "fixed_same_shape"
    all_same = all(i == o for i, o in zip(input_shapes, output_shapes))
    all_expand = all(o[0] >= i[0] and o[1] >= i[1] and o != i for i, o in zip(input_shapes, output_shapes))
    all_shrink = all(o[0] <= i[0] and o[1] <= i[1] and o != i for i, o in zip(input_shapes, output_shapes))
    if all_same:
        return "same_shape_variable_size"
    if all_expand:
        return "expands"
    if all_shrink:
        return "shrinks"
    return "mixed_shape_change"


def classify_task(task_id, data):
    examples = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
    input_shapes = [shape(ex["input"]) for ex in examples]
    output_shapes = [shape(ex["output"]) for ex in examples]
    input_shape_counts = Counter(input_shapes)
    output_shape_counts = Counter(output_shapes)
    relation = shape_relation(input_shapes, output_shapes)
    same_shape_count = sum(1 for i, o in zip(input_shapes, output_shapes) if i == o)
    expand_count = sum(1 for i, o in zip(input_shapes, output_shapes) if o[0] >= i[0] and o[1] >= i[1] and o != i)
    shrink_count = sum(1 for i, o in zip(input_shapes, output_shapes) if o[0] <= i[0] and o[1] <= i[1] and o != i)
    mixed_count = len(examples) - same_shape_count - expand_count - shrink_count

    all_identity = all(same_grid(ex["input"], ex["output"]) for ex in examples)
    mapping, mapping_conflicts = infer_global_mapping(examples)
    is_mapping = mapping is not None and mapping_conflicts == 0
    non_identity_mapping = bool(is_mapping and any(k != v for k, v in mapping.items()))
    geom = fixed_geometric_transform(examples[:80])
    local_score, local_conflicts, local_total = local_rule_consistency(examples)
    preserved_ratio, added_nonzero, changed_cells, comparable = input_preserved_ratio(examples)

    input_color_set = sorted(set().union(*(colors(ex["input"]) for ex in examples))) if examples else []
    output_color_set = sorted(set().union(*(colors(ex["output"]) for ex in examples))) if examples else []
    new_output_colors = sorted(set(output_color_set) - set(input_color_set))

    flags = []
    confidence = "low"
    primary = "composite_or_unknown"

    if all_identity:
        primary = "identity_noop"
        confidence = "high"
        flags.append("identity")
    elif is_mapping and non_identity_mapping:
        primary = "global_color_remap"
        confidence = "high"
        flags.append("global_color_remap")
    elif geom:
        primary = "geometric_transform"
        confidence = "high"
        flags.append("fixed_geometric_" + "|".join(geom))
    elif relation == "expands":
        primary = "expansion_tiling"
        confidence = "medium"
        flags.append("shape_expands")
    elif relation == "shrinks":
        primary = "cropping_extraction"
        confidence = "medium"
        flags.append("shape_shrinks")
    elif same_shape_count == len(examples):
        if added_nonzero > 0 and preserved_ratio >= 0.90 and new_output_colors:
            primary = "fill_enclosed_regions"
            confidence = "medium"
            flags.append("adds_new_color_preserves_input")
        elif local_score >= 0.995 and changed_cells > 0:
            primary = "same_shape_local_rule"
            confidence = "medium"
            flags.append("local_3x3_consistent")
        elif changed_cells > 0 and preserved_ratio >= 0.80:
            primary = "pattern_completion"
            confidence = "low"
            flags.append("mostly_preserves_input_adds_or_changes")
        elif changed_cells > 0:
            primary = "mask_object_selection"
            confidence = "low"
            flags.append("same_shape_nonlocal_changes")
    else:
        flags.append("mixed_shape_relation")

    if is_mapping and "mapping_consistent" not in flags:
        flags.append("mapping_consistent")
    if local_score >= 0.995 and "local_3x3_consistent" not in flags:
        flags.append("local_3x3_consistent")
    if new_output_colors and "new_output_colors" not in flags:
        flags.append("new_output_colors")

    mapping_str = ""
    if mapping:
        mapping_str = json.dumps({str(k): v for k, v in sorted(mapping.items())}, separators=(",", ":"))

    return {
        "task_id": f"task{task_id:03d}",
        "task_num": task_id,
        "primary_family": primary,
        "confidence": confidence,
        "candidate_flags": "|".join(flags),
        "n_train": len(data.get("train", [])),
        "n_test": len(data.get("test", [])),
        "n_arc_gen": len(data.get("arc-gen", [])),
        "n_examples": len(examples),
        "shape_relation": relation,
        "same_shape_examples": same_shape_count,
        "expand_examples": expand_count,
        "shrink_examples": shrink_count,
        "mixed_examples": mixed_count,
        "input_shape_modes": ";".join(f"{h}x{w}:{n}" for (h, w), n in input_shape_counts.most_common(5)),
        "output_shape_modes": ";".join(f"{h}x{w}:{n}" for (h, w), n in output_shape_counts.most_common(5)),
        "input_colors": "".join(str(c) for c in input_color_set),
        "output_colors": "".join(str(c) for c in output_color_set),
        "new_output_colors": "".join(str(c) for c in new_output_colors),
        "input_color_list": json.dumps(input_color_set, separators=(",", ":")),
        "output_color_list": json.dumps(output_color_set, separators=(",", ":")),
        "new_output_color_list": json.dumps(new_output_colors, separators=(",", ":")),
        "all_identity": all_identity,
        "global_mapping_consistent": is_mapping,
        "global_mapping": mapping_str,
        "mapping_conflicts": mapping_conflicts,
        "fixed_geometric_transforms": "|".join(geom),
        "local_3x3_score": round(local_score, 6),
        "local_3x3_conflicts": local_conflicts,
        "local_3x3_samples": local_total,
        "input_nonzero_preserved_ratio": round(preserved_ratio, 6),
        "added_nonzero_cells": added_nonzero,
        "changed_cells": changed_cells,
        "notes": FAMILIES[primary],
    }


def main():
    records = []
    for task_num in range(1, 401):
        path = TASK_DIR / f"task{task_num:03d}.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        records.append(classify_task(task_num, data))

    csv_path = OUT_DIR / "task_type_map.csv"
    json_path = OUT_DIR / "task_type_map.json"
    groups_path = OUT_DIR / "task_type_groups.json"

    fieldnames = list(records[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")

    grouped = defaultdict(list)
    for row in records:
        grouped[row["primary_family"]].append(row["task_id"])
    grouped = {family: grouped.get(family, []) for family in FAMILIES}
    with groups_path.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2)
        f.write("\n")

    counts = Counter(row["primary_family"] for row in records)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {groups_path}")
    for family, count in counts.most_common():
        print(f"{family}: {count}")


if __name__ == "__main__":
    main()
