"""Validate task group artifacts against task files and each other."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "competition_material" / "taskfiles"
GROUP_DIR = ROOT / "task_groups"
MAP_CSV = GROUP_DIR / "task_type_map.csv"
MAP_JSON = GROUP_DIR / "task_type_map.json"
GROUPS_JSON = GROUP_DIR / "task_type_groups.json"
REPORT_MD = GROUP_DIR / "task_group_validation_report.md"


EXPECTED_FAMILIES = [
    "identity_noop",
    "global_color_remap",
    "same_shape_local_rule",
    "mask_object_selection",
    "fill_enclosed_regions",
    "expansion_tiling",
    "cropping_extraction",
    "geometric_transform",
    "pattern_completion",
    "counting_relational",
    "composite_or_unknown",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shape(grid):
    return len(grid), len(grid[0]) if grid else 0


def raw_task_stats(task_id):
    data = load_json(TASK_DIR / f"{task_id}.json")
    examples = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
    same = expand = shrink = mixed = 0
    identity = True
    input_colors = set()
    output_colors = set()
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        ish, osh = shape(inp), shape(out)
        if ish == osh:
            same += 1
        elif osh[0] >= ish[0] and osh[1] >= ish[1]:
            expand += 1
        elif osh[0] <= ish[0] and osh[1] <= ish[1]:
            shrink += 1
        else:
            mixed += 1
        if ish != osh or inp != out:
            identity = False
        input_colors.update(cell for row in inp for cell in row)
        output_colors.update(cell for row in out for cell in row)
    return {
        "n_examples": len(examples),
        "same": same,
        "expand": expand,
        "shrink": shrink,
        "mixed": mixed,
        "identity": identity,
        "input_colors": sorted(input_colors),
        "output_colors": sorted(output_colors),
        "new_output_colors": sorted(output_colors - input_colors),
    }


def parse_csv_rows():
    with MAP_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_value(value):
    if value in {"True", "False"}:
        return value == "True"
    return value


def main():
    csv_rows = parse_csv_rows()
    json_rows = load_json(MAP_JSON)
    groups = load_json(GROUPS_JSON)

    issues = []
    warnings = []

    if len(csv_rows) != 400:
        issues.append(f"CSV row count is {len(csv_rows)}, expected 400.")
    if len(json_rows) != 400:
        issues.append(f"JSON row count is {len(json_rows)}, expected 400.")

    csv_ids = [row["task_id"] for row in csv_rows]
    json_ids = [row["task_id"] for row in json_rows]
    expected_ids = [f"task{i:03d}" for i in range(1, 401)]

    if csv_ids != expected_ids:
        issues.append("CSV task ids are not exactly task001..task400 in order.")
    if json_ids != expected_ids:
        issues.append("JSON task ids are not exactly task001..task400 in order.")
    if len(set(csv_ids)) != len(csv_ids):
        issues.append("CSV has duplicate task ids.")
    if len(set(json_ids)) != len(json_ids):
        issues.append("JSON has duplicate task ids.")

    json_by_id = {row["task_id"]: row for row in json_rows}
    for csv_row in csv_rows:
        json_row = json_by_id.get(csv_row["task_id"])
        if not json_row:
            issues.append(f"{csv_row['task_id']} missing from JSON map.")
            continue
        for key in ["task_id", "primary_family", "confidence", "shape_relation", "candidate_flags"]:
            if str(csv_row[key]) != str(json_row[key]):
                issues.append(f"{csv_row['task_id']} mismatch for {key}: CSV={csv_row[key]!r}, JSON={json_row[key]!r}.")

    if list(groups.keys()) != EXPECTED_FAMILIES:
        warnings.append("task_type_groups.json family order or set differs from expected family list.")

    grouped_from_rows = {family: [] for family in EXPECTED_FAMILIES}
    for row in json_rows:
        grouped_from_rows.setdefault(row["primary_family"], []).append(row["task_id"])
    for family in EXPECTED_FAMILIES:
        if groups.get(family, []) != grouped_from_rows.get(family, []):
            issues.append(f"Grouped JSON mismatch for family {family}.")

    family_counts = Counter(row["primary_family"] for row in json_rows)
    confidence_counts = Counter((row["primary_family"], row["confidence"]) for row in json_rows)

    family_property_checks = Counter()
    family_property_failures = Counter()
    family_examples = {family: [] for family in EXPECTED_FAMILIES}

    for row in json_rows:
        task_id = row["task_id"]
        family = row["primary_family"]
        stats = raw_task_stats(task_id)
        family_examples.setdefault(family, []).append(task_id)

        if family == "identity_noop":
            family_property_checks[family] += 1
            if not stats["identity"]:
                family_property_failures[family] += 1
        elif family == "global_color_remap":
            family_property_checks[family] += 1
            if not row["global_mapping_consistent"]:
                family_property_failures[family] += 1
        elif family in {"same_shape_local_rule", "mask_object_selection", "fill_enclosed_regions", "pattern_completion"}:
            family_property_checks[family] += 1
            if stats["same"] != stats["n_examples"]:
                family_property_failures[family] += 1
        elif family == "expansion_tiling":
            family_property_checks[family] += 1
            if stats["expand"] != stats["n_examples"]:
                family_property_failures[family] += 1
        elif family == "cropping_extraction":
            family_property_checks[family] += 1
            if stats["shrink"] != stats["n_examples"]:
                family_property_failures[family] += 1
        elif family == "geometric_transform":
            family_property_checks[family] += 1
            if not row["fixed_geometric_transforms"]:
                family_property_failures[family] += 1

    lines = []
    lines.append("# Task Group Validation Report")
    lines.append("")
    lines.append("This report validates `task_groups` artifacts against the raw task JSON files.")
    lines.append("")
    lines.append("## Artifact Consistency")
    lines.append("")
    lines.append(f"- CSV rows: {len(csv_rows)}")
    lines.append(f"- JSON rows: {len(json_rows)}")
    lines.append(f"- Grouped families: {len(groups)}")
    lines.append(f"- Issues: {len(issues)}")
    lines.append(f"- Warnings: {len(warnings)}")
    lines.append("")

    if issues:
        lines.append("### Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    if warnings:
        lines.append("### Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Family Counts")
    lines.append("")
    lines.append("| Family | Count | High | Medium | Low | Property check failures |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for family in EXPECTED_FAMILIES:
        high = confidence_counts[(family, "high")]
        medium = confidence_counts[(family, "medium")]
        low = confidence_counts[(family, "low")]
        failures = family_property_failures[family]
        lines.append(f"| `{family}` | {family_counts[family]} | {high} | {medium} | {low} | {failures} |")
    lines.append("")

    lines.append("## Validation Interpretation")
    lines.append("")
    lines.append("- The families are heuristic routing labels for solver development, not proven ARC semantic labels.")
    lines.append("- Shape-driven families are strongly validated by raw grid dimensions.")
    lines.append("- `global_color_remap` is strongly validated by consistent cellwise color mappings.")
    lines.append("- Same-shape semantic families are weaker: they separate tasks by observable signals such as local consistency, input preservation, and added colors.")
    lines.append("- `fill_enclosed_regions` should be read as an additive fill/marking family; some tasks may fill enclosed regions, while others preserve input cells and add new marked cells.")
    lines.append("- `mask_object_selection`, `pattern_completion`, and `composite_or_unknown` should be manually reviewed before relying on a specialized solver.")
    lines.append("")

    lines.append("## Sample Tasks By Family")
    lines.append("")
    for family in EXPECTED_FAMILIES:
        sample = family_examples.get(family, [])[:12]
        lines.append(f"- `{family}`: {', '.join(sample) if sample else '(none)'}")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {REPORT_MD}")
    print(f"issues={len(issues)} warnings={len(warnings)}")
    for family in EXPECTED_FAMILIES:
        print(f"{family}: count={family_counts[family]} failures={family_property_failures[family]}")

    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
