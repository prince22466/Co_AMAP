"""Create solver-family starter notebooks for NeuroGolf."""

from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent


FAMILY_SPECS = [
    ("01_identity_noop.ipynb", "identity_noop", "Identity / No-Op"),
    ("02_global_color_remap.ipynb", "global_color_remap", "Global Color Remapping"),
    ("03_same_shape_local_rule.ipynb", "same_shape_local_rule", "Same-Shape Local Rules"),
    ("04_mask_object_selection.ipynb", "mask_object_selection", "Mask / Object Selection"),
    ("05_fill_additive_marking.ipynb", "fill_enclosed_regions", "Fill / Additive Marking"),
    ("06_expansion_tiling.ipynb", "expansion_tiling", "Expansion / Tiling"),
    ("07_cropping_extraction.ipynb", "cropping_extraction", "Cropping / Extraction"),
    ("08_geometric_transform.ipynb", "geometric_transform", "Geometric Transformations"),
    ("09_pattern_completion.ipynb", "pattern_completion", "Pattern Completion"),
    ("10_counting_relational.ipynb", "counting_relational", "Counting / Relational"),
    ("11_composite_unknown.ipynb", "composite_or_unknown", "Composite / Unknown"),
]


TRAINERS = {
    "identity_noop": """
def train_family_task(task):
    # Valid only for tasks where every visible input already equals output.
    examples = all_examples(task)
    if not examples or any(ex['input'] != ex['output'] for ex in examples):
        return None, {'ok': False, 'reason': 'not visible identity'}
    return make_identity_model(), {'ok': True, 'trainer': 'identity'}
""",
    "global_color_remap": """
def train_family_task(task):
    # Infers a global cellwise color mapping over train + test + arc-gen.
    return train_color_remap_model(task)
""",
    "geometric_transform": """
def train_family_task(task):
    # This notebook detects fixed transforms. ONNX export for spatial
    # permutation is intentionally left as the next implementation step.
    # The identity fallback lets you build a structurally valid submission
    # while developing the transform graph.
    transforms = infer_fixed_geometric_transform(all_examples(task))
    if not transforms:
        return None, {'ok': False, 'reason': 'no fixed transform detected'}
    return None, {'ok': False, 'reason': 'detected transform; ONNX exporter pending', 'transforms': transforms}
""",
    "same_shape_local_rule": """
def train_family_task(task):
    # Starter trainer: try global color remap first, then leave task unsolved.
    # Next step for this family is fitting local 3x3/5x5 convolution rules.
    model, info = train_color_remap_model(task)
    if model is not None:
        info['trainer'] = 'color_remap_fallback'
        return model, info
    return None, {'ok': False, 'reason': 'local-rule ONNX trainer pending'}
""",
    "fill_enclosed_regions": """
def train_family_task(task):
    # Starter trainer for additive fill/marking tasks.
    # The visible map identifies these tasks by preserved foreground plus
    # added cells. True flood-fill ONNX generation is a later step.
    return None, {'ok': False, 'reason': 'fill/additive marking ONNX trainer pending'}
""",
    "expansion_tiling": """
def train_family_task(task):
    # Starter trainer for larger outputs. Next implementations should cover
    # fixed scale factors such as 3x3->6x6 and 3x3->9x9.
    return None, {'ok': False, 'reason': 'expansion/tiling ONNX trainer pending'}
""",
    "cropping_extraction": """
def train_family_task(task):
    # Starter trainer for smaller outputs. Dynamic extraction is hard under
    # static-shape ONNX, so begin with fixed crop or fixed bounding-box cases.
    return None, {'ok': False, 'reason': 'crop/extraction ONNX trainer pending'}
""",
    "mask_object_selection": """
def train_family_task(task):
    # Starter trainer for same-shape object/mask tasks.
    # First useful sub-solvers: color masks, remove-background, keep-one-color.
    return None, {'ok': False, 'reason': 'mask/object selection ONNX trainer pending'}
""",
    "pattern_completion": """
def train_family_task(task):
    # Starter trainer for same-shape completion tasks.
    # First useful sub-solvers: line extension, mirror completion, periodic copy.
    return None, {'ok': False, 'reason': 'pattern completion ONNX trainer pending'}
""",
    "counting_relational": """
def train_family_task(task):
    # No tasks are currently auto-routed here. Keep this notebook for manually
    # moved tasks that need count, size, frequency, or relational reasoning.
    return None, {'ok': False, 'reason': 'counting/relational trainer pending'}
""",
    "composite_or_unknown": """
def train_family_task(task):
    # Manual workbench for tasks that do not fit a simple primary family.
    return None, {'ok': False, 'reason': 'manual composite solver pending'}
""",
}


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def notebook_for(family, title):
    version = {
        "fill_enclosed_regions": "fill-additive-v0.1",
    }.get(family, f"{family.replace('_', '-')}-v0.1")
    cells = [
            md_cell(
                f"""
# NeuroGolf Solver Family: {title}

This notebook is a starter pipeline for the `{family}` task family.

Workflow:

1. Load task ids from `task_groups/task_type_groups.json`.
2. Inspect the family metadata from `task_type_map.csv`.
3. Train or infer a per-task ONNX model with `train_family_task`.
4. Save `taskNNN.onnx` files.
5. Build `submission.zip` from the generated models.

The generated maps are heuristic solver-routing labels. Validate against visible examples before submitting.
"""
            ),
            code_cell(
                f"""
from pathlib import Path
import sys
import json
import pandas as pd

ROOT = Path.cwd()
if str(ROOT / 'submission_nbs') not in sys.path:
    sys.path.append(str(ROOT / 'submission_nbs'))

from neurogolf_nb_common import *

FAMILY = '{family}'
MODEL_VERSION = '{version}'
DATA_DIR, BASE_OUT_DIR = default_paths()
OUT_DIR = BASE_OUT_DIR / FAMILY
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('DATA_DIR =', DATA_DIR)
print('OUT_DIR =', OUT_DIR)
print('MODEL_VERSION =', MODEL_VERSION)
"""
            ),
            code_cell(
                """
task_map = load_task_type_map()
task_ids = family_task_ids(FAMILY)
family_df = task_map[task_map.primary_family == FAMILY].copy()

print('family:', FAMILY)
print('tasks:', len(task_ids))
display(family_df.head(20))
"""
            ),
            code_cell(
                """
# Inspect one task quickly.
if task_ids:
    sample_task_id = task_ids[0]
    sample_task = load_task(DATA_DIR, sample_task_id)
    print(sample_task_id, 'examples:', len(all_examples(sample_task)))
    print('first input shape:', grid_shape(sample_task['train'][0]['input']))
    print('first output shape:', grid_shape(sample_task['train'][0]['output']))
    print('first input:', sample_task['train'][0]['input'])
    print('first output:', sample_task['train'][0]['output'])
else:
    print('No tasks currently mapped to this family.')
"""
            ),
            code_cell(TRAINERS[family]),
            code_cell(
                """
# Dry-run training on the first few tasks without saving.
dry_rows = []
for task_id in task_ids[:10]:
    task = load_task(DATA_DIR, task_id)
    model, info = train_family_task(task)
    dry_rows.append({'task_id': task_id, 'has_model': model is not None, **info})

pd.DataFrame(dry_rows)
"""
            ),
            code_cell(
                """
# Build family models.
# Set fallback_identity=True only when you explicitly want placeholder models
# for tasks whose trainer is not implemented yet.
rows, zip_path = build_family_submission(
    FAMILY,
    train_family_task,
    DATA_DIR,
    OUT_DIR,
    fallback_identity=False,
    validate=False,
)

result_df = pd.DataFrame(rows)
display(result_df.head(30))
print('models saved:', int(result_df.get('saved', pd.Series(dtype=bool)).sum()) if len(result_df) else 0)
print('zip:', zip_path)
"""
            ),
            code_cell(
                """
# Optional: validate saved ONNX models on visible examples.
# This can be slow for large families and requires onnxruntime.
validate_rows = []
for row in rows:
    if not row.get('saved'):
        continue
    task = load_task(DATA_DIR, row['task_id'])
    summary = visible_validation_summary(row['path'], task)
    validate_rows.append({
        'task_id': row['task_id'],
        'right': summary['right'],
        'wrong': summary['wrong'],
    })

pd.DataFrame(validate_rows)
"""
            ),
            code_cell(
                """
# Submission helper.
# For a full competition submission, combine models from multiple family
# folders into one directory, then call create_submission_zip(combined_dir).
submission_zip = create_submission_zip(OUT_DIR)
print(submission_zip)
"""
            ),
        ]

    if family == "fill_enclosed_regions":
        cells.insert(
            2,
            md_cell(
                """
## Fill / Additive Marking Version Contract

This notebook should track each solver version explicitly:

- `MODEL_VERSION`: human-readable solver version.
- selected tasks: rows from `task_type_map.csv` where `primary_family == "fill_enclosed_regions"`.
- architecture: ONNX nodes, ops, initializer shapes, file size, parameter count.
- performance: exact-match accuracy on `train`, `test`, `arc-gen`, and all visible examples.
- memory profile: parameter count, static tensor memory, runtime profile memory when `onnxruntime` is available.

The competition score for a correct task is driven by `params + memory_bytes`, so keep both visible.
"""
            ),
        )
        cells.insert(
            5,
            code_cell(
                """
# Family selection table: these are the tasks this notebook is responsible for.
selection_cols = [
    'task_id',
    'confidence',
    'n_train',
    'n_test',
    'n_arc_gen',
    'shape_relation',
    'input_shape_modes',
    'output_shape_modes',
    'input_color_list',
    'output_color_list',
    'new_output_color_list',
    'candidate_flags',
]
fill_selection = family_df[selection_cols].reset_index(drop=True)
print('selected fill/additive tasks:', len(fill_selection))
display(fill_selection)
"""
            ),
        )
        cells.insert(
            9,
            code_cell(
                """
# Model/version manifest for this notebook run.
run_manifest = {
    'family': FAMILY,
    'model_version': MODEL_VERSION,
    'task_count': len(task_ids),
    'out_dir': str(OUT_DIR),
}
run_manifest
"""
            ),
        )
        cells.insert(
            12,
            code_cell(
                """
# Architecture, performance, and memory report for saved models.
# This cell expects train_family_task to save one or more ONNX models.
# It reports the metrics the competition cares about: file size, parameter
# count, and memory profile, plus train/test/arc-gen exact-match performance.

report_rows = []
for row in rows:
    if not row.get('saved'):
        continue
    task = load_task(DATA_DIR, row['task_id'])
    try:
        report = model_report(row['path'], task=task)
        arch = report['architecture']
        mem = report['memory_profile']
        perf = report['performance']
        report_rows.append({
            'task_id': row['task_id'],
            'model_version': MODEL_VERSION,
            'file_size_bytes': arch.get('file_size_bytes'),
            'params': arch.get('params'),
            'nodes': arch.get('nodes'),
            'op_counts': json.dumps(arch.get('op_counts', {}), sort_keys=True),
            'static_memory_bytes': mem.get('static_memory_bytes'),
            'runtime_memory_bytes': mem.get('runtime_memory_bytes'),
            'train_right': perf['train']['right'],
            'train_total': perf['train']['total'],
            'train_accuracy': perf['train']['accuracy'],
            'test_right': perf['test']['right'],
            'test_total': perf['test']['total'],
            'test_accuracy': perf['test']['accuracy'],
            'arc_gen_right': perf['arc_gen']['right'],
            'arc_gen_total': perf['arc_gen']['total'],
            'arc_gen_accuracy': perf['arc_gen']['accuracy'],
            'visible_right': perf['visible_all']['right'],
            'visible_total': perf['visible_all']['total'],
            'visible_accuracy': perf['visible_all']['accuracy'],
        })
    except Exception as exc:
        report_rows.append({
            'task_id': row['task_id'],
            'model_version': MODEL_VERSION,
            'profile_error': repr(exc),
        })

profile_df = pd.DataFrame(report_rows)
display(profile_df)
"""
            ),
        )
        cells.insert(
            13,
            code_cell(
                """
# Persist run metadata next to the generated models.
if 'profile_df' in globals() and len(profile_df):
    profile_path = OUT_DIR / f'{FAMILY}_{MODEL_VERSION}_profile.csv'
    profile_df.to_csv(profile_path, index=False)
    print('wrote profile:', profile_path)

manifest_path = OUT_DIR / f'{FAMILY}_{MODEL_VERSION}_manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(run_manifest, f, indent=2)
print('wrote manifest:', manifest_path)
"""
            ),
        )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    for filename, family, title in FAMILY_SPECS:
        path = OUT_DIR / filename
        path.write_text(json.dumps(notebook_for(family, title), indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
