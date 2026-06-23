"""Shared helpers for NeuroGolf submission notebooks.

The notebooks in this folder are solver-family workbooks. They should be
copied into Kaggle or run locally with the competition files available.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
except Exception:  # Notebook analysis cells can still run without ONNX.
    onnx = None
    ort = None
    TensorProto = None
    helper = None


BATCH, CH, H, W = 1, 10, 30, 30
MODEL_VERSION = "starter-v0.1"


def default_paths():
    kaggle_dir = Path("/kaggle/input/competitions/neurogolf-2026")
    if kaggle_dir.exists():
        data_dir = kaggle_dir
        root = Path("/kaggle/working")
    else:
        root = Path.cwd()
        data_dir = root / "competition_material" / "taskfiles"
        if not data_dir.exists():
            data_dir = root / "competition_material"
    out_dir = root / "working_submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, out_dir


def load_task_type_map(path="task_groups/task_type_map.csv"):
    import pandas as pd

    return pd.read_csv(path, dtype={"task_id": str})


def load_task_groups(path="task_groups/task_type_groups.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def family_task_ids(family, groups_path="task_groups/task_type_groups.json"):
    groups = load_task_groups(groups_path)
    return groups.get(family, [])


def task_num(task_id):
    return int(str(task_id).replace("task", ""))


def task_path(data_dir, task_id):
    data_dir = Path(data_dir)
    name = f"{task_id}.json" if str(task_id).startswith("task") else f"task{int(task_id):03d}.json"
    direct = data_dir / name
    if direct.exists():
        return direct
    nested = data_dir / "taskfiles" / name
    if nested.exists():
        return nested
    raise FileNotFoundError(name)


def load_task(data_dir, task_id):
    with task_path(data_dir, task_id).open("r", encoding="utf-8") as f:
        return json.load(f)


def all_examples(task):
    return task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])


def grid_shape(grid):
    return len(grid), len(grid[0]) if grid else 0


def grid_to_tensor(grid):
    arr = np.zeros((BATCH, CH, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= r < H and 0 <= c < W:
                arr[0, int(color), r, c] = 1.0
    return arr


def tensor_to_grid(arr):
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    grid = []
    for r in range(H):
        row = []
        for c in range(W):
            vals = np.where(arr[:, r, c] > 0.5)[0]
            row.append(int(vals[0]) if len(vals) == 1 else 0)
        while row and row[-1] == 0:
            row.pop()
        grid.append(row)
    while grid and not grid[-1]:
        grid.pop()
    return grid


def require_onnx():
    if onnx is None or helper is None or TensorProto is None:
        raise ImportError("onnx is required to build models")


def make_model(nodes, initializers, opset=10):
    require_onnx()
    dt = TensorProto.FLOAT
    inp = helper.make_tensor_value_info("input", dt, [BATCH, CH, H, W])
    out = helper.make_tensor_value_info("output", dt, [BATCH, CH, H, W])
    graph = helper.make_graph(nodes, "graph", [inp], [out], initializers)
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", opset)])


def make_identity_model():
    return make_1x1_color_model({c: c for c in range(CH)})


def make_1x1_color_model(mapping):
    require_onnx()
    dt = TensorProto.FLOAT
    weights = np.zeros((CH, CH, 1, 1), dtype=np.float32)
    bias = np.full((CH,), -0.5, dtype=np.float32)
    for ic in range(CH):
        oc = int(mapping.get(ic, ic))
        weights[oc, ic, 0, 0] = 1.0
    w = helper.make_tensor("W", dt, [CH, CH, 1, 1], weights.flatten().tolist())
    b = helper.make_tensor("B", dt, [CH], bias.tolist())
    node = helper.make_node("Conv", ["input", "W", "B"], ["output"], kernel_shape=[1, 1])
    return make_model([node], [w, b])


def infer_global_color_mapping(examples):
    mapping = {}
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if grid_shape(inp) != grid_shape(out):
            return None
        for r, row in enumerate(inp):
            for c, ic in enumerate(row):
                oc = out[r][c]
                prev = mapping.get(int(ic))
                if prev is None:
                    mapping[int(ic)] = int(oc)
                elif prev != int(oc):
                    return None
    for c in range(CH):
        mapping.setdefault(c, c)
    return mapping


def train_color_remap_model(task):
    mapping = infer_global_color_mapping(all_examples(task))
    if mapping is None:
        return None, {"ok": False, "reason": "no consistent global color mapping"}
    return make_1x1_color_model(mapping), {"ok": True, "mapping": mapping}


def fixed_transform(grid, transform):
    arr = np.array(grid, dtype=int)
    if transform == "rot90":
        return np.rot90(arr, -1).tolist()
    if transform == "rot180":
        return np.rot90(arr, 2).tolist()
    if transform == "rot270":
        return np.rot90(arr, 1).tolist()
    if transform == "flip_h":
        return np.fliplr(arr).tolist()
    if transform == "flip_v":
        return np.flipud(arr).tolist()
    if transform == "transpose":
        return arr.T.tolist()
    raise ValueError(transform)


def infer_fixed_geometric_transform(examples):
    names = ["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"]
    matches = []
    for name in names:
        if all(fixed_transform(ex["input"], name) == ex["output"] for ex in examples):
            matches.append(name)
    return matches


def save_model(model, out_dir, task_id):
    require_onnx()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}.onnx"
    onnx.save(model, path)
    return path


def run_model(model_or_path, input_grid):
    require_onnx()
    if ort is None:
        raise ImportError("onnxruntime is required to run validation")
    if isinstance(model_or_path, (str, Path)):
        session = ort.InferenceSession(str(model_or_path), providers=["CPUExecutionProvider"])
    else:
        session = ort.InferenceSession(model_or_path.SerializeToString(), providers=["CPUExecutionProvider"])
    output = session.run(["output"], {"input": grid_to_tensor(input_grid)})[0]
    return (output > 0).astype(np.float32)


def visible_validation_summary(model_or_path, task, max_examples=None):
    examples = all_examples(task)
    if max_examples is not None:
        examples = examples[:max_examples]
    right = 0
    wrong = 0
    first_wrong = None
    for ex in examples:
        expected = grid_to_tensor(ex["output"])
        actual = run_model(model_or_path, ex["input"])
        if np.array_equal(actual, expected):
            right += 1
        else:
            wrong += 1
            if first_wrong is None:
                first_wrong = ex
    return {"right": right, "wrong": wrong, "first_wrong": first_wrong}


def split_examples(task):
    return {
        "train": task.get("train", []),
        "test": task.get("test", []),
        "arc_gen": task.get("arc-gen", []),
    }


def validation_summary_for_examples(model_or_path, examples, max_examples=None):
    if max_examples is not None:
        examples = examples[:max_examples]
    right = 0
    wrong = 0
    first_wrong = None
    for ex in examples:
        expected = grid_to_tensor(ex["output"])
        actual = run_model(model_or_path, ex["input"])
        if np.array_equal(actual, expected):
            right += 1
        else:
            wrong += 1
            if first_wrong is None:
                first_wrong = ex
    total = right + wrong
    accuracy = right / total if total else None
    return {"right": right, "wrong": wrong, "total": total, "accuracy": accuracy, "first_wrong": first_wrong}


def split_validation_summary(model_or_path, task, max_examples=None):
    rows = {}
    for split, examples in split_examples(task).items():
        summary = validation_summary_for_examples(model_or_path, examples, max_examples=max_examples)
        summary.pop("first_wrong", None)
        rows[split] = summary
    visible = validation_summary_for_examples(model_or_path, all_examples(task), max_examples=max_examples)
    visible.pop("first_wrong", None)
    rows["visible_all"] = visible
    return rows


def count_model_params(model_or_path):
    require_onnx()
    model = onnx.load(str(model_or_path)) if isinstance(model_or_path, (str, Path)) else model_or_path
    params = 0
    for init in model.graph.initializer:
        if init.dims:
            params += math.prod(init.dims)
        else:
            params += 1
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                params += math.prod(attr.t.dims) if attr.t.dims else 1
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return int(params)


def model_architecture_summary(model_or_path):
    require_onnx()
    path = Path(model_or_path) if isinstance(model_or_path, (str, Path)) else None
    model = onnx.load(str(path)) if path else model_or_path
    op_counts = Counter(node.op_type for node in model.graph.node)
    init_shapes = {init.name: list(init.dims) for init in model.graph.initializer}
    return {
        "ir_version": model.ir_version,
        "opsets": {op.domain or "ai.onnx": op.version for op in model.opset_import},
        "nodes": len(model.graph.node),
        "op_counts": dict(op_counts),
        "initializers": init_shapes,
        "params": count_model_params(model),
        "file_size_bytes": path.stat().st_size if path and path.exists() else None,
    }


def approximate_memory_from_model_shapes(model_or_path):
    """Approximate scored tensor memory from static value_info shapes.

    The official helper uses ONNX Runtime profiling to refine tensor memory.
    This approximation is useful in notebooks before running the full profiler.
    """
    require_onnx()
    model = onnx.load(str(model_or_path)) if isinstance(model_or_path, (str, Path)) else model_or_path
    inferred = onnx.shape_inference.infer_shapes(model)
    graph = inferred.graph
    total = 0
    for value in list(graph.value_info):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                dims = []
                break
            dims.append(dim.dim_value)
        if dims:
            total += math.prod(dims) * 4
    return int(total)


def runtime_memory_profile(model_or_path, sample_input_grid):
    """Return an approximate competition memory/param profile.

    This uses ONNX Runtime profiling if available. It is not a replacement for
    the official Kaggle validator, but it tracks the same concerns: parameters,
    intermediate tensor memory, and file size.
    """
    require_onnx()
    if ort is None:
        raise ImportError("onnxruntime is required for runtime memory profiling")
    path = Path(model_or_path) if isinstance(model_or_path, (str, Path)) else None
    model = onnx.load(str(path)) if path else model_or_path
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    session.run(["output"], {"input": grid_to_tensor(sample_input_grid)})
    trace_path = session.end_profiling()
    trace_memory = 0
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
        for event in trace:
            args = event.get("args", {})
            for shape_dict in args.get("output_type_shape", []) or []:
                for dims in shape_dict.values():
                    if dims and all(isinstance(d, int) and d > 0 for d in dims):
                        trace_memory += math.prod(dims) * 4
    except Exception:
        trace_memory = approximate_memory_from_model_shapes(model)
    return {
        "params": count_model_params(model),
        "runtime_memory_bytes": int(trace_memory),
        "static_memory_bytes": approximate_memory_from_model_shapes(model),
        "file_size_bytes": path.stat().st_size if path and path.exists() else None,
        "profile_trace_path": trace_path,
    }


def model_report(model_or_path, task=None, sample_input_grid=None, max_validation_examples=None):
    report = {"architecture": model_architecture_summary(model_or_path)}
    if task is not None:
        report["performance"] = split_validation_summary(
            model_or_path,
            task,
            max_examples=max_validation_examples,
        )
        if sample_input_grid is None:
            examples = all_examples(task)
            if examples:
                sample_input_grid = examples[0]["input"]
    if sample_input_grid is not None and ort is not None:
        report["memory_profile"] = runtime_memory_profile(model_or_path, sample_input_grid)
    else:
        report["memory_profile"] = {
            "params": report["architecture"]["params"],
            "static_memory_bytes": approximate_memory_from_model_shapes(model_or_path),
            "runtime_memory_bytes": None,
            "file_size_bytes": report["architecture"]["file_size_bytes"],
            "profile_trace_path": None,
        }
    return report


def create_submission_zip(model_dir, zip_path=None):
    model_dir = Path(model_dir)
    if zip_path is None:
        zip_path = model_dir / "submission.zip"
    else:
        zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(model_dir.glob("task*.onnx")):
            zf.write(path, path.name)
    return zip_path


def build_family_submission(family, trainer, data_dir, out_dir, fallback_identity=False, validate=False):
    task_ids = family_task_ids(family)
    rows = []
    for task_id in task_ids:
        task = load_task(data_dir, task_id)
        model, info = trainer(task)
        if model is None and fallback_identity:
            model = make_identity_model()
            info = {**info, "fallback": "identity"}
        if model is None:
            rows.append({"task_id": task_id, "saved": False, **info})
            continue
        path = save_model(model, out_dir, task_id)
        row = {"task_id": task_id, "saved": True, "path": str(path), **info}
        if validate:
            try:
                row.update({f"visible_{k}": v for k, v in visible_validation_summary(path, task).items() if k != "first_wrong"})
            except Exception as exc:
                row["visible_error"] = repr(exc)
        rows.append(row)
    zip_path = create_submission_zip(out_dir)
    return rows, zip_path
