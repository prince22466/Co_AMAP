"""Load exact notebook controller cells for local evaluation, without Kaggle gateway.

Only the Kaggle Policy import is replaced with an equivalent local base.
The actual HVAC controller code remains unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import numpy as np

VERSIONS = (7, 23, 25)


class LocalPolicy:
    def __init__(self, building_meta_data: dict[str, Any], action_spec: Any):
        self.building_meta_data = building_meta_data
        self.action_spec = action_spec


def notebook_path(version: int) -> Path:
    if version not in VERSIONS:
        raise ValueError(f"Supported baseline versions: {VERSIONS}, got {version}")
    return (Path(__file__).resolve().parents[2] / "submissions"
            / f"submission-nb - Version {version}.ipynb")


def load_controller(version: int, *, path: Path | None = None) -> type:
    source_path = path if path is not None else notebook_path(version)
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    matches = ["".join(cell["source"]) for cell in notebook["cells"]
               if cell["cell_type"] == "code"
               and "class TransferPolicy(" in "".join(cell["source"])]
    if len(matches) != 1:
        raise ValueError(f"Expected one controller code cell in {source_path}")
    source = matches[0]
    kaggle_import = "from kaggle_evaluation import smart_buildings_inference_server"
    if source.count(kaggle_import) != 1:
        raise ValueError("Expected exactly one Kaggle inference-server import")
    source = source.replace(kaggle_import, "# Local Policy base supplied below")
    module = ModuleType(f"g8_original_submission_v{version}")
    module.__dict__["smart_buildings_inference_server"] = SimpleNamespace(
        Policy=LocalPolicy)
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    policy_class = getattr(module, "TransferPolicy", None)
    if not isinstance(policy_class, type) or not issubclass(policy_class, LocalPolicy):
        raise TypeError(f"Missing TransferPolicy in {source_path}")
    return policy_class


def validate_action(action: dict[str, Any], action_spec: Any) -> dict[str, np.ndarray]:
    if set(action) != {"continuous_action", "discrete_action"}:
        raise ValueError(f"Unexpected action keys: {sorted(action)}")
    continuous = np.asarray(action["continuous_action"])
    discrete = np.asarray(action["discrete_action"])
    if continuous.dtype != np.float32 or discrete.dtype != np.int32:
        raise TypeError(f"Expected float32/int32, got {continuous.dtype}/{discrete.dtype}")
    if continuous.shape != action_spec["continuous_action"].shape:
        raise ValueError("Invalid continuous action shape")
    if discrete.shape != action_spec["discrete_action"].shape:
        raise ValueError("Invalid discrete action shape")
    if not np.isfinite(continuous).all() or np.any(np.abs(continuous) > 1.0):
        raise ValueError("Continuous commands must be finite and within [-1, 1]")
    if not np.all((discrete == 0) | (discrete == 1)):
        raise ValueError("Discrete commands must be 0 or 1")
    return {"continuous_action": continuous, "discrete_action": discrete}
