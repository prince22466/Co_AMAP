"""Shared building-disjoint split, paired scoring and atomic JSON reports."""
from __future__ import annotations

import json
import math

import numpy as np


SPLIT = {"train": (1, 2, 4, 7), "validation": (5,), "test": (8,)}


def split_scenarios(scenarios):
    scenarios = list(scenarios)
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("Duplicate scenario")
    groups = {key: [s for s in scenarios
                    if s.split("/")[0] in {f"building_train_{i}" for i in ids}]
              for key, ids in SPLIT.items()}
    if sorted(s for group in groups.values() for s in group) != sorted(scenarios):
        raise ValueError("Unassigned or duplicated scenario")
    for key, ids in SPLIT.items():
        for building in ids:
            prefix = f"building_train_{building}/"
            if sum(s.startswith(prefix) for s in groups[key]) != 5:
                raise ValueError(f"Expected five scenarios for building {building}")
    return groups


def mean_loss(rows):
    losses = [r["mean_loss"] for r in rows]
    if not losses or not all(map(math.isfinite, losses)):
        raise ValueError("Empty or nonfinite evaluation")
    result = float(np.mean(losses))
    if not math.isfinite(result):
        raise ValueError("Nonfinite mean loss")
    return result


def comparison(baseline, learned):
    def identity(row):
        return (row["scenario"], row["steps"], row["full_episode"],
                row.get("terminated"), row.get("truncated"))

    if (not baseline or [identity(r) for r in baseline] != [identity(r) for r in learned]
            or len({r["scenario"] for r in baseline}) != len(baseline)
            or any(r["steps"] <= 0 for r in baseline)):
        raise ValueError("Unpaired evaluation")
    baseline_loss, rl_loss = mean_loss(baseline), mean_loss(learned)
    delta = rl_loss - baseline_loss
    if not math.isfinite(delta):
        raise ValueError("Nonfinite evaluation delta")
    return {"baseline_loss": baseline_loss, "rl_loss": rl_loss, "delta": delta,
            "baseline": baseline, "rl": learned}


def assert_initial_parity(report):
    """Compare every scenario and component, not just canceling mean errors."""
    # Validate pairing even when a caller supplies a report directly.
    comparison(report["baseline"], report["rl"])
    for baseline, learned in zip(report["baseline"], report["rl"]):
        same_loss = math.isclose(baseline["mean_loss"], learned["mean_loss"],
                                 abs_tol=1e-9, rel_tol=1e-9)
        same_parts = (
            baseline["loss_components"].keys() == learned["loss_components"].keys()
            and all(math.isclose(v, learned["loss_components"][k],
                                 abs_tol=1e-9, rel_tol=1e-9)
                    for k, v in baseline["loss_components"].items()))
        histogram = learned["profile_steps"]
        baseline_only = bool(histogram and histogram[0] == learned["steps"]
                             and all(v == 0 for v in histogram[1:]))
        if not same_loss or not same_parts or not baseline_only:
            raise RuntimeError(f"Initial greedy policy differs from V23: {baseline['scenario']}")


def save_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
