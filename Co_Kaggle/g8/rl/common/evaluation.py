"""Shared building-disjoint split, paired scoring and atomic JSON reports."""
from __future__ import annotations

import json
import math

import numpy as np


SPLIT = {"train": (1, 2, 4, 7), "validation": (5,), "test": (8,)}


def split_scenarios(scenarios):
    groups = {key: [s for s in scenarios
                    if s.split("/")[0] in {f"building_train_{i}" for i in ids}]
              for key, ids in SPLIT.items()}
    if sorted(s for group in groups.values() for s in group) != sorted(scenarios):
        raise ValueError("Unassigned or duplicated scenario")
    if [len(groups[k]) for k in SPLIT] != [20, 5, 5]:
        raise ValueError("Expected 20/5/5 building-disjoint scenarios")
    return groups


def mean_loss(rows):
    return float(np.mean([r["mean_loss"] for r in rows]))


def comparison(baseline, learned):
    if not baseline or [(r["scenario"], r["steps"], r["full_episode"]) for r in baseline] != [
            (r["scenario"], r["steps"], r["full_episode"]) for r in learned]:
        raise ValueError("Unpaired evaluation")
    return {"baseline_loss": mean_loss(baseline), "rl_loss": mean_loss(learned),
            "delta": mean_loss(learned) - mean_loss(baseline),
            "baseline": baseline, "rl": learned}


def assert_initial_parity(report):
    """Compare every scenario and component, not just canceling mean errors."""
    for baseline, learned in zip(report["baseline"], report["rl"]):
        same_loss = math.isclose(baseline["mean_loss"], learned["mean_loss"],
                                 abs_tol=1e-9, rel_tol=1e-9)
        same_parts = all(math.isclose(v, learned["loss_components"][k],
                                      abs_tol=1e-9, rel_tol=1e-9)
                         for k, v in baseline["loss_components"].items())
        if not same_loss or not same_parts or learned["profile_steps"][0] != learned["steps"]:
            raise RuntimeError(f"Initial greedy policy differs from V23: {baseline['scenario']}")


def save_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
