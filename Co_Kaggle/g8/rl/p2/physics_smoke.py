"""Verify shared deterministic rollout, baseline parity and checkpoint replay."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ..p0.benchmark import default_data_root
from ..common.evaluation import assert_initial_parity, comparison, mean_loss
from .model import SearchPolicy, signature
from .train import run_jobs, save_checkpoint


def main():
    root = default_data_root().resolve()
    scenarios = ["building_train_1/atlanta_2", "building_train_1/denver_7"]
    model = SearchPolicy()

    def run(policy, mode="eval"):
        return run_jobs(None, root, scenarios, 12, 3, policy, mode, 42)

    baseline = run(model, "baseline")
    assert_initial_parity(comparison(baseline, run(model)))
    with torch.no_grad():
        model.coefficients[0, -1] = 1.
    changed = run(model)
    assert all(r["profile_steps"][1] == 12 for r in changed)
    # Windows and Linux use the same spawn path in production.
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        parallel = run_jobs(pool, root, scenarios, 12, 3, model, "eval", 42)
    assert abs(comparison(changed, parallel)["delta"]) < 1e-9
    for serial_row, parallel_row in zip(changed, parallel):
        assert abs(serial_row["mean_loss"] - parallel_row["mean_loss"]) < 1e-9
        assert serial_row["profile_steps"] == parallel_row["profile_steps"]
        for key, value in serial_row["loss_components"].items():
            assert abs(value - parallel_row["loss_components"][key]) < 1e-9
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.pt"
        save_checkpoint(path, model, 3, 1, mean_loss(changed), 12)
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["signature"] == signature()
        restored = SearchPolicy()
        restored.load_state_dict(state["model"])
        replay = run(restored)
        assert abs(comparison(changed, replay)["delta"]) < 1e-9
        for original, restored_row in zip(changed, replay):
            assert abs(original["mean_loss"] - restored_row["mean_loss"]) < 1e-9
            assert original["profile_steps"] == restored_row["profile_steps"]
    print("P2 official simulator integration passed", flush=True)


if __name__ == "__main__":
    main()
