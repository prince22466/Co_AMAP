"""Verify shared deterministic rollout, baseline parity and checkpoint replay."""
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ..p0.benchmark import default_data_root
from ..p1.train import assert_initial_parity, comparison
from .model import SearchPolicy, signature
from .train import run_jobs, save_checkpoint


def main():
    root = default_data_root().resolve()
    scenarios = ["building_train_1/atlanta_2"]
    model = SearchPolicy()

    def run(policy, mode="eval"):
        return run_jobs(None, root, scenarios, 12, 3, policy, mode, 42)

    baseline = run(model, "baseline")
    assert_initial_parity(comparison(baseline, run(model)))
    with torch.no_grad():
        model.coefficients[0, -1] = 1.
    changed = run(model)
    assert changed[0]["profile_steps"][1] == 12
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.pt"
        save_checkpoint(path, model, 3, 1, changed[0]["mean_loss"], 12)
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["signature"] == signature()
        restored = SearchPolicy()
        restored.load_state_dict(state["model"])
        replay = run(restored)
        assert abs(comparison(changed, replay)["delta"]) < 1e-9
    print("P2 official simulator integration passed", flush=True)


if __name__ == "__main__":
    main()
