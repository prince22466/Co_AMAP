"""Small official-simulator integration test, including independent V23 parity."""
from pathlib import Path
from tempfile import TemporaryDirectory
import math

import torch

from ..p0.benchmark import default_data_root
from ..p0.runner import run_scenario
from .model import ActorCritic
from .train import rollout, snapshot, update, comparison, assert_initial_parity, checkpoint_signature


def main():
    root = str(default_data_root().resolve())
    scenario = "building_train_1/atlanta_2"
    model = ActorCritic()
    weights = snapshot(model)
    def run(mode):
        return rollout((root, scenario, 12, 3, weights, mode, 42))
    original = run_scenario((root, scenario, 23, 12))
    baseline, initial = run("baseline"), run("eval")
    assert math.isclose(original["mean_loss"], baseline["mean_loss"], abs_tol=1e-9)
    assert_initial_parity(comparison([baseline], [initial]))
    rng_before = torch.random.get_rng_state().clone()
    train = run("train")
    assert torch.equal(rng_before, torch.random.get_rng_state()), "Rollout modified PPO RNG"
    assert train["steps"] == 12
    assert sum(train["trajectory"]["duration"]) == 12
    assert len(train["trajectory"]["x"]) == 4
    loss = update(model, torch.optim.Adam(model.parameters(), lr=3e-4),
                  [train], "cpu", 1, 4)
    assert math.isfinite(loss)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        torch.save({"model": snapshot(model), "signature": checkpoint_signature()}, path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["signature"] == checkpoint_signature()
        restored = ActorCritic()
        restored.load_state_dict(state["model"])
        result = rollout((root, scenario, 12, 3, snapshot(restored), "eval", 42))
        assert math.isfinite(result["mean_loss"])
        assert sum(result["profile_steps"]) == 12
    print("P1 simulator integration smoke passed", flush=True)


if __name__ == "__main__":
    main()
