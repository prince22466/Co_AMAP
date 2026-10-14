"""Static-replay agent using a v25 PPO checkpoint for worker actions.

Select a checkpoint with the V25_WORKER_CHECKPOINT environment variable.
Without it, the best checkpoint from the animal-reward run is used.
"""

import os
from pathlib import Path, WindowsPath

import torch

import v25_rl
from worker_policy import (
    ActorCritic,
    CANDIDATE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    WorkerPolicy,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    HERE / "runs" / "worker_ppo_static_v20_v13_plant_signal"
    / "checkpoints" / "best.pt"
)
CHECKPOINT = Path(os.environ.get("V25_WORKER_CHECKPOINT", DEFAULT_CHECKPOINT)).resolve()

if not CHECKPOINT.is_file():
    raise FileNotFoundError(f"Worker checkpoint not found: {CHECKPOINT}")

with torch.serialization.safe_globals([WindowsPath]):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
if checkpoint.get("algorithm") != "v25_static_worker_ppo_gae_v4_animal_reward":
    raise ValueError(f"Incompatible worker checkpoint: {CHECKPOINT}")
if tuple(checkpoint.get("candidate_feature_names", ())) != CANDIDATE_FEATURE_NAMES:
    raise ValueError("Checkpoint candidate features differ from worker_policy.py")
if tuple(checkpoint.get("global_feature_names", ())) != GLOBAL_FEATURE_NAMES:
    raise ValueError("Checkpoint global features differ from worker_policy.py")

hidden = int(checkpoint.get("args", {}).get("hidden", 128))
model = ActorCritic(len(CANDIDATE_FEATURE_NAMES), len(GLOBAL_FEATURE_NAMES), hidden)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
policy = WorkerPolicy(v25_rl, model, torch.device("cpu"), deterministic=True, collect=False)


def agent(obs):
    # v25_rl.agent keeps its crop, animal, and market planning. Only its worker
    # scheduler is replaced by the checkpoint policy.
    v25_rl.unit_actions = policy.unit_actions
    return v25_rl.agent(obs)
