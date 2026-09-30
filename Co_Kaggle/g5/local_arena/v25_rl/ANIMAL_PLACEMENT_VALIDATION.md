# Planned animal placement and PASS validation

The uploaded v10 run reduced validation animal placement from 10.6 at its starting baseline to 7.0 at update 157. Zero escapes alone therefore does not establish survival at the original production capacity. Avoidable PASS was zero at every uploaded validation checkpoint, but totaled 9 actions across its 40 stochastic training episodes.

This change gives capacity-safe planned animal setup priority after critical WATER/FEED, emergency wheat supply, and first-loss harvest/cleanup deadlines. The setup pipeline covers clearing the planned site, BUILD, animal PICKUP, and PLACE. Ready PLACE is preferred over PICKUP, which is preferred over starting more structures. Existing committed routes continue until completion or survival/deadline preemption. Animal placement and seed planting share same-turn maintenance reservations. PASS is excluded when that worker has feasible unreserved work; genuine idle remains legal.

## Paired replay

- Base: main commit `3250615ba226e36fda79cb3eda5229da5e31f027`.
- Actor: uploaded v10 `checkpoints/latest.pt`, update 157, Git blob `fdb30efe85d2a8eda69bf93b41c8b85e7ef0ceb1`.
- Same five histories from `Co_Kaggle/g5/game_history/v20`, unchanged model weights, executor, market replay, and deterministic inference.
- Linux, Python 3.12.14, PyTorch 2.14.0+cpu, kaggle-environments 1.32.7, one Torch thread per process.
- Local base results differ from the uploaded validation log. All changes below use the paired local base, rather than mixing those populations. These are scheduler replays, not new PPO training results.

| Mean per episode | Local base | Changed scheduler | Change |
| --- | ---: | ---: | ---: |
| Worker reward | 37,331.29 | 40,916.01 | +9.60% |
| Animals placed | 7.0 | 8.4 | +20.00% |
| Animal units generated | 209.0 | 231.6 | +10.81% |
| Animal units delivered | 206.4 | 227.8 | +10.37% |
| Crop units harvested | 507.8 | 488.8 | -3.74% |
| Total units delivered | 419.0 | 395.6 | -5.58% |
| Animal escapes | 0 | 0 | 0 |
| Crop weeds | 0.6 | 0 | -0.6 |
| Crop deaths | 0 | 0 | 0 |
| Lost harvestable units | 2.4 | 0 | -2.4 |
| Avoidable PASS | 0 | 0 | 0 |

| History | Base animals placed | Changed animals placed | Changed reward | Changed escapes / weeds / lost |
| --- | ---: | ---: | ---: | --- |
| 111551968 | 6 | 7 | 40,587.40 | 0 / 0 / 0 |
| 111558730 | 8 | 9 | 39,484.95 | 0 / 0 / 0 |
| 111565456 | 6 | 8 | 40,114.90 | 0 / 0 / 0 |
| 111587965 | 8 | 9 | 42,588.65 | 0 / 0 / 0 |
| 111592475 | 7 | 9 | 41,804.15 | 0 / 0 / 0 |

More animal placement improved animal output and reward, with lower crop and total-unit throughput. This is a five-history deterministic result; stochastic exploration and other histories still need evaluation. Maintenance admission is a bounded route estimate with uncertain future stock and hiring, not a proof of zero losses.

## Checks and reproduction

110 unit tests pass, including actual-engine decay checks, deadline priority, remote placement commitment, stochastic placement priority, unavailable stock, next-day maintenance overload, PASS masks and filtered PPO samples. `git diff --check` passes.

Run from `Co_Kaggle/g5/local_arena/v25_rl`:

```bash
python -m unittest discover -p 'test_*.py'
```

For each of the two scheduler versions, evaluate the same checkpoint and history files with the trainer's replay API; no optimizer or checkpoint resume reset is involved:

```python
from pathlib import Path
import torch
from train_v25_worker_ppo import (
    ActorCritic, CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES,
    DEFAULT_EXECUTOR, run_static_episode,
)

torch.set_num_threads(1)
checkpoint = torch.load(
    "runs/worker_ppo_static_v20_v10_lossless_harvest/checkpoints/latest.pt",
    map_location="cpu", weights_only=False,
)
model = ActorCritic(len(CANDIDATE_FEATURE_NAMES), len(GLOBAL_FEATURE_NAMES), 128)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
histories = Path("../../game_history/v20")
for episode in ("111551968", "111558730", "111565456", "111587965", "111592475"):
    with torch.inference_mode():
        result, _ = run_static_episode(
            histories / (episode + ".json"), model, torch.device("cpu"),
            DEFAULT_EXECUTOR, True, False,
        )
    print(result.episode, result.worker_reward, result.reward_breakdown)
```

Linux loading of Windows-created checkpoints may require `pathlib.WindowsPath = pathlib.PosixPath` before loading because saved arguments contain Windows Path objects.

The v11 trainer writes to a separate default output directory and changes its execution contract. A normal training resume keeps actor weights/update numbering, resets critic/optimizer/best statistics, and measures a fresh baseline. Reward coefficients and reward-maximizing checkpoint selection are preserved. Total PASS is now reported as `pass_actions`/`mean_pass_actions` alongside `avoidable_passes`/`mean_avoidable_passes`; zero avoidable PASS does not detect missing candidate tasks or ineffective movement.

