# Smart Buildings RL — P0 baseline and profiling

This package evaluates the original V7, V23, and V25 notebook controller cells
against the uploaded competition physics simulator. P0 establishes an accurate
reward decomposition and simulator throughput benchmark BEFORE RL training.

## Resources

Required repository paths:

- Co_Kaggle/g8/competition_material/Data/smart_buildings/
- Co_Kaggle/g8/submissions/submission-nb - Version 7.ipynb
- Co_Kaggle/g8/submissions/submission-nb - Version 23.ipynb
- Co_Kaggle/g8/submissions/submission-nb - Version 25.ipynb

Run from repository root with the official starter-kit Python dependencies
installed (numpy, scipy, numba, gin-config, gymnasium, pandas, matplotlib,
Pillow, etc.). The uploaded Data folder does not have an installable
package or pinned environment file. CPU simulation does not use a GPU.

## Python 3.12 setup

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r Co_Kaggle/g8/rl/p0/requirements-simulator.txt

The above uses the organizer's pinned Python dependency versions. On Windows,
activate the venv using .venv\Scripts\activate instead.

## Commands

    # Dependency-light tests (numpy required)
    python -m unittest discover -s Co_Kaggle/g8/rl/p0/tests -v

    # Inventory the 30 uploaded training scenarios
    python -m Co_Kaggle.g8.rl.p0.benchmark --list

    # Smoke-test all three controllers, 20 steps each
    python -m Co_Kaggle.g8.rl.p0.benchmark --steps 20

    # Short parallel profile over all training scenarios
    python -m Co_Kaggle.g8.rl.p0.benchmark --all --steps 20 --workers 4

    # Full, expensive, 30-day training-set evaluation
    python -m Co_Kaggle.g8.rl.p0.benchmark --full --workers 4

Outputs: g8_rl_p0_results/report.json and per_scenario.csv.
Each scenario reports normalized weighted comfort/energy/carbon loss, and
separate controller, simulator and reward-diagnostics execution times.
Throughput is measured using overall wall-clock time across all workers.

IMPORTANT: Scores from short episodes are not leaderboard scores.
Only a full 8640-step evaluation on all 30 uploaded training scenarios
is labelled TRAINING score comparable in definition to the competition.
It is NOT a hidden leaderboard score, and the V23 public 0.03706 does not
necessarily match the uploaded training scenarios.

## Design constraints

- The exact controller source cells are loaded, not rewritten. Only the
  Kaggle Policy base import is replaced; the gateway run cell is excluded.
- Each scenario gets a fresh environment and policy.
- Gin config and current working directory are process-global. Parallelism
  uses separate spawned processes, never threads.
- Match competition gateway: set env.metrics_path = None (not the typo
  env._metrics_path) and occupancy normalization = 125.0.
- Do not allocate SimulationTracker full-grid snapshots: 8640-step runs
  otherwise require multiple GB of memory per scenario.
- Reward diagnostics recompute the normalized components post-step.
  Simulator timing excludes the diagnostic overhead.
- Check that the official returned reward equals the sum of the negative
  weighted loss components, to 1e-5 tolerance.
- Respect the 30-scenario, 8640-step scoring contract.

## Next steps

P0: verify local runtime and measure V7/V23/V25 on all scenarios.
P1: add trajectory collection, residual PPO/SAC and a held-out scenario split.
P2: compare hierarchical RL and GPU surrogate-based rollouts.

Targets are first below 0.010, then 0.006, ultimately below 0.00484.
