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

Outputs: Co_Kaggle/g8/rl/p0/results/report.json and per_scenario.csv.
Each scenario reports normalized weighted comfort/energy/carbon loss, and
separate controller, simulator and reward-diagnostics execution times.
Throughput is measured using overall wall-clock time across all workers.

IMPORTANT: Scores from short episodes are not leaderboard scores.
Only a run to actual simulator termination on all 30 uploaded training
scenarios is labelled TRAINING score comparable to the competition gateway.
It is NOT a hidden leaderboard score, and the V23 public 0.03706 does not
necessarily match the uploaded training scenarios.

## Episode length: important gateway-vs-documentation discrepancy

The episode is configured for 30 days, or 8640 five-minute intervals.
In the uploaded Environment.step implementation, the terminal flag is tested
against the current step counter *before* incrementing it. Starting from
counter zero, this makes the current gateway consume **8641 rewards** and
score the mean over the actual number of completed steps.

The organizer's standalone score_leaderboard.py helper instead assumes
**8640 rewards**. P0 deliberately follows the *competition gateway* for
--full by stepping until termination and dividing by the actual reward count.
The full benchmark verifies its scenario set against training_set.yaml.
Do not divide the gateway-equivalent 8641-step return by 8640, or silently
label 8640 collected steps as a complete gateway run.

A short smoke run (e.g. --steps 72) is purely diagnostic and is never marked
as a completed episode.

## Design constraints

- The exact controller source cells are loaded, not rewritten. Only the
  Kaggle Policy base import is replaced; the gateway run cell is excluded.
- Each scenario gets a fresh environment and policy.
- Gin config and current working directory are process-global. Parallelism
  uses separate spawned processes, never threads.
- Match the competition gateway: set env.metrics_path = None (not
  env._metrics_path), and preserve its assignment
  env._occupancy_normalization_constant = 125.0. This second assignment is
  an upstream typo: actual observations use env.occupancy_normalization_constant,
  which remains 0.0 under the shipped config. Do not silently fix it in only
  this benchmarking harness; that would destroy gateway parity.
- Do not allocate SimulationTracker full-grid snapshots: 8640-step runs
  otherwise require multiple GB of memory per scenario.
- Reward diagnostics recompute the normalized components post-step.
  Simulator timing excludes the diagnostic overhead.
- Check that the official returned reward equals the sum of the negative
  weighted loss components, to 1e-9 tolerance.
- Respect the 30-scenario training manifest and run to actual gateway termination.

## Next steps

P0: verify local runtime and measure V7/V23/V25 on all scenarios.
P1: add trajectory collection, residual PPO/SAC and a held-out scenario split.
P2: compare hierarchical RL and GPU surrogate-based rollouts.

Targets are first below 0.010, then 0.006, ultimately below 0.00484.
