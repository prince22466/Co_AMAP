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


## P0.1: targeted reward / temperature / actuator diagnostics

P0.1 answers three focused questions *before* RL training. It does not
modify any controller, simulator dynamics, objective, or the P0 report.

1. **A — Reward timeline / occupancy:** stream step-level loss decomposition,
   electricity and gas cost, carbon emissions, occupancy, hour and comfort-mode
   status. Summary includes means by hour, observed occupancy and comfort/eco
   schedule, with separate occupied, unoccupied comfort, upcoming comfort
   (preconditioning opportunity), and unoccupied eco phases. Hour buckets use
   the actual reward timestamp in UTC; the action timestamp and reward interval
   end are also retained. Comfort flags are from the post-step observation.
   An active comfort schedule is not proof that a zone is actually occupied.
2. **B — Temperature violations:** calculate actual zone deviation beyond the
   heating/cooling setpoint band using the simulator's reward-info zone
   temperatures. Count exceedances strictly above 1/3/5/10 K, separately for
   all zones, occupied zones, and occupant-weighted samples. Attribute the
   exact weighted comfort loss to disjoint severity bands using the simulator's
   piecewise productivity curve and per-step occupancy normalization.
3. **C — Actuator utilization:** report mean normalized command, fraction
   within the outer 5% of the normalized action range, mean absolute step
   changes, and run/switch duty cycles for AHUs / water equipment and zone
   damper/reheat actuators. These are **command-space** bounds, not evidence
   that the physical device is saturated or that delivered flow is sufficient.
   Per-step command summaries support timing analysis; additionally report
   occupied-zone delivered airflow at or above 95% of design capacity, including
   the subset with >1 K discomfort. This is a capacity indicator, not a
   causal diagnosis of insufficient HVAC capacity.

Run from repository root after the same P0 simulator setup:

    # Targeted 3-day trajectories: 3 climates x V7/V23/V25
    # All selected scenarios begin on a Sunday; three days also covers weekdays.
    python -m Co_Kaggle.g8.rl.p0.diagnostics --steps 864 --workers 4

    # One targeted 12-hour evaluation (not a leaderboard score)
    python -m Co_Kaggle.g8.rl.p0.diagnostics \
      --scenarios building_train_1/fairbanks_1 --versions 23 --steps 144

    # Optional full episode for one scenario (slow)
    python -m Co_Kaggle.g8.rl.p0.diagnostics \
      --scenarios building_train_2/phoenix_6 --versions 7 23 --full

    python -m unittest discover -s Co_Kaggle/g8/rl/p0/tests -v

Outputs are under `results/diagnostics/v<version>/` as one CSV timeline and
one JSON summary per scenario; `results/diagnostics/index.json` lists the
runs. They are intentionally not written into the existing `report.json` or
`per_scenario.csv`. Runs with `--steps` are **partial diagnostics**: they
must not be described as training-set or public leaderboard scores. `--full`
follows actual gateway termination (currently 8641 rewards) and is flagged
separately.

P0.1 keeps only compact summaries in memory and streams rows to CSV.
Use fewer workers if RAM is constrained. The default is one worker; parallel
workers use spawned processes. Interrupted runs leave a partial CSV rather
than a new completed summary/index. A zero-occupancy warning indicates that
comfort diagnostics need a longer trajectory.
The loop obtains the organizer's post-step RewardInfo/RewardResponse for
exact accounting; it is intended solely for local investigation (the Kaggle
submission policy cannot access those simulator internals). No policy is
trained or changed in this PR.
