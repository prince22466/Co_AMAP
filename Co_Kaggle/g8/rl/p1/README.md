# P1 — first RL model: hierarchical residual-profile PPO

P0 established gateway-equivalent simulation. P1 learns a categorical policy
over seven bounded V23 operating profiles, chosen every 12 five-minute steps
(one hour); V23 still controls all continuous/discrete equipment every step.
Profile zero is identical to V23. The initial greedy actor selects profile zero
for every state, and training checks each scenario and reward component against independent,
unmodified V23 before updating. A small initial baseline preference avoids
making the deterministic actor unnecessarily difficult to move away from V23.
The policy does not read the simulator internals, reward components, scenario
identity, future occupancy, or weather forecasts. Reward is the actual simulator
reward scaled by a positive constant for numerical conditioning. The PPO actor
and critic use two 64-unit layers and 25 fixed-length features (zone comfort
errors, outdoor temperature, clock, schedule, sensor coverage, previous profile).
Soft command-space capacity budgets are varied, not physical capacity guarantees.
No independent actuator safety guarantee is implied.

Shared profile definitions, simulator rollout and paired evaluation live in
`../common/`. PPO, GAE and checkpoint selection remain in P1; the original
CLI and checkpoint schema are unchanged.

## Install and run

From the repository root (the directory containing Co_Kaggle), in the Python
3.12 P0 environment:

    python -m pip install -r Co_Kaggle/g8/rl/p0/requirements-simulator.txt
    python -m pip install -r Co_Kaggle/g8/rl/p1/requirements.txt
    python -m unittest Co_Kaggle.g8.rl.p1.test_p1 -v

CPU smoke test (one update; validation still covers five scenarios):

    python -m Co_Kaggle.g8.rl.p1.train --out Co_Kaggle/g8/rl/p1/results/smoke --updates 1 --steps 24 --interval 12 --scenarios-per-update 1 --workers 1

Initial experiment (three-day windows; Sunday plus weekdays):

    python -m Co_Kaggle.g8.rl.p1.train --out Co_Kaggle/g8/rl/p1/results/run1 --updates 20 --steps 864 --workers 4 --device cpu

Use --device cuda for PPO optimization if a compatible CUDA PyTorch build is
installed. The simulator and rollout inference are CPU-bound. Do not assume
a 64 GB GPU accelerates physics. Use fewer workers if RAM is constrained.
Training defaults to 4 randomly selected train scenarios per update, 4 PPO
epochs, 128 minibatch, gamma .999 per five-minute step, macro GAE lambda .95,
clip .2, Adam 3e-4, entropy .01, value coefficient .5, gradient norm .5.
Partial training windows bootstrap the critic at their artificial time limit.
Training horizons must be divisible by the decision interval. Rollout action
sampling uses an independent CPU generator, separate from PPO minibatch RNG.

## Evaluation protocol

Train on buildings 1,2,4,7 (20 scenarios); validation on building 5 (5);
final test on building 8 (5). All contain the five climate stations; the
weather strips differ. Test is never used by training or checkpoint selection.
Every five updates and at the last update, greedy RL is compared against the
cached V23 validation baseline using the same scenario seeds and horizon.
best.pt starts as the baseline actor and is replaced only on strictly improved
validation loss. history.json includes paired per-scenario and decomposed loss
for validation, profile utilization, transition counts and PPO loss. The
checkpoint contains model weights, interval and profile schema, not
optimizer/resume state. Evaluation reports include the horizon in their filename
and will not overwrite a previous report.

Evaluate once on the independent test building after choosing the checkpoint:

    python -m Co_Kaggle.g8.rl.p1.train --checkpoint Co_Kaggle/g8/rl/p1/results/run1/best.pt --out Co_Kaggle/g8/rl/p1/results/run1/test --steps 864 --workers 4

For complete gateway-length test episodes, add --full (slow). Short windows
are diagnostics only. Full test is still a five-scenario local proxy, NOT a
30-scenario training-set score or hidden leaderboard score. Compare raw mean
loss against V23 and review comfort/energy/carbon components and each scenario.
The target sequence remains 0.010, 0.006, 0.00484 on the competition score;
no improvement or leaderboard performance is claimed until measured.
This PR does not create a trained checkpoint or a Kaggle submission notebook.
