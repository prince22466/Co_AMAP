# P3 — baseline-gated Double DQN

P1 PPO collapsed toward comfort; P2's 20-update paired search retained V23
at validation loss 0.0136183537 (864 steps). Three P2 mutations were accepted
on training, but none improved validation. P3 tests a distinct hypothesis:
reuse every informative macro-transition across updates with off-policy
value learning instead of searching episode-level returns.

## Model and objective

The shared seven V23 profiles and 25 gateway-only features are unchanged.
A two-layer 64-unit Q-network predicts normalized discounted reward for
each profile. V23 remains the low-level controller every five minutes;
the learned policy selects a profile every 12 steps by default.
Zero output initialization makes greedy action exactly V23; a nonbaseline
profile must exceed the baseline Q by 0.03 normalized reward units to deploy.
This margin is a heuristic, not a guarantee of safety or improvement.

Training collects epsilon-greedy trajectories (0.4 to 0.1), samples a bounded
50k-transition replay buffer, uses Huber TD loss, Adam, gradient clipping
and a lagged target network. The online Q argmax selects the bootstrap
action; the target Q evaluates it (Double DQN). The training target uses
the ungated argmax; deployment uses the conservative margin. For duration
d, reward is the shared discounted simulator reward scaled by 100 and
(1-gamma), and the bootstrap multiplier is gamma^d (gamma=.99 per five-minute
step, zero at actual termination). A partial-window final transition is
omitted because common.rollout lacks its next observation. No episode
boundaries are joined, and no reward shaping or future observations are
used as policy inputs. The discounted training objective is a surrogate;
checkpoint selection uses the raw undiscounted simulator loss.

## Run from repository root, Python 3.12 P0 environment

    python -m pip install -r Co_Kaggle/g8/rl/p0/requirements-simulator.txt
    python -m pip install -r Co_Kaggle/g8/rl/common/requirements.txt
    python -m unittest Co_Kaggle.g8.rl.p3.test_p3 -v

Short end-to-end simulator smoke (small batch to exercise optimization):

    python -m Co_Kaggle.g8.rl.p3.train --out Co_Kaggle/g8/rl/p3/results/smoke --updates 1 --steps 24 --workers 1 --batch-size 4 --gradient-steps 2 --target-sync 1

Initial matched-horizon experiment:

    python -m Co_Kaggle.g8.rl.p3.train --out Co_Kaggle/g8/rl/p3/results/run1 --updates 20 --steps 864 --workers 4 --device cpu

Four training rollouts per update (one random climate per train building);
five validation scenarios every five updates and the final update, plus
initial V23 parity. Same building-disjoint 20/5/5 split as P1/P2. No test
scenarios are used in learning or selection. CPU simulator workers run in
parallel; CUDA optionally accelerates only the learner, not physics.
The initial baseline checkpoint is replaced only on strictly improved
validation loss. History records exploratory training loss, replay size,
TD loss, action counts, optimizer steps, per-scenario profile utilization,
and paired validation loss/components. TD loss is not competition loss,
and exploratory train losses across different scenarios are not comparable.

Independent checkpoint test:

    python -m Co_Kaggle.g8.rl.p3.train --checkpoint Co_Kaggle/g8/rl/p3/results/run1/best.pt --out Co_Kaggle/g8/rl/p3/results/run1/test --steps 864 --workers 4

Add --full for the full gateway-length test. Use a new output directory.
Checkpoints are inference-only; no optimizer or replay resume support.
No trained checkpoint, Kaggle notebook, or measured improvement is included.
A successful diagnostic first beats the paired 0.013618 baseline, then
approaches 0.010; the competition targets 0.006 and below 0.00484 remain
unverified. Short-window local scores are not leaderboard scores.
