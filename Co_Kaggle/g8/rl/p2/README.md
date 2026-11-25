# P2 — paired, baseline-anchored direct RL policy search

P1 run1 (864 steps, building 5 validation) retained V23 at 0.013618.
Greedy validation at updates 10, 15 and 20 deteriorated to 0.014123,
0.019651 and 0.019403 respectively; by update 20 every validation
step used the comfort profile. This is evidence of profile collapse,
not proof that all residual-profile methods fail.

P2 keeps the seven operating profiles, V23 low-level actions, official
simulator, reward and 20/5/5 building-disjoint split. It replaces PPO
with gradient-free direct policy search (episodic reinforcement learning):
a deterministic affine score for each nonbaseline profile, a fixed zero
score for V23, and 18 physical/clock features. Previous-profile one-hot
is excluded to avoid self-reinforcement. The initial greedy model exactly
reproduces V23. Each update cycles through one nonbaseline profile and
proposes two opposite coefficient perturbations. Parent and proposals run
on identical scenarios, horizon and seeds. A candidate is accepted only
if mean training loss improves and no sampled scenario worsens by more
than the configured absolute threshold (default .005). This is an
empirical guard, not an actuator safety guarantee or a generalization
guarantee. Global collapse is still possible; profile histograms expose it.

Validation is never used for accepting training mutations; only for
selecting best.pt, which starts at V23. Test building 8 is untouched until
explicit checkpoint evaluation. The training incumbent may differ from
best.pt; no optimizer state or resume support. P1's runner is reused through
an optional model factory whose default is unchanged.

## Run from the repository root (Python 3.12, P0 simulator environment)

    python -m pip install -r Co_Kaggle/g8/rl/p0/requirements-simulator.txt
    python -m pip install -r Co_Kaggle/g8/rl/p1/requirements.txt
    python -m unittest Co_Kaggle.g8.rl.p2.test_p2 -v
    python -m Co_Kaggle.g8.rl.p2.physics_smoke

Short end-to-end training:

    python -m Co_Kaggle.g8.rl.p2.train --out Co_Kaggle/g8/rl/p2/results/smoke --updates 1 --steps 24 --workers 4

Initial comparable-horizon experiment:

    python -m Co_Kaggle.g8.rl.p2.train --out Co_Kaggle/g8/rl/p2/results/run1 --updates 20 --steps 864 --workers 4 --device cpu

Four train scenarios per update by default (one per train building).
Use --scenarios-per-building 5 for the entire train set. Parent and two
candidates require 12 scenario rollouts per update by default, versus
P1's four. For a comparable training-rollout budget to P1's 20 updates,
start with six P2 updates. Validation adds separate rollout overhead.
CPU workers parallelize simulator episodes; GPU is not used. Each
history entry records paired per-scenario losses and components,
profile utilization, acceptance and simulator steps. Train losses
across different sampled scenarios are not directly comparable.

Independent test after checkpoint selection:

    python -m Co_Kaggle.g8.rl.p2.train --checkpoint Co_Kaggle/g8/rl/p2/results/run1/best.pt --out Co_Kaggle/g8/rl/p2/results/run1/test --steps 864 --workers 4

Add --full for full-length evaluation. Short-window results are diagnostic,
not leaderboard scores. No trained checkpoint or Kaggle submission is
included. Success criterion: improve paired V23 validation loss, inspect
per-climate comfort/energy/carbon, then verify on independent full test.
The first local diagnostic target is .010; no score is claimed until measured.
