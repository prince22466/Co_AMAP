# Shared RL infrastructure

Both P1 and P2 depend on this package; neither should import the other's
implementation. Future profile-based models can supply their own model factory
and training algorithm.

- profiles.py: V23 operating profile definitions, controller adapter, features.
- rollout.py: fresh simulator environment, ordered process-based dispatch,
  independent baseline actions, raw reward/component accounting, optional
  stochastic macro-step trajectory collection, model snapshots.
- evaluation.py: building-disjoint split, paired raw loss, initial baseline
  parity checks, atomic JSON reporting.
- requirements.txt: shared PyTorch dependency; simulator dependencies remain in P0.

The rollout task tuple is (data_root, scenario, steps, interval, weights, mode,
seed). Supply a top-level, picklable model factory to rollout; its model returns
(profile logits, scalar value). Deterministic eval and independent baseline
modes return raw loss, components and profile histogram. Train mode additionally
returns raw macro-step trajectory and bootstrap, with explicit gamma and
reward_scale; P1 computes GAE outside this package. Dispatch requires a
picklable worker; it preserves input order and per-scenario seeds.

P0 remains the underlying official simulator benchmark, controller loader and
reward metric implementation. It does not depend on common or PyTorch.
P1's legacy imports, CLI, checkpoint signature and tensor keys remain compatible.
