# v13: stronger and correctly attributed useful planting credit

## Problem and resulting behavior

v12's +16 raw completion credit was divided by rollout GAE standard deviations
of 417–528, leaving only 0.030–0.038 normalized advantage units per useful plant.
The uploaded run's planting-delay penalty fired zero times. u170 improved reward
by 0.28% and achieved zero validation weeds/lost units/escapes, but still created
32.8 unproductive late plants per game versus 30 at its starting baseline.

Add +0.25 directly to the originating worker choice's normalized actor advantage
after a successful, useful planned PLANT. Retain the +16 raw credit, so effective
completion preference is about 0.28–0.29 at the observed scales. This coefficient
is a modest experiment, not a proven optimum or a promised performance gain.

Also fix attribution: the old code credited the first N planting origins from a
single successful-completion count. A turn containing a late MELON and useful
WHEAT could therefore reward the MELON choice and omit the WHEAT choice.
`planned_plants_completed_by_worker` now identifies the exact worker confirmed
by the reward transition. Both bonuses go to that worker's originating PLANT
decision, including when its route started on an earlier turn.

Failed, interrupted and unproductive late planting receive no completion bonus.
The additional credit does not enter turn reward, critic returns or unrelated
worker choices. There is no new candidate mask or forced action. Existing seed,
watering, maintenance, harvest, animal-survival and animal-placement guards apply.
The delay penalty's maintenance/output exemptions remain; zero activations alone
does not establish that the worker could safely have planted instead.

## Verification and observable metrics

- 129 unit/engine tests pass, including mixed useful/late completions, local and
  remote origin credit, failed/interrupted/late planting, no duplicate credit,
  no unrelated-worker/critic credit, and GAE scales of 400, 1000 and 4000.
- `git diff --check` passes.
- Actual v12 u170 checkpoint resumes at u171 with identical actor parameters.
- A full stochastic 719-turn replay and one PPO epoch check that all confirmed
  useful plantings have exactly one matched originating choice, each with +0.25,
  finite losses/parameters and the expected event metrics.

Training metrics now include `plant_completion_actor_events`,
`plant_completion_actor_bonus_mean` (over all sampled actor choices), and
`plant_completion_actor_bonus_per_event` (0.25 whenever there are events).
Per-worker completion counters are recorded in episode breakdowns.

The output directory defaults to `worker_ppo_static_v20_v13_plant_signal`.
The new contract triggers the existing critic/optimizer/best-score reset while
preserving the actor and update number. A fresh deterministic validation baseline
is established. The additional shaping is actor-only, so validation worker reward
with unchanged weights stays comparable to v12. Any learned improvement needs
validation after training; blocked seed-backed plans may still need scheduling
or capacity diagnosis.

## Resume (PowerShell)

```powershell
cd E:\Projects\Co_AMAP
git fetch origin
git switch codex/v25-plant-signal
cd Co_Kaggle\g5\local_arena\v25_rl
python train_v25_worker_ppo.py `
  --resume runs/worker_ppo_static_v20_v12_productive_planting/checkpoints/best.pt `
  --output-dir runs/worker_ppo_static_v20_v13_plant_signal `
  --updates 500 `
  --episodes-per-update 4 `
  --max-training-hours 2 `
  --device cpu
```

The time limit starts after baseline validation and is checked between updates.
Use a fresh output directory. Judge progress by useful planting, seed-backed
empty tile-time, actual delivery and loss counts alongside reward. Raising
completion preference can crowd ordinary work despite admission guards; inspect
animal output and maintenance as well as crop utilization.
