# v12: useful planting rewards and land-use measurement

The v11 worker reward could credit planting and movement toward a crop that
cannot produce a deliverable harvest before the game ends. It also penalized
deferring planting for maintenance or delivery. Empty tiles were absent from
training logs, hiding both planting delays and missing seed supply.

## Changes

- Increase engine-confirmed, useful planned PLANT actor credit from 8 to 16 raw
  reward-equivalent units. It uses the existing GAE normalization and route-origin
  attribution; the bonus does not enter the critic or unrelated worker choices.
- Gate both successful-plant turn credit and planned completion credit on time
  to mature, travel from a shed, HARVEST, return, and PLACE before day 29 hour 23.
  Use engine first-yield ages, including early partial WHEAT/CARROT harvests.
  Movement toward an unproductive late planting gets no route-progress reward.
- Keep existing observed-seed, same-day watering and shared today/tomorrow
  maintenance admission. Keep the existing candidate set and scheduling: the
  horizon changes learning credit, rather than immediately forcing different
  actions with unchanged weights. Late planting can still occur until PPO learns.
- Apply the existing -0.75 normalized actor deferral penalty only when the same
  worker has a feasible, useful, unreserved planned PLANT alternative and chooses
  optional work. WATER, FEED, CARE, HARVEST, DELIVER, urgent crop retirement and
  planned animal setup are exempt. Required wheat supply remains exempt.
- Increase each animal escape cost from -100 to -512: four animal-product
  lifecycle values. This is an experimental coefficient, not a guarantee of zero
  escapes or an empirically optimal setting. Production/delivery rewards remain
  8 per crop unit and 128 per animal-product unit across their lifecycle.
- Log occupancy every pre-action worker turn, including crop/animal-reserved
  empty tiles, seed-backed and unallocated-seed crop empties, productive crop
  potential, unproductive plant creations, and avoidable planting delays.
- Update the reward contract and default output to
  `runs/worker_ppo_static_v20_v12_productive_planting`. Resume keeps actor weights
  and update numbering, resetting the critic, optimizer moments and best score.
  Best selection still maximizes validation worker reward under the new contract.

## Measurement definitions

`*_tile_turns` sum tile counts over pre-action observations, excluding `LOCKED`.
Crop-eligible means outside `ANIMAL_POINTS`. An empty structure is occupied land,
not an empty tile. Seed-backed means an empty crop tile occurs in `crop_plan`;
unseeded means no seed is allocated to that empty tile by that plan. A seed-backed
plan can still be blocked by capacity or time. These counters do not penalize
unavoidable seed shortages, reserved animal sites, or required idle workers.

`productive_crop_tile_turns` is potential occupancy: held yield that can mature
before the game ends, or a remaining yield event in the executor's crop schedule
before the end. It excludes weeds and exhausted zero-yield crops. It is a proxy,
not proof that a harvest will be maintained, collected or delivered. Evaluate it
alongside actual delivered units and losses.

`owned_empty_tile_fraction`, `crop_empty_tile_fraction`, and
`productive_crop_tile_fraction` divide pooled tile-turn totals, not unweighted
per-episode percentages. `midgame_*` fractions use days 5 through 24 inclusive,
matching the prior empty-tile diagnosis. Per-episode counters appear in
`episodes.jsonl` and `validation_episodes.jsonl`; aggregate means and fractions
appear in `metrics.jsonl` and `validation.jsonl`. Console output includes
`empty_crop_avg` and `plant_delays`.

## Validation

Source: main `72539802bda1997a5e7f9849434e8cb1bc61180f`. Checkpoint: v11 `best.pt`
at u165, Git blob `58019e8929b195ed3490f2b05c1cf36bbedb92ce`. Paired deterministic
Linux CPU replays used unchanged actor weights, hidden 128, torch threads 1,
Python 3.12 and kaggle-environments 1.32.7. Windows uploaded validations may
follow different numerical trajectories; compare within the same runtime.

| Metric, mean per episode | v11 | v12 |
| --- | ---: | ---: |
| Worker reward under each contract | 43,900.21 | 43,864.95 |
| Animals placed | 9.0 | 9.0 |
| Animal units delivered | 252.2 | 252.2 |
| Crop units harvested | 472.8 | 472.8 |
| Total units delivered | 417.2 | 417.2 |
| Animal escapes | 0 | 0 |
| Crop weeds | 0.2 | 0.2 |
| Lost harvestable units | 1.0 | 1.0 |
| PASS actions | 537.2 | 537.2 |
| Avoidable PASS | 0 | 0 |

The final change preserves these output/loss measurements with unchanged
weights. The small reward decrease removes credit for late planting; it is not
an output regression. Reward totals across contracts are not directly comparable.
v12 recorded 31.2 unproductive new plants per episode: a concrete training target,
not an improvement already achieved. Midgame empty crop tile-time was 27.61%.
The same actor has not learned the new rewards yet.

| History | v12 reward | Escapes | Crop weeds | Lost units |
| --- | ---: | ---: | ---: | ---: |
| 111551968 | 41,782.60 | 0 | 1 | 5 |
| 111558730 | 44,445.15 | 0 | 0 | 0 |
| 111565456 | 48,450.65 | 0 | 0 | 0 |
| 111587965 | 39,858.85 | 0 | 0 | 0 |
| 111592475 | 44,787.50 | 0 | 0 | 0 |

122 unit/engine tests pass. Tests cover last viable planting days, delivery travel,
real-engine early WHEAT harvest legality, failed and late planting credit,
maintenance/output exemptions, actual optional-work deferral, route credit,
locked/animal-reserved tiles, seed allocation, exhausted crops, pooled ratios,
metadata isolation, PPO attribution, and checkpoint resume. Actual u165 resume
was checked: starts at u166, actor parameters identical, fresh optimizer/best stats.

## Resume training (PowerShell)

```powershell
cd E:\Projects\Co_AMAP
git fetch origin
git switch codex/v25-productive-planting
cd Co_Kaggle\g5\local_arena\v25_rl
python train_v25_worker_ppo.py `
  --resume runs/worker_ppo_static_v20_v11_animal_placement/checkpoints/best.pt `
  --output-dir runs/worker_ppo_static_v20_v12_productive_planting `
  --updates 500 `
  --episodes-per-update 4 `
  --max-training-hours 2 `
  --device cpu
```

The time limit is checked between updates and starts after initial validation;
baseline validation and the final in-progress update add wall time. For a strict
two-hour wall-clock window, use `--max-training-hours 1.5` to leave headroom.
Keep a fresh v12 output directory so the two contracts' logs stay distinct.

Judge the next run against its fresh v12 baseline: more delivered output, fewer
seed-backed empty tile-turns and unproductive new plants, with weeds, lost units
and escapes approaching zero. Total empty tiles may remain when seed procurement
is inadequate: market orders are still frozen v20 replay inputs and worker PPO
cannot change them. A jump to 60k is not promised by these coefficient changes.
