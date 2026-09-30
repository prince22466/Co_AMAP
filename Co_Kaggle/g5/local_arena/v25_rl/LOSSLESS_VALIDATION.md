# First-loss scheduler validation

Deterministic replay on the same five v9 held-out histories, using the existing
v9 `latest.pt` actor at update 147. No PPO updates were performed. Both schedulers
were measured with the corrected reward extractor in this PR; recorded market
orders and opponent actions stayed frozen. The before scheduler is from main
commit `9397a30f7d0b9090d6ec6133101e62eb24bc0471`.

Environment: Python 3.12, `kaggle-environments==1.32.7`, CPU PyTorch 2.14.0,
one PyTorch thread per episode. Checkpoint SHA-256:
`773e6e43b3b945af533d88c195635291339b06d88f3399e6b1546e542c34407b`.

| History | Weeds before / after | Lost units before / after | Escapes before / after | Crop harvest before / after | Delivery before / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 111551968 | 20 / 0 | 70 / 0 | 1 / 0 | 493 / 495 | 407 / 378 |
| 111558730 | 15 / 0 | 26 / 0 | 0 / 0 | 548 / 454 | 448 / 368 |
| 111565456 | 24 / 0 | 38 / 0 | 0 / 0 | 401 / 359 | 388 / 371 |
| 111587965 | 15 / 0 | 8 / 0 | 0 / 0 | 444 / 400 | 420 / 416 |
| 111592475 | 22 / 0 | 59 / 0 | 0 / 0 | 408 / 416 | 380 / 381 |
| **Mean** | **19.2 / 0** | **40.2 / 0** | **0.2 / 0** | **458.8 / 424.8** | **408.6 / 382.8** |

All ten replays finished successfully. The after scheduler also had zero crop
deaths and zero gradual decay units. Mean planting decreased from 202.2 to 187;
animal-product harvest increased from 249.2 to 259.6. Mean worker reward under
the corrected accounting increased from 37,719.85 to 44,327.33.

The zero-loss result is not achieved by stopping production, but crop harvest
is 7.4% lower and delivery is 6.3% lower. The unchanged actor was trained with
the old scheduler; a subsequent training run should track these throughput
metrics as well as losses. These five cases establish a regression baseline,
not a guarantee for unseen histories, stochastic rollouts, or future PPO
checkpoints. The admission guard uses a bounded greedy route estimate, not an
optimal scheduling proof or guaranteed future feed supply.

The historical v9 log reported 19.4 weeds and 12.8 lost units per episode.
Those numbers use different accounting: they miss gradual decay and include
a random weed spawned immediately after a successful one-time harvest. The
paired table above applies the same corrected measurement to both schedulers.
For ongoing crops, harvest-at-expiry can still create a real zero-yield weed;
that failure remains counted.

To reproduce the after baseline from `v25_rl`, with the full original history
directory and the unchanged v9 update-147 checkpoint present:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_v25_worker_ppo.py \
  --device cpu \
  --history-dir ../../game_history/v20 \
  --resume runs/worker_ppo_static_v20_v9_animal_survival/checkpoints/latest.pt \
  --output-dir runs/worker_ppo_static_v20_v10_preflight \
  --split-seed 20260928 \
  --updates 148 \
  --max-training-hours 0
```

This checks recorded-action parity, resets stale critic/optimizer state on the
contract change, evaluates a fresh deterministic baseline, and performs no
training updates. The generated `validation_episodes.jsonl` contains all five
episode breakdowns. `best.pt` retains main's reward-based selection rule.

Regression checks:

```bash
python -m unittest discover -p 'test_*.py'
```

The 95 tests cover scheduler pressure and specialist assignment, planting
admission, emergency field wheat for feeding, ongoing-crop retirement, complete
decay accounting, random-weed attribution, checkpoint compatibility, and PPO
subdecision ratios. Five boundary tests use the real Kaggriculture engine.
