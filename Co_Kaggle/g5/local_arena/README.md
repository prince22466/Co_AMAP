# Kaggriculture round-robin arena

This runner evaluates the exact `main.py` embedded in submission notebooks with
the official Kaggle environment. By default it discovers every
`submission_nb/kaggriculture-sub_v*.ipynb` notebook (currently v1-v5),
runs every unordered pairing from both player seats on every seed, and writes:

- `matches.csv`: one row per game;
- `pairings.csv`: seat-balanced, matched-seed head-to-head summaries;
- `leaderboard.csv`: aggregate wins, rewards, margins, and error counts;
- `results.json`: metadata plus all three result tables.

It never modifies the source notebooks. Agents are extracted into a temporary
directory with unique filenames so every match exercises the file-loader path.

## Setup

```powershell
python -m pip install -r local_arena/requirements.txt
```

## Validate source extraction

This does not require `kaggle-environments`:

```powershell
python local_arena/arena.py --validate-only
```

## Run all saved versions

The quick default uses two seeds. With v1-v5 it runs 40 games: every unordered
pairing, both seats, and both seeds.

```powershell
python local_arena/arena.py --jobs 2
```

## Add a candidate

The candidate can be an `.ipynb`, `.py`, `submission.tar.gz`, or a directory
containing `main.py`:

```powershell
python local_arena/arena.py `
  --candidate submission_nb/kaggriculture-sub_v6.ipynb `
  --candidate-name v6 `
  --seeds 3101,3102,3103,3104,3105 `
  --jobs 2 `
  --output local_arena/results_v6
```

Additional agents can be supplied repeatedly:

```powershell
python local_arena/arena.py `
  --agent experimental=path/to/main.py `
  --agent archive=path/to/submission.tar.gz
```

Use `--no-defaults` with at least two `--agent NAME=PATH` arguments for a custom
arena. `--self-play` adds one validation game per agent and seed. Full
competition-length games use the default `--episode-steps 720`.

For promotion decisions, prioritize zero errors and positive seat-balanced
`mean_paired_seed_margin` over raw mean reward. Use at least 10-20 seeds for a
serious comparison; two seeds are only a smoke test.
