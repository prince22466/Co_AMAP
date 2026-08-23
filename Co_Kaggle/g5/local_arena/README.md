# Kaggriculture round-robin arena

This runner evaluates the exact `main.py` embedded in submission notebooks with
an audited, pinned release of Kaggle's public environment package. By default it discovers every
`submission_nb/kaggriculture-sub_v*.ipynb` notebook (currently v1-v6),
runs every unordered pairing from both player seats on every seed, and writes:

- `matches.csv`: one row per game;
- `pairings.csv`: seat-balanced, matched-seed head-to-head summaries;
- `leaderboard.csv`: aggregate wins, rewards, margins, and error counts;
- `results.json`: metadata plus all three result tables.

It never modifies the source notebooks. Each agent is extracted into an isolated
temporary directory as a root-level `main.py`, so every match exercises the
file-loader path with Kaggle's required filename.

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

The quick default uses two seeds. With v1-v6 it runs 60 games: every unordered
pairing, both seats, and both seeds.

```powershell
python local_arena/arena.py --jobs 2
```

## Add a candidate

The candidate can be an `.ipynb`, `.py`, `submission.tar.gz`, or a directory
containing `main.py`. Archives and directories are deliberately limited to
single-file submissions; the arena rejects extra files instead of silently
testing an incomplete bundle:

```powershell
python local_arena/arena.py `
  --candidate path/to/experimental-agent.ipynb `
  --candidate-name experimental `
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
competition-length games use the default `--episode-steps 720`. The runner also
refuses to rank games when the installed engine version or documented game
defaults differ from the audited public baseline, unless
`--allow-unverified-engine` is supplied explicitly.

For promotion decisions, prioritize zero errors and head-to-head win/tie/loss
(`points_rate`), because Kaggle's rating ignores coin-margin magnitude. Use
seat-balanced `mean_paired_seed_margin` only as a robustness diagnostic. An
agent-side failure counts as a forfeit, and zero-error agents rank ahead of
erroring agents. Run
`--self-play` at least once to mirror Kaggle's validation episode, and use at
least 10-20 fresh seeds for a serious comparison; two seeds are only a smoke
test.

## Fidelity boundary

The runner is mechanics-faithful to the audited public package: it uses the
pinned release, default 10x10/3000-coin configuration, 720 turns, the real file
loader, the official terminal bank reward, checked/resolved engine seeds, and both
seats. It does not establish that Kaggle's private production runner uses the
same package build, nor reproduce the live opponent pool, similar-rating
matchmaking, continuing skill updates, or final Bradley-Terry fit. Multi-file
bundles and production CPU/RAM limits are also outside its fidelity boundary.
Accordingly, local win rate is a promotion test—not a guaranteed leaderboard score. See
[`AUDIT.md`](AUDIT.md) for the rule-by-rule comparison.
