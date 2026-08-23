# Kaggriculture local-arena fidelity audit

Audit date: 2026-08-22.

## Conclusion

The arena is consistent with Kaggle's published rules for **individual game
mechanics and terminal win/tie/loss outcomes** when it reports
`engine_verified=true` and `official_episode_length=true`. Here,
`engine_verified` means that the installed public package and material defaults
match this repository's audited baseline; it does not claim binary identity
with Kaggle's private production runner. The arena is not an exact reproduction
of the live ladder, whose opponent selection, continuing skill updates, and
final Bradley-Terry fit depend on private/live competition state.

The promotion signal is therefore head-to-head win rate on fresh seeds. Coin
margin is useful for robustness, but it is not part of Kaggle's rating update.

## Rule-by-rule comparison

| Official behavior | Local implementation | Finding |
| --- | --- | --- |
| Two agents play the advanced `kaggriculture` environment for 720 turns. | `make("kaggriculture")`; default `--episode-steps 720`; two agents per job. | Consistent. |
| Each farm starts with the documented 10x10 board, 3,000 coins, one unlocked quadrant, 24 turns/day, 100-item shed, and the documented town/market defaults. | The runner supplies only `episodeSteps` and a deterministic test `seed`, leaving game defaults intact. It audits the material defaults and verifies that the engine resolves the requested hidden seed before ranking. | Consistent with the pinned public engine. |
| The winner is the player with the most banked coins at the end; unsold inventory is not counted. | The runner reads the official engine's terminal `reward` and compares the two rewards. | Consistent. |
| Kaggle changes skill rating from win/loss/tie only; coin-difference magnitude does not affect the rating. | Zero agent errors rank first, then `points_rate` (win = 1, tie = 0.5); coin margin is reported but not used to rank. Agent-side failures are forfeits rather than dropped games. | Consistent outcome signal; this is not a skill-rating replica. |
| Kaggle runs a self-play validation episode before ladder admission. | `--self-play` runs the exact file-loader source against itself; self-play is excluded from ranking. | Consistent when the option is used. |
| Submission code exposes root-level `main.py` and an agent callable; Kaggle executes it through its agent loader. | Notebook `%%writefile main.py` content is extracted into a unique directory as `main.py` and passed to `env.run`. Archives/directories with extra files are rejected. | Consistent for the single-file notebooks under test; multi-file bundles are unsupported. |
| The official starter installs `kaggle-environments>=1.32.2`; Kaggle does not publicly pin the production runner build in the reviewed material. | `requirements.txt` pins public release `1.32.7`, the latest official PyPI release on the audit date. The runner refuses an unexpected version/default configuration unless explicitly overridden and records that production identity is unconfirmed. | Reproducible best-available public baseline, not proof of server binary equivalence. |
| Matchmaking uses similarly rated live opponents and the final leaderboard uses a Bradley-Terry tournament over collected episodes. | The arena uses a chosen local opponent set, equal game counts, fixed fresh seeds, and both seat orientations. It does not fit or claim a Kaggle skill rating. | Deliberate, unavoidable limitation. |
| Kaggle's agent container has platform resource and submission-size limits. | The official loader timeouts remain active, but local CPU/RAM scheduling and concurrent `--jobs` execution are host-dependent. The proof separately checks the single-file archive size. | Partial; use `--jobs 1` for the most conservative timing check. |

## Verification sources

- [Official competition evaluation](https://www.kaggle.com/competitions/kaggriculture/overview/evaluation)
- [Official competition rules](https://www.kaggle.com/competitions/kaggriculture/rules)
- [Official Kaggriculture environment guide](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/AGENTS.md)
- [Official game mechanics and configuration](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/README.md)
- [Official engine implementation](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/kaggriculture.py)
- [PyPI release history for `kaggle-environments`](https://pypi.org/project/kaggle-environments/)

## Promotion protocol

1. Keep v4, v5, and the candidate fixed before choosing proof seeds.
2. Run at least 20 fresh seeds, both seats, at 720 turns with the audited public
   engine baseline and zero non-`DONE` games.
3. Require the candidate to beat both predecessors on win rate. Use paired
   margin and worst game only to judge whether the lead is fragile.
4. Run self-play through the file loader and verify the generated archive has a
   root-level `main.py` and remains below the published agent size limit.
