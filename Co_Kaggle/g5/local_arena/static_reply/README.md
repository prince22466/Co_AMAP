# Static replay runner

From the project root:

```powershell
python local_arena/static_reply/reply_template.py v20_bench.py
```

Or from this directory:

```powershell
python reply_template.py v20_bench.py
```

The agent must define `agent(obs)` or `agent(obs, config)` and return the usual
`farmer` / `hands` / `market` action dictionary. Names resolve from the current
directory first, then beside the runner. Full Python file paths also work.
Install any extra dependencies required by your agent separately.

By default every JSON in `game_history/v20` is replayed. The candidate replaces
the losing seat, as this corpus contains v20 losses. The opponent's commands
are read verbatim. Both players' states and rewards are recomputed by the engine;
the opponent does not adapt.

```powershell
# One game with full recorded-action parity checking (the default)
python local_arena/static_reply/reply_template.py v20_bench.py --limit 1

# Specific games and an explicit new output directory
python local_arena/static_reply/reply_template.py v20_bench.py --episodes 111548564 111549675 --output local_arena/static_reply/runs/my_run
```

Results go to `runs/<agent>/<UTC timestamp>/` by default:

- `histories/<source episode ID>.json`: full replay from official `env.toJSON()`,
  with all steps, actions, observations, rewards, statuses, configuration,
  specification, versions, and seed. IDs and team labels describe the new local
  run; original metadata is under `info.static_replay.source_info`.
- `summary.json`: wins/losses/ties, errors, seats, original/new rewards, margins,
  replay paths, timing, and source-agent SHA256.

Before running each candidate, the runner verifies that both original action
streams reproduce every saved observation/reward/status. A mismatch is an
error, never a scored game. `--skip-parity` disables that check explicitly.
Use `--seat 0` or `--seat 1` for a corpus where the intended seat is not the loser;
ties require an explicit seat.

Outputs must be separate from inputs and cannot overwrite an earlier run.
Agent failures are reported and cause a nonzero exit code; only complete games
produce history files. Agent code runs locally in-process with fresh main-module
globals each game (imported helper modules may be cached). Competition timeouts
are not enforced; maximum observed action time is reported. The runner contains
no training, optimizer, or model implementation.
