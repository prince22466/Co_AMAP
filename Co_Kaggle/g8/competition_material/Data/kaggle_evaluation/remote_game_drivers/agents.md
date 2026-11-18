# Remote Game Drivers — agents.md

Scoped to this package. For repo-wide setup, test commands, code standards, and the
gateway/inference-server security model, see the top-level [`AGENTS.md`](../../AGENTS.md);
this file covers only what's specific to the drivers.

Bridge library that runs whole game-environment libraries (Gymnasium, PettingZoo,
OpenSpiel) on Kaggle's competition infrastructure, with the game engine and each
agent in **separate containers**. One driver per library, not per game — the point
is to scale supported games by porting libraries. Agents and engines talk over gRPC
via the Hearth relay; writing an agent should feel like writing one against the
native library.

Design doc: [go/kaggle-game-driver-design-proposal](https://docs.google.com/document/d/1-FjYCMz7VfJ6EuhNGRoKJ_xxlYjVB-qiUe2aC0pRE-I/) ·
Handoff notes: [doc](https://docs.google.com/document/d/1B0iZQWUTNGRF1KWUDZnIu0yDJ7DWgJypKA1hwD21qgk/)

## Architecture

```
Agent ↔ RemoteAgent ↔ Host (Hearth gateway or kaggle-environments) ↔ GameDriver ↔ Game library ↔ Game logic
```

- **Gateway container** runs the `GameDriver`: owns game state, drives the loop,
  pushes turn data to agents over relay.
- **Inference server container** runs the `RemoteAgent` (competitor code): receives
  state, returns an action.

Serialization goes through the Hearth relay, which natively handles primitives,
enums, dataclasses, and Pydantic models. Only library-specific types (e.g.
`gymnasium.spaces.Space`) need an explicit `Serializer`. This dependency on the
relay is a deliberate deviation from the original loosely-coupled design — see the
handoff notes.

## Layout

```
core/base_classes.py     # BaseGameDriver, BaseRemoteAgent, BaseSerializer — start here
gymnasium_remote_driver/ # single-agent; serialization.py handles spaces ↔ Pydantic
pettingzoo_remote_driver/# multi-agent; AECGameDriver (sequential) + ParallelGameDriver
shimmy_remote_driver/    # OpenSpiel via Shimmy; ~60 2-player games (chess, go, poker, …)
hearth_integrations/     # gateway + inference-server templates per library
```
PettingZoo and Shimmy drivers extend the Gymnasium driver and share its
serialization and agent interface.

## Core abstractions

Read the docstrings in `core/base_classes.py` for exact signatures; the essentials:

- **`BaseGameDriver[GameData, Action]`** (gateway): implement `start_new_game`,
  `step`, `get_agent_action`, `get_still_playing`, `get_results`, `run_game`.
  `get_full_state` / `render` are optional. Configured via `driver_config`
  (`agent_ids`, `agent_config`, `game_config`), mirroring kaggle-environments args.
  Every subclass is auto-wrapped with OpenTelemetry spans (opt-in, off by default).
- **`BaseRemoteAgent`** (agent): competitors implement only `choose_action(action_space,
  observation, info) -> action | (action, reasoning)`. `process_turn` is the relay
  endpoint that de/serializes around it — a pass-through when the library only uses
  relay-native types.
- **`BaseSerializer`**: converts library types to/from relay-safe formats. Return
  relay-native types unchanged; JSON-encode only what the relay can't handle.

## Writing an agent

```python
from remote_game_drivers.gymnasium_remote_driver.remote_agent import RemoteAgent
from remote_game_drivers.hearth_integrations.gymnasium.gym_inference_server import GymInferenceServer


class MyAgent(RemoteAgent):
    def choose_action(self, action_space, observation, info=None):
        return action_space.sample()  # random baseline; or return (action, reasoning)


GymInferenceServer(agent_class=MyAgent).serve()
```

## Adding a game library

1. Create `<library>_remote_driver/` with `GameDriver(BaseGameDriver)` and
   `RemoteAgent(BaseRemoteAgent)`.
2. Add a `Serializer(BaseSerializer)` only if the library uses non-relay-native types.
3. Add `hearth_integrations/<library>/` templates if applicable.
4. Add the library to the dependency group in `pyproject.toml`.

Tests live in `tests/remote_game_drivers/`; run them with `./test_in_docker.sh -t
remote_game_drivers` (see the top-level AGENTS.md for details).
