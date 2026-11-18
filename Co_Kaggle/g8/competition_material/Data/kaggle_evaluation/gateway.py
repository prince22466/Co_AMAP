"""Gateway for the Smart Buildings HVAC control competition (ID 157303).

The competitor supplies an HVAC control policy. This gateway runs that policy against
a set of held-out building/weather scenarios in the `smart_buildings` simulator and
reports how each scenario went.

Scoring
-------
`submission.parquet` holds one row per scenario, carrying the mean reward the simulator
gave over that scenario's episode and nothing else. The leaderboard metric is the stock
scikit-learn mean absolute error against a solution whose target is 0.0 on every row. The
simulator's reward lies in [-1, 0], so the score reduces to the mean per-step cost the
policy incurred: 0 for a policy that is simultaneously perfectly comfortable and free, 1
for the worst possible policy, lower is better.

One row per scenario rather than one per step, even though the metric would read either.
The solution file is built by the prep pipeline before any gateway run exists, so its row
IDs are a prediction of what this file will contain, and a prediction of the scenario list
is a directory listing while a prediction of the step count is arithmetic over `config.gin`
that an episode ending a step early would silently invalidate. Averaging here also makes
every scenario weigh the same regardless of episode length, which per-step rows would only
do for as long as every episode happened to be the same length.

Protocol
--------
The standard gymnasium one: a single `process_turn(action_space, observation, info)`
endpoint, driven by `SmartBuildingsDriver` below, which is `HearthGymDriver` with the
simulator's loading, validation and recording needs layered on.

An episode is 8640 steps (30 days at a 300 s time step) and a scenario's static
metadata is a few MiB, so what crosses the wire per step matters. It stays small: the
metadata rides gym's info dict, which the simulator populates only on `reset()`, so a
policy receives it on the first turn of its scenario and nothing but a ~1.2 KiB
observation and a 319 byte action space thereafter.

`action_space` travels as a real `gymnasium.spaces.Dict` via the shared
`GymSerializer`, so the policy receives the same subscriptable action spec the starter
kit hands it locally.

Read-only competition data
--------------------------
The simulator is mounted read-only. Two things in the upstream code would otherwise
write into that tree, and both are neutralized in `_load_environment`:
  * `Environment.metrics_path` -- per-episode metric dumps written next to the
    scenario. `evaluate.py` disables this correctly, but `run_environment.py` and the
    RL collection scripts assign `env._metrics_path`, which is a typo for the real
    attribute and therefore disables nothing. Copy from `evaluate.py`, not from those.
  * numba's `cache=True` kernels and matplotlib's font cache, redirected to a
    scratch directory by `_redirect_caches_to_scratch()` before the simulator import.

Scenario `config.gin` files reference their assets by paths relative to the simulator
root, so environments are constructed with the working directory temporarily moved
there. It is restored before any submission file is written.
"""

import contextlib
import math
import os
import pathlib
import sys
import tempfile
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl

import kaggle_evaluation.core.templates
from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType
from remote_game_drivers.gymnasium_remote_driver.game_driver import GymGameData
from remote_game_drivers.hearth_integrations.gymnasium.gym_gateway import HearthGymDriver

# A scenario is a directory holding a config.gin plus its floor plan and zone map.
CONFIG_FILENAME = 'config.gin'

# Key the scenario's static metadata is delivered under, in gym's info dict. Kept in step
# with the constant of the same name in smart_buildings_inference_server.py: the two run in
# separate containers and share only what crosses the relay.
BUILDING_META_DATA = 'building_meta_data'

# Every scenario the host mounts is run, and there is deliberately no wall-clock check
# here: the rerun allowance is enforced outside this container, so a check inside it could
# only end the run early, spending the competitor's remaining time on nothing. Likewise no
# cap on the scenario count, which would let a run report success having quietly skipped
# scenarios.

# A scenario's first turn is where a policy is constructed, which may involve loading
# model weights, so it gets a far longer allowance than an individual turn.
FIRST_TURN_TIMEOUT_SECONDS = 15 * 60
PROCESS_TURN_TIMEOUT_SECONDS = 60

# What identifies a submission row, as opposed to what is being reported about it. Passed
# to write_submission as the row IDs, which is what puts it first in the output.
#
# `row_id` is the scenario name, e.g. `building_0003/phoenix_jul`, which is unique because it is
# the key `_discover_scenarios` builds its map on. Named generically rather than `scenario_id`
# because it is what scoring aligns on, and because it is the only identifying column the file
# can carry: mean absolute error differences every column it is handed except this one.
ROW_ID_COLUMN = 'row_id'

# The mean over the episode of the simulator's per-step reward, `RewardResponse.agent_reward_value`,
# itself a weighted average of three normalized cost terms:
#
#   reward = (regret * w_p - energy * w_e - carbon * w_c) / (w_p + w_e + w_c)
#
# All three are normalized, so the reward lies in [-1, 0] and so does its mean. See
# SetpointEnergyCarbonRegretFunction.compute_reward in the starter kit.
REWARD_COLUMN = 'reward'

# One row per scenario. Everything else the simulator's `RewardInfo` and `RewardResponse`
# carry either feeds this number, is a scenario constant recoverable from `config.gin`, or
# is physical state the metric does not read.
SUBMISSION_SCHEMA: dict[str, Any] = {ROW_ID_COLUMN: pl.String, REWARD_COLUMN: pl.Float32}

# What the simulator scores a step at when the building refuses the action, instead of
# raising: `environment.ACTION_REJECTION_REWARD`, mirrored here because the simulator is not
# importable until a scenario has put the data mount on sys.path.
ACTION_REJECTION_REWARD: float = -math.inf


# Set by the prep pipeline to truncate every episode so that a draft run of the whole pipeline
# finishes in minutes rather than the hours a full 48 scenario sweep takes. `emulate_hearth_on_kaggle`
# forwards `HEARTH_HOST_*` into the containers it starts; a real Kaggle rerun does not, so episodes
# there always run to the length their own config.gin specifies. It changes the scores a draft run
# produces, being a mean over a different number of steps, but not the shape of the file: the row
# per scenario is there either way, so the solution file does not have to know about this.
MAX_STEPS_ENV_VAR = 'HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS'


def _max_steps_per_scenario_from_env() -> int | None:
    raw = os.getenv(MAX_STEPS_ENV_VAR)
    if not raw:
        return None
    max_steps = int(raw)
    if max_steps < 1:
        # A host misconfiguration rather than anything the competitor did, but the gateway has no
        # other way to report from here and silently accepting it would emit an empty submission.
        raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'{MAX_STEPS_ENV_VAR} must be a positive integer, got {raw!r}')
    print(f'[gateway] {MAX_STEPS_ENV_VAR}={max_steps}: this is a truncated smoke test, not a full run.', flush=True)
    return max_steps


def _redirect_caches_to_scratch() -> None:
    """Point libraries that cache to disk at a writable scratch directory.

    numba compiles the simulator's hot loops with `cache=True` and matplotlib builds a
    font cache on import. Both default to writing inside the package tree, which is a
    read-only mount here.
    """
    scratch = pathlib.Path(tempfile.gettempdir()) / 'smart_buildings_cache'
    for var, subdir in (('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib')):
        path = scratch / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(var, str(path))


@contextlib.contextmanager
def _working_directory(path: pathlib.Path) -> Iterator[None]:
    """Temporarily move the working directory, restoring it even on failure."""
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class SmartBuildingsDriver(HearthGymDriver):
    """Runs one scenario at a time through the shared gymnasium game loop.

    `HearthGymDriver` already supplies everything generic: the turn loop, serializing the
    action space, calling `process_turn` over the gateway's relay client, accumulating
    reward and closing the environment. Layered on here is what the simulator needs --
    building an environment from a scenario directory rather than `gym.make`, delivering
    the building metadata, attributing an action the simulator rejects, and counting the
    steps the episode ran for.

    One instance runs every scenario in turn; `start_new_game` resets all per-scenario state.
    """

    def __init__(self, gateway: 'SmartBuildingsGateway'):
        super().__init__(gateway)
        self.gateway = gateway

    def start_new_game(
        self,
        game_name: str,
        random_seed: int | None = None,
        game_state: str | None = None,
        action_timeout: int | None = None,
    ) -> None:
        """Build the environment for the scenario named by `game_name`.

        Scenarios are configured directories rather than registered gym ids, so this
        replaces the base class's `gym.make` outright.
        """
        self.game_name = game_name
        # Reset per scenario rather than accumulated across the run: it is the divisor that
        # turns this scenario's accumulated reward into a mean.
        self.step_index = 0
        self.env, building_meta_data = self.gateway._load_environment(self.gateway.scenarios[game_name])
        self._game_over = False
        self.total_reward = 0.0
        self.reward = 0.0
        self.observation, _info = self.env.reset()
        # The loop forwards this to the agent every turn, and the simulator's own info dict
        # is empty from here on, so the metadata is delivered exactly once per scenario. It
        # is a few MiB; resending it every step would add minutes to the scenario.
        self.info = {BUILDING_META_DATA: building_meta_data}
        self.gateway.set_response_timeout_seconds(FIRST_TURN_TIMEOUT_SECONDS)

    def step(self, action: Any) -> GymGameData | None:
        try:
            game_data = super().step(action)
        except ValueError as exception:
            # The simulator checks most of the action itself -- out of bounds, wrong shape and a
            # missing key all raise ValueError out of env.step -- so the gateway does not duplicate
            # those checks. It does have to attribute the failure: `step` is not on
            # `get_agent_action`'s path, so an unclassified error here would be reported as a
            # Kaggle-side fault rather than against the submission that caused it.
            #
            # NaN is the gap in the simulator's checks, and `find_non_action` covers it upstream in
            # `get_agent_action`. BoundedActionNormalizer.setpoint_value bounds-checks with `<` and
            # `>`, which a NaN fails both of, so a NaN setpoint reaches the building unrejected and
            # poisons the simulation and every reward term after it. A NaN in the discrete slot is
            # caught, but as a NotImplementedError that this handler would not have classified.
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION,
                f'{self.game_name} step {self.step_index}: the simulator rejected the action: {exception}',
            ) from None
        if self.reward == ACTION_REJECTION_REWARD:
            # A refused action is the simulator's other way of saying the action was invalid,
            # and the only one that does not raise: the building declined to apply it, the
            # step ran on the setpoints it already had, and the reward terms describe a
            # building the policy did not control. Recording those terms would be recording
            # fiction, so the run stops here rather than submitting them.
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION,
                f'{self.game_name} step {self.step_index}: the building refused the action.',
            )
        self.step_index += 1
        if self.step_index == 1:
            self.gateway.set_response_timeout_seconds(PROCESS_TURN_TIMEOUT_SECONDS)
        if self.gateway.max_steps_per_scenario is not None and self.step_index >= self.gateway.max_steps_per_scenario:
            self._game_over = True
            return None
        return game_data


class SmartBuildingsGateway(kaggle_evaluation.core.templates.GameGateway):
    """Runs a competitor's policy across every held-out scenario in the data folder."""

    def __init__(
        self,
        competition_data_folder: str | pathlib.Path | None = None,
        file_share_dir: str | None = None,
        server_names: list[str] | None = None,
        server_ports: dict[str, int] | None = None,
    ):
        super().__init__(
            competition_data_folder=competition_data_folder,
            file_share_dir=file_share_dir,
            server_names=server_names,
            server_ports=server_ports,
        )
        self.set_response_timeout_seconds(FIRST_TURN_TIMEOUT_SECONDS)
        self.simulator_root = self._resolve_simulator_root()
        self.scenarios = self._discover_scenarios()
        # Truncates every episode. For host smoke tests only: a full run leaves this None
        # so that episode length is decided by the scenario's own config.gin.
        self.max_steps_per_scenario: int | None = _max_steps_per_scenario_from_env()

    def _resolve_simulator_root(self) -> pathlib.Path:
        """Return the directory that the scenario `config.gin` paths are relative to.

        That is the directory containing the `smart_buildings` package, which is also
        what must be importable for the simulator itself.
        """
        if self.competition_data_folder is None:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, 'competition_data_folder is required.')
        root = pathlib.Path(self.competition_data_folder).resolve()
        if not (root / 'smart_buildings').is_dir():
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION,
                f'Expected a `smart_buildings` package directory under {root}, found: {sorted(p.name for p in root.iterdir())[:10]}',
            )
        return root

    def _discover_scenarios(self) -> dict[str, pathlib.Path]:
        """Map each scenario's name, e.g. `building_0203/riyadh_jul`, to its directory.

        Which scenarios run is controlled entirely by what the host mounts, so no
        scenario list is hardcoded here. Insertion order is the order they are run in.

        `evaluation_scenarios` only, never `training_scenarios`, even though the mount
        carries both: the training scenarios are published in full, so falling back to
        them would silently score the competition on data every entrant already has.
        """
        scenario_root = self.simulator_root / 'smart_buildings' / 'evaluation_scenarios'
        scenarios = {
            f'{config.parent.parent.name}/{config.parent.name}': config.parent for config in sorted(scenario_root.glob(f'*/*/{CONFIG_FILENAME}'))
        }
        if not scenarios:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'No {CONFIG_FILENAME} files found under {scenario_root}.')
        return scenarios

    def _load_environment(self, scenario_dir: pathlib.Path) -> Any:
        """Build a simulator environment for one scenario.

        Imports are deferred to the first call so that the cache redirection above is in
        place, and so the gateway can raise a clean error if the mount is malformed.
        """
        _redirect_caches_to_scratch()
        if str(self.simulator_root) not in sys.path:
            sys.path.insert(0, str(self.simulator_root))

        # Neither module is a build-time dependency of Hearth: matplotlib is pulled in
        # transitively by the simulator, and the simulator itself arrives on the data mount.
        import matplotlib  # ty: ignore[unresolved-import]

        # Select a headless backend before the simulator imports pyplot.
        matplotlib.use('Agg')
        # Reuse the starter kit's own loader and metadata builder rather than
        # reimplementing them; the metadata keys are a contract with competitors and must
        # not drift from what they see locally.
        from smart_buildings import evaluate  # ty: ignore[unresolved-import]

        with _working_directory(self.simulator_root):
            env = evaluate.load_hybrid_environment(scenario_dir / CONFIG_FILENAME)
            # Disable the per-episode metric dump: it writes into the read-only scenario tree.
            env.metrics_path = None
            # Mirrors evaluate.py, typo included: the attribute the observation reads is
            # `occupancy_normalization_constant`, so both this line and the starter kit's
            # leave it at the 0.0 that config.gin sets. Kept rather than corrected because
            # matching what competitors see locally is the point; correcting it here alone
            # would rescale NUM_OCCUPANTS for the gateway only. Fix it upstream first.
            env._occupancy_normalization_constant = 125.0
            building_meta_data = evaluate.create_building_meta_data(env, scenario_dir)
        return env, building_meta_data

    def generate_game_results(self) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        """Run every scenario, yielding its mean per-step reward as it finishes.

        A scenario/mean-reward row stands in for the base class's agent_id/score row, because
        the leaderboard is per scenario rather than per agent -- there is only ever one agent.
        `run_game` is not implemented for the same reason.
        """
        driver = SmartBuildingsDriver(self)
        scenario_rewards = []
        for scenario_id in self.scenarios:
            started = time.monotonic()
            results = driver.run_game(scenario_id)
            # `get_results` reports the episode total, one entry because there is one agent.
            # Divided by the steps it accumulated over so that a long episode is not scored as a
            # worse one; `run_game` cannot exit before a step, so the divisor is never zero.
            scenario_rewards.append(float(next(iter(results.values()))) / driver.step_index)
            print(
                f'[gateway] {scenario_id}: mean reward={scenario_rewards[-1]:.4f} over {driver.step_index} steps in {time.monotonic() - started:.0f}s',
                flush=True,
            )
            table = pl.DataFrame([{ROW_ID_COLUMN: scenario_id, REWARD_COLUMN: scenario_rewards[-1]}], schema=SUBMISSION_SCHEMA)
            yield table.select(REWARD_COLUMN), table.select(ROW_ID_COLUMN)

        print(f'[gateway] mean scenario reward total, for logs only: {np.mean(scenario_rewards):.4f}', flush=True)


if __name__ == '__main__':
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        gateway = SmartBuildingsGateway()
        gateway.run()
    else:
        print('Skipping run: not a competition rerun.')
