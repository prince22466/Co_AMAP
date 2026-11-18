"""Inference server for the Smart Buildings HVAC control competition (ID 157303).

Competitors subclass `Policy` and hand it to `SmartBuildingsInferenceServer`. The
interface is deliberately identical to the one `smart_buildings/evaluate.py` uses
locally, so a policy developed against the starter kit runs here unchanged:

    class MyPolicy(Policy):
        def __init__(self, building_meta_data, action_spec):
            ...

        def get_action(self, obs):
            return {'continuous_action': ..., 'discrete_action': ...}

    if __name__ == '__main__':
        SmartBuildingsInferenceServer(MyPolicy).run()

The server is constructed once and reused across every scenario. A fresh policy is
built whenever the gateway sends a new scenario's metadata, which happens on the first
turn of each scenario, so carrying state between scenarios is not possible by
construction.

The relay protocol itself is the standard gymnasium one: a single `process_turn`
endpoint, supplied by `RemoteAgent`, receiving the action space, the observation and
gym's info dict.
"""

import abc
import functools
from typing import Any, cast

import numpy as np
from gymnasium import spaces

from remote_game_drivers.gymnasium_remote_driver.remote_agent import RemoteAgent
from remote_game_drivers.hearth_integrations.gymnasium.gym_inference_server import GymInferenceServer

# Key the gateway puts the scenario's static metadata under, in gym's info dict. Kept in
# step with the constant of the same name in gateway.py: the two run in separate
# containers and share only what crosses the relay.
BUILDING_META_DATA = 'building_meta_data'


class Policy(abc.ABC):
    """An HVAC control policy for one building/weather scenario."""

    def __init__(self, building_meta_data: dict[str, Any], action_spec: spaces.Dict):
        """
        Args:
            building_meta_data: Static description of the building. Includes the floor plan
                and zone map arrays, AHU/zone assignments, observation field names, the
                index groups needed to address individual actions (`temp_indices`,
                `pressure_indices`, `water_indices`, `zone_temp_indices`), the occupancy
                schedule, and the simulator's observation/action normalization constants.
            action_spec: Subscriptable action space with `continuous_action` and
                `discrete_action` entries, each a `gymnasium.spaces.Box`.
        """
        self.building_meta_data = building_meta_data
        self.action_spec = action_spec

    @abc.abstractmethod
    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Return the action for the current observation.

        Args:
            obs: float32 array of observations, named by
                `building_meta_data['observation_field_names']`.

        Returns:
            A dict with a float32 `continuous_action` and an int32 `discrete_action`, each
            matching the shape and bounds of the corresponding `action_spec` entry. The
            gateway rejects the submission if either is out of range.
        """
        raise NotImplementedError


class DefaultPolicy(Policy):
    """Trivial reference policy: hold every setpoint mid-range and every device on."""

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        # Both entries are always Box; spaces.Dict.__getitem__ is only typed as Space.
        continuous = cast(spaces.Box, self.action_spec['continuous_action'])
        discrete = cast(spaces.Box, self.action_spec['discrete_action'])
        return {
            'continuous_action': np.zeros(continuous.shape, dtype=np.float32),
            'discrete_action': np.ones(discrete.shape, dtype=np.int32),
        }


class PolicyAgent(RemoteAgent):
    """Bridges the `Policy` interface onto the gymnasium remote agent protocol.

    `RemoteAgent` supplies the `process_turn` endpoint, which deserializes the action
    space and wraps the returned action for the relay, leaving only the choice of action
    to implement here.
    """

    def __init__(self, policy_class: type[Policy] = DefaultPolicy):
        super().__init__()
        self.policy_class = policy_class
        self._policy: Policy | None = None

    def choose_action(self, action_space: spaces.Space[Any], observation: Any, info: Any | None = None) -> dict[str, np.ndarray]:
        # A scenario's metadata is a few MiB, so the gateway sends it on the scenario's
        # first turn only; the simulator's info dict is empty for every step after that.
        # Its arrival is therefore what marks the start of a new scenario.
        if info and BUILDING_META_DATA in info:
            self._policy = self.policy_class(info[BUILDING_META_DATA], cast(spaces.Dict, action_space))
        if self._policy is None:
            raise RuntimeError(f'The first turn of a scenario did not carry {BUILDING_META_DATA}, so no policy could be built.')
        return self._policy.get_action(observation)


class SmartBuildingsInferenceServer(GymInferenceServer):
    """Serves a `Policy` subclass to the Smart Buildings gateway."""

    def __init__(self, policy_class: type[Policy] = DefaultPolicy):
        # GymInferenceServer constructs the agent itself, with no arguments.
        super().__init__(cast(type[RemoteAgent], functools.partial(PolicyAgent, policy_class)))

    def _get_gateway_for_test(self, competition_data_folder=None, file_share_dir=None, **kwargs):
        """Gateway used by `run_local_game()`, which lets a competitor test against a scenario."""
        from gateway import SmartBuildingsGateway

        return SmartBuildingsGateway(competition_data_folder=competition_data_folder, file_share_dir=file_share_dir, **kwargs)


if __name__ == '__main__':
    SmartBuildingsInferenceServer(DefaultPolicy).run()
