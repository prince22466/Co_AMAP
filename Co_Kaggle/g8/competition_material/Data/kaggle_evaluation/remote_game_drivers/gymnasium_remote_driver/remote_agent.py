import abc
from typing import Any

import gymnasium.core
from gymnasium import spaces

from remote_game_drivers.gymnasium_remote_driver.serialization import GymSerializer, SpaceModelUnion

GymAction = gymnasium.core.ActType


class RemoteAgent(abc.ABC):
    """Base class for all agents that interact with a gymnasium environment.

    Agents receive game data via relay networking. The action_space arrives as a dict of fields,
    which the annotation on process_turn rebuilds into the matching Pydantic SpaceModel before it
    is converted to a Gymnasium Space and passed to choose_action(). The annotation names the union
    of concrete models rather than the abstract SpaceModel base, because the base declares no
    fields and so describes nothing to rebuild.

    From the agent's perspective, they are operating in a normal Gymnasium environment.
    """

    def __init__(self) -> None:
        self._serializer = GymSerializer()

    def process_turn(self, action_space: SpaceModelUnion, observation: Any, info: Any | None = None) -> dict[str, Any]:
        """Entry point called by relay server. Converts Pydantic models to Gymnasium types.

        Args:
            action_space: Pydantic SpaceModel (rebuilt from this method's annotation by the relay)
            observation: Native observation value (tuple, np.array, dict, etc.)
            info: Optional additional game-specific data (used by OpenSpiel/Shimmy)

        Returns:
            Dictionary with 'action' and optionally 'reasoning' keys.
        """
        gym_action_space = self._serializer.deserialize(action_space)
        result = self.choose_action(gym_action_space, observation, info=info)

        if isinstance(result, tuple) and len(result) == 2:
            action, reasoning = result
            response: dict[str, Any] = {'action': action}
            if reasoning is not None:
                response['reasoning'] = reasoning
            return response
        else:
            return {'action': result}

    @abc.abstractmethod
    def choose_action(
        self, action_space: spaces.Space[Any], observation: gymnasium.core.ObsType, info: Any | None = None
    ) -> GymAction | tuple[GymAction, str | None]:
        """Choose an action given the current game state.

        This is the main method agents implement. They receive native Gymnasium types.

        Args:
            action_space: Gymnasium action space (e.g., spaces.Discrete, spaces.Box)
            observation: Gymnasium observation (tuple, np.array, dict, etc.)
            info: Optional additional game-specific data

        Returns:
            The chosen action, or a tuple of (action, reasoning)
        """
        raise NotImplementedError()


class RandomAgent(RemoteAgent):
    """Default agent implementation: move at random."""

    def choose_action(self, action_space: spaces.Space[Any], observation: tuple[int], info: Any = None) -> GymAction:
        return action_space.sample()
