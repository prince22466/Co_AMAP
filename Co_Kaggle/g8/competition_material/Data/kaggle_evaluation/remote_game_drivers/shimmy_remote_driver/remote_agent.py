import abc
from typing import Any

import numpy as np
from gymnasium import spaces

from remote_game_drivers.gymnasium_remote_driver import remote_agent


class RemoteAgent(remote_agent.RemoteAgent, abc.ABC):
    """OpenSpiel via Shimmy uses Gymnasium spaces (same as PettingZoo).

    The main difference is that OpenSpiel provides action_mask in the info dict
    rather than in the observation dict.
    """

    @abc.abstractmethod
    def choose_action(self, action_space: Any, observation: Any, info: Any = None) -> Any:
        raise NotImplementedError()


class RandomAgent(RemoteAgent):
    """An agent that plays random legal actions using the action_mask from info."""

    def choose_action(self, action_space: spaces.Space[Any], observation: Any, info: Any = None) -> Any:
        mask = None
        if info is not None:
            mask = info.get('action_mask')
            if mask is not None:
                mask = np.array(mask, dtype=np.int8)
        return action_space.sample(mask=mask)
