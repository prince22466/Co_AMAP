from typing import Any

import numpy as np
from gymnasium import spaces

from remote_game_drivers.gymnasium_remote_driver import remote_agent


class RemoteAgent(remote_agent.RemoteAgent):
    """PettingZoo uses Gymnasium spaces. Ex:
    https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/connect_four/connect_four.py#L67

    From the agent side, this means there's actually no difference from gymnasium.
    """


class RandomAgent(RemoteAgent):
    """Default agent implementation: move at random."""

    def choose_action(self, action_space: spaces.Space[Any], observation: Any, info=None) -> remote_agent.GymAction:
        mask = observation.get('action_mask')
        if mask is not None:
            mask = np.array(mask, dtype=np.int8)
        return action_space.sample(mask=mask)
