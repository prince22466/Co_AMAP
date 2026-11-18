import abc
import collections
import importlib
from types import ModuleType
from typing import Any

import pettingzoo.utils.env

from remote_game_drivers.core.base_classes import KaggleAgentId
from remote_game_drivers.gymnasium_remote_driver import game_driver
from remote_game_drivers.gymnasium_remote_driver.game_driver import GymGameData


class PettingZooBaseDriver(game_driver.GameDriver[Any], abc.ABC):
    @abc.abstractmethod
    def _initialize_env(self, game_module: ModuleType) -> None:
        raise NotImplementedError

    def get_still_playing(self) -> dict[KaggleAgentId, bool]:
        agents_still_playing: dict[KaggleAgentId, bool] = {}
        for agent_id in self.env.terminations:  # ty: ignore
            # TODO: QC this per-agent liveness logic (https://github.com/Kaggle/competitions/pull/630/files#r2603439878).
            # It is now the single source of truth for game-over; episode_complete derives from it.
            agents_still_playing[self._pz_to_kaggle_agent_id_map[agent_id]] = not (
                self.env.terminations[agent_id] or self.env.truncations[agent_id]  # ty: ignore
            )
        return agents_still_playing

    def _get_agent_inputs(self) -> GymGameData:
        raise NotImplementedError('_get_agent_inputs is not used by PettingZoo drivers')

    def start_new_game(self, game_name: str, random_seed=None, game_state=None, action_timeout=None):
        """
        TODO: finish docstring
        game_name needs to be a full module name, such as pettingzoo.classic.chess_v6
        """
        self.game_name: str = game_name
        self.env: pettingzoo.AECEnv[Any, Any, Any] | pettingzoo.ParallelEnv[Any, Any, Any]
        # TODO: refactor to accept game module rather than a str?
        game_module = importlib.import_module(name=game_name)
        self._initialize_env(game_module=game_module)
        self.rewards: dict[int, list[float | int]] = collections.defaultdict(list)
        self.random_seed = random_seed
        self.game_state = game_state
        self.action_timeout = action_timeout
        # PettingZoo uses ints for agent IDs, while Kaggle may need more expressive names like "Gemini_2.5_Rethink".
        self._kaggle_to_pz_agent_id_map: dict[KaggleAgentId, int] = {}
        self._pz_to_kaggle_agent_id_map: dict[int, KaggleAgentId] = {}
        for k_agent_id, p_agent_id in zip(self.agent_ids, self.env.agents):
            pz_id = p_agent_id
            self._kaggle_to_pz_agent_id_map[k_agent_id] = pz_id
            self._pz_to_kaggle_agent_id_map[pz_id] = k_agent_id
        self.turn_count = 0
        self._align_agent_ids(env_agent_ids=self.env.agents)

    def get_results(self) -> dict[KaggleAgentId, float | int]:
        kaggle_rewards = {}
        for pz_agent_id, reward in self.rewards.items():
            kaggle_id = self._pz_to_kaggle_agent_id_map[pz_agent_id]
            kaggle_rewards[kaggle_id] = sum(reward)
        return kaggle_rewards


class AECGameDriver(PettingZooBaseDriver):
    """Driver for games using the PettingZoo AEC API.
    https://pettingzoo.farama.org/api/aec/
    """

    def _initialize_env(self, game_module: ModuleType) -> None:
        self.env: pettingzoo.utils.env.AECEnv[Any, Any, Any] = game_module.env(**self.game_config)
        # TODO: add support for random seed. Might need to be handled in start_new_game instead.
        self.env.reset()

    def step(self, action: Any) -> GymGameData | None:
        self.env.step(action)
        return None

    def run_game(self, game_name: str) -> dict[KaggleAgentId, float | int]:
        self.start_new_game(game_name=game_name)
        for pz_agent_id in self.env.agent_iter():
            _raw_observation, reward, termination, truncation, _info = self.env.last()
            # env.reward() does not track the total rewards over time, so we need to manage it manually.
            # This is generally true for games that support multiple cycles without a reset, like RockPaperScissors.
            self.rewards[pz_agent_id].append(reward)
            action_space = self.env.action_space(pz_agent_id)
            kaggle_agent_id: KaggleAgentId = self._pz_to_kaggle_agent_id_map[pz_agent_id]
            if termination or truncation:
                # The AEC spec permits a None observation on an agent's final turn, and we don't
                # need one: a terminated agent must be stepped with a None action.
                action = None
            else:
                if _raw_observation is None:
                    raise ValueError(f'{game_name} provided no observation for agent {pz_agent_id} on turn {self.turn_count}.')
                self.observation: dict[str, Any] = _raw_observation
                action, _reasoning = self.get_agent_action((action_space, self.observation), agent_id=kaggle_agent_id)
            self.step(action)
            self.turn_count += 1

        results = self.get_results()
        self.env.close()
        return results


class ParallelGameDriver(PettingZooBaseDriver):
    """Driver for games using the PettingZoo parallel API.
    https://pettingzoo.farama.org/api/parallel/
    """

    def _initialize_env(self, game_module: ModuleType) -> None:
        self.env: pettingzoo.utils.env.ParallelEnv[Any, Any, Any] = game_module.parallel_env(**self.game_config)
        # TODO: add support for random seed. Might need to be handled in start_new_game instead.
        self.observations, self.infos = self.env.reset()

    def step(self, action: dict[str, Any]) -> GymGameData | None:
        raise NotImplementedError

    def run_game(self, game_name: str) -> dict[KaggleAgentId, float | int]:
        self.start_new_game(game_name=game_name)
        while self.env.agents:
            actions: dict[int, Any] = {}
            pz_agent_id: int
            for pz_agent_id in self.env.agents:
                kaggle_agent_id: KaggleAgentId = self._pz_to_kaggle_agent_id_map[pz_agent_id]
                # TODO: determine best way to parallelize this upon network host integration.
                action, _reasoning = self.get_agent_action(
                    game_data=(self.env.action_space(pz_agent_id), self.observations[pz_agent_id]),
                    agent_id=kaggle_agent_id,
                )
                actions[pz_agent_id] = action
            _raw_observations, rewards, _terminations, _truncations, _ = self.env.step(actions)
            self.observations: dict[int, Any] = _raw_observations
            for pz_agent_id, reward in rewards.items():
                self.rewards[pz_agent_id].append(reward)
            self.turn_count += 1
        results = self.get_results()
        self.env.close()
        return results
