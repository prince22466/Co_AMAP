import warnings
from typing import Any

warnings.filterwarnings('ignore', message='^.*pkg_resources is deprecated')  # noqua
import gymnasium as gym
import gymnasium.core
import gymnasium.spaces

from kaggle_evaluation.core import relay
from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType
from remote_game_drivers.core import base_classes
from remote_game_drivers.core.base_classes import PROCESS_TURN_ENDPOINT, KaggleAgentId
from remote_game_drivers.gymnasium_remote_driver.serialization import GymSerializer

# action_space, observation
GymGameData = tuple[gymnasium.spaces.Space[Any], Any]
GymAction = gymnasium.core.ActType


class GameDriver(base_classes.BaseGameDriver[GymGameData, GymAction]):
    """Game driver for Gymnasium environments using relay for agent communication.

    All agent communication goes through relay.Client, even for local testing.
    This ensures consistent behavior between local development and production.
    """

    def __init__(
        self,
        driver_config: dict[str, Any] | None = None,
        relay_clients: dict[KaggleAgentId, relay.Client] | None = None,
    ) -> None:
        """Initialize the game driver.

        Args:
            driver_config: Game-specific configuration (agent_ids, game_config, etc.)
            relay_clients: Dict mapping agent IDs to relay clients.
        """
        super().__init__(driver_config, relay_clients=relay_clients)
        self._serializer = GymSerializer()

    def _get_agent_inputs(self) -> GymGameData:
        return self.env.action_space, self.observation

    def start_new_game(
        self,
        game_name: str,
        random_seed: int | None = None,
        game_state: str | None = None,
        action_timeout: int | None = None,
    ) -> None:
        # TODO: use game_state
        # TODO: use action_timeout or move to __init__
        self.game_name: str = game_name
        # render_mode=None disables rendering, which is desirable as we want to support a default headless runtime.
        self.env: gym.Env[Any, Any] = gym.make(game_name, render_mode=None)
        # Single stored game-over primitive; get_still_playing() and episode_complete derive from it.
        self._game_over: bool = False
        self.total_reward: float = 0
        self.reward: float = 0
        # Seed the environment reset for reproducibility when a seed is provided.
        if random_seed is not None:
            observation, info = self.env.reset(seed=random_seed)
        else:
            observation, info = self.env.reset()
        self.observation: Any = observation
        # Gym's info dict carries game-specific metadata (e.g. building_meta_data); forward it to agents.
        self.info: dict[str, Any] = info
        self.random_seed = random_seed
        self.game_state = game_state
        self.action_timeout = action_timeout

    def _get_agent_action(self, game_data: GymGameData, agent_id: KaggleAgentId, info: dict[str, Any] | None = None) -> tuple[GymAction, str | None]:
        """Get an action from the agent via relay.

        Serializes the action_space to a Pydantic model (which relay handles natively),
        sends to the agent's process_turn endpoint, and returns the action.
        """
        if agent_id not in self.relay_clients:
            raise RuntimeError(f'No relay client for agent {agent_id}. Available: {list(self.relay_clients.keys())}')

        action_space, observation = game_data
        # Serialize action_space to Pydantic model - relay handles Pydantic natively
        serialized_action_space = self._serializer.serialize(action_space)
        # Observation is already a native type (tuple, np.array, etc.) - relay handles these

        # Send to agent via relay - agent's process_turn will deserialize and call choose_action
        result = self.relay_clients[agent_id].send(PROCESS_TURN_ENDPOINT, serialized_action_space, observation, info)

        if not isinstance(result, dict) or 'action' not in result:
            # Attributed here rather than left to get_agent_action's catch-all: only this check knows
            # the response is unusable, and an unclassified error is reported as a Kaggle-side fault.
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION, f'agent {agent_id}: expected a dict with an "action" key, got {result}'
            )

        action: GymAction = result['action']
        reasoning: str | None = result.get('reasoning')

        return action, reasoning

    def step(self, action: Any) -> GymGameData | None:
        self.observation, reward, terminated, truncated, self.info = self.env.step(action)
        # The raw reward from env.step() is annotated as `SupportsFloat` which is an unsupported type.
        # Kept alongside the total because an individual step's reward cannot be recovered from a
        # running sum, and subclasses that record per-step outcomes need it.
        self.reward = float(reward)
        self.total_reward += self.reward
        self._game_over = terminated or truncated
        if self._game_over:
            return None
        return self._get_agent_inputs()

    def get_still_playing(self) -> dict[KaggleAgentId, bool]:
        # Gym native games share a single game over state for all agents.
        return {agent_id: not self._game_over for agent_id in self.agent_ids}

    def get_results(self) -> dict[KaggleAgentId, float | int]:
        # Gymnasium only supports one agent by default, multi-agent games typically use PettingZoo.
        # Some third party gym games are set up for multiple agents and will need a book keeping update.
        # TODO: remove the single agent restriction here once a passing multiplayer Gym test case exists.
        assert len(self.agent_ids) == 1
        return {agent_id: self.total_reward for agent_id in self.agent_ids}

    def run_game(self, game_name: str) -> dict[KaggleAgentId, float | int]:
        self.start_new_game(game_name=game_name)
        turn_count = 0
        while not self._game_over:
            for agent_id in self.agent_ids:
                action_space, observation = self._get_agent_inputs()
                action, _reasoning = self.get_agent_action((action_space, observation), agent_id, info=self.info)
                step_result = self.step(action)
                if step_result:
                    action_space, observation = step_result
                elif step_result is None and not self._game_over:
                    raise ValueError('No game data, but episode not complete.')
                turn_count += 1
        results = self.get_results()
        print(f'Episode finished! Total reward: {results}')
        self.env.close()
        return results
