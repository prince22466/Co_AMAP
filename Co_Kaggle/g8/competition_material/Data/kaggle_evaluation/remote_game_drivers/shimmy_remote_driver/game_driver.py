import collections
from typing import Any

from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0

from remote_game_drivers.core.base_classes import KaggleAgentId
from remote_game_drivers.gymnasium_remote_driver import game_driver
from remote_game_drivers.gymnasium_remote_driver.game_driver import GymGameData

# Allowlist of supported 2-player OpenSpiel games that work with Shimmy
# See the docstring of _game_is_supported for details on why some games are excluded.
SUPPORTED_GAMES: frozenset[str] = frozenset(
    [
        # Board games
        'amazons',
        'backgammon',
        'breakthrough',
        'checkers',
        'chess',
        'clobber',
        'connect_four',
        'dots_and_boxes',
        'go',
        'havannah',
        'hex',
        'hive',
        'lines_of_action',
        'mancala',
        'nine_mens_morris',
        'othello',
        'oware',
        'pentago',
        'quoridor',
        'tic_tac_toe',
        'twixt',
        'ultimate_tic_tac_toe',
        'y',
        # Card/dice games
        'gin_rummy',
        'kuhn_poker',
        'leduc_poker',
        'liars_dice',
        # Abstract/mathematical games
        'blotto',
        'coin_game',
        'goofspiel',
        'markov_soccer',
        'matrix_rps',
        'nim',
        'oshi_zumo',
        'pig',
        # Imperfect information games
        'battleship',
        'dark_chess',
        'dark_hex',
        'phantom_ttt',
        # Variant games
        'einstein_wurfelt_nicht',
        'latent_ttt',
    ]
)


class OpenSpielGameDriver(game_driver.GameDriver[Any]):
    """Driver for OpenSpiel games via the Shimmy wrapper.

    Shimmy converts OpenSpiel games to PettingZoo AEC API.
    https://shimmy.farama.org/environments/open_spiel/

    Not all OpenSpiel games are supported. See the __game_is_supported() method for details.
    """

    def _get_agent_inputs(self) -> GymGameData:
        raise NotImplementedError('_get_agent_inputs is not used by OpenSpiel drivers')

    def get_still_playing(self) -> dict[KaggleAgentId, bool]:
        agents_still_playing: dict[KaggleAgentId, bool] = {}
        for agent_id in self.env.terminations:
            agents_still_playing[self._openspiel_to_kaggle_agent_id_map[agent_id]] = not (
                self.env.terminations[agent_id] or self.env.truncations[agent_id]
            )
        return agents_still_playing

    def start_new_game(self, game_name: str, random_seed=None, game_state=None, action_timeout=None):
        """Initialize a new OpenSpiel game.

        Args:
            game_name: OpenSpiel game name (e.g., "tic_tac_toe", "connect_four", "chess")
        """
        self.game_name: str = game_name
        self.env: OpenSpielCompatibilityV0 = OpenSpielCompatibilityV0(game_name=game_name, **self.game_config)
        self.env.reset(seed=random_seed)
        self.rewards: dict[str, list[float | int]] = collections.defaultdict(list)
        self.random_seed = random_seed
        self.game_state = game_state
        self.action_timeout = action_timeout
        # OpenSpiel uses string agent IDs (e.g., "player_0", "player_1")
        self._kaggle_to_openspiel_agent_id_map: dict[KaggleAgentId, str] = {}
        self._openspiel_to_kaggle_agent_id_map: dict[str, KaggleAgentId] = {}
        for k_agent_id, os_agent_id in zip(self.agent_ids, self.env.agents):
            self._kaggle_to_openspiel_agent_id_map[k_agent_id] = os_agent_id
            self._openspiel_to_kaggle_agent_id_map[os_agent_id] = k_agent_id
        self.turn_count = 0
        self._align_agent_ids(env_agent_ids=self.env.agents)

    def step(self, action: Any) -> GymGameData | None:
        self.env.step(action)
        return None

    def get_results(self) -> dict[KaggleAgentId, float | int]:
        kaggle_rewards = {}
        for os_agent_id, reward in self.rewards.items():
            kaggle_id = self._openspiel_to_kaggle_agent_id_map[os_agent_id]
            kaggle_rewards[kaggle_id] = sum(reward)
        return kaggle_rewards

    def _game_is_supported(self, game_name: str) -> bool:
        """Check if a game is in the supported allowlist.

        Unsupported Games:
        The following OpenSpiel games are NOT supported due to Shimmy/OpenSpiel limitations:

        - **Single-player games**: blackjack (1 player), 2048, solitaire, cliff_walking, deep_sea, etc.
        - **Games with 3+ players**: crazy_eights (5 players), hearts (4 players), spades (4 players),
          bridge (4 players), euchre (4 players), oh_hell, skat, dou_dizhu, etc.
        - **Games without observation tensors**: cribbage, tarok, and others that raise
          NotImplementedError("No information/observation tensor/string implemented").
        - **Wrapper games requiring special syntax**: misere, repeated_game, turn_based_simultaneous_game,
          add_noise, cached_tree, coop_to_1p, start_at, etc.
        - **Mean-field games (MFG)**: mfg_crowd_modelling, mfg_dynamic_routing, mfg_garnet, etc.
        - **Games requiring external configuration**: efg_game, nfg_game, universal_poker.

        For a full list of OpenSpiel games, see: pyspiel.registered_names()
        """
        # TODO: cover games with more than 2 players
        # TODO: handle games without observation tensors
        # TODO: handle wrapper games requiring special syntax
        # TODO: handle mean-field games (MFG)
        # TODO: handle games requiring external configuration
        return game_name in SUPPORTED_GAMES

    def run_game(self, game_name: str) -> dict[KaggleAgentId, float | int]:
        if not self._game_is_supported(game_name):
            raise ValueError(f'Game {game_name} is not yet supported. See the self_game_is_supported() docstring for details.')

        self.start_new_game(game_name=game_name)
        for os_agent_id in self.env.agent_iter():
            _raw_observation, reward, termination, truncation, info = self.env.last()
            self.rewards[os_agent_id].append(reward)
            action_space = self.env.action_space(os_agent_id)
            kaggle_agent_id: KaggleAgentId = self._openspiel_to_kaggle_agent_id_map[os_agent_id]
            if termination or truncation:
                # The AEC spec permits a None observation on an agent's final turn, and we don't
                # need one: a terminated agent must be stepped with a None action.
                action = None
            else:
                if _raw_observation is None:
                    raise ValueError(f'{game_name} provided no observation for agent {os_agent_id} on turn {self.turn_count}.')
                self.observation: dict[str, Any] = _raw_observation
                action, _reasoning = self.get_agent_action((action_space, self.observation), agent_id=kaggle_agent_id, info=info)
            self.step(action)
            self.turn_count += 1

        results = self.get_results()
        self.env.close()
        return results
