"""Inference server for gymnasium-based games using the game_arena remote driver."""

from pathlib import Path

import kaggle_evaluation.core.templates

# Resolves in the deployed layout too: the whole remote_game_drivers tree is copied alongside
# the kaggle_evaluation package, so this does not depend on where this file itself ends up.
from remote_game_drivers.gymnasium_remote_driver.remote_agent import RemoteAgent
from remote_game_drivers.hearth_integrations.gymnasium import gym_gateway


class GymInferenceServer(kaggle_evaluation.core.templates.GameInferenceServer):
    """Inference server for gymnasium-based games.

    Creates an agent instance and registers its process_turn method directly.
    Extends GameInferenceServer to support multiplayer games via run_local_game().
    """

    def __init__(self, agent_class: type[RemoteAgent]):
        self.agent = agent_class()
        # Register the agent's process_turn method directly as a bound method
        super().__init__(self.agent.process_turn)

    def _get_gateway_for_test(
        self,
        competition_data_folder: str | Path | None = None,
        file_share_dir: str | None = None,
        **kwargs,
    ) -> kaggle_evaluation.core.templates.GameGateway:
        return gym_gateway.DefaultGymGateway(competition_data_folder=competition_data_folder, file_share_dir=file_share_dir)
