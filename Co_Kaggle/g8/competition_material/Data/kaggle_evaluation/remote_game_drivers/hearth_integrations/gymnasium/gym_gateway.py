"""Hearth gateway that supports the game arena gymasium_remote_driver."""

from pathlib import Path
from typing import Any

import kaggle_evaluation.core.templates
import remote_game_drivers.gymnasium_remote_driver.game_driver as gym_game_driver
from remote_game_drivers.core.base_classes import DEFAULT_AGENT_ID


class HearthGymDriver(gym_game_driver.GameDriver[Any]):
    """GameDriver that uses the Gateway's relay client for agent communication.

    The Gateway already has a relay.Client connected to the inference server.
    This driver simply passes that client to the base GameDriver.
    """

    def __init__(self, gateway: kaggle_evaluation.core.templates.GameGateway):
        # For single-agent Gymnasium games, use the same client for the default agent
        relay_clients = {DEFAULT_AGENT_ID: gateway.client}
        super().__init__(relay_clients=relay_clients)


class DefaultGymGateway(kaggle_evaluation.core.templates.GameGateway):
    """Gateway for Gymnasium-based game competitions."""

    def __init__(self, competition_data_folder: str | Path | None = None, file_share_dir: str | None = None):
        super().__init__(competition_data_folder=competition_data_folder, file_share_dir=file_share_dir)
        self.set_response_timeout_seconds(60 * 30)

        # Game configuration
        self.game_name: str = 'Blackjack-v1'
        self.random_seed: int | None = None
        self.game_state: str | None = None
        self.action_timeout: int | None = None

    def run_game(self) -> dict[str, float | int]:
        """Run the gymnasium game and return agent scores."""
        gym_driver = HearthGymDriver(gateway=self)
        # KaggleAgentId is a NewType of str, so this cast is safe
        results = gym_driver.run_game(self.game_name)
        return {str(k): v for k, v in results.items()}
