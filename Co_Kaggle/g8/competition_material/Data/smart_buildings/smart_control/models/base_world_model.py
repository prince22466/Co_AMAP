"""An abstract base class to be used by a model-based RL algorithm (MPPI)."""

import abc
from collections.abc import Mapping
from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np

StateKey: TypeAlias = str
StateValue: TypeAlias = Any
StateType: TypeAlias = Mapping[StateKey, StateValue]
DiscreteActionMapping: TypeAlias = dict[int, list[float]]


class BaseWorldModel(abc.ABC):
  """Abstract base class for a world model used by the MPPI planner."""

  def __init__(self):
    """Initializes the world model's parameters."""
    self.all_z_dim: int = 0
    self.all_z_max: list[float] = []
    self.all_z_min: list[float] = []
    self.continuous_action_normalizers: Mapping[str, Any] = {}
    self.discrete_action_normalizers: Mapping[str, Any] = {}
    self.k_dim: int = 0
    self.offset: list[int] = []
    self.par_size: list[int] = []
    self.z_max: list[float] = []
    self.z_min: list[float] = []

  @abc.abstractmethod
  def next(
      self, action: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predicts the next state, reward, and continuation probability.

    Args:
        action: A tensor representing the action to take.

    Returns:
        A tuple of:
          s_pred_tensor: Predicted next state.
          reward_tensor: Predicted reward.
          continue_prob: Probability of continuing.
          r2: An optional secondary reward tensor, which may be used for
          specific environments or for intrinsic rewards. Current implementation
          of MPPI does not use it.
    """

  @abc.abstractmethod
  def get_state(self) -> StateType:
    """Saves a snapshot of the internal state of the model's environment.

    This is used by the planner to save the state of the simulation environment
    so that it can be restored later to explore different trajectories.
    Basically, it is required to be able to reset the environment to the
    pristine state. While reset might work for some environments, it is not
    always possible to restore the environment to a previous state exactly.
    Hence, we use this method to save a snapshot of the environment's state,
    which might not be possible to achieve using the environment's reset()
    method.

    Returns:
      A dictionary representing the state of the environment.
    """

  @abc.abstractmethod
  def set_state(self, state_dict: StateType):
    """Restores the internal state of the model's environment.

    This is used by the planner to restore the state of the simulation
    environment to a previously saved state.

    Args:
      state_dict: A dictionary representing the state of the environment, as
        returned by `get_state`
    """

  @property
  @abc.abstractmethod
  def env(self) -> gym.Env:
    """Returns the underlying environment."""

  @property
  @abc.abstractmethod
  def discrete_action_mapping(self) -> DiscreteActionMapping:
    """Returns the mapping from discrete action index to supervisor commands."""

  @discrete_action_mapping.setter
  @abc.abstractmethod
  def discrete_action_mapping(self, value: DiscreteActionMapping):
    pass

  @abc.abstractmethod
  def create_action_dict(self, k: int, z: np.ndarray) -> Mapping[str, Any]:
    """Creates a dictionary of actions for the environment."""

  @abc.abstractmethod
  def synchronize(self, source_env: gym.Env):
    """Synchronizes the planning environment with the acting environment."""
