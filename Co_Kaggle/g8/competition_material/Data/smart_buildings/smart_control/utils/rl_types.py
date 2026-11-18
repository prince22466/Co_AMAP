"""RL type aliases backed by gymnasium, replacing tf_agents.typing.types.

The environment historically depended on TF-Agents for its action/observation
specs and type annotations. TF-Agents has no Python 3.12 wheels, so the specs
are now plain ``gymnasium.spaces`` objects. Only these lightweight aliases are
needed for annotations across the codebase.
"""

from collections.abc import Mapping
from typing import Any, Union

import gymnasium as gym
import numpy as np

# A single spec is a gymnasium space (e.g. Box). A nested spec is a space or a
# mapping of names to specs (the hybrid action space is a dict of spaces).
ArraySpec = gym.spaces.Space
NestedArraySpec = Union[gym.spaces.Space, Mapping[str, Any]]

# Actions/observations are arbitrary numpy arrays or (for hybrid actions) dicts
# of arrays, matching TF-Agents' loose NestedArray contract.
NestedArray = Any


def bounded_spec(shape, dtype, minimum, maximum) -> gym.spaces.Box:
  """A gymnasium Box with explicit bounds (replaces BoundedArraySpec)."""
  return gym.spaces.Box(
      low=minimum, high=maximum, shape=tuple(shape), dtype=dtype
  )


def unbounded_spec(shape, dtype) -> gym.spaces.Box:
  """A gymnasium Box spanning the full range of dtype (replaces ArraySpec)."""
  return gym.spaces.Box(
      low=-np.inf, high=np.inf, shape=tuple(shape), dtype=dtype
  )
