"""Minimal bidirectional dict, replacing the external ``bidict`` package.

Only the small surface the environment relies on is implemented: forward
lookup/containment/assignment plus a reverse view exposed as ``.inv``. Values,
like keys, must be unique (the environment guarantees this when generating ids).
"""

from collections.abc import Mapping
from typing import Any


class _Inverse(Mapping):
  """Read-only view of the value -> key direction of a :class:`bidict`."""

  def __init__(self, inv: dict):
    self._inv = inv

  def __getitem__(self, value: Any) -> Any:
    return self._inv[value]

  def __contains__(self, value: Any) -> bool:
    return value in self._inv

  def __iter__(self):
    return iter(self._inv)

  def __len__(self) -> int:
    return len(self._inv)


class bidict(dict):
  """A dict that also maintains an inverse (value -> key) mapping."""

  def __init__(self, *args, **kwargs):
    super().__init__()
    self._inv: dict = {}
    for key, value in dict(*args, **kwargs).items():
      self[key] = value

  @property
  def inv(self) -> _Inverse:
    return _Inverse(self._inv)

  def __setitem__(self, key: Any, value: Any) -> None:
    if key in self:
      del self._inv[super().__getitem__(key)]
    self._inv[value] = key
    super().__setitem__(key, value)

  def __delitem__(self, key: Any) -> None:
    del self._inv[super().__getitem__(key)]
    super().__delitem__(key)
