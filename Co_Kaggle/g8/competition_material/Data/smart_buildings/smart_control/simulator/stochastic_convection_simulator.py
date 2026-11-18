"""Stochastic simulator of convection flow in bldg.

A convection simulator that randomly shuffles control volumes as a stochastic
imitation of convection
We specify probability of a control volume getting shuffled, as well as maximum
distance that any given
control volume can be moved.
"""

import collections
import copy
import random
from collections.abc import MutableSequence

import gin
import numba
import numpy as np

from smart_buildings.smart_control.simulator import base_convection_simulator


@numba.njit(cache=True)
def _seed_kernel_rng(seed):  # pylint: disable=missing-function-docstring
  np.random.seed(seed)


@numba.njit(cache=True)
def _convect_room(cvs, cand_ptr, cand_idx, unlimited, p, temp):
  """Samples control volumes, pairs each with a neighbour, and swaps them.

  Same algorithm as the Python version it replaces: draw once per CV, keep a
  fraction p, pair each kept CV with a uniformly chosen candidate, shuffle the
  pairs, then apply the swaps in order. The swaps must stay sequential because a
  CV can appear in more than one pair, so a vectorised assignment would not
  agree with itself.

  The random stream differs from the Python original by design. This is numba's
  generator, not CPython's, so a given seed produces a different realisation of
  the same process rather than the same one.

  Args:
    cvs: (n, 2) int64 row/column of each control volume in the room.
    cand_ptr: (n + 1,) int64 CSR offsets into cand_idx.
    cand_idx: int64 candidate indices, as positions within cvs.
    unlimited: True when any CV in the room may pair with any other, in which
      case cand_ptr and cand_idx are unused.
    p: probability that a CV is sampled.
    temp: the temperature grid, modified in place.
  """
  n = cvs.shape[0]
  left = np.empty(n, dtype=np.int64)
  right = np.empty(n, dtype=np.int64)
  m = 0
  for i in range(n):
    if np.random.random() > p:
      continue
    if unlimited:
      j = np.random.randint(n)
    else:
      start = cand_ptr[i]
      end = cand_ptr[i + 1]
      if end <= start:
        continue
      j = cand_idx[start + np.random.randint(end - start)]
    left[m] = i
    right[m] = j
    m += 1

  for i in range(m - 1, 0, -1):  # Fisher-Yates, matching random.shuffle
    j = np.random.randint(i + 1)
    left[i], left[j] = left[j], left[i]
    right[i], right[j] = right[j], right[i]

  for k in range(m):
    a = left[k]
    b = right[k]
    r1 = cvs[a, 0]
    c1 = cvs[a, 1]
    r2 = cvs[b, 0]
    c2 = cvs[b, 1]
    held = temp[r1, c1]
    temp[r1, c1] = temp[r2, c2]
    temp[r2, c2] = held

@gin.configurable
class StochasticConvectionSimulator(
    base_convection_simulator.BaseConvectionSimulator
):
  """Stochastic method of simulating air convection.

  Attributes:
    _p: shuffling probability
    _distance: max distance a CV can move
  """

  def __init__(self, p: float, distance: int, seed: int | None):
    """Initializes the stochastic convection simulator.

    Args:
      p: shuffling probability
      distance: max distance a CV can move
      seed: random seed
    """
    self._p = p
    self._distance = distance
    # cache is used to store precalculated shuffling patterns
    # so we do not need to calculate them again
    self._cache = collections.defaultdict(dict)
    # Offsets depend only on the radius, room membership only on the room.
    self._offsets_cache = {}
    self._room_set_cache = {}
    self._room_arrays_cache = {}

    if seed is not None:
      random.seed(seed)
      # The kernel uses numba's generator, which is separate from
      # CPython's and must be seeded on its own.
      _seed_kernel_rng(seed)

  @property
  def p(self) -> float:
    """The shuffling probability."""
    return self._p

  @property
  def distance(self) -> int:
    """The max distance a CV can move."""
    return self._distance

  def apply_convection(
      self,
      room_dict: dict[str, MutableSequence[tuple[int, int]]],
      temp: np.ndarray,
  ) -> None:
    """Applies convection to the temperature array given.

    Splits up rooms via room_dict.

    Args:
      room_dict: A dictionary mapping of room coordinates.
      temp: An array of temperatures.
    """
    p = self._p
    distance = self._distance
    if p == 0 or distance == 0:
      return

    for k, v in room_dict.items():
      if k == "exterior_space":
        continue
      if k == "interior_wall":
        continue
      if distance == -1 and p == 1:  # special case, can be more efficient
        self._shuffle_no_max_dist(v, temp)
      else:
        self._shuffle_max_dist(p, v, distance, temp, k)

  def _shuffle_no_max_dist(self, v, temp):
    """Special case of shuffling when no max dist is specified.and p=1.

    Args:
      v: list of CVs to shuffle
      temp: temperature array
    """
    v = copy.deepcopy(v)
    vals = {}
    for cv in v:
      vals[cv] = temp[cv[0], cv[1]]
    v_shuffle = copy.deepcopy(v)
    random.shuffle(v_shuffle)
    for i, _ in enumerate(v_shuffle):
      cv = v_shuffle[i]
      val = vals[v[i]]
      temp[cv[0], cv[1]] = val

  def _room_arrays(self, room_key, v, max_dist, unlimited):
    """Returns (cvs, cand_ptr, cand_idx) for a room, built once and cached.

    Candidates are stored in CSR form: cand_idx holds every candidate as a
    position within cvs, and cand_ptr[i]..cand_ptr[i+1] is the slice belonging
    to CV i. Under `unlimited` every CV may pair with any other, so the lists
    would be n squared and are not built at all; the kernel draws directly.
    """
    key = (room_key, max_dist, unlimited)
    cached = self._room_arrays_cache.get(key)
    if cached is not None:
      return cached

    cvs = np.asarray(v, dtype=np.int64)
    if unlimited:
      empty = np.zeros(1, dtype=np.int64)
      out = (cvs, empty, empty)
      self._room_arrays_cache[key] = out
      return out

    offsets = self._offsets_cache.get(max_dist)
    if offsets is None:
      offsets = self._offsets_cache[max_dist] = [
          (d_0, d_1)
          for d_0 in range(-max_dist, max_dist + 1)
          for d_1 in range(-max_dist, max_dist + 1)
          if d_0**2 + d_1**2 <= max_dist**2
      ]

    position = {cv: i for i, cv in enumerate(v)}
    ptr = np.zeros(len(v) + 1, dtype=np.int64)
    idx = []
    for i, (r, c) in enumerate(v):
      for d_0, d_1 in offsets:
        j = position.get((r + d_0, c + d_1))
        if j is not None:
          idx.append(j)
      ptr[i + 1] = len(idx)
    out = (cvs, ptr, np.asarray(idx, dtype=np.int64))
    self._room_arrays_cache[key] = out
    return out

  def candidates_for(self, room_key, max_dist, cv):
    """Returns the coordinates a control volume may be swapped with.

    The candidate lists live in CSR arrays for the kernel's benefit. This
    unpacks one CV's slice back into coordinates, for tests and diagnostics.
    """
    cvs, ptr, idx = self._room_arrays_cache[(room_key, max_dist, max_dist == -1)]
    rows = [tuple(x) for x in cvs.tolist()]
    i = rows.index(tuple(cv))
    return [rows[j] for j in idx[ptr[i] : ptr[i + 1]]]

  def _shuffle_max_dist(self, p, v, max_dist, temp, room_key=None):
    """Shuffles a room's control volumes, in compiled code.

    Args:
      p: shuffle probability
      v: list of CVs to shuffle
      max_dist: max distance to shuffle, or -1 for no limit
      temp: temperature array, modified in place
      room_key: room identifier, used to cache the per-room arrays
    """
    unlimited = max_dist == -1
    if not v:
      return
    cvs, cand_ptr, cand_idx = self._room_arrays(room_key, v, max_dist, unlimited)
    _convect_room(cvs, cand_ptr, cand_idx, unlimited, p, temp)
