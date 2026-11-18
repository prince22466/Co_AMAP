"""A randomized occupancy model for the building simulation.

In this model, N occupants arrive between the earliest and latest arrival
hour and depart between the earliest and latest departure hour. The probability
of departure is specified so that the expected arrival and departure times
occur halfway in the interval. For a standard Bernoulli RV, E[X] = n*p, so
p = E[X] / n / 2, where E[X] is the expected number of arrivals, which equals 1.
"""

import datetime
import enum
from typing import Any

import gin
import numpy as np
import pandas as pd

from smart_buildings.smart_control.models import base_occupancy
from smart_buildings.smart_control.utils import conversion_utils

SerializableData = dict[str, Any]


class OccupancyStateEnum(enum.Enum):
  AWAY = 1
  WORK = 2


class ZoneOccupant:
  """Represents a single occupant in a zone.

  Attributes:
    earliest_expected_arrival_hour: earliest arrivel, 0 - 22
    latest_expected_arrival_hour: latest arrivel, 1 - 23
    earliest_expected_departure_hour: earliest departure, 0 - 22
    latest_expected_departure_hour: latest departure, 1 - 23
    random_state: random state used to generate events
  """

  def __init__(
      self,
      earliest_expected_arrival_hour: int,
      latest_expected_arrival_hour: int,
      earliest_expected_departure_hour: int,
      latest_expected_departure_hour: int,
      step_size: pd.Timedelta,
      random_state: np.random.RandomState,
      time_zone: datetime.tzinfo | str = 'UTC',
  ):
    assert (
        earliest_expected_arrival_hour
        < latest_expected_arrival_hour
        < earliest_expected_departure_hour
        < latest_expected_departure_hour
    )

    self._earliest_expected_arrival_hour = earliest_expected_arrival_hour
    self._latest_expected_arrival_hour = latest_expected_arrival_hour
    self._earliest_expected_departure_hour = earliest_expected_departure_hour
    self._latest_expected_departure_hour = latest_expected_departure_hour
    self._step_size = step_size
    self._occupancy_state = OccupancyStateEnum.AWAY
    self._p_arrival = self._get_event_probability(
        earliest_expected_arrival_hour, latest_expected_arrival_hour
    )
    self._p_departure = self._get_event_probability(
        earliest_expected_departure_hour, latest_expected_departure_hour
    )
    self._random_state = random_state
    self._time_zone = time_zone

  def _to_local_time(self, timestamp: pd.Timestamp) -> pd.Timestamp:
    """Converts timestamp to local time, surviving both DST transitions.

    In a zone that observes daylight saving there are two hours a year that are
    not a simple map from wall clock to instant, and a bare `tz_localize`
    raises on both:

      spring forward  the 02:00 hour does not exist -> NonExistentTimeError
      fall back       the 01:00 hour occurs twice   -> AmbiguousTimeError

    Any episode whose window contains either date would otherwise fail
    mid-rollout, so both cases are resolved explicitly rather than left to
    raise.

    The choices suit an occupancy schedule. `shift_forward` maps the hour that
    does not exist onto the next one that does, so an occupant due to arrive
    during it arrives when the clock next reads a real time. `ambiguous=True`
    takes the first pass through the repeated hour. Only `.hour` is read from
    the result, so either resolution is defensible; what matters is that the
    simulation continues.
    """
    if timestamp.tz is None:
      return timestamp.tz_localize(
          self._time_zone, nonexistent='shift_forward', ambiguous=True)
    else:
      return timestamp.tz_convert(self._time_zone)

  def _get_event_probability(self, start_hour, end_hour):
    """Returns the probability of an event based on the number of time steps."""
    assert start_hour < end_hour
    # The window is the number of Bernoulli trials (i.e. tests for arrival).
    window = pd.Timedelta(end_hour - start_hour, unit='hour')
    # The halfway point is the firts half of the trials.
    n_halfway = window / self._step_size / 2.0
    # We'd like to return the probability of event happening in a single time-
    # step. This follow a geometric distribution, where E[X] = 1/p, where
    # E[x] is the expected number of events before the first success. If
    # E[X] is the halfway point, then p = 1 / n_halfway.
    return 1.0 / n_halfway

  def _occupant_arrived(self, timestamp):
    """Makes a random draw to determine whether occupant arrives."""

    local_timestamp = self._to_local_time(timestamp)
    # TODO(sipple): Consider effects when time crosses DST>
    if (
        local_timestamp.hour < self._earliest_expected_arrival_hour
        or local_timestamp.hour > self._latest_expected_arrival_hour
    ):
      return False
    else:
      return self._random_state.rand() < self._p_arrival

  def _occupant_departed(self, timestamp):
    """Makes a random draw to determine whether the occupant departs.

    The window is bounded on both sides, as the arrival window already is.
    Without the upper bound, departure was a per-step coin flip that could
    only be tossed from _earliest_expected_departure_hour onward, so the
    roughly 2.5% of occupants still present at midnight became unable to
    leave: the hour was then below the earliest departure hour, and stayed
    below it until 13:00 the following day. Those occupants sat in the
    building all night, and because the reward normalises productivity by
    total occupancy, one stranded occupant carries the same comfort weight
    as a full floor.
    """
    local_timestamp = self._to_local_time(timestamp)
    if local_timestamp.hour < self._earliest_expected_departure_hour:
      return False
    if local_timestamp.hour >= self._latest_expected_departure_hour:
      return True
    return self._random_state.rand() < self._p_departure

  def peek(self, current_time: pd.Timestamp) -> OccupancyStateEnum:
    """Returns the state (WORK or AWAY) of the occupant for the current_time."""

    local_timestamp = self._to_local_time(current_time)
    day = pd.Timestamp(
        year=local_timestamp.year,
        month=local_timestamp.month,
        day=local_timestamp.day,
    )

    if not conversion_utils.is_work_day(day):
      self._occupancy_state = OccupancyStateEnum.AWAY

    elif self._occupancy_state == OccupancyStateEnum.AWAY:
      if self._occupant_arrived(current_time):
        self._occupancy_state = OccupancyStateEnum.WORK

    else:
      if self._occupant_departed(current_time):
        self._occupancy_state = OccupancyStateEnum.AWAY

    return self._occupancy_state


@gin.configurable
class RandomizedArrivalDepartureOccupancy(base_occupancy.BaseOccupancy):
  """Provides the RL agent information about how many people are in a zone.

  Attributes:
    zone_assignment: number of occupants in a zone
    earliest_expected_arrival_hour: earliest arrivel, 0 - 22
    latest_expected_arrival_hour: latest arrivel, 1 - 23
    earliest_expected_departure_hour: earliest departure, 0 - 22
    latest_expected_departure_hour: latest departure, 1 - 23
    seed: integer used to set the random state for repeatability
  """

  def __init__(
      self,
      zone_assignment: int,
      earliest_expected_arrival_hour: int,
      latest_expected_arrival_hour: int,
      earliest_expected_departure_hour: int,
      latest_expected_departure_hour: int,
      time_step_sec: int,
      seed: int | None = 17321,
      time_zone: str = 'UTC',
  ):
    self._zone_assignment = zone_assignment
    self._zone_occupants = {}
    self._step_size = pd.Timedelta(time_step_sec, unit='second')
    self._earliest_expected_arrival_hour = earliest_expected_arrival_hour
    self._latest_expected_arrival_hour = latest_expected_arrival_hour
    self._earliest_expected_departure_hour = earliest_expected_departure_hour
    self._latest_expected_departure_hour = latest_expected_departure_hour
    self._random_state = np.random.RandomState(seed)
    self._time_zone = time_zone
    # ZoneOccupant.peek() advances an occupant's arrival/departure Markov chain
    # and consumes _random_state while the occupant is still AWAY inside the
    # arrival window. The chain must therefore tick once per simulated interval,
    # not once per caller. These two fields memoise the result for the interval
    # currently being asked about; the cache is dropped as soon as a different
    # interval arrives, so it holds at most one step's worth of values.
    self._occupancy_cache_key = None
    self._occupancy_cache = {}

  @property
  def json_metadata(self) -> SerializableData:
    """Returns JSON-serializable data about the occupancy."""
    metadata = super().json_metadata
    metadata.update({
        'zone_assignment': self._zone_assignment,
        'earliest_arrival': self.earliest_expected_arrival_hour,
        'latest_arrival': self.latest_expected_arrival_hour,
        'earliest_departure': self.earliest_expected_departure_hour,
        'latest_departure': self.latest_expected_departure_hour,
        'time_step_sec': self.step_size.total_seconds(),
        'time_zone': self.time_zone,
    })
    return metadata

  @property
  def zone_assignment(self) -> int:
    return self._zone_assignment

  @property
  def time_zone(self) -> datetime.tzinfo | str:
    return self._time_zone

  @property
  def step_size(self) -> pd.Timedelta:
    return self._step_size

  @property
  def earliest_expected_arrival_hour(self) -> int:
    return self._earliest_expected_arrival_hour

  @property
  def latest_expected_arrival_hour(self) -> int:
    return self._latest_expected_arrival_hour

  @property
  def earliest_expected_departure_hour(self) -> int:
    return self._earliest_expected_departure_hour

  @property
  def latest_expected_departure_hour(self) -> int:
    return self._latest_expected_departure_hour

  def average_zone_occupancy(
      self, zone_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> float:
    """Returns the occupancy within start_time, end_time for the zone.

    If the zone is not found, implementations should raise a ValueError.

    Args:
      zone_id: specific zone identifier for the building.
      start_time: **local time** w/ TZ for the beginning of the interval.
      end_time: **local time** w/ TZ for the end of the interval.

    Returns:
      average number of people in the zone for the interval.
    """

    if zone_id not in self._zone_occupants:
      self._zone_occupants[zone_id] = []
      for _ in range(self._zone_assignment):
        self._zone_occupants[zone_id].append(
            ZoneOccupant(
                self._earliest_expected_arrival_hour,
                self._latest_expected_arrival_hour,
                self._earliest_expected_departure_hour,
                self._latest_expected_departure_hour,
                self._step_size,
                self._random_state,
                self._time_zone,
            )
        )

    # Reuse the value already computed for this interval rather than advancing
    # every occupant's chain again. Both the reward path and
    # building.num_occupants query the same interval within a single step.
    cache_key = (start_time, end_time)
    if self._occupancy_cache_key != cache_key:
      self._occupancy_cache_key = cache_key
      self._occupancy_cache = {}
    elif zone_id in self._occupancy_cache:
      return self._occupancy_cache[zone_id]

    # Average across [start_time, end_time) as the docstring promises and as
    # both sibling models do (step_function_occupancy, stochastic_occupancy).
    # The previous implementation sampled start_time only and never read
    # end_time at all.
    total_occupancy = 0.0
    steps = 0
    current_time = start_time
    while current_time < end_time:
      num_occupants = 0.0
      for occupant in self._zone_occupants[zone_id]:
        if occupant.peek(current_time) == OccupancyStateEnum.WORK:
          num_occupants += 1.0
      total_occupancy += num_occupants
      steps += 1
      current_time += self._step_size

    if steps:
      average_occupancy = total_occupancy / steps
    else:
      # Degenerate interval (end_time <= start_time): fall back to a single
      # sample so the return value stays defined.
      average_occupancy = float(
          sum(
              occupant.peek(start_time) == OccupancyStateEnum.WORK
              for occupant in self._zone_occupants[zone_id]
          )
      )

    self._occupancy_cache[zone_id] = average_occupancy
    return average_occupancy
