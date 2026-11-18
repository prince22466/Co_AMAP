"""Code for representing the control volumes within a building."""

import abc
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import gin
import numpy as np

from smart_buildings.smart_control.simulator import (
  base_convection_simulator,
  building_utils,
  constants,
  thermal_diffuser_utils,
)

Coordinates2D = tuple[int, int]
Shape2D = tuple[int, int]

RoomIndicesDict = dict[str, Sequence[Coordinates2D]]

@gin.configurable
@dataclasses.dataclass
class MaterialProperties:
  """Holds the physical constants for a material."""

  conductivity: float
  heat_capacity: float
  density: float


@dataclasses.dataclass(frozen=True)
class InteriorMass:
  """A lumped thermal mass hanging off the occupied control volumes.

  The floor slab and the furniture in a room hold far more heat than the room
  air does, and they give it up through a surface film rather than instantly.
  Folding them into the air CV's own heat capacity therefore gets the storage
  right and the dynamics wrong: the room ends up with one time constant when a
  real one has two, a fast one for the air and a slow one for the slab. This
  gives each masked CV a second node instead, of the same volume as the CV and
  coupled to the air across the CV's height.

  Attributes:
    material: effective properties of the mass. density * heat_capacity is the
      volumetric capacity of the whole CV, and conductivity sets the coupling
      conductance k * u * v / z; neither is a handbook value for concrete.
    mask: True at every CV that carries a mass node.
  """

  material: MaterialProperties
  mask: np.ndarray


def _check_room_sizes(matrix_shape: Shape2D, room_shape: Shape2D) -> None:
  """Raises a ValueError if room_shape is not compatible with matrix_shape.

  The matrix for the building includes 2 outer wall layers, then rooms divided
  by walls.

  Args:
    matrix_shape: 2-Tuple representing shape of a matrix.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
  """
  if (matrix_shape[0] - 3) % (room_shape[0] + 1) != 0:
    raise ValueError("Room_shape[0] is not compatible with matrix_shape[0]")

  if (matrix_shape[1] - 3) % (room_shape[1] + 1) != 0:
    raise ValueError("Room_shape[1] is not compatible with matrix_shape[1]")


def assign_building_exterior_values(array: np.ndarray, value: float) -> None:
  """Assigns value to the building's exterior locations.

  The outer 2 layers of the matrix are special CVs which represent the thicker
  exterior walls as well as the ambient air.

  Args:
    array: Numpy array to assign values to.
    value: Value to assign.
  """
  array[:, [0, 1, -2, -1]] = value
  array[[0, 1, -2, -1], :] = value


def assign_interior_wall_values(
    array: np.ndarray, value: float, room_shape: Shape2D
) -> None:
  """Assigns value to interior wall locations.

  These are the walls dividing the rooms. None of these walls are on the
  outer 2 layers of the matrix which are reserved for the thicker outer walls.

  Args:
    array: Numpy array to assign values to.
    value: Value to assign.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
  """
  _check_room_sizes(array.shape, room_shape)
  nrows, ncols = array.shape

  for x in range(room_shape[0] + 2, nrows - 2, room_shape[0] + 1):
    for y in range(2, ncols - 2):
      array[x, y] = value
  for x in range(2, nrows - 2):
    for y in range(room_shape[1] + 2, ncols - 2, room_shape[1] + 1):
      array[x, y] = value


def generate_thermal_diffusers(
    matrix_shape: Shape2D, room_shape: Shape2D
) -> np.ndarray:
  """Returns a matrix with four thermal air diffusers for a VAV in each zone.

  This function places 4 diffusers in each room. The function aims to distribute
  them evenly in the room regardless of room size.

  Args:
    matrix_shape: 2-Tuple representing shape of a matrix.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
  """
  _check_room_sizes(matrix_shape, room_shape)

  n_diffusers_per_dim = 2

  # The sum of the diffuser's values in each room sum to 1.
  diffuser_value = 1 / n_diffusers_per_dim**2

  diffusers = np.zeros(shape=matrix_shape, dtype=np.float32)
  nrows, ncols = matrix_shape

  # First, number of non-diffuser spaces across each dimension is calculated
  empty_spaces_x = room_shape[0] - n_diffusers_per_dim

  # The empty spaces are distributed evenly between the diffusers and the walls.
  # This leads to 3 zones: wall to diff_1, diff_1 to diff_2, diff_2 to wall.
  diff_1_step_x = empty_spaces_x // 3

  # Put the second diffuser the same distance from the far wall.
  diff_2_step_x = room_shape[0] - diff_1_step_x - 1

  # Same steps for y dimension
  empty_spaces_y = room_shape[1] - n_diffusers_per_dim
  diff_1_step_y = empty_spaces_y // 3
  diff_2_step_y = room_shape[1] - diff_1_step_y - 1

  # room_start is the first empty space in each room, stop at the end of the
  # building.
  for room_start_x in range(2, nrows - 3, room_shape[0] + 1):
    for room_start_y in range(2, ncols - 3, room_shape[1] + 1):
      diffusers[room_start_x + diff_1_step_x, room_start_y + diff_1_step_y] = (
          diffuser_value
      )
      diffusers[room_start_x + diff_2_step_x, room_start_y + diff_1_step_y] = (
          diffuser_value
      )
      diffusers[room_start_x + diff_1_step_x, room_start_y + diff_2_step_y] = (
          diffuser_value
      )
      diffusers[room_start_x + diff_2_step_x, room_start_y + diff_2_step_y] = (
          diffuser_value
      )
  return diffusers


def get_zone_bounds(
    zone_coordinates: Coordinates2D, room_shape: Shape2D
) -> tuple[int, int, int, int]:
  """Returns (min_x, max_x, min_y, max_y) index bounds for specified zone.

  Helper function to get the minimum and maximum indices excluding walls in
  each dimension for specified zone.

  Args:
    zone_coordinates: Tuple containing x and y coordinates for zone.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
  """
  zone_x, zone_y = zone_coordinates
  x_min = zone_x * (room_shape[0] + 1) + 2
  x_max = x_min + room_shape[0] - 1
  y_min = zone_y * (room_shape[1] + 1) + 2
  y_max = y_min + room_shape[1] - 1
  return (x_min, x_max, y_min, y_max)


#### Helper code below here marks the updated helper functions that Lucas wrote:


def enlarge_exterior_walls(
    exterior_walls: building_utils.ExteriorWalls,
    interior_walls: building_utils.InteriorWalls,
) -> tuple[building_utils.ExteriorWalls, building_utils.InteriorWalls]:
  """Sequentially calls helper functions for expanding exterior walls.

  This function links together some necessary helper functions in
  building_utils.py so that it is clear and straightforward what they do when
  put in sequence. Given a FileInputFloorPlan, this function breaks out the
  necessary pieces of information for further processing.

  Args:
    exterior_walls: an ExteriorWalls noting where exterior walls are.
    interior_walls: an InteriorWalls noting where interior walls are.

  Returns:
    an ExteriorWalls with exterior walls expanded by
    constants.EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT.
    an InteriorWalls with interior walls shrunk by
    constants.EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT.
  """

  exterior_walls_binary = exterior_walls.copy()
  interior_walls_binary = interior_walls.copy()
  exterior_walls_binary = (
      exterior_walls_binary == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
  ).astype(np.uint8)
  interior_walls_binary = (
      interior_walls_binary == constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  ).astype(np.uint8)
  exterior_walls_augmented_temp = building_utils.enlarge_component(
      exterior_walls_binary, constants.EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT
  )
  walls_or_expanded = (
      exterior_walls_augmented_temp
      + interior_walls_binary
      + exterior_walls_binary
  )
  exterior_walls_augmented = np.int16(
      walls_or_expanded >= constants.WALLS_AND_EXPANDED_BOOLS
  ) * (constants.EXTERIOR_WALL_VALUE_IN_FUNCTION)
  interior_walls_shrunk = np.int16(
      interior_walls + exterior_walls_augmented
      == constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  ) * (constants.INTERIOR_WALL_VALUE_IN_FUNCTION)

  return exterior_walls_augmented, interior_walls_shrunk


def _assign_interior_and_exterior_values(
    exterior_walls: np.ndarray,
    interior_walls: np.ndarray,
    interior_wall_value: float,
    exterior_wall_value: float,
    interior_and_exterior_space_value: float,
) -> np.ndarray:
  """Assigns properties to interior and exterior walls.

  This differs from the original implementation in that it reads from
  pre-processed arrays noting where the exterior and interior inds are,
  whereas the original implementation simply counted, assuming rectangular
  rooms.

  Args:
    exterior_walls: an nd.array with constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
      as exterior walls and 0 otherwise.
    interior_walls: an nd.array with constants.INTERIOR_WALL_VALUE_IN_FUNCTION
      as interior walls and 0 otherwise.
    interior_wall_value: the value to assign to interior walls.
    exterior_wall_value: the value to assign to exterior spaces.
    interior_and_exterior_space_value: the value to assign to interior and
      exterior space.

  Returns:
    an np.ndarray with the appropriate values set.
  """
  array_to_return = np.where(
      interior_walls == constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
      interior_wall_value,
      np.where(
          exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION,
          exterior_wall_value,
          interior_and_exterior_space_value,
      ),
  )
  return array_to_return


def _construct_cv_type_array(
    exterior_walls: np.ndarray, exterior_space: np.ndarray
) -> np.ndarray:
  """Fills once the CV type matrix and save it.

  In the original imlementation,
      the sweep() function would call the get_cv_type() function every time,
      repeating logic that only needed to be computed once and saved.

  Args:
    exterior_walls: np.ndarray noting where exterior walls are
    exterior_space: np.ndarray noting where outside air is

  Returns:
    an np.array filled with strings of the cv type.
  """

  return np.where(
      exterior_space == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
      constants.LABEL_FOR_EXTERIOR_SPACE,
      np.where(
          exterior_walls == constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
          constants.LABEL_FOR_INTERIOR_SPACE,
          constants.LABEL_FOR_WALLS,
      ),
  )


def _assign_thermal_diffusers(
    array_to_fill: np.ndarray,
    room_dict: RoomIndicesDict,
    interior_walls: building_utils.InteriorWalls,
    diffuser_spacing: int = 10,
    buffer_from_walls: int = 5,
    min_room_size: int = 350,
) -> np.ndarray:
  """Places as many thermal diffusers in a zone as "diffuser_spacing" allows.

  The method by which assign_thermal_diffusers works has been updated to deal
  with rooms with differing geometries. It works as follows:

    First, test if the room is rectangular enough.
      If so, then allocate diffusers evenly in a 2D grid whose distance is
        formed by diffuser_spacing, and included only if the index is within the
        ind list for the entry to room_dict
      If not (and this can be fairly rare), then allocate the diffusers randomly

  It is different from the original method, which simply dispersed thermal
    diffusers in a grid determined by a value, "n_diffusers_per_zone", and
    did not consider any non-rectangular room. It would not work if, say,
    we considered the "room" made up of a windy hallway, or an "L" shaped room.

  assign_thermal_diffusers() is a placeholder until we have data on
    exactly where the diffusers are.

  Args:
    array_to_fill: an array prefilled with interior space values that this
      function will fill appropriately
    room_dict: a dict mapping room names to indices
    interior_walls: additional check to see if the allocated diffusers were
      placed in walls.
    diffuser_spacing: how many diffusers to have per control volume spacing.
    buffer_from_walls: how many CVs to leave in between each wall and each
      thermal diffuser
    min_room_size: minimum room size to place diffusers

  Returns:
    an np.ndarray with the appropriate values set.
  """

  for key, value in room_dict.items():
    if constants.is_non_physical_space(key):
      continue
    inds = thermal_diffuser_utils.diffuser_allocation_switch(
        room_cv_indices=value,
        spacing=diffuser_spacing,
        interior_walls=interior_walls,
        buffer_from_walls=buffer_from_walls,
        min_room_size=min_room_size,
    )
    num_inds = len(inds)
    for ind in inds:
      array_to_fill[tuple(ind)] = 1.0 / float(num_inds)

  return array_to_fill


class BaseSimulatorBuilding[ZoneKeyT](abc.ABC):
  """Base class for building simulators.

  The two building families identify zones differently: Building keys them by
  grid coordinates, FloorPlanBasedBuilding by room name. Code that works with
  either (the Simulator, for instance) parameterizes on this class with
  ZoneKeyT left unbound.
  """

  # Set by every concrete subclass in __init__ / reset. Declared here because
  # the Simulator drives the finite-differences sweep through them and only
  # ever sees a BaseSimulatorBuilding.
  cv_size_cm: float
  floor_height_cm: float
  temp: np.ndarray
  input_q: np.ndarray
  neighbors: list[list[list[Coordinates2D]]]

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the building to its initial parameters."""

  @abc.abstractmethod
  def get_zone_average_temps(self) -> dict[ZoneKeyT, Any]:
    """Returns the average temperature of each zone."""

  @abc.abstractmethod
  def apply_thermal_power_zone(self, zone: ZoneKeyT, /, power: float) -> None:
    """Applies thermal power to a zone, spread evenly across its diffusers.

    The zone argument is positional-only so subclasses are free to name it
    after whatever identifies a zone in their coordinate system.
    """

  @property
  @abc.abstractmethod
  def density(self) -> np.ndarray:
    """Returns the density array of the building."""

  @property
  @abc.abstractmethod
  def heat_capacity(self) -> np.ndarray:
    """Returns the heat capacity array of the building."""

  @property
  @abc.abstractmethod
  def conductivity(self) -> np.ndarray:
    """Returns the conductivity array of the building."""

  @property
  @abc.abstractmethod
  def cv_type(self) -> np.ndarray:
    """Returns the CV type array of the building."""

  @property
  def interior_mass(self) -> InteriorMass | None:
    """Returns the building's lumped interior thermal mass, if it has one.

    A building that models none returns None, and the solver then treats each
    CV's own heat capacity as all the storage there is.
    """
    return None


@gin.configurable
class Building(BaseSimulatorBuilding[Coordinates2D]):
  """Represents a matrix of volumes of material in a building.

  Attributes:
    cv_size_cm: Scalar in cm representing width, length and height of control
      volume.
    floor_height_cm: Height in cm floor to ceiling of each room.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
    building_shape: 2-Tuple representing the number of rooms in the width and
      length of the building.
    temp: The current temp in K of each control volume.
    conductivity: Thermal conductivity in of each control volume W/m/K.
    heat_capacity: Thermal heat cpacity of each control volume in J/kg/K.
    density: Material density in kg/m3 of each control volume.
    input_q: Heat energy applied (sign indicates heating/cooling) at the CV in W
      (J/s).
    diffusers: Proportion of the heat applied per VAV; sums to 1 for each zone.
    neighbors: Matrix containing list of neighbor coordinates for each control
      volume.
    cv_type: a matrix noting whether each CV is outside air, interior space, or
      a wall. cv_type will be used in the sweep() function.
    inside_air_properties: MaterialProperties for interior air.
    inside_wall_properties: MaterialProperties for interior walls.
    building_exterior_properties: MaterialProperties for building's exterior.
  """

  def __init__(
      self,
      cv_size_cm: float,
      floor_height_cm: float,
      room_shape: Shape2D,
      building_shape: Shape2D,
      initial_temp: float,
      inside_air_properties: MaterialProperties,
      inside_wall_properties: MaterialProperties,
      building_exterior_properties: MaterialProperties,
      deprecation: bool = False,
  ):
    """Initializes the ControlVolumes.

    Creates a matrix of control volumes representing the air and walls of a
    building. The size of each room (in terms of control volumes of air) is
    controlled by room_shape. The number of rooms in each building is controlled
    by building_shape. The outer 2 layers of the matrix represent special cells
    where the exterior walls and ambient air interact.

    Args:
      cv_size_cm: Width, length and height of control volume.
      floor_height_cm: Height in cm floor to ceiling of each room.
      room_shape: 2-Tuple representing the number of air control volumes in the
        width and length of each room.
      building_shape: 2-Tuple representing the number of rooms in the width and
        length of the building.
      initial_temp: Initial temperature for each control volume.
      inside_air_properties: MaterialProperties for interior air.
      inside_wall_properties: MaterialProperties for interior walls.
      building_exterior_properties: MaterialProperties for building's exterior.
      deprecation: if true, the old code has been deprecated and transitioned to
        the new, geometrically flexible code. TODO(spangher): change to True
        when the former code is deprecated.
    """

    self.cv_size_cm = cv_size_cm
    self.floor_height_cm = floor_height_cm
    self.room_shape = room_shape
    self.building_shape = building_shape
    self._initial_temp = initial_temp

    self.inside_air_properties = inside_air_properties
    self.inside_wall_properties = inside_wall_properties
    self.building_exterior_properties = building_exterior_properties

    if not deprecation:
      # TODO(sipple): delete the class when deprecation is finished.

      nrows = (self.room_shape[0] + 1) * self.building_shape[0] + 3
      ncols = (self.room_shape[1] + 1) * self.building_shape[1] + 3

      self._conductivity = np.full(
          (nrows, ncols), inside_air_properties.conductivity
      )
      assign_interior_wall_values(
          self._conductivity,
          inside_wall_properties.conductivity,
          self.room_shape,
      )
      assign_building_exterior_values(
          self._conductivity, building_exterior_properties.conductivity
      )

      self._heat_capacity = np.full(
          (nrows, ncols), inside_air_properties.heat_capacity
      )
      assign_interior_wall_values(
          self._heat_capacity,
          inside_wall_properties.heat_capacity,
          self.room_shape,
      )
      assign_building_exterior_values(
          self._heat_capacity, building_exterior_properties.heat_capacity
      )

      self._density = np.full((nrows, ncols), inside_air_properties.density)
      assign_interior_wall_values(
          self._density, inside_wall_properties.density, self.room_shape
      )
      assign_building_exterior_values(
          self._density, building_exterior_properties.density
      )

      self.diffusers = generate_thermal_diffusers(
          (nrows, ncols), self.room_shape
      )

      self.neighbors = self._calculate_neighbors((nrows, ncols))

      self.reset()

  @property
  def density(self) -> np.ndarray:
    return self._density

  @property
  def heat_capacity(self) -> np.ndarray:
    return self._heat_capacity

  @property
  def conductivity(self) -> np.ndarray:
    return self._conductivity

  @property
  def cv_type(self) -> np.ndarray:
    raise NotImplementedError()

  def reset(self) -> None:
    """Resets the building to its initial parameters."""
    nrows = (self.room_shape[0] + 1) * self.building_shape[0] + 3
    ncols = (self.room_shape[1] + 1) * self.building_shape[1] + 3
    self.temp = np.full((nrows, ncols), self._initial_temp)
    self.input_q = np.full((nrows, ncols), 0.0)

  def _calculate_neighbors(
      self, shape: Shape2D
  ) -> list[list[list[Coordinates2D]]]:
    """Returns matrix of list of neighbor indices for each location in a matrix.

    Args:
      shape: 2-Tuple representing the shape of a matrix.
    """
    neighbors = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

    for x in range(shape[0]):
      for y in range(shape[1]):
        possible_neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in possible_neighbors:
          if nx >= 0 and nx < shape[0] and ny >= 0 and ny < shape[1]:
            neighbors[x][y].append((nx, ny))
    return neighbors

  def get_zone_thermal_energy_rate(
      self, zone_coordinates: Coordinates2D
  ) -> float:
    """Returns energy rate in W being input to specified zone.

    Sums its CVs contributions.

    Calculates and returns sum of input_q of all air CVs in a given zone.

    Args:
      zone_coordinates: Tuple containing x and y coordinates for zone.
    """

    x_min, x_max, y_min, y_max = get_zone_bounds(
        zone_coordinates, self.room_shape
    )
    submat = self.input_q[x_min : x_max + 1, y_min : y_max + 1]
    return np.sum(submat)

  def get_zone_temp_stats(
      self, zone_coordinates: Coordinates2D
  ) -> tuple[float, float, float]:
    """Returns the min, max, and mean temp of all air CVs in zone.

    Args:
      zone_coordinates: Tuple containing x and y coordinates for zone.
    """

    x_min, x_max, y_min, y_max = get_zone_bounds(
        zone_coordinates, self.room_shape
    )
    submat = self.temp[x_min : x_max + 1, y_min : y_max + 1]
    return np.min(submat), np.max(submat), np.mean(submat)

  def get_zone_average_temps(self) -> dict[tuple[int, int], Any]:
    """Returns a dict of zone average temps.

    The dict is formatted as {`zone_coordinates`: `temp`}.
    """
    avg_temps = {}
    for zone_x in range(self.building_shape[0]):
      for zone_y in range(self.building_shape[1]):
        zone_coordinates = (zone_x, zone_y)
        _, _, avg_temp = self.get_zone_temp_stats(zone_coordinates)
        avg_temps[zone_coordinates] = avg_temp
    return avg_temps

  def apply_thermal_power_zone(
      self, zone_coordinates: Coordinates2D, power: float
  ) -> None:
    """Applies thermal power to zones, spread evenly across diffusers.

    The thermal power [W] is applied to zones `zone_x` and `zone_y`.

    Args:
       zone_coordinates: Tuple containing x and y coordinates for zone.
       power: Watts to apply to zone.
    """

    x_min, x_max, y_min, y_max = get_zone_bounds(
        zone_coordinates, self.room_shape
    )
    for x in range(x_min, x_max + 1):
      for y in range(y_min, y_max + 1):
        if self.diffusers[x, y] > 0.0:
          self.input_q[x, y] = power * self.diffusers[x, y]


@gin.configurable
class FloorPlanBasedBuilding(BaseSimulatorBuilding[str]):
  """Creates a Building that is floor plan based to avoid a messy deprecation.

  Attributes:
    cv_size_cm: Scalar in cm representing width, length and height of control
      volume.
    floor_height_cm: Height in cm floor to ceiling of each room.
    room_shape: 2-Tuple representing the number of air control volumes in the
      width and length of each room.
    building_shape: 2-Tuple representing the number of rooms in the width and
      length of the building.
    floor_plan_filepath: path to the floor plan npy file.
    zone_map_filepath: path to the zone map npy file.
    floor_plan: an np.ndarray representing the building's floor plan.
    temp: The current temp in K of each control volume.
    conductivity: Thermal conductivity in of each control volume W/m/K.
    heat_capacity: Thermal heat capacity of each control volume in J/kg/K.
    density: Material density in kg/m3 of each control volume.
    input_q: Heat energy applied (sign indicates heating/cooling) at the CV in W
      (J/s).
    diffusers: Proportion of the heat applied per VAV; sums to 1 for each zone.
    cv_type: a matrix noting whether each CV is outside air, interior space, or
      a wall. cv_type will be used in the sweep() function.
    neighbors: Matrix containing list of neighbor coordinates for each control
      volume.
    len_neighbors: matrix containing the length of neighbors
    inside_air_properties: MaterialProperties for interior air.
    inside_wall_properties: MaterialProperties for interior walls.
    building_exterior_properties: MaterialProperties for building's exterior.
    interior_mass: the slab and furnishings in the occupied space, or None.
  """

  def _check_floor_plan_and_zone_inputs(
      self,
      floor_plan: np.ndarray | None,
      floor_plan_filepath: str | None,
      zone_map: np.ndarray | None,
      zone_map_filepath: str | None,
      custom_room_dict: dict[str, list[tuple[int, int]]] | None,
      custom_zone_to_vavs: dict[str, list[str]] | None,
  ) -> None:
    """Checks for valid combinations of floor plan and zone inputs."""
    if floor_plan_filepath is not None and floor_plan is not None:
      raise ValueError(
          "You have provided both a floor_plan and a floor_plan_filepath. "
          "Please provide only one."
      )
    if floor_plan is None and floor_plan_filepath is None:
      raise ValueError(
          "Both floor_plan and floor_plan_filepath cannot be None."
      )

    if (
        zone_map_filepath is None
        and zone_map is None
        and custom_room_dict is None
    ):
      raise ValueError(
          "Please provide a zone_map_filepath or a zone_map or a"
          " custom_room_dict."
      )

    if zone_map_filepath is not None and zone_map is not None:
      raise ValueError(
          "You have provided both zone_map_filepath and a zone_map."
      )
    if custom_room_dict is not None and (
        zone_map is not None or zone_map_filepath is not None
    ):
      raise ValueError(
          "You have provided both custom_room_dict and a zone_map or"
          " zone_map_filepath."
      )

    if custom_zone_to_vavs is not None and custom_room_dict is None:
      raise ValueError(
          "custom_zone_to_vavs cannot be provided without custom_room_dict."
      )

  def __init__(
      self,
      cv_size_cm: float,
      floor_height_cm: float,
      initial_temp: float,
      inside_air_properties: MaterialProperties,
      inside_wall_properties: MaterialProperties,
      building_exterior_properties: MaterialProperties,
      interior_mass_properties: MaterialProperties | None = None,
      zone_map: np.ndarray | None = None,
      zone_map_filepath: str | None = None,
      custom_room_dict: dict[str, list[tuple[int, int]]] | None = None,
      custom_zone_to_vavs: dict[str, list[str]] | None = None,
      floor_plan: np.ndarray | None = None,
      floor_plan_filepath: str | None = None,
      buffer_from_walls: int = 3,
      convection_simulator: base_convection_simulator.BaseConvectionSimulator | None = None,
      reset_temp_values: np.ndarray | None = None,
      min_room_size: int = 350,
      min_zone_floor_area_m2: float = 20.0,
  ):
    """Initializes the New Building.

    Args:
      cv_size_cm: Width, length and height of control volume.
      floor_height_cm: Height in cm floor to ceiling of each room.
      initial_temp: Initial temperature for each control volume.
      inside_air_properties: MaterialProperties for interior air.
      inside_wall_properties: MaterialProperties for interior walls.
      building_exterior_properties: MaterialProperties for building's exterior.
      interior_mass_properties: MaterialProperties for the slab and furnishings
        in the occupied space. None, the default, models no interior mass, and
        leaves the air CVs' own heat capacity as the only storage in a room.
      zone_map: an np.ndarray noting where the VAV zones are.
      zone_map_filepath: a string of where to find the zone_map in CNS. Note
        that the user requires only to provide one of either zone_map_filepath
        or zone_map.
      custom_room_dict: a dictionary that maps custom zone names to their
        corresponding control volumes. Example: {'zone_1': [(2, 3), (2, 4)],
          'zone_2': [(10, 11), (10, 12)]}
      custom_zone_to_vavs: a dictionary that maps custom zone names to their
        corresponding VAVs. This should only be provided if a custom_room_dict
        is provided. Example:
        {'zone_1': ['vav_1'], 'zone_2': ['vav_2', 'vav_3']}
      floor_plan: an np.ndarray to pass into the function if one has this. If
        this is None, then the user must pass in a filepath.
      floor_plan_filepath: a string of where to find the floor_plan in CNS. Both
        floor_plan and floor_plan_filepath may not be None in the new code.
        debugging purposes.
      buffer_from_walls: int to note the space to put between thermal diffusers
        and walls
      convection_simulator: object to simulate air convection
      reset_temp_values: Temp values to use when resetting the building
      min_room_size: The minimum number of control volumes a room must have to
        be considered for diffuser placement.
      min_zone_floor_area_m2: Smallest floor area, in m2, a zone may have. Zones
        below this are rejected at construction rather than simulated. Set to
        0.0 to disable the check. See _check_zone_floor_areas for why it exists.
    """
    # consider super call!
    self.floor_plan_filepath = floor_plan_filepath
    self.zone_map_filepath = zone_map_filepath
    self.cv_size_cm = cv_size_cm
    self.floor_height_cm = floor_height_cm
    self.inside_air_properties = inside_air_properties
    self.inside_wall_properties = inside_wall_properties
    self.building_exterior_properties = building_exterior_properties
    self._initial_temp = initial_temp
    self._convection_simulator = convection_simulator
    self._reset_temp_values = reset_temp_values
    self._min_room_size = min_room_size

    self._check_floor_plan_and_zone_inputs(
        floor_plan=floor_plan,
        floor_plan_filepath=floor_plan_filepath,
        zone_map=zone_map,
        zone_map_filepath=zone_map_filepath,
        custom_room_dict=custom_room_dict,
        custom_zone_to_vavs=custom_zone_to_vavs,
    )

    if floor_plan_filepath is not None:
      self.floor_plan, _ = building_utils.read_floor_plan_from_filepath(
          floor_plan_filepath
      )
    elif floor_plan is not None:
      self.floor_plan = np.asarray(floor_plan)
    else:
      # This case should be caught by _check_floor_plan_and_zone_inputs,
      # but included for type consistency.
      raise ValueError(
          "Both floor_plan and floor_plan_filepath cannot be None."
      )

    if zone_map is not None:
      self._zone_map = np.asarray(zone_map)
      zone_map = self._zone_map
    elif zone_map_filepath is not None:
      zone_map, _ = building_utils.read_floor_plan_from_filepath(
          zone_map_filepath
      )
      self._zone_map = zone_map
    else:
      self._zone_map = None

    effective_zone_map = zone_map if zone_map is not None else self.floor_plan
    (
        self._room_dict,
        exterior_walls,
        interior_walls,
        self._exterior_space,
        offset,
    ) = building_utils.construct_building_data_types(
        floor_plan=self.floor_plan, zone_map=effective_zone_map
    )
    # quick fix is to overwrite, might delete the autodetected roomdict later
    # we keep this call for backward compatibility with the old approach
    # where zones were automatically detected.
    if custom_room_dict is not None:
      self._room_dict = {
          k: [(r + offset[0], c + offset[1]) for (r, c) in v]
          for k, v in custom_room_dict.items()
      }

    self._custom_room_dict = custom_room_dict
    self._custom_zone_to_vavs = custom_zone_to_vavs
    self._zone_indices_cache = {}
    self._min_zone_floor_area_m2 = min_zone_floor_area_m2
    self._check_zone_floor_areas()

    self._exterior_walls, self._interior_walls = enlarge_exterior_walls(
        exterior_walls=exterior_walls, interior_walls=interior_walls
    )

    self._conductivity = _assign_interior_and_exterior_values(
        exterior_walls=self._exterior_walls,
        interior_walls=self._interior_walls,
        interior_wall_value=inside_wall_properties.conductivity,
        exterior_wall_value=building_exterior_properties.conductivity,
        interior_and_exterior_space_value=inside_air_properties.conductivity,
    )

    self._heat_capacity = _assign_interior_and_exterior_values(
        exterior_walls=self._exterior_walls,
        interior_walls=self._interior_walls,
        interior_wall_value=inside_wall_properties.heat_capacity,
        exterior_wall_value=building_exterior_properties.heat_capacity,
        interior_and_exterior_space_value=inside_air_properties.heat_capacity,
    )

    self._density = _assign_interior_and_exterior_values(
        exterior_walls=self._exterior_walls,
        interior_walls=self._interior_walls,
        interior_wall_value=inside_wall_properties.density,
        exterior_wall_value=building_exterior_properties.density,
        interior_and_exterior_space_value=inside_air_properties.density,
    )

    self.diffusers = np.zeros(self._exterior_walls.shape)
    self.diffusers = _assign_thermal_diffusers(
        self.diffusers,
        room_dict=self._room_dict,
        interior_walls=interior_walls,
        buffer_from_walls=buffer_from_walls,
        min_room_size=self._min_room_size,
    )

    self._cv_type = _construct_cv_type_array(
        self._exterior_walls, self._exterior_space
    )

    # The slab and the furnishings stand in the occupied space; the walls carry
    # their own mass already, and the CVs outside the building carry none.
    self._interior_mass = (
        None
        if interior_mass_properties is None
        else InteriorMass(
            interior_mass_properties,
            (self._exterior_walls != constants.EXTERIOR_WALL_VALUE_IN_FUNCTION)
            & (
                self._interior_walls
                != constants.INTERIOR_WALL_VALUE_IN_FUNCTION
            )
            & (
                self._exterior_space
                != constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION
            ),
        )
    )

    self.neighbors = self._calculate_neighbors()
    self.len_neighbors = self._calculate_length_of_neighbors()

    self.reset()

  @property
  def density(self) -> np.ndarray:
    """Returns the density array."""
    return self._density

  @property
  def heat_capacity(self) -> np.ndarray:
    """Returns the heat capacity array."""
    return self._heat_capacity

  @property
  def conductivity(self) -> np.ndarray:
    """Returns the conductivity array."""
    return self._conductivity

  @property
  def cv_type(self) -> np.ndarray:
    """Returns the cv_type array."""
    return self._cv_type

  @property
  def interior_mass(self) -> InteriorMass | None:
    """Returns the slab and furnishings in the occupied space, if modelled."""
    return self._interior_mass

  @property
  def room_dict(self) -> Mapping[str, list[tuple[int, int]]]:
    """The room dictionary mapping room names to CV coordinates."""
    return self._room_dict

  @property
  def custom_zone_to_vavs(self) -> Mapping[str, list[str]] | None:
    """The custom zone to vavs mapping."""
    return self._custom_zone_to_vavs

  @property
  def initial_temp(self) -> float:
    """Returns the initial temperature for the building."""
    return self._initial_temp

  @property
  def convection_simulator(
      self,
  ) -> base_convection_simulator.BaseConvectionSimulator | None:
    """The convection simulator for the building."""
    return self._convection_simulator

  def reset(self) -> None:
    self.temp = np.full(
        shape=self._exterior_walls.shape, fill_value=self._initial_temp
    )

    if self._reset_temp_values is not None:
      self.temp = np.copy(self._reset_temp_values)

    self.input_q = np.zeros(self._exterior_walls.shape)

  def _calculate_neighbors(self) -> list[list[list[Coordinates2D]]]:
    """Returns matrix of list of neighbor indices for each location in a matrix.

    Returns:
      A list of CVs that are neighbors with respect to the building.
    """
    shape = self._exterior_walls.shape
    neighbors = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

    for x in range(shape[0]):
      for y in range(shape[1]):
        if self.cv_type[x][y] == constants.LABEL_FOR_EXTERIOR_SPACE:
          continue

        possible_neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in possible_neighbors:
          if nx >= 0 and nx < shape[0] and ny >= 0 and ny < shape[1]:
            if self.cv_type[nx][ny] != constants.LABEL_FOR_EXTERIOR_SPACE:
              neighbors[x][y].append((nx, ny))
    return neighbors

  def _calculate_length_of_neighbors(self) -> np.ndarray:
    """Calculates an array with the length of neighbors noted."""

    len_neighbors = np.full(shape=self._exterior_walls.shape, fill_value=0)
    for x in range(len_neighbors.shape[0]):
      for y in range(len_neighbors.shape[1]):
        len_neighbors[x][y] = len(self.neighbors[x][y])

    return len_neighbors

  def _zone_indices(self, zone_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns a zone's CV coordinates as (row, column) index arrays.

    Zone membership is fixed once the building is constructed, so the index
    arrays are built on first use and cached. Indexing a grid with them is
    what lets the zone accessors below work on whole zones at once instead of
    looping in Python over every CV.

    Args:
      zone_name: a string with the name of the zone. Needs to be present in
        self.room_dict.

    Returns:
      the row indices and the column indices of the zone's CVs.
    """

    indices = self._zone_indices_cache.get(zone_name)
    if indices is None:
      if zone_name not in self._room_dict.keys():
        raise ValueError("Zone name is not present in room_dict.")
      coordinates = np.asarray(
          self._room_dict[zone_name], dtype=np.intp
      ).reshape(-1, 2)
      indices = (coordinates[:, 0], coordinates[:, 1])
      self._zone_indices_cache[zone_name] = indices
    return indices

  def get_zone_thermal_energy_rate(self, zone_name: str) -> float:  # pylint: disable=arguments-renamed
    """Returns energy rate in W being input to specified zone.

    Sums its CVs contributions.

    Calculates and returns sum of input_q of all air CVs in a given zone.

    Args:
      zone_name: a string with the name of the zone to calculate over. Needs to
        be present in self.room_dict.

    Returns:
      the thermal energy rate of the zone.
    """

    return np.sum(self.input_q[self._zone_indices(zone_name)])

  def get_zone_temp_stats(self, zone_name: str) -> tuple[float, float, float]:  # pylint: disable=arguments-renamed
    """Returns the min, max, and mean temp of all air CVs in zone.

    Args:
      zone_name: a string with the name of the zone to calculate over. Needs to
        be present in self.room_dict.

    Returns:
      the thermal energy rate of the zone.
    """

    temps = self.temp[self._zone_indices(zone_name)]
    return np.min(temps), np.max(temps), np.mean(temps)

  def get_zone_average_temps(self) -> dict[str, Any]:
    """Returns a dict of zone average temps.

    The dict is formatted as: {`zone_coordinates`: `temp`}.
    """
    avg_temps = {}

    for zone in self._room_dict.keys():
      if constants.is_non_physical_space(zone):
        continue
      _, _, avg_temp = self.get_zone_temp_stats(zone)
      avg_temps[zone] = avg_temp
    return avg_temps

  def apply_thermal_power_zone(self, zone_name: str, power: float) -> None:
    """Applies thermal power to zones, spread evenly across diffusers.

    The thermal power [W] is applied to zones `zone_x` and `zone_y`.

    Args:
      zone_name: a string with the name of the zone to calculate over. Needs to
        be present in self.room_dict.
      power: Watts to apply to zone.
    """

    rows, columns = self._zone_indices(zone_name)
    if rows.size == 0:
      return
    # The zone's thermal power is spread over every cell in the zone, not
    # concentrated on the cells beneath a diffuser.
    #
    # A diffuser blows air into a room and it mixes. Depositing a zone's entire
    # load into the small parcel of air at the vent is a modelling artifact
    # rather than physics, and it is also numerically hostile: a single air
    # control volume holds on the order of 100 J/K, so kilowatts applied to it
    # over a 300 s step imply a temperature change of hundreds of kelvin and
    # drive the finite-difference solve outside physical bounds.
    #
    # Spreading the load raises the receiving heat capacity by two to three
    # orders of magnitude, which is both more realistic and stable.
    self.input_q[rows, columns] = power / float(rows.size)

  def _check_zone_floor_areas(self) -> None:
    """Rejects a floor plan carrying zones too small to be rooms.

    The floor plan generator honours min_room_size when it splits the BSP tree,
    but the composite shape mask is applied afterwards and clips any leaf that
    straddles its boundary. What survives is a connected component whose size
    nothing ever checked, and it can be a sliver: building_train_5 and
    building_train_7 share min_room_size 25 and composite_coverage 0.67 and
    come out with smallest zones of 9.36 m2 and 50.88 m2. So the realised
    minimum is not settable from the floor plan parameters and has to be
    measured here, after the zones exist.

    A sliver is not merely unrealistic, it diverges. Occupant heat is assigned
    per zone -- zone_assignment occupants at occupant_sensible_heat_w each --
    while internal gains, the VAV's share of airflow, and the mean temperature
    its thermostat reads are all weighted by floor area. So a small zone takes
    a full zone's worth of occupant heat into very little air while contributing
    almost nothing to the signal the loop controlling it can see. Measured on
    building_train_8 before this check existed: room_1, 9.88 m2 and 35,746 J/K,
    reached 149.2 C over a scored episode, and the 9.36 m2 room_23 of
    building_train_5 reached 141.4 C. Nothing bounded them but the solver's own
    np.clip(temp, 150, 500), which tops out near 227 C.

    It also corrupts the reward rather than just the trajectory. Comfort is 0.6
    of the reward, its shortfall is unbounded beyond the deviation reference,
    and productivity is summed over zones weighted by each zone's own occupancy
    rather than its area. A zone 167 K outside the band scores a productivity
    factor near -1050 against about +1 for an in-band zone, with the same
    weight, so one sliver decides the scenario.

    Twenty square metres is the floor because the runaway stops by about 17 m2
    (roughly 60,000 J/K) in measurement, and because 20 m2 is 10 m2 per
    occupant at the shipped zone_assignment of 2, an ordinary office density.
    That is what makes the per-zone occupancy assumption sound, which is why
    this is a geometry check and not a change to the occupancy model.

    Raises:
      ValueError: if any zone's floor area is below min_zone_floor_area_m2.
    """
    if self._min_zone_floor_area_m2 <= 0.0:
      return
    cv_area_m2 = (self.cv_size_cm / 100.0) ** 2
    offenders = []
    for zone_name, coords in self._room_dict.items():
      if zone_name in ("exterior_space", "interior_wall"):
        continue
      area_m2 = len(coords) * cv_area_m2
      if area_m2 < self._min_zone_floor_area_m2:
        offenders.append((zone_name, area_m2, len(coords)))
    if not offenders:
      return
    offenders.sort(key=lambda o: o[1])
    listed = ", ".join(
        f"{name} {area:.2f} m2 ({cells} cells)" for name, area, cells in offenders
    )
    raise ValueError(
        f"{len(offenders)} zone(s) are below the {self._min_zone_floor_area_m2}"
        f" m2 minimum floor area: {listed}. A zone this small is a floor plan"
        " artifact, not a room, and it diverges: occupant heat is assigned per"
        " zone while the HVAC authority acting on it scales with area. Draw the"
        " building with a different seed and check the realised minimum, since"
        " min_room_size does not control it. Pass"
        " min_zone_floor_area_m2=0.0 to disable this check."
    )

  def zone_air_heat_capacity(self, zone_name: str) -> float:
    """Heat capacity in J/K of the cells a zone's thermal power is spread over.

    Matches the distribution apply_thermal_power_zone uses, so a caller can ask
    how much temperature change a given power implies over one timestep.
    """
    rows, columns = self._zone_indices(zone_name)
    if rows.size == 0:
      return 0.0
    cv_volume = (self.cv_size_cm / 100.0) ** 2 * (self.floor_height_cm / 100.0)
    return float(
        np.sum(self.density[rows, columns] * self.heat_capacity[rows, columns])
        * cv_volume
    )

  def apply_convection(self) -> None:
    if self._convection_simulator is not None:
      self._convection_simulator.apply_convection(self._room_dict, self.temp)
