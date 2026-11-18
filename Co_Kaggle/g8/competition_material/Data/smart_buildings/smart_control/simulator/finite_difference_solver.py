"""Finite difference calculator for buildings with many control volumes.

Advancing the temperature field by one timestep is a sparse linear solve, and
this module hands that solve to scipy. FiniteDifferenceSolver assembles the
backward-Euler system once from the building's geometry and materials,
factorizes it once, and reuses the factorization for the rest of the episode.
It reads nothing but the thermal grid, so it serves every building family and
every Simulator.

Historically the discretization was written in TensorFlow and solved by
repeated Jacobi sweeps, because a padded-grid tensor program cannot express
anything better: there is no sparse matrix to hand to a real solver. The
sweeps were stopped on a temperature-change threshold and capped by an
iteration limit, so on stiff steps the solve was truncated part-way and the
building acquired an artificial thermal lag that varied with the weather. A
direct factorization removes both the truncation and the knobs that controlled
it.
"""

import enum
import functools
from collections.abc import Mapping, Sequence

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from absl import logging

from smart_buildings.smart_control.simulator import building as building_py

# List of neighbors for a specific CV.
NeighborCoordinates = Sequence[building_py.Coordinates2D]
# Neighors, indexable by a specific i,j coord.
Neighbors = Sequence[Sequence[NeighborCoordinates]]


# All CVs must be interior, exterior, or boundary positions.
class CVPositionType(enum.Enum):
  INTERIOR = 1
  EXTERIOR = 2
  BOUNDARY = 3


# All Boundary CVs must be either edge (three non-exterior neighbors), or
# corner (two non-exterior neighbors).
class CVBoundaryType(enum.Enum):
  EDGE = 1
  CORNER = 2


# All edge CVs must be top, bottom, left or right edges.
class CVEdgeOrientationType(enum.Enum):
  """Classifies edge CVs by orientation."""

  TOP = (1, 0.5, 1.0)
  BOTTOM = (2, 0.5, 1.0)
  LEFT = (3, 1.0, 0.5)
  RIGHT = (4, 1.0, 0.5)

  def __init__(
      self,
      value: int,
      vertical_scale_factor: float,
      horizontal_scale_factor: float,
  ):
    self._value_ = value
    self.vertical_scale_factor = vertical_scale_factor
    self.horizontal_scale_factor = horizontal_scale_factor


# All corner CVs must be top left, bottom right, bottom left, or top right.
class CVCornerOrientationType(enum.Enum):
  TOP_LEFT = (1, 0.5)
  BOTTOM_LEFT = (2, 0.5)
  BOTTOM_RIGHT = (3, 0.5)
  TOP_RIGHT = (4, 0.5)

  def __init__(self, value: int, scale_factor: float):
    self._value_ = value
    self.scale_factor = scale_factor


class CVType:
  """Classifies a CV based on its neighborhood and orientation.

  A CV can be interior with four neighbors, exterior with zero or one
  neighbor, or boundary with two or three neighbors. A boundary CV
  can be an edge with three neighbors, or a corner with two neighbors.

  Using a regular class instead of a data class to ensure consistent
  assignment of CV proprties.
  """

  def __init__(
      self,
      position: CVPositionType,
      boundary: CVBoundaryType | None = None,
      edge: CVEdgeOrientationType | None = None,
      corner: CVCornerOrientationType | None = None,
  ):
    self.position = position
    self.boundary = boundary
    self.edge = edge
    self.corner = corner

    if self.position in [CVPositionType.EXTERIOR, CVPositionType.INTERIOR]:
      if (
          self.boundary is not None
          or self.edge is not None
          or self.corner is not None
      ):
        raise ValueError(
            'Cannot assign boundary properties to a non-boundary CV.'
        )
    elif self.position == CVPositionType.BOUNDARY:
      if self.edge is None and self.corner is None:
        raise ValueError('Boundary CVs must have edge or corner properties')

      if self.edge is not None and self.corner is not None:
        raise ValueError('Boundary CVs cannot be edge AND corner.')

      if self.boundary == CVBoundaryType.EDGE and self.edge is None:
        raise ValueError('Edge CVs must be assigned an edge orientation.')

      if self.boundary == CVBoundaryType.CORNER and self.corner is None:
        raise ValueError('Corner CVs must be assigned a corner orientation.')

  def __eq__(self, other) -> bool:

    if isinstance(other, CVType):
      return (
          self.position == other.position
          and self.boundary == other.boundary
          and self.edge == other.edge
          and self.corner == other.corner
      )
    return False

  @property
  def horizontal_scale_factor(self) -> float:
    """Returns the horizontal scale factor for this CV."""
    if self.position != CVPositionType.BOUNDARY:
      return 1.0
    match self.boundary:
      case CVBoundaryType.CORNER:
        if self.corner is None:
          raise ValueError('Corner CVs must be assigned a corner orientation.')
        return self.corner.scale_factor
      case CVBoundaryType.EDGE:
        if self.edge is None:
          raise ValueError('Edge CVs must be assigned an edge orientation.')
        return self.edge.horizontal_scale_factor
    raise ValueError('horizontal_scale_factor is not defined for this CV type.')

  @property
  def vertical_scale_factor(self) -> float:
    """Returns the vertical scale factor for this CV."""
    if self.position != CVPositionType.BOUNDARY:
      return 1.0
    match self.boundary:
      case CVBoundaryType.CORNER:
        if self.corner is None:
          raise ValueError('Corner CVs must be assigned a corner orientation.')
        return self.corner.scale_factor
      case CVBoundaryType.EDGE:
        if self.edge is None:
          raise ValueError('Edge CVs must be assigned an edge orientation.')
        return self.edge.vertical_scale_factor
    raise ValueError('vertical_scale_factor is not defined for this CV type.')


BoundaryCVMapping = Mapping[building_py.Coordinates2D, CVType]


def classify_cv(
    coords: building_py.Coordinates2D, neighbors: Neighbors
) -> CVType:
  """Classifies a CV based on its neighborhood.

  0-1 neighbors: EXTERIOR
  2 neighbors: CORNER
  3 neighbors: EDGE
  4 neighbors: INTERIOR

  Args:
    coords: i, j coords of the CV
    neighbors:  a lookup table of all the neighbors.

  Returns:
    a string that represents the CV
  """

  def _cv_type_corner_factory(
      corner: CVCornerOrientationType,
  ) -> CVType:

    return CVType(
        position=CVPositionType.BOUNDARY,
        boundary=CVBoundaryType.CORNER,
        corner=corner,
    )

  def _cv_corner_type(cv_neighbors: Sequence[tuple[int, int]]) -> CVType:
    if set([(i + 1, j), (i, j + 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(corner=CVCornerOrientationType.TOP_LEFT)
    if set([(i - 1, j), (i, j + 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(corner=CVCornerOrientationType.BOTTOM_LEFT)
    if set([(i + 1, j), (i, j - 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(corner=CVCornerOrientationType.TOP_RIGHT)
    if set([(i - 1, j), (i, j - 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(
          corner=CVCornerOrientationType.BOTTOM_RIGHT
      )
    raise ValueError(
        f"Wasn't able to determine which corner the CV {(i, j)} is."
    )

  def _cv_type_edge_factory(
      edge: CVEdgeOrientationType,
  ) -> CVType:

    return CVType(
        position=CVPositionType.BOUNDARY,
        boundary=CVBoundaryType.EDGE,
        edge=edge,
    )

  def _cv_edge_type(cv_neighbors: Sequence[tuple[int, int]]) -> CVType:
    edge = functools.partial(_cv_type_edge_factory)
    if set([(i, j + 1), (i, j - 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.TOP)
    if set([(i, j - 1), (i, j + 1), (i - 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.BOTTOM)
    if set([(i - 1, j), (i, j + 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.LEFT)
    if set([(i - 1, j), (i, j - 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.RIGHT)
    raise ValueError(f"Wasn't able to determine which edge the CV {(i, j)} is.")

  i, j = coords

  cv_neighbors = neighbors[i][j]
  match len(cv_neighbors):
    case 0:
      return CVType(position=CVPositionType.EXTERIOR)
    case 1:
      return CVType(position=CVPositionType.EXTERIOR)
    case 2:
      return _cv_corner_type(cv_neighbors)
    case 3:
      return _cv_edge_type(cv_neighbors)
    case 4:
      return CVType(position=CVPositionType.INTERIOR)
    case _:
      raise ValueError(
          f"Wasn't able to determine which CV type the CV {(i, j)} is."
      )


def get_cv_mapping(
    neighbors: Neighbors,
    position_criterion: CVPositionType = CVPositionType.BOUNDARY,
) -> BoundaryCVMapping:
  """Gets a map of all the non-interior cv_types."""

  boundary_cv_mapping = {}
  ni = len(neighbors)
  nj = len(neighbors[0])
  for i in range(ni):
    for j in range(nj):
      cv_type = classify_cv((i, j), neighbors)
      if cv_type.position == position_criterion:
        boundary_cv_mapping[(i, j)] = cv_type

  return boundary_cv_mapping


def get_cv_dimension_tensors(
    control_volume_cm: float,
    boundary_cv_mapping: BoundaryCVMapping,
    shape=tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
  """Returns horizontal and vertical CV dimension tensors.

  Boundary control volumes have varying heights and widths, since they share
  a part with the outside. For example, edge boundary CVs have one side
  that is half width, depending on orientation, and corner CVs have both edges
  half width.

  Args:
    control_volume_cm: centimeters width of each interior control volume.
    boundary_cv_mapping: dict of all boundary elements.
    shape: shape of the target matrix.

  Returns:
    horizontal (u), vertical (v) dimension tensors.
  """

  def _compute_cv_dimension(cv_type: CVType) -> tuple[float, float]:
    """Sets the horizontal and vertical CV dimension for one CV."""

    return (
        control_volume_cm * cv_type.horizontal_scale_factor,
        control_volume_cm * cv_type.vertical_scale_factor,
    )

  horizontal_cv_dimension = np.full(shape, control_volume_cm, dtype=np.float64)
  vertical_cv_dimension = np.full(shape, control_volume_cm, dtype=np.float64)

  for (i, j), cv_type in boundary_cv_mapping.items():

    cv_horizontal_cv_dimension, cv_vertical_cv_dimension = (
        _compute_cv_dimension(cv_type)
    )
    vertical_cv_dimension[i][j] = cv_vertical_cv_dimension
    horizontal_cv_dimension[i][j] = cv_horizontal_cv_dimension

  return horizontal_cv_dimension, vertical_cv_dimension


# The two faces a corner CV is missing a neighbour on, and so exchanges heat
# with the outside air across.
_CORNER_EXPOSED_EDGES: Mapping[
    CVCornerOrientationType, tuple[CVEdgeOrientationType, ...]
] = {
    CVCornerOrientationType.TOP_LEFT: (
        CVEdgeOrientationType.TOP,
        CVEdgeOrientationType.LEFT,
    ),
    CVCornerOrientationType.TOP_RIGHT: (
        CVEdgeOrientationType.TOP,
        CVEdgeOrientationType.RIGHT,
    ),
    CVCornerOrientationType.BOTTOM_LEFT: (
        CVEdgeOrientationType.BOTTOM,
        CVEdgeOrientationType.LEFT,
    ),
    CVCornerOrientationType.BOTTOM_RIGHT: (
        CVEdgeOrientationType.BOTTOM,
        CVEdgeOrientationType.RIGHT,
    ),
}


def get_ambient_face_areas(
    boundary_cv_mapping: BoundaryCVMapping,
    horizontal_cv_dimension: np.ndarray,
    vertical_cv_dimension: np.ndarray,
    floor_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns each CV's exposed area, split by the axis the face normal runs on.

  Heat leaving a boundary CV for the outside air crosses two resistances in
  series: half of the CV's own thickness, then the surface film. The half
  thickness differs by axis -- a left or right face is half a
  horizontal_cv_dimension in from the surface, a top or bottom face is half a
  vertical_cv_dimension -- so the two groups cannot be summed into one area and
  are returned separately for `get_ambient_conductances` to weight.

  Args:
    boundary_cv_mapping: dict of boundary CVs.
    horizontal_cv_dimension: width (u) of each CV, in m.
    vertical_cv_dimension: height (v) of each CV, in m.
    floor_height_m: depth (z) of every CV, in m.

  Returns:
    (horizontal, vertical): exposed area in m2 for every CV, for faces whose
    normal runs along the horizontal axis and along the vertical axis
    respectively. Both are zero away from the boundary.
  """
  # Left and right faces span the CV's height; top and bottom faces its width.
  face_area = {
      CVEdgeOrientationType.LEFT: floor_height_m * vertical_cv_dimension,
      CVEdgeOrientationType.RIGHT: floor_height_m * vertical_cv_dimension,
      CVEdgeOrientationType.TOP: floor_height_m * horizontal_cv_dimension,
      CVEdgeOrientationType.BOTTOM: floor_height_m * horizontal_cv_dimension,
  }
  along_horizontal = (CVEdgeOrientationType.LEFT, CVEdgeOrientationType.RIGHT)

  areas = (
      np.zeros_like(horizontal_cv_dimension),
      np.zeros_like(vertical_cv_dimension),
  )
  for (i, j), cv_type in boundary_cv_mapping.items():
    if cv_type.edge is not None:
      exposed = (cv_type.edge,)
    elif cv_type.corner is not None:
      exposed = _CORNER_EXPOSED_EDGES[cv_type.corner]
    else:
      raise ValueError(f'Boundary CV {(i, j)} is neither an edge nor a corner.')
    for edge in exposed:
      areas[0 if edge in along_horizontal else 1][i][j] += face_area[edge][i][j]
  return areas


def get_ambient_conductances(
    ambient_face_areas: tuple[np.ndarray, np.ndarray],
    horizontal_cv_dimension: np.ndarray,
    vertical_cv_dimension: np.ndarray,
    conductivity: np.ndarray,
    convection_coefficient: float,
) -> np.ndarray:
  """Returns the conductance in W/K from every CV to the outside air.

  The path out of a boundary CV runs from its centre to its exposed surface
  through half the CV's thickness, then across the surface film into the wind:

      1 / G = dx / (2 k A)  +  1 / (h A)   ->   G = h A / (1 + h dx / 2 k)

  Leaving the wall term out -- treating the exchange as h * A alone -- makes
  the film the entire envelope, so no choice of material can insulate the
  building and the effective U is the film coefficient itself, tens of W/m2K.
  With the wall term in, `h` keeps its physical meaning as a wind-driven film
  coefficient and the exterior CV's conductivity sets the insulation level.

  Args:
    ambient_face_areas: (horizontal, vertical) exposed areas from
      `get_ambient_face_areas`.
    horizontal_cv_dimension: width (u) of each CV, in m.
    vertical_cv_dimension: height (v) of each CV, in m.
    conductivity: thermal conductivity of each CV, in W/mK.
    convection_coefficient: surface film coefficient h, in W/m2K.

  Returns:
    Conductance to ambient in W/K for every CV, zero away from the boundary.
  """
  # Written as h A / (1 + h dx / 2 k) rather than the reciprocal-sum form so
  # that still air, h = 0, gives no exchange instead of dividing by zero.
  conductivity = np.asarray(conductivity, dtype=np.float64)
  total = np.zeros_like(conductivity)
  for area, thickness in zip(
      ambient_face_areas, (horizontal_cv_dimension, vertical_cv_dimension)
  ):
    exposed = area > 0.0
    total[exposed] += (convection_coefficient * area[exposed]) / (
        1.0
        + convection_coefficient
        * 0.5
        * thickness[exposed]
        / conductivity[exposed]
    )
  return total


def get_interior_mass_terms(
    interior_mass: building_py.InteriorMass,
    horizontal_cv_dimension: np.ndarray,
    vertical_cv_dimension: np.ndarray,
    floor_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns each mass node's heat capacity in J/K and its coupling in W/K.

  The node fills the same volume as its CV and exchanges with the air across
  the CV's height, so

      C = rho c (u v z)        G = k (u v) / z

  and the pair sets both how much heat the slab and furnishings can hold and
  how fast the room can get at it. C / G is the time constant that separates
  the two, a few hours in a real building.

  Args:
    interior_mass: the mass material and the CVs that carry a node.
    horizontal_cv_dimension: width (u) of each CV, in m.
    vertical_cv_dimension: height (v) of each CV, in m.
    floor_height_m: depth (z) of every CV, in m.

  Returns:
    (capacity, conductance) for every CV, both zero away from the mask.
  """
  material = interior_mass.material
  footprint = horizontal_cv_dimension * vertical_cv_dimension
  capacity = np.where(
      interior_mass.mask,
      material.density * material.heat_capacity * footprint * floor_height_m,
      0.0,
  )
  conductance = np.where(
      interior_mass.mask,
      material.conductivity * footprint / floor_height_m,
      0.0,
  )
  return capacity, conductance


def get_oriented_conductivity_tensors(
    conductivity: np.ndarray,
    boundary_cv_mapping: BoundaryCVMapping,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Returns oriented conductivity tensors.

  Conduction is a function of two adjacent CVs. On boundary CVs, there will be
  no conduction transfer, k = 0. On internal CVs there will be heat transfer,
  k > 0.

  The conductivity is split by sides of the CV, top, bottom, left and right.
  Each entry is the CV's own material conductivity, so it describes only one
  half of a face; get_face_conductances combines the two halves.

  Args:
    conductivity: conductivity of the CV itself.
    boundary_cv_mapping: mapping that contains the coordinates of boundary CVs

  Returns:
    tensors for left, right, top, bottom conductivities.
  """

  k_left_edge = conductivity.copy()
  k_right_edge = conductivity.copy()
  k_top_edge = conductivity.copy()
  k_bottom_edge = conductivity.copy()

  for (i, j), cv_type in boundary_cv_mapping.items():
    if cv_type.position == CVPositionType.BOUNDARY:
      if cv_type.boundary == CVBoundaryType.EDGE:
        if cv_type.edge == CVEdgeOrientationType.TOP:
          k_top_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.LEFT:
          k_left_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.RIGHT:
          k_right_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.BOTTOM:
          k_bottom_edge[i][j] = 0.0

      if cv_type.boundary == CVBoundaryType.CORNER:
        if cv_type.corner == CVCornerOrientationType.TOP_LEFT:
          k_top_edge[i][j] = 0.0
          k_left_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.TOP_RIGHT:
          k_top_edge[i][j] = 0.0
          k_right_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.BOTTOM_LEFT:
          k_bottom_edge[i][j] = 0.0
          k_left_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.BOTTOM_RIGHT:
          k_bottom_edge[i][j] = 0.0
          k_right_edge[i][j] = 0.0

  return (
      k_left_edge.astype(np.float64),
      k_right_edge.astype(np.float64),
      k_top_edge.astype(np.float64),
      k_bottom_edge.astype(np.float64),
  )


def get_face_conductances(
    conductivity_tensors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    horizontal_cv_dimension: np.ndarray,
    vertical_cv_dimension: np.ndarray,
    floor_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns the conductance across every face between two CVs, in W/K.

  A face is shared by exactly two CVs, so its conductance has to be a single
  number, obtained by putting the half-cell resistance on either side in
  series:

      1 / G = dx_a / (2 k_a A) + dx_b / (2 k_b A)

  with A the shared face area. Reading the conductance off each CV's own
  material instead gives the two CVs different views of the same face, so the
  heat leaving one does not equal the heat arriving in the other: energy is
  created at every material interface, and the resulting matrix is not
  symmetric, which rules out every solver that needs positive definiteness.

  classify_cv calls a CV a LEFT edge when its j-1 neighbour is the missing
  one, so k_left_edge is the conductivity across the face toward j-1 and pairs
  with k_right_edge of the CV at j-1. Likewise k_top_edge faces i-1 and pairs
  with k_bottom_edge of the CV at i-1. A zero on either side marks a face that
  opens onto the outside air rather than onto a neighbour, and gives that face
  no conductance at all: the exchange there runs through the surface film, and
  get_ambient_conductances accounts for it.

  Args:
    conductivity_tensors: left, right, top and bottom oriented conductivities,
      as returned by get_oriented_conductivity_tensors.
    horizontal_cv_dimension: width (u) of each CV, in m.
    vertical_cv_dimension: height (v) of each CV, in m.
    floor_height_m: depth (z) of every CV, in m.

  Returns:
    Conductance across the faces separating (i, j) from (i, j + 1), of shape
    (rows, columns - 1), and across the faces separating (i, j) from
    (i + 1, j), of shape (rows - 1, columns).
  """
  k_left, k_right, k_top, k_bottom = conductivity_tensors

  def _series(
      k_a: np.ndarray,
      thickness_a: np.ndarray,
      k_b: np.ndarray,
      thickness_b: np.ndarray,
      face_area: np.ndarray,
  ) -> np.ndarray:
    conducts = (k_a > 0.0) & (k_b > 0.0)
    resistance = np.divide(
        thickness_a, 2.0 * k_a, out=np.zeros_like(face_area), where=conducts
    ) + np.divide(
        thickness_b, 2.0 * k_b, out=np.zeros_like(face_area), where=conducts
    )
    return np.divide(
        face_area, resistance, out=np.zeros_like(face_area), where=conducts
    )

  # Where two CVs of different height meet, only the overlap is a shared face.
  column_conductance = _series(
      k_right[:, :-1],
      horizontal_cv_dimension[:, :-1],
      k_left[:, 1:],
      horizontal_cv_dimension[:, 1:],
      floor_height_m
      * np.minimum(vertical_cv_dimension[:, :-1], vertical_cv_dimension[:, 1:]),
  )
  row_conductance = _series(
      k_bottom[:-1, :],
      vertical_cv_dimension[:-1, :],
      k_top[1:, :],
      vertical_cv_dimension[1:, :],
      floor_height_m
      * np.minimum(
          horizontal_cv_dimension[:-1, :], horizontal_cv_dimension[1:, :]
      ),
  )
  return column_conductance, row_conductance


def _neighbour_conductance_matrix(
    column_conductance: np.ndarray,
    row_conductance: np.ndarray,
    shape: tuple[int, int],
) -> scipy.sparse.csr_matrix:
  """Returns face conductances as a symmetric matrix over flattened CVs.

  Args:
    column_conductance: conductance between (i, j) and (i, j + 1), in W/K.
    row_conductance: conductance between (i, j) and (i + 1, j), in W/K.
    shape: shape of the thermal grid.

  Returns:
    An N x N matrix with a zero diagonal, whose (a, b) entry is the
    conductance between CVs a and b.
  """
  n_rows, n_columns = shape
  index = np.arange(n_rows * n_columns).reshape(shape)
  lower = np.concatenate([index[:, :-1].ravel(), index[:-1, :].ravel()])
  upper = np.concatenate([index[:, 1:].ravel(), index[1:, :].ravel()])
  conductance = np.concatenate(
      [column_conductance.ravel(), row_conductance.ravel()]
  )
  return scipy.sparse.csr_matrix(
      (
          np.concatenate([conductance, conductance]),
          (np.concatenate([lower, upper]), np.concatenate([upper, lower])),
      ),
      shape=(n_rows * n_columns, n_rows * n_columns),
  )


# Right-hand sides per triangular solve when building the air-side response.
# See FiniteDifferenceSolver._add_air_side.
_RESPONSE_BLOCK = 16


class FiniteDifferenceSolver:
  """Advances a building's temperature field by one backward-Euler timestep.

  The discretization is the usual implicit one,

      (C / dt + K) T = (C / dt) T_previous + Q + boundary terms,

  where C is each CV's heat capacity, K collects conduction between
  neighbouring CVs and convection to the outside air, and Q is the thermal
  power the HVAC applies. Only Q and T_previous change from step to step, so
  the matrix is assembled once here and factorized once, and each timestep
  costs a triangular solve.

  Two properties of the assembly are what make that cheap and correct, and
  both are checked by the tests:

    * Face conductances are shared by the pair of CVs either side of the face
      (see get_face_conductances), so the matrix is symmetric and, being
      diagonally dominant with a positive diagonal, positive definite.
    * CVs outside the building are held at the ambient air temperature, so
      they are not unknowns at all. Their columns are substituted into the
      right-hand side rather than their rows being pinned to the identity;
      pinning would leave the columns in place and destroy the symmetry above.
      On the shipped floor plans this also removes about 28% of the unknowns.

  A building may also give its occupied CVs a lumped interior mass, for the
  floor slab and the furniture. That node touches nothing but its own air CV,
  so it is eliminated into that CV's diagonal rather than added to the
  unknowns: the sparsity pattern, the symmetry and the factorization are all
  unchanged, the mass temperatures come back out of the solved air field, and
  the coupling stays implicit instead of lagging a step behind. Those
  temperatures are the only state the solver carries between timesteps, so
  solve advances them and reset clears them.

  Part of Q is not fixed forcing at all: a VAV blows air at a supply
  temperature across a zone and delivers whatever the resulting temperature
  difference calls for. couple_air_side registers those terms so they too are
  evaluated at the end of the step rather than the start.
  """

  def __init__(
      self,
      building: building_py.BaseSimulatorBuilding,
      time_step_sec: float,
  ):
    """Assembles the timestep matrix for a building's geometry and materials.

    Args:
      building: the building whose thermal grid is to be advanced. Its
        geometry, materials and control volume sizes are read once, here.
      time_step_sec: size of the timestep the matrix is assembled for.
    """
    n_rows, n_columns = building.temp.shape
    self._shape = (n_rows, n_columns)

    boundary_cv_mapping = get_cv_mapping(
        building.neighbors, position_criterion=CVPositionType.BOUNDARY
    )
    exterior_cv_mapping = get_cv_mapping(
        building.neighbors, position_criterion=CVPositionType.EXTERIOR
    )
    logging.info('Number of boundary CVs: %d', len(boundary_cv_mapping))
    logging.info('Number of exterior CVs: %d', len(exterior_cv_mapping))
    logging.info(
        'Number of interior CVs: %d',
        n_rows * n_columns
        - len(boundary_cv_mapping)
        - len(exterior_cv_mapping),
    )

    horizontal, vertical = get_cv_dimension_tensors(
        building.cv_size_cm / 100.0, boundary_cv_mapping, self._shape
    )
    floor_height_m = building.floor_height_cm / 100.0

    column_conductance, row_conductance = get_face_conductances(
        get_oriented_conductivity_tensors(
            building.conductivity, boundary_cv_mapping
        ),
        horizontal,
        vertical,
        floor_height_m,
    )
    conductance = _neighbour_conductance_matrix(
        column_conductance, row_conductance, self._shape
    )

    # Exterior CVs are pinned to ambient, so they are eliminated rather than
    # solved for. Everything below is indexed by position in _solved.
    self._solved = np.flatnonzero(
        ~self._exterior_mask(exterior_cv_mapping).ravel()
    )
    coupling = conductance[self._solved][:, self._solved]

    # The diagonal keeps the conductance to the eliminated CVs; only the
    # off-diagonal entry moves, into _dirichlet_conductance below.
    total_conductance = np.asarray(conductance.sum(axis=1)).ravel()[
        self._solved
    ]
    self._dirichlet_conductance = total_conductance - np.asarray(
        coupling.sum(axis=1)
    ).ravel()

    heat_capacity = (
        floor_height_m
        * horizontal
        * vertical
        * np.asarray(building.density, dtype=np.float64)
        * np.asarray(building.heat_capacity, dtype=np.float64)
    )
    self._capacitance_rate = (
        heat_capacity.ravel()[self._solved] / time_step_sec
    )

    # Eliminating the mass node from the pair
    #
    #     (C_a/dt + K + G) T   = C_a/dt T- + Q + G T_m
    #     (C_m/dt + G) T_m     = C_m/dt T_m- + G T
    #
    # leaves the air equation with one extra diagonal term and one extra
    # forcing term, both equal to G and C_m/dt in series. At a long timestep
    # that tends to G, the mass following the air exactly; at a short one it
    # tends to C_m/dt, the air seeing a fixed heat sink. Steady state is the
    # dt -> infinity limit of the forcing minus the diagonal, which is zero:
    # storage holds heat, it does not conduct any.
    #
    # Every term is carried at full length with zeros off the mask rather than
    # gathered through an index array. The mask covers most of the building, so
    # a gather costs more than the arithmetic it saves, and zero coupling on a
    # CV without a node makes it contribute nothing anyway.
    capacity_rate = np.zeros(len(self._solved))
    conductance = np.zeros(len(self._solved))
    interior_mass = building.interior_mass
    if interior_mass is not None:
      capacity, conductance = get_interior_mass_terms(
          interior_mass, horizontal, vertical, floor_height_m
      )
      capacity_rate = capacity.ravel()[self._solved] / time_step_sec
      conductance = conductance.ravel()[self._solved]
    total = capacity_rate + conductance
    inverse = np.divide(
        1.0, total, out=np.zeros_like(total), where=total > 0.0
    )
    self._mass_coupling = capacity_rate * conductance * inverse
    # The mass update rearranged as a weighted mean, so a step costs two
    # multiplies rather than a divide: the node keeps this much of where it
    # was and takes the rest from the air it ends the step touching.
    self._mass_retained = capacity_rate * inverse
    self._mass_gained = conductance * inverse

    # The conductance to ambient depends on h, which varies with the weather,
    # so only its geometry and material arguments are kept here.
    self._ambient_face_areas = get_ambient_face_areas(
        boundary_cv_mapping, horizontal, vertical, floor_height_m
    )
    self._horizontal = horizontal
    self._vertical = vertical
    self._conductivity = np.asarray(building.conductivity, dtype=np.float64)
    self._ambient_conductance = np.empty(0)
    self._ambient_conductance_at = None

    self._conduction = (
        scipy.sparse.diags(
            total_conductance + self._capacitance_rate + self._mass_coupling
        )
        - coupling
    ).tocsc()
    self._factorization = None
    self._factorized_at = None
    self._mass_temp = np.zeros(len(self._solved))
    # No air-side coupling until a caller registers one; see couple_air_side.
    self._applied = None
    self._sensed = None
    self._sensed_response = None
    self.reset(building.temp)

  def reset(self, temp: np.ndarray) -> None:
    """Puts every interior mass node at the temperature of its own CV.

    The mass temperatures are an internal node of the discretization rather
    than something the building exposes, so the solver keeps them; this is
    what an episode reset has to clear, and the constructor's initial call is
    what makes a freshly built solver usable without one.

    Args:
      temp: Temperature field to seed from, normally the building's at reset.
    """
    self._mass_temp = np.where(
        self._mass_coupling > 0.0,
        np.asarray(temp, dtype=np.float64).reshape(-1)[self._solved],
        0.0,
    )

  @property
  def interior_mass_temp(self) -> np.ndarray:
    """Temperature in K of each interior mass node, if the building has any.

    In the order the CVs carrying one appear in the field, so a building with
    no mass gives back nothing rather than a field of zeros.
    """
    return self._mass_temp[self._mass_coupling > 0.0]

  def ambient_conductance(self, convection_coefficient: float) -> np.ndarray:
    """Returns the W/K from each solved CV to ambient, for a given h.

    Args:
      convection_coefficient: Current wind convection coefficient (W/m2/K).
    """
    if convection_coefficient != self._ambient_conductance_at:
      self._ambient_conductance = get_ambient_conductances(
          self._ambient_face_areas,
          self._horizontal,
          self._vertical,
          self._conductivity,
          convection_coefficient,
      ).ravel()[self._solved]
      self._ambient_conductance_at = convection_coefficient
    return self._ambient_conductance

  def _exterior_mask(
      self, exterior_cv_mapping: BoundaryCVMapping
  ) -> np.ndarray:
    """Returns a binary tensor mask of all CVs that are exterior = True."""
    exterior_mask = np.full(self._shape, False)
    for i, j in exterior_cv_mapping:
      exterior_mask[i][j] = True
    return exterior_mask

  def matrix(self, convection_coefficient: float) -> scipy.sparse.csc_matrix:
    """Returns the timestep matrix for a given convection coefficient.

    Convection reaches the matrix only through the diagonal of the exposed
    CVs, so the whole geometry-dependent part is assembled once in the
    constructor and the coefficient is added in here.

    Args:
      convection_coefficient: Current wind convection coefficient (W/m2/K).
    """
    return (
        self._conduction
        + scipy.sparse.diags(self.ambient_conductance(convection_coefficient))
    ).tocsc()

  def couple_air_side(
      self,
      applied: scipy.sparse.spmatrix,
      sensed: scipy.sparse.spmatrix,
  ) -> None:
    """Registers thermal power that depends on the temperature it acts on.

    A VAV does not deliver a fixed number of watts. It blows air at T_supply
    over the zones it serves and delivers g * (T_supply - Tbar), where g is the
    capacity rate of the airflow and Tbar the mean temperature its thermostat
    reads. Handed to solve inside input_q, that has to be evaluated from the
    temperature at the START of the step, which makes the air side forward
    Euler on the zone-air capacitance and unstable once dt * g / C > 2. The
    conduction being backward Euler does not save it, because the loop closes
    outside the matrix. Registered here it is evaluated at the end of the step
    with everything else, and is unconditionally stable.

    Each column is one such term, and the two matrices say where its power
    lands and which temperature it responds to: column j of applied holds the
    share reaching each CV (the VAV's diffusers, summing to one) and column j
    of sensed the weights of the mean it reads (the CVs its thermostat
    averages, also summing to one). Both are indexed by position in the
    flattened temperature field, so a caller builds them from the same zone
    geometry it would pass to building.apply_thermal_power_zone.

    The pair perturbs the timestep matrix by applied * diag(g) * sensed^T,
    which is dense but has only one column per VAV, so solve applies it by a
    Woodbury update on the existing factorization instead of refactorizing.
    That is the difference between a few hundred milliseconds a step and a
    second triangular solve; see _add_air_side.

    Args:
      applied: (n_cvs, n_terms) share of each term's power reaching each CV.
      sensed: (n_cvs, n_terms) weights of the mean each term responds to.

    Raises:
      ValueError: if the shapes disagree with each other or with the field, or
        if either matrix touches a CV pinned to ambient, whose temperature is
        not solved for and so can be neither driven nor read.
    """
    n_cvs = self._shape[0] * self._shape[1]
    if applied.shape != sensed.shape or applied.shape[0] != n_cvs:
      raise ValueError(
          f'applied {applied.shape} and sensed {sensed.shape} must agree and'
          f' have one row per CV ({n_cvs}).'
      )
    pinned = np.ones(n_cvs, dtype=bool)
    pinned[self._solved] = False
    applied = scipy.sparse.csc_matrix(applied)
    sensed = scipy.sparse.csc_matrix(sensed)
    if applied[pinned].count_nonzero() or sensed[pinned].count_nonzero():
      raise ValueError(
          'An air-side term acts on or reads a CV that is pinned to ambient'
          ' and therefore not solved for.'
      )

    self._applied = applied[self._solved]
    self._sensed = sensed[self._solved].T.tocsr()
    self._sensed_response = None

  def solve(
      self,
      temp: np.ndarray,
      input_q: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
      air_side_conductance: np.ndarray | None = None,
      supply_temperature: np.ndarray | None = None,
  ) -> np.ndarray:
    """Returns the temperature field one timestep after temp.

    Advances the interior mass nodes by the same timestep, so calling this
    twice on one temperature field steps the building twice rather than
    recomputing the same answer. reset puts them back.

    Args:
      temp: Temperature in K of every CV at the start of the timestep.
      input_q: External thermal power in W applied to each CV.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).
      air_side_conductance: W/K of each term registered with couple_air_side,
        or None to leave them all switched off.
      supply_temperature: Temperature in K each of those terms drives its zones
        towards. Required whenever air_side_conductance is given.
    """
    factorization = self._factorization
    if factorization is None or convection_coefficient != self._factorized_at:
      # Refactorizing costs a few hundred timesteps' worth of solves, so a
      # weather model that varied the wind would want to quantize h to a
      # handful of values and keep a factorization per bucket. Every shipped
      # weather controller returns a constant, so one is enough.
      factorization = scipy.sparse.linalg.splu(
          self.matrix(convection_coefficient), permc_spec='MMD_AT_PLUS_A'
      )
      self._factorization = factorization
      self._factorized_at = convection_coefficient
      self._sensed_response = None

    exposure = self.ambient_conductance(convection_coefficient)
    forcing = (
        self._capacitance_rate
        * np.asarray(temp, dtype=np.float64).reshape(-1)[self._solved]
        + np.asarray(input_q, dtype=np.float64).reshape(-1)[self._solved]
        + ambient_temperature * (self._dirichlet_conductance + exposure)
    )
    forcing += self._mass_coupling * self._mass_temp

    solved = factorization.solve(forcing)
    if air_side_conductance is not None:
      solved = self._add_air_side(
          factorization, solved, air_side_conductance, supply_temperature
      )
    self._mass_temp *= self._mass_retained
    self._mass_temp += self._mass_gained * solved

    updated = np.full(self._shape[0] * self._shape[1], ambient_temperature)
    updated[self._solved] = solved
    return updated.reshape(self._shape)

  def _add_air_side(
      self,
      factorization: scipy.sparse.linalg.SuperLU,
      uncoupled: np.ndarray,
      conductance: np.ndarray,
      supply_temperature: np.ndarray,
  ) -> np.ndarray:
    """Returns the field with the registered air-side terms folded back in.

    Writing P for the applied shares, Q for the sensed weights, G for
    diag(conductance) and y for the field solved without any of it, the
    coupled field is the T that satisfies

        T = y + A^-1 P q,     q = G (T_supply - Q^T T),

    the second equation being the VAV's own output law read at the end of the
    step. Substituting the first into the second leaves one small dense system
    for the delivered power q,

        (I + G Q^T A^-1 P) q = G (T_supply - Q^T y),

    with one row per term rather than one per CV. That is the Woodbury
    identity written so a term with no flow through it is the identity rather
    than a division by zero, which matters because a closed damper is the
    normal state of most of a building most of the time.

    Only Q^T A^-1 P is kept: it is one small matrix, it depends on nothing but
    the geometry and the factorization, and it is rebuilt whenever either
    changes. A^-1 P itself would save the second triangular solve below, but
    it is one dense column per VAV -- on the shipped floor plans the same order
    as the factorization -- and streaming it once a step evicts the
    factorization from cache, which costs more than the solve it saves. Both
    ways were measured; this one is about a third faster end to end.

    Args:
      factorization: The current factorization of the timestep matrix.
      uncoupled: Solved field over the unknowns with no air-side power at all.
      conductance: W/K of each registered term.
      supply_temperature: Temperature in K each term drives its zones towards.
    """
    if self._applied is None:
      raise ValueError(
          'solve was given air-side conductances but no term was registered'
          ' with couple_air_side.'
      )
    if supply_temperature is None:
      raise ValueError(
          'solve needs a supply temperature for every air-side conductance.'
      )
    if self._sensed_response is None:
      # A column at a time would be a dense n-vector; all of them at once is a
      # dense n-by-k, half a gigabyte on the largest shipped floor plan, to be
      # collapsed straight back down to k-by-k. Solving in blocks keeps enough
      # right-hand sides in flight for the solve to stay vectorized without
      # ever holding more than a slice.
      n_terms = self._applied.shape[1]
      self._sensed_response = np.hstack(
          [np.zeros((n_terms, 0))]
          + [
              self._sensed
              @ factorization.solve(
                  self._applied[:, start : start + _RESPONSE_BLOCK].toarray()
              )
              for start in range(0, n_terms, _RESPONSE_BLOCK)
          ]
      )

    conductance = np.asarray(conductance, dtype=np.float64).reshape(-1)
    supply_temperature = np.asarray(
        supply_temperature, dtype=np.float64
    ).reshape(-1)
    power = np.linalg.solve(
        np.eye(len(conductance))
        + conductance[:, np.newaxis] * self._sensed_response,
        conductance * (supply_temperature - self._sensed @ uncoupled),
    )
    return uncoupled + factorization.solve(self._applied @ power)
