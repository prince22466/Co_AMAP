"""Property tests for the sparse backward-Euler solve.

These check the four things the migration from truncated Jacobi sweeps to a
scipy factorization is supposed to buy, and that the previous implementation
could not offer:

  * the matrix is symmetric, so a conductance is shared by the two CVs either
    side of a face and heat leaving one arrives in the other;
  * it is a strictly diagonally dominant M-matrix, hence positive definite,
    which is what makes a sparse Cholesky-like factorization safe;
  * CVs outside the building are eliminated rather than pinned, so they are
    not unknowns at all;
  * the returned field satisfies the discrete energy balance exactly, rather
    than part-way, on every step regardless of how stiff the step is.

The last one is the point of the exercise. The sweeps stopped on a
temperature-change threshold and were capped by an iteration limit, so a stiff
step was truncated and left the building part-way to where the physics said it
should be - indistinguishable from a larger thermal mass, and varying with the
weather.
"""

from typing import Any

import numpy as np
import scipy.sparse
from absl.testing import absltest, parameterized

from smart_buildings.smart_control.simulator import building as building_py
from smart_buildings.smart_control.simulator import (
  finite_difference_solver as fd_solver,
)

_CV_SIZE_CM = 20.0
_FLOOR_HEIGHT_CM = 300.0
_TIME_STEP_SEC = 300.0
_AMBIENT = 275.0
_CONVECTION = 19.5


class _GridBuilding(building_py.BaseSimulatorBuilding[Any]):
  """A building given directly as a grid, rather than read off a floor plan.

  FloorPlanBasedBuilding infers rooms, walls and diffuser placement from an
  image, which makes it awkward to say "this CV is outside and that one is
  masonry" - and those are exactly the things these tests need to vary. The
  solver only ever reads the thermal grid, so this supplies one and leaves the
  zone interface unimplemented.
  """

  def __init__(
      self,
      inside: np.ndarray,
      conductivity: np.ndarray,
      density: float = 1.0,
      heat_capacity: float = 70000.0,
      initial_temp: float = 293.0,
      interior_mass: building_py.InteriorMass | None = None,
  ):
    self.cv_size_cm = _CV_SIZE_CM
    self.floor_height_cm = _FLOOR_HEIGHT_CM
    self.temp = np.full(inside.shape, initial_temp)
    self.input_q = np.zeros(inside.shape)
    self.inside = inside
    self._conductivity = conductivity
    self._density = np.full(inside.shape, density)
    self._heat_capacity = np.full(inside.shape, heat_capacity)
    self._interior_mass = interior_mass

    n_rows, n_columns = inside.shape
    self.neighbors = [
        [
            [
                (x + dx, y + dy)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= x + dx < n_rows
                and 0 <= y + dy < n_columns
                and inside[x + dx][y + dy]
            ]
            if inside[x][y]
            else []
            for y in range(n_columns)
        ]
        for x in range(n_rows)
    ]

  @property
  def conductivity(self) -> np.ndarray:
    return self._conductivity

  @property
  def density(self) -> np.ndarray:
    return self._density

  @property
  def heat_capacity(self) -> np.ndarray:
    return self._heat_capacity

  @property
  def cv_type(self) -> np.ndarray:
    raise NotImplementedError()

  @property
  def interior_mass(self) -> building_py.InteriorMass | None:
    return self._interior_mass

  def reset(self) -> None:
    raise NotImplementedError()

  def get_zone_average_temps(self) -> dict[Any, Any]:
    raise NotImplementedError()

  def apply_thermal_power_zone(self, zone: Any, /, power: float) -> None:
    raise NotImplementedError()


def _building() -> _GridBuilding:
  """Returns a 10x14 building inset in a 14x18 grid, with a dividing wall.

  The inset leaves a ring of exterior CVs for the solver to eliminate, and the
  wall makes the conductivity non-uniform so that face conductances have to
  combine two different materials.
  """
  inside = np.zeros((14, 18), dtype=bool)
  inside[2:12, 2:16] = True
  conductivity = np.full(inside.shape, 50.0)
  conductivity[2:12, 8] = 2.0  # a masonry wall down the middle
  conductivity[~inside] = 5.5
  return _GridBuilding(inside, conductivity)


def _envelope_building(boundary_conductivity: float) -> _GridBuilding:
  """Returns the same building with a given material around its boundary.

  Only the ring of CVs that face the outside air differs, so any change in how
  much heat the building loses is attributable to the envelope material alone.
  """
  inside = np.zeros((14, 18), dtype=bool)
  inside[2:12, 2:16] = True
  conductivity = np.full(inside.shape, 50.0)
  conductivity[2, 2:16] = conductivity[11, 2:16] = boundary_conductivity
  conductivity[2:12, 2] = conductivity[2:12, 15] = boundary_conductivity
  conductivity[~inside] = 5.5
  return _GridBuilding(inside, conductivity)


def _mass_building(
    conductivity: float = 2000.0, density: float = 350.0
) -> _GridBuilding:
  """Returns the standard building with a mass node on every interior CV.

  The mask deliberately excludes the boundary ring, so the tests also cover
  the case of a building where only some CVs carry a node. The material is
  scaled to _building's deliberately overweight air rather than to a real
  slab: a node holding several times the CV's own capacity, coupled loosely
  enough that it takes tens of minutes to follow the room.
  """
  building = _building()
  mask = np.zeros(building.temp.shape, dtype=bool)
  mask[3:11, 3:15] = True
  return _GridBuilding(
      building.inside,
      building.conductivity,
      interior_mass=building_py.InteriorMass(
          building_py.MaterialProperties(
              conductivity=conductivity, heat_capacity=1000.0, density=density
          ),
          mask,
      ),
  )


def _solver(building: _GridBuilding) -> fd_solver.FiniteDifferenceSolver:
  return fd_solver.FiniteDifferenceSolver(building, _TIME_STEP_SEC)


def _air_side_terms(
    shape: tuple[int, int],
) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix]:
  """Returns applied and sensed for two zones, one either side of the wall.

  Each has a single diffuser and reads the mean over its own CVs, which is the
  shape a VAV serving one zone takes. Two of them rather than one because the
  small system solve is a no-op at rank one: only with a second term can the
  heat one puts into the building reach the temperature the other reads.
  """
  applied = np.zeros((shape[0] * shape[1], 2))
  sensed = np.zeros((shape[0] * shape[1], 2))
  for column, (columns, diffuser) in enumerate(
      ((slice(3, 8), (5, 5)), (slice(9, 14), (5, 11)))
  ):
    zone = np.zeros(shape, dtype=bool)
    zone[3:8, columns] = True
    sensed[zone.ravel(), column] = 1.0 / np.count_nonzero(zone)
    applied[np.ravel_multi_index(diffuser, shape), column] = 1.0
  return scipy.sparse.csc_matrix(applied), scipy.sparse.csc_matrix(sensed)


def _steady_state_ua(building: _GridBuilding, watts: float = 1000.0) -> float:
  """Returns the building's W/K to ambient, measured by injecting power.

  At an effectively infinite timestep the capacitance term C/dt drops out of
  the backward-Euler matrix, so one solve lands exactly on the steady state and
  UA is the injected power over the temperature lift it sustains.
  """
  solver = fd_solver.FiniteDifferenceSolver(building, 1e14)
  input_q = np.zeros(building.temp.shape)
  input_q[building.inside] = watts / np.count_nonzero(building.inside)
  settled = solver.solve(
      np.full(building.temp.shape, _AMBIENT), input_q, _AMBIENT, _CONVECTION
  )
  return watts / (settled[building.inside].mean() - _AMBIENT)


class MatrixPropertyTest(absltest.TestCase):
  """Properties of the assembled timestep matrix."""

  def setUp(self):
    super().setUp()
    self.building = _building()
    self.matrix = _solver(self.building).matrix(_CONVECTION)

  def test_matrix_is_exactly_symmetric(self):
    """Not nearly symmetric: the two CVs either side of a face share one number.

    Taking each CV's own conductivity across a face instead leaves the pair
    disagreeing about how much heat crossed it, which creates energy at every
    material interface and rules out a factorization that assumes symmetry.
    """
    difference = self.matrix - self.matrix.T
    self.assertEqual(abs(difference).max(), 0.0)

  def test_off_diagonals_are_non_positive(self):
    """Conduction can only pull a CV toward its neighbour, never push it away."""
    off_diagonal = self.matrix - np.diag(self.matrix.diagonal())
    self.assertLessEqual(off_diagonal.max(), 0.0)

  def test_matrix_is_a_strictly_dominant_m_matrix(self):
    """Diagonal dominance plus symmetry is a positive-definiteness certificate.

    The margin is at least the smallest CV's heat capacity over the timestep,
    because that term appears on the diagonal and nowhere else.
    """
    diagonal = self.matrix.diagonal()
    off_diagonal_sum = np.asarray(
        abs(self.matrix).sum(axis=1)
    ).ravel() - abs(diagonal)
    margin = diagonal - off_diagonal_sum

    cv_area = (_CV_SIZE_CM / 100.0) ** 2
    volume = cv_area * _FLOOR_HEIGHT_CM / 100.0
    # Boundary CVs are half or quarter size, hence the 0.25.
    smallest_capacitance_rate = (
        0.25 * volume * 1.0 * 70000.0 / _TIME_STEP_SEC
    )
    self.assertGreaterEqual(margin.min(), smallest_capacitance_rate)

  def test_exterior_cvs_are_eliminated_not_pinned(self):
    """The system is only as large as the number of CVs actually solved for."""
    self.assertEqual(self.matrix.shape[0], np.count_nonzero(self.building.inside))
    self.assertLess(self.matrix.shape[0], self.building.temp.size)

  def test_convection_touches_only_the_diagonal_of_exposed_cvs(self):
    """The factorization is reused until the coefficient changes, so it must be
    the only thing the coefficient can reach.
    """
    solver = _solver(self.building)
    difference = solver.matrix(2.0 * _CONVECTION) - solver.matrix(_CONVECTION)
    off_diagonal = difference - np.diag(difference.diagonal())

    self.assertEqual(abs(off_diagonal).max(), 0.0)
    self.assertGreater(difference.diagonal().max(), 0.0)
    # Only the CVs on the building's boundary see the outside air at all.
    exposed = difference.diagonal() > 0.0
    self.assertLess(np.count_nonzero(exposed), self.matrix.shape[0])


class AmbientPathTest(absltest.TestCase):
  """The envelope material, not just the wind, governs heat loss.

  The outward face of every boundary CV is zeroed by
  get_oriented_conductivity_tensors, so it contributes no neighbour conductance
  and the ambient exchange is the only way out of the building. If that
  exchange is taken as h * area, the surface film becomes the entire envelope:
  the wall material stops mattering at all, the effective U is pinned at h -
  tens of W/m2K against a real facade's ~1 - and wind alone decides how well a
  building is insulated. These tests pin the series path that avoids that.
  """

  def test_envelope_material_governs_heat_loss(self):
    """An insulating envelope must lose far less heat than a conducting one."""
    insulated = _steady_state_ua(_envelope_building(0.21))
    conducting = _steady_state_ua(_envelope_building(500.0))

    self.assertLess(insulated, conducting / 5.0)

  def test_heat_loss_saturates_at_the_film_coefficient(self):
    """However conductive the wall, the film still limits the exchange.

    This is the other half of the series path: the wall term alone would let a
    conductive envelope lose heat without bound.
    """
    building = _envelope_building(1e6)
    areas = fd_solver.get_ambient_face_areas(
        fd_solver.get_cv_mapping(building.neighbors),
        *fd_solver.get_cv_dimension_tensors(
            _CV_SIZE_CM / 100.0,
            fd_solver.get_cv_mapping(building.neighbors),
            building.temp.shape,
        ),
        _FLOOR_HEIGHT_CM / 100.0,
    )
    film_limit = _CONVECTION * (areas[0] + areas[1]).sum()

    self.assertLess(_steady_state_ua(building), film_limit)
    self.assertGreater(_steady_state_ua(building), 0.9 * film_limit)

  def test_still_air_exchanges_no_heat(self):
    """h = 0 is a valid weather input and must not divide by zero."""
    building = _envelope_building(0.21)
    conductance = _solver(building).ambient_conductance(0.0)

    self.assertEqual(abs(conductance).max(), 0.0)


def _energy_balance_residual(
    building: _GridBuilding,
    previous: np.ndarray,
    updated: np.ndarray,
    input_q: np.ndarray,
    convection_coefficient: float,
    interior_mass_temp: np.ndarray | None = None,
) -> np.ndarray:
  """Returns each solved CV's unbalanced power, in W.

  Rebuilt from the public helpers rather than from anything the solver
  cached, so this checks the assembly and the solve together: at the end of
  an implicit step, the heat a CV stored must equal the heat applied to it
  plus the heat that conducted in from its neighbours, convected in from
  outside, and come back out of its interior mass, all evaluated at the new
  temperatures.

  Args:
    building: the building the field belongs to.
    previous: temperature field at the start of the step.
    updated: temperature field the solver returned.
    input_q: thermal power applied to each CV.
    convection_coefficient: the h the step was solved at.
    interior_mass_temp: mass temperatures after the step, from the solver.
      Required whenever the building has an interior mass, since the air
      balance does not close without the heat that came out of it.
  """
  horizontal, vertical = fd_solver.get_cv_dimension_tensors(
      building.cv_size_cm / 100.0,
      fd_solver.get_cv_mapping(building.neighbors),
      building.temp.shape,
  )
  floor_height_m = building.floor_height_cm / 100.0
  column_conductance, row_conductance = fd_solver.get_face_conductances(
      fd_solver.get_oriented_conductivity_tensors(
          building.conductivity,
          fd_solver.get_cv_mapping(building.neighbors),
      ),
      horizontal,
      vertical,
      floor_height_m,
  )

  conducted = np.zeros_like(updated)
  across_columns = column_conductance * np.diff(updated, axis=1)
  conducted[:, :-1] += across_columns
  conducted[:, 1:] -= across_columns
  across_rows = row_conductance * np.diff(updated, axis=0)
  conducted[:-1, :] += across_rows
  conducted[1:, :] -= across_rows

  convected = fd_solver.get_ambient_conductances(
      fd_solver.get_ambient_face_areas(
          fd_solver.get_cv_mapping(building.neighbors),
          horizontal,
          vertical,
          floor_height_m,
      ),
      horizontal,
      vertical,
      building.conductivity,
      convection_coefficient,
  ) * (_AMBIENT - updated)
  stored = (
      floor_height_m
      * horizontal
      * vertical
      * building.density
      * building.heat_capacity
      * (updated - previous)
      / _TIME_STEP_SEC
  )

  released = np.zeros_like(updated)
  interior_mass = building.interior_mass
  if interior_mass is not None:
    assert interior_mass_temp is not None
    _, conductance = fd_solver.get_interior_mass_terms(
        interior_mass, horizontal, vertical, floor_height_m
    )
    mass_field = np.zeros_like(updated)
    mass_field[interior_mass.mask] = interior_mass_temp
    released = conductance * (mass_field - updated)

  return (stored - conducted - convected - released - input_q)[
      building.inside
  ]


class SolveTest(parameterized.TestCase):
  """Properties of the temperature field the solver returns."""

  def setUp(self):
    super().setUp()
    self.building = _building()
    self.solver = _solver(self.building)

  def test_step_satisfies_the_discrete_energy_balance(self):
    """Every solved CV closes its energy balance, on one step, exactly.

    This is what the sweeps could not do. They stopped when the largest
    temperature change fell below a threshold, or when they ran out of
    iterations, and on a stiff step that left every CV short of where the
    balance above puts it.
    """
    previous = np.full(self.building.temp.shape, 293.0)
    input_q = np.zeros_like(previous)
    input_q[6, 4] = 4000.0
    input_q[9, 12] = -2500.0

    updated = self.solver.solve(previous, input_q, _AMBIENT, _CONVECTION)

    residual = _energy_balance_residual(
        self.building, previous, updated, input_q, _CONVECTION
    )
    self.assertLess(abs(residual).max(), 1e-6)

  def test_a_stiff_step_is_not_truncated(self):
    """A step with a large driving temperature difference balances too.

    With the shipped materials one 5-minute step is many hundreds of grid
    diffusion times, which is exactly the regime the sweeps ran out of
    iterations in.
    """
    previous = np.full(self.building.temp.shape, 320.0)
    input_q = np.zeros_like(previous)

    updated = self.solver.solve(previous, input_q, _AMBIENT, _CONVECTION)

    residual = _energy_balance_residual(
        self.building, previous, updated, input_q, _CONVECTION
    )
    self.assertLess(abs(residual).max(), 1e-6)
    # And it really is a large move, not a step the solver barely had to take.
    self.assertGreater(abs(updated - previous)[self.building.inside].max(), 5.0)

  @parameterized.named_parameters(
      ('still_air', 0.0), ('shipped', _CONVECTION), ('gale', 100.0)
  )
  def test_ambient_everywhere_is_a_fixed_point(self, convection_coefficient):
    """A building already at ambient, with nothing applied, must not drift.

    This pins the convection coefficient into the matrix and the right-hand
    side consistently: scaling one but not the other leaves the building
    coupled to an outside that is at the wrong temperature, and shows up here
    as drift that grows with the coefficient.
    """
    previous = np.full(self.building.temp.shape, _AMBIENT)

    updated = self.solver.solve(
        previous, np.zeros_like(previous), _AMBIENT, convection_coefficient
    )

    self.assertLess(abs(updated - _AMBIENT).max(), 1e-9)

  def test_exterior_cvs_are_returned_at_ambient(self):
    """Eliminating them must not change what a caller reads back."""
    previous = np.full(self.building.temp.shape, 293.0)

    updated = self.solver.solve(
        previous, np.zeros_like(previous), _AMBIENT, _CONVECTION
    )

    outside = updated[~self.building.inside]
    np.testing.assert_allclose(outside, _AMBIENT)

  def test_heat_applied_to_a_zone_warms_it(self):
    """Direction check, in case a sign error survives the balance above."""
    previous = np.full(self.building.temp.shape, _AMBIENT)
    input_q = np.zeros_like(previous)
    input_q[6, 4] = 4000.0

    updated = self.solver.solve(previous, input_q, _AMBIENT, _CONVECTION)

    self.assertGreater(updated[6, 4], _AMBIENT)
    self.assertGreater(updated[6, 5], _AMBIENT)  # and it spreads

  def test_changing_the_convection_coefficient_refactorizes(self):
    """A cached factorization must never be reused for a different wind."""
    previous = np.full(self.building.temp.shape, 320.0)
    zero_q = np.zeros_like(previous)

    self.solver.solve(previous, zero_q, _AMBIENT, _CONVECTION)
    reused = self.solver.solve(previous, zero_q, _AMBIENT, 2.0 * _CONVECTION)
    fresh = _solver(self.building).solve(
        previous, zero_q, _AMBIENT, 2.0 * _CONVECTION
    )

    np.testing.assert_allclose(reused, fresh)
    # A doubled coefficient must actually cool the building faster.
    once = _solver(self.building).solve(
        previous, zero_q, _AMBIENT, _CONVECTION
    )
    self.assertLess(reused[self.building.inside].mean(),
                    once[self.building.inside].mean())


class InteriorMassTest(absltest.TestCase):
  """The slab and furnishings as a second node per CV, condensed away.

  A room's air responds in minutes and the slab it sits on in hours, so a
  building has two time constants where a single lumped CV has one. The node
  that supplies the second one touches nothing but its own air CV, which is
  what lets it be eliminated into that CV's diagonal instead of enlarging the
  system: these tests pin both halves of that claim, that the elimination is
  free and that the algebra it stands on is exact.
  """

  def setUp(self):
    super().setUp()
    self.building = _mass_building()
    self.solver = _solver(self.building)

  def _start(self, temperature: float) -> np.ndarray:
    """Returns a uniform field with the solver's mass nodes on it."""
    field = np.full(self.building.temp.shape, temperature)
    self.solver.reset(field)
    return field

  def _step(self, previous: np.ndarray, watts: float = 0.0):
    input_q = np.zeros_like(previous)
    input_q[self.building.inside] = watts / np.count_nonzero(
        self.building.inside
    )
    updated = self.solver.solve(previous, input_q, _AMBIENT, _CONVECTION)
    return updated, input_q

  def test_the_mass_adds_no_unknowns_and_no_fill(self):
    """Condensation must leave the matrix the solver factorizes untouched.

    Anywhere else the node would show up as extra rows and columns, and the
    factorization would have to be redone; here the only trace of it is a
    larger diagonal on the CVs that carry one.
    """
    plain = _solver(_building()).matrix(_CONVECTION).tocsr()
    massed = self.solver.matrix(_CONVECTION).tocsr()

    self.assertEqual(plain.shape, massed.shape)
    np.testing.assert_array_equal(plain.indptr, massed.indptr)
    np.testing.assert_array_equal(plain.indices, massed.indices)
    difference = massed - plain
    off_diagonal = difference - scipy.sparse.diags(difference.diagonal())
    self.assertEqual(abs(off_diagonal).max(), 0.0)
    self.assertGreater(difference.diagonal().max(), 0.0)

  def test_stored_heat_does_not_change_the_steady_state(self):
    """A mass holds heat; it does not conduct any, so UA is unaffected.

    This is what keeps the envelope calibration a separate question from the
    thermal mass one: adding storage cannot move the measured U. Exactly so in
    the limit, since the condensed term carries C/dt as a factor; at the finite
    1e14 s this measures at, what is left of it is a part in 1e9.
    """
    np.testing.assert_allclose(
        _steady_state_ua(_mass_building()),
        _steady_state_ua(_building()),
        rtol=1e-6,
    )

  def test_step_satisfies_the_discrete_energy_balance(self):
    """Every CV closes its balance, including the heat out of its mass.

    Evaluated at the mass temperature the step ended on, not the one it
    started from. Upstream advances the air against the previous mass
    temperature and updates the mass afterwards, which leaves this residual
    non-zero by whatever the mass moved during the step.
    """
    previous = self._start(293.0)
    updated, input_q = self._step(previous, watts=6000.0)

    residual = _energy_balance_residual(
        self.building,
        previous,
        updated,
        input_q,
        _CONVECTION,
        self.solver.interior_mass_temp,
    )
    self.assertLess(abs(residual).max(), 1e-6)

  def test_the_mass_node_closes_its_own_balance(self):
    """What the mass stored equals what crossed the coupling into it."""
    previous = self._start(293.0)
    before = self.solver.interior_mass_temp.copy()
    updated, _ = self._step(previous, watts=6000.0)
    after = self.solver.interior_mass_temp

    interior_mass = self.building.interior_mass
    assert interior_mass is not None
    horizontal, vertical = fd_solver.get_cv_dimension_tensors(
        _CV_SIZE_CM / 100.0,
        fd_solver.get_cv_mapping(self.building.neighbors),
        self.building.temp.shape,
    )
    capacity, conductance = fd_solver.get_interior_mass_terms(
        interior_mass, horizontal, vertical, _FLOOR_HEIGHT_CM / 100.0
    )
    mask = interior_mass.mask
    stored = capacity[mask] * (after - before) / _TIME_STEP_SEC
    crossed = conductance[mask] * (updated[mask] - after)

    self.assertGreater(abs(stored).max(), 0.0)  # the step did something
    np.testing.assert_allclose(stored, crossed, atol=1e-9)

  def test_the_mass_lags_the_air_and_then_catches_it_up(self):
    """The point of the node: a second, slower temperature in the room."""
    field = self._start(_AMBIENT)
    interior_mass = self.building.interior_mass
    assert interior_mass is not None
    mask = interior_mass.mask

    for _ in range(12):  # one hour of heating
      field, _ = self._step(field, watts=6000.0)
    air = field[mask].mean()
    mass = self.solver.interior_mass_temp.mean()
    self.assertGreater(air, _AMBIENT + 1.0)
    self.assertLess(mass, air)
    self.assertGreater(mass, _AMBIENT)

    for _ in range(2000):
      field, _ = self._step(field, watts=6000.0)
    self.assertAlmostEqual(
        self.solver.interior_mass_temp.mean(), field[mask].mean(), places=3
    )

  def test_the_mass_damps_the_swing_and_outlasts_it(self):
    """Lower peak while the heat is on, warmer room once it stops.

    A capacity folded into the air CV would do the first but not the second:
    it would give the room more inertia in both directions at once. Two nodes
    separated by a resistance charge and discharge on their own timescale,
    which is the behaviour a controller has to learn. The crossover is what
    dates it: for the first hour after the heat goes off the massed room is
    still the colder of the two, because the heat it kept is in the slab and
    the slab has not handed it back yet.
    """
    def run(building):
      solver = _solver(building)
      field = np.full(building.temp.shape, _AMBIENT)
      solver.reset(field)
      input_q = np.zeros_like(field)
      input_q[building.inside] = 6000.0 / np.count_nonzero(building.inside)
      peak = _AMBIENT
      for _ in range(12):
        field = solver.solve(field, input_q, _AMBIENT, _CONVECTION)
        peak = max(peak, field[building.inside].mean())
      for _ in range(24):
        field = solver.solve(
            field, np.zeros_like(field), _AMBIENT, _CONVECTION
        )
      return peak, field[building.inside].mean()

    massed_peak, massed_after = run(_mass_building())
    plain_peak, plain_after = run(_building())

    self.assertLess(massed_peak, plain_peak)
    self.assertGreater(massed_after, plain_after)

  def test_reset_puts_the_mass_back(self):
    """The mass is the solver's only state across a step, so reset must clear
    it or an episode starts wherever the last one left off.
    """
    field = self._start(320.0)
    for _ in range(24):
      field, _ = self._step(field)
    self.assertGreater(self.solver.interior_mass_temp.mean(), 300.0)

    self.solver.reset(np.full(self.building.temp.shape, 293.0))

    np.testing.assert_allclose(self.solver.interior_mass_temp, 293.0)

  def test_a_building_without_a_mass_carries_no_state(self):
    """The default is off, and off must cost nothing."""
    solver = _solver(_building())
    self.assertEqual(solver.interior_mass_temp.size, 0)


class AirSideCouplingTest(absltest.TestCase):
  """Power that reads the temperature it is applied to, solved implicitly.

  A VAV delivers g * (T_supply - Tbar), not a fixed number of watts. Handed to
  solve as watts it has to be evaluated before the step, which is forward Euler
  across g on the zone-air capacitance and diverges once dt * g / C > 2 -- and
  on the shipped floor plans that ratio reaches 20. couple_air_side moves it
  inside the matrix; these check that doing so changes only the timing, not the
  physics.
  """

  def setUp(self):
    super().setUp()
    self.building = _building()
    self.solver = _solver(self.building)
    self.applied, self.sensed = _air_side_terms(self.building.temp.shape)
    self.solver.couple_air_side(self.applied, self.sensed)
    self.previous = np.full(self.building.temp.shape, 293.0)
    self.input_q = np.zeros_like(self.previous)
    self.input_q[6, 4] = 400.0
    self.conductance = np.array([9000.0, 4000.0])
    self.supply_temp = np.array([305.0, 285.0])

  def _coupled(self) -> np.ndarray:
    self.solver.reset(self.previous)
    return self.solver.solve(
        self.previous,
        self.input_q,
        _AMBIENT,
        _CONVECTION,
        self.conductance,
        self.supply_temp,
    )

  def test_it_delivers_exactly_the_power_its_own_output_law_asks_for(self):
    """The coupled field is a fixed point of the uncoupled solve.

    Whatever the temperature ends up being, the power that ends up applied has
    to be the power a VAV looking at that temperature would have delivered.
    Feeding that power back in as ordinary watts must therefore reproduce the
    same field exactly -- which is the whole claim, stated without reassembling
    the perturbed matrix here and hoping the two assemblies agree.
    """
    coupled = self._coupled()

    delivered = self.conductance * (
        self.supply_temp - self.sensed.T @ coupled.reshape(-1)
    )
    equivalent = self.input_q.reshape(-1) + self.applied @ delivered

    self.solver.reset(self.previous)
    np.testing.assert_allclose(
        self.solver.solve(
            self.previous,
            equivalent.reshape(self.building.temp.shape),
            _AMBIENT,
            _CONVECTION,
        ),
        coupled,
        atol=1e-9,
    )

  def test_switching_the_terms_off_reproduces_the_uncoupled_solve(self):
    """A closed damper has to be free, not a division by zero.

    Most of a building is off most of the time, so this is the common case
    rather than an edge one, and it is the reason the small system is written
    as I + G S rather than G^-1 + S.
    """
    self.conductance = np.zeros(2)

    self.solver.reset(self.previous)
    np.testing.assert_allclose(
        self._coupled(),
        self.solver.solve(self.previous, self.input_q, _AMBIENT, _CONVECTION),
        atol=1e-9,
    )

  def test_it_is_stable_where_the_explicit_form_diverges(self):
    """The point of the exercise, on a zone chosen to be well over the limit.

    The gain here is dt * g / C = 13, so the explicit form multiplies any
    departure from the supply temperature by -12 every step. Both are run from
    the same state with the same physics and only the timing differs.
    """
    zone = self.sensed[:, 0].toarray().reshape(self.building.temp.shape) > 0
    capacitance = np.count_nonzero(zone) * 0.2 * 0.2 * 3.0 * 70000.0
    self.assertGreater(_TIME_STEP_SEC * self.conductance[0] / capacitance, 10.0)

    implicit = self.previous
    explicit = self.previous
    for _ in range(24):
      self.solver.reset(implicit)
      implicit = self.solver.solve(
          implicit,
          self.input_q,
          _AMBIENT,
          _CONVECTION,
          self.conductance,
          self.supply_temp,
      )
      lagged = self.input_q.reshape(-1) + self.applied @ (
          self.conductance
          * (self.supply_temp - self.sensed.T @ explicit.reshape(-1))
      )
      self.solver.reset(explicit)
      explicit = self.solver.solve(
          explicit,
          lagged.reshape(self.building.temp.shape),
          _AMBIENT,
          _CONVECTION,
      )

    # The implicit zone settles between the two supply temperatures it is being
    # driven towards; the explicit one leaves the building's own bounds.
    self.assertBetween(implicit[zone].mean(), 285.0, 305.0)
    self.assertGreater(abs(explicit[zone].mean() - 293.0), 1e4)

  def test_a_term_may_not_touch_a_cv_that_is_pinned_to_ambient(self):
    """Exterior CVs are not unknowns, so nothing can drive or read them.

    Silently dropping such an entry would leave a VAV heating the outdoors, or
    averaging the weather into its zone temperature, with no sign that it had
    happened.
    """
    outside = np.ravel_multi_index((0, 0), self.building.temp.shape)
    applied = self.applied.tolil()
    applied[outside, 0] = 1.0

    with self.assertRaises(ValueError):
      self.solver.couple_air_side(applied.tocsc(), self.sensed)


if __name__ == '__main__':
  absltest.main()
