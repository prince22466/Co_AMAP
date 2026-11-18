"""Tests that no zone is small enough to diverge.

A zone's air heat capacity is rho * c * area * height, so it falls with floor
area while two things acting on it do not fall with it:

  * occupant heat is assigned per zone -- zone_assignment occupants at
    occupant_sensible_heat_w each -- regardless of how big the zone is;
  * the VAV serving it senses the area-weighted mean temperature of all its
    zones and spreads its output by area, so a small zone is a small fraction
    of both the signal and the authority.

Internal gains do scale with area, which is what makes the occupant term the
odd one out. The result is a zone that heats at a fixed rate the loop
controlling it cannot see. Measured on the shipped training set before this
check existed: building_train_8's room_1, 9.88 m2 and 35,746 J/K, reached
149.2 C over a scored episode, and building_train_5's 9.36 m2 room_23 reached
141.4 C. Nothing bounded either but the solver's own np.clip(temp, 150, 500),
which tops out near 227 C. Six of the 48 shipped scenarios carried a zone above
40 C, five above 100 C. Zones of 15.16 m2 and larger stayed under 30 C. Those
counts are over the 48 scenarios shipped when this was measured; the set is
now 30.

It corrupts the score and not only the trajectory. Comfort is 0.6 of the
reward, its shortfall is unbounded past the deviation reference, and
productivity is summed over zones weighted by each zone's own occupancy rather
than its area. A zone 167 K outside the band scores a productivity factor near
-1050 where an in-band zone scores about +1, at the same weight, so a single
sliver decides the scenario and therefore r_best and r_worst with it.

The floor plan parameters cannot prevent this. min_room_size bounds the BSP
leaves, but the composite shape mask is applied after the split and clips any
leaf crossing its boundary; what survives is a connected component whose size
nothing checked. building_train_5 and building_train_7 share min_room_size 25
and composite_coverage 0.67 and come out with smallest zones of 9.36 m2 and
50.88 m2. So the realised minimum has to be measured, which is what these tests
do.
"""

import pathlib

from absl.testing import absltest, parameterized

from smart_buildings.smart_control.simulator import building as building_py
from smart_buildings.smart_control.utils import scenario_generator

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCENARIOS = _REPO_ROOT / "smart_buildings" / "training_scenarios"

# 20 m2 is 10 m2 per occupant at the shipped zone assignment of 2, an ordinary
# office density, and measurement puts the runaway's end at about 17 m2. Keeping
# zones at least this big is what makes per-zone occupancy a sound assumption,
# which is why the fix is a geometry limit rather than a change to the
# occupancy model.
_MIN_AREA_M2 = 20.0
_CV_SIZE_CM = 20.0
_FLOOR_HEIGHT_CM = 300.0

# Only the thermal grid is built here, so these stand in for the per-building
# drawn materials. Nothing in these tests depends on their values.
_AIR = building_py.MaterialProperties(
    conductivity=0.05, heat_capacity=1005.0, density=1.2
)
_WALL = building_py.MaterialProperties(
    conductivity=0.5, heat_capacity=900.0, density=127.0
)
_EXTERIOR = building_py.MaterialProperties(
    conductivity=0.2, heat_capacity=850.0, density=500.0
)


def _scenarios() -> list[tuple[str, pathlib.Path]]:
  """Every shipped training scenario, as (test name, directory)."""
  found = sorted(p.parent for p in _SCENARIOS.glob("*/*/config.gin"))
  return [(f"{d.parent.name}_{d.name}", d) for d in found]


def _build(scenario_dir: pathlib.Path, **kwargs):
  """Constructs the thermal grid for one scenario, without gin or HVAC."""
  return building_py.FloorPlanBasedBuilding(
      cv_size_cm=_CV_SIZE_CM,
      floor_height_cm=_FLOOR_HEIGHT_CM,
      initial_temp=293.0,
      inside_air_properties=_AIR,
      inside_wall_properties=_WALL,
      building_exterior_properties=_EXTERIOR,
      floor_plan_filepath=str(scenario_dir / "floor_plan.npy"),
      zone_map_filepath=str(scenario_dir / "zone_map.npy"),
      **kwargs,
  )


def _zone_areas(building) -> dict[str, float]:
  cv_area_m2 = (_CV_SIZE_CM / 100.0) ** 2
  return {
      name: len(coords) * cv_area_m2
      for name, coords in building.room_dict.items()
      if name not in ("exterior_space", "interior_wall")
  }


class ShippedScenarioZoneAreaTest(parameterized.TestCase):
  """Every shipped scenario must satisfy the limit its own default enforces.

  Parameterized per scenario rather than looped so a failure names the building
  and strip that regressed, which is what tells you which seed to redraw.
  """

  def test_scenarios_exist(self):
    # A glob that silently matches nothing would make every other test in this
    # class vacuous.
    self.assertLen(_scenarios(), 30)

  @parameterized.named_parameters(*_scenarios())
  def test_no_zone_below_minimum(self, scenario_dir: pathlib.Path):
    building = _build(scenario_dir)
    areas = _zone_areas(building)
    self.assertNotEmpty(areas)
    offenders = {n: a for n, a in areas.items() if a < _MIN_AREA_M2}
    self.assertEmpty(
        offenders,
        f"{scenario_dir.parent.name}/{scenario_dir.name} carries zones under "
        f"{_MIN_AREA_M2} m2: {offenders}. Redraw the building with a different "
        "seed and measure the realised minimum; min_room_size does not set it.",
    )


class GuardTest(absltest.TestCase):
  """The check itself, exercised on real shipped geometry."""

  def setUp(self):
    super().setUp()
    scenarios = _scenarios()
    self.assertNotEmpty(scenarios)
    self._scenario = scenarios[0][1]

  def test_constructs_at_the_default_limit(self):
    # The shipped default is the limit the shipped data satisfies, so this is
    # the ordinary path and it must not raise.
    self._build_ok()

  def _build_ok(self):
    building = _build(self._scenario)
    self.assertNotEmpty(_zone_areas(building))
    return building

  def test_raises_when_a_zone_is_below_the_limit(self):
    # Rather than synthesize a sliver, raise the limit above the real minimum.
    # That exercises the same comparison on the same geometry.
    smallest = min(_zone_areas(self._build_ok()).values())
    with self.assertRaises(ValueError) as caught:
      _build(self._scenario, min_zone_floor_area_m2=smallest + 1.0)
    message = str(caught.exception)
    self.assertIn("minimum floor area", message)
    # The message has to name the offenders, because the fix is to redraw the
    # seed and you need to know how far short the plan fell.
    self.assertIn("m2", message)

  def test_zero_disables_the_check(self):
    smallest = min(_zone_areas(self._build_ok()).values())
    # Same limit that raised above, but switched off.
    building = _build(self._scenario, min_zone_floor_area_m2=0.0)
    self.assertLessEqual(smallest, min(_zone_areas(building).values()) + 1e-9)

  def test_generator_and_building_agree_on_the_limit(self):
    # Two defaults in two modules describing one rule. If they drift, a
    # scenario generates cleanly and then refuses to load.
    self.assertEqual(
        scenario_generator.MIN_ZONE_FLOOR_AREA_M2, _MIN_AREA_M2
    )

  def test_generator_and_configs_agree_on_control_volume_size(self):
    # The area check converts cells to m2 with this, and the simulator reads it
    # out of config.gin. A literal in each place is how the two come apart.
    config = (self._scenario / "config.gin").read_text()
    self.assertIn(
        f"control_volume_cm = {scenario_generator.CONTROL_VOLUME_CM}", config
    )
    self.assertIn(
        f"floor_height_cm = {scenario_generator.FLOOR_HEIGHT_CM}", config
    )


if __name__ == "__main__":
  absltest.main()
