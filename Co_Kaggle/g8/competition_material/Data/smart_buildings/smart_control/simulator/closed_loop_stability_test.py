"""Closed-loop stability of the shipped scenarios.

Every other test in this package drives the physics open loop: it hands the
solver a temperature field and a fixed forcing and checks what comes back. That
is enough to catch a wrong conductance and blind to the failure that actually
matters, because the HVAC is not fixed forcing -- it reads the zone temperature
and pushes back on it, and the loop that closes through it has a stability
condition of its own that no matrix property can express.

That loop used to be explicit. A VAV's air stream delivers

    q = Vdot * rho * cp * (T_supply - T_zone)

and it was evaluated from the zone temperature at the START of a step, stored
in building.input_q and held constant by the solve. So while the conduction was
backward Euler and unconditionally stable, the air side was forward Euler
across a conductance G = Vdot * rho * cp acting on the zone-air capacitance:

    T[n+1] - T_supply = (1 - dt*G/C) * (T[n] - T_supply)

which grows without bound once dt*G/C > 2, however exactly the conduction is
solved. On these floor plans that ratio runs from 3 to 21, and seven of the
eight buildings blew past 400 K within a few hours of any policy that turned
the heat on. finite_difference_solver.couple_air_side now puts the air side in
the matrix with everything else, so the ratio no longer bounds anything.

What is worth guarding is the end of that argument rather than its middle: the
field stays where a building's field can be, under actions an agent will
actually reach for. A stability condition stated in terms of the current
discretization would have to be rewritten the next time the discretization
changes; this survives that.
"""

import functools
from pathlib import Path

import numpy as np
from absl.testing import absltest, parameterized

from smart_buildings.evaluate import load_hybrid_environment

_SCENARIO_ROOT = Path('smart_buildings/training_scenarios')
_SCENARIOS = sorted(p.name for p in _SCENARIO_ROOT.iterdir() if p.is_dir())


def _weather_for(building: str) -> str:
  """One weather strip for this building, whichever it happens to be paired with.

  The material constants, VAV flow and timestep that set the gain are
  per-building blocks that scenario_generator emits into every strip
  identically, so a second strip for the same building buys nothing; what
  varies between buildings is the floor plan, and that is what sizes the zones.

  The name cannot be hardcoded. Buildings are paired with one strip per station
  and which strip depends on the building, so no single strip name exists
  across all of them.
  """
  strips = sorted(p.name for p in (_SCENARIO_ROOT / building).iterdir() if p.is_dir())
  if not strips:
    raise FileNotFoundError(f'no generated scenarios for {building}')
  return strips[0]

# The bounds are not invented here: simulator_flexible_floor_plan clips the
# field to them and logs "the finite-difference solve has diverged" when it has
# to, on the stated grounds that no weather and no habitable building can reach
# them. These tests assert that guard never has to fire.
_COLDEST_K = 150.0
_HOTTEST_K = 500.0
_STEPS = 24


@functools.cache
def _env(building: str):
  """Returns the environment for one shipped scenario, built at most once.

  Construction dominates the cost of this file at several seconds a scenario,
  and both tests want the same eight, so they share them. gin is global state
  and load_hybrid_environment clears it, so an env must be finished with before
  the next is built -- which holds here because each test consumes one at a
  time and the cache hands back the already-built object.
  """
  return load_hybrid_environment(
      _SCENARIO_ROOT / building / _weather_for(building) / 'config.gin'
  )


def _drive(building: str, actions) -> tuple[float, float]:
  """Returns the coldest and hottest CV seen while stepping through actions."""
  env = _env(building)
  env.reset()
  coldest, hottest = np.inf, -np.inf
  for action in actions:
    env.step(action)
    temp = env.building.simulator.building.temp
    coldest = min(coldest, float(temp.min()))
    hottest = max(hottest, float(temp.max()))
  return coldest, hottest


class ClosedLoopStabilityTest(parameterized.TestCase):

  def _assert_inside_the_guard(self, building, coldest, hottest, description):
    self.assertGreater(
        coldest,
        _COLDEST_K,
        msg=(
            f'{building} fell to {coldest:.1f} K within {_STEPS} steps of'
            f' {description}; the solve has diverged.'
        ),
    )
    self.assertLess(
        hottest,
        _HOTTEST_K,
        msg=(
            f'{building} rose to {hottest:.1f} K within {_STEPS} steps of'
            f' {description}; the solve has diverged.'
        ),
    )

  @parameterized.named_parameters((b, b) for b in _SCENARIOS)
  def test_a_constant_maximum_command_does_not_blow_the_building_up(
      self, building
  ):
    """Two hours of the hottest legal action, which is the first thing tried.

    Constant, in-spec, and reachable on step one: every continuous setpoint at
    its maximum and every air handler running. This is the case the explicit
    coupling could not survive, because holding every damper open holds every
    zone's gain at its nameplate value for hours on end.
    """
    env = _env(building)
    spec = env.action_spec()
    action = {
        'continuous_action': np.ones(spec['continuous_action'].shape, dtype=np.float32),
        'discrete_action': np.ones(spec['discrete_action'].shape, dtype=np.int32),
    }

    coldest, hottest = _drive(building, [action] * _STEPS)

    self._assert_inside_the_guard(
        building, coldest, hottest, 'a constant maximum in-spec action'
    )

  @parameterized.named_parameters((b, b) for b in _SCENARIOS)
  def test_an_action_that_changes_every_step_does_not_either(self, building):
    """The same two hours, with the whole action space rattled.

    An agent early in training does not hold anything constant, and every
    setpoint it moves changes the conductance the solver is handed for that
    step. Holding the action still would leave that path -- rebuilding and
    re-solving the coupled system against a different damper position each
    step -- untested, and it is the path every real episode takes.
    """
    env = _env(building)
    spec = env.action_spec()
    rng = np.random.default_rng(0)
    actions = [
        {
            'continuous_action': rng.uniform(
                0.0, 1.0, spec['continuous_action'].shape
            ).astype(np.float32),
            'discrete_action': rng.integers(
                0, 2, spec['discrete_action'].shape
            ).astype(np.int32),
        }
        for _ in range(_STEPS)
    ]

    coldest, hottest = _drive(building, actions)

    self._assert_inside_the_guard(
        building, coldest, hottest, 'an action redrawn at random every step'
    )


if __name__ == '__main__':
  absltest.main()
