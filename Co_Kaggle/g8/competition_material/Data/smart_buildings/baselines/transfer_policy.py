"""Starting point for your own policy. Edit this file.

This is the template a submission is built from. As shipped it returns the zero
action on every step, which is a valid action but a poor controller: it does
nothing at all. It exists so `python -m smart_buildings.evaluate` runs out of
the box and so you have a working shape to edit rather than a blank file.

Run it with:

    python -m smart_buildings.evaluate --policies transfer \\
        --buildings building_train_1 --weathers fairbanks_1

A policy needs exactly two things: ``__init__(building_meta_data, action_spec)``
and ``get_action(obs)``. Nothing else is required, there is no base class to
subclass, and nothing needs registering unless you rename the class -- in which
case update ``POLICY_MAP`` in ``evaluate.py``.

## The action

Hybrid, and its shape depends on the building. For a building with ``K`` air
handlers and ``V`` VAV boxes (one per zone):

    discrete_action    int32,   shape (K + 1,)       on/off per AHU, plus the pump
    continuous_action  float32, shape (3K + 2 + 2V,) three setpoints per AHU, two
                                                     for the hot-water system, and
                                                     two per VAV box

Continuous values are normalised to ``[-1, 1]`` and the environment maps them to
physical units. Discrete values are ``{0, 1}``.

**The continuous layout is interleaved, not grouped.** The plant entries run
``[temp_0, pressure_0, recirculation_0, temp_1, ...]`` with the two water-system
entries appended, and the zone entries follow those. Do not assume the ordering:
read ``building_meta_data["temp_indices"]``, ``["pressure_indices"]``,
``["recirculation_indices"]``, ``["water_indices"]`` and
``["zone_action_indices"]`` and scatter into those positions. A policy that
hardcodes the layout will appear to work and quietly drive the wrong actuator.

The first four are positional lists in AHU order. ``zone_action_indices`` is not:
it maps VAV id (``"VAV_7"``) to setpoint name to index, because a zone decision is
about one particular box and there is no order to count along.
``building_meta_data["ahu_zone_assignments"]`` tells you which zones each AHU
serves, in ``room_N`` form -- the matching box is ``VAV_N``.

## What the continuous setpoints do

Per air handler:

| setpoint | effect |
|---|---|
| supply air temperature | how cold or hot the air arriving at the zones is |
| supply air static pressure | how hard the fan pushes, so how much air moves |
| recirculation | fraction of return air reused instead of drawn from outside |

Per VAV box, so per zone:

| setpoint | effect |
|---|---|
| supply air damper | this zone's share of the air the fan is moving |
| reheat valve | hot water to this zone's coil, to temper air that is too cold |

And for the hot-water system:

| setpoint | effect |
|---|---|
| supply water temperature | how hot the heating loop runs |
| differential pressure | how hard the pump pushes water round the loop |

The air side is what moves zone temperature quickly. The water side sets how much
heating capacity is available for the air side to use. The AHU picks one supply
air temperature for every zone it serves, so the dampers and reheat valves are
the only place a per-zone decision can be made at all -- that is where the
headroom over the shipped baselines mostly is.

**The air handler can only cool. All heat in this building comes from the VAV
reheat coils, fed by the hot-water loop.** The AHU mixes return air with outside
air and then cools that mixture down to your setpoint, so what actually leaves it
is::

    supply air = min(mixed air temperature, your setpoint)

Read that formula before you tune the supply air temperature action, because it
means the action goes dead whenever the mixed air is already colder than the
coldest value you are allowed to command. That is not a rare corner.

Work out when it happens. Mixing gives ``mixed = r*return + (1-r)*outside``
with ``r`` capped at 0.70, the coldest setpoint you can command is 285 K, and
return air from a building in band is about 293 K. So the setpoint is inert
whenever outside air is below **-6.8 C**, whatever you command. Over the scored
window of the shipped strips that works out as:

| station | ambient range | hours the setpoint is inert |
|---|---|---|
| singapore | 22.9 to 34.0 C | 0.0% |
| phoenix | 0.4 to 45.6 C | 0.0% |
| atlanta | -3.9 to 35.0 C | 0.0% |
| denver | -21.1 to 37.2 C | 3.8% |
| fairbanks | -42.0 to 28.9 C | 31.5% |

So in the cold the real levers are the hot-water temperature and the reheat
valves, and the supply air temperature action is mostly along for the ride. In
the heat it is one of the strongest levers you have. This is a property of the
plant, stated here so you do not have to discover it by ablation.

**Ventilation floors are enforced by the simulator, not by you.** A running AHU
holds a minimum static pressure, cannot recirculate more than 70%, and every VAV
damper is floored at 10% open. Commanding below a floor is not an error; you
simply get the floor. Occupied zones have to keep receiving outside air, so
"close everything" is not available as an energy strategy.

## The observation

One flat float vector whose length varies by building: 352 for
``building_train_1``, 271 for ``building_train_8``.
``building_meta_data["observation_field_names"]`` gives a name per element, so
build an index table once in ``__init__`` and slice with it, rather than
assuming offsets.

It contains, per zone: air temperature, current heating and cooling setpoints,
damper position, reheat valve position, airflow setpoint. Per air handler: supply
air temperature and setpoint, static pressure and setpoint, airflow and setpoint,
outside air temperature, recirculation, cooling request count, fan status. Plus
the hot-water system, building occupancy, comfort-schedule flags, and cyclically
encoded time of day and day of week.

The damper and reheat positions are what the boxes actually took, which is not
always what you asked for: a command below the ventilation floor is reported back
at the floor.

**There is no weather label, month, station name or building id.** Which climate
you are in has to be inferred from ``outside_air_temperature_sensor``, building
state and time of day. This is deliberate: hardcoding a reference to a held-out
config or weather seed is disqualifying under the competition rules.

## What you are scored on

Per step, a weighted trade-off: 0.6 occupant comfort, 0.2 energy cost, 0.2
carbon. Comfort is judged against a band that widens outside comfort hours, and
the penalty is **unbounded** -- it grows quadratically once a zone is more than
5 K outside the band, so one badly neglected zone can dominate a whole episode.

Doing nothing is not safe, and the zero action is not "nothing". Zero is the
*middle* of each normalised range, so leaving this file as shipped commands every
damper half open and every valve half shut, not off. Look at the shipped
baselines in this directory for reference points. The four that define the
score are ``bang_bang``, ``fixed_setpoint``, ``pid`` and ``pid_water``, and the
two weak ones fail in opposite climates: ``bang_bang`` is the worst of the four
in the cold, ``fixed_setpoint`` the worst of the four in the heat. So the
bottom of the scale is not one controller's bad day, and a policy that beats
the floor everywhere has to be good in both. ``all_off`` is runnable too, as a
sense of what an actually-off building costs, but it is not one of the four.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np


class TransferPolicy:
    """Returns the zero action every step. Replace ``get_action`` with control."""

    name = "transfer"

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        action_spec: Any,
    ) -> None:
        """Set up whatever your policy needs, once per building.

        Args:
            building_meta_data: Action indices, observation field names,
                zone-to-AHU assignment, floor plan, comfort schedule and
                normalisation constants for this building. Build any index
                tables here, not in ``get_action`` -- this runs once, that runs
                8640 times.
            action_spec: The hybrid action spec for this building.
        """
        self.metadata = building_meta_data
        self.action_spec = action_spec
        self.n_discrete = int(building_meta_data["n_discrete_action"])
        self.n_continuous = int(building_meta_data["n_continuous_action"])

        # Number of air handlers. Counted from the index table rather than from
        # the action width, which also carries the water system and two entries
        # per VAV box.
        self.num_ahus = len(building_meta_data["temp_indices"])

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Return one action for the current observation.

        Args:
            obs: Flat observation vector for this building.

        Returns:
            ``{"continuous_action": float32 (3K+2+2V,),
               "discrete_action": int32 (K+1,)}``
        """
        return {
            "continuous_action": np.zeros(self.n_continuous, dtype=np.float32),
            "discrete_action": np.zeros(self.n_discrete, dtype=np.int32),
        }
