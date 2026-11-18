"""A reference controller that ignores the occupancy schedule.

Most of the reference policies in this package are one controller with
successive refinements added. A set built from a single family is a weak ruler:
it explores one idea, and its weakest member tends to be weak everywhere. This
one deliberately does something structurally different.

It subclasses the PID policy so it shares the air-handler mapping, band
reconstruction and zone commands. Only the decision under test differs.
"""

from typing import Any

import numpy as np

from smart_buildings.baselines.pid import PIDPolicy


class FixedSetpointPolicy(PIDPolicy):
    """One comfort band, all day, every day. No eco setback.

    Many real buildings are run exactly this way, so this is a plausible
    controller rather than a degenerate one. It is identical to `pid` in every
    other respect, so the difference measured against `pid` is the value of the
    schedule and nothing else.

    Worth keeping even where it scores poorly. A reference set whose weakest
    member is always the same controller inherits that controller's quirks. A
    second plausible-but-imperfect policy that fails for a *different* reason
    makes the lower end of the range a real minimum rather than one
    controller's worst case.
    """

    @property
    def name(self) -> str:
        return "fixed_setpoint"

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        # `pid.get_action` picks the band from the hour. Rather than copy the
        # whole method to change that one choice, pin both windows to the
        # comfort band and delegate. `_get_hour_and_day` still runs; its answer
        # simply stops mattering.
        saved = self.eco_temp_window
        self.eco_temp_window = self.comfort_temp_window
        try:
            return super().get_action(obs)
        finally:
            self.eco_temp_window = saved
