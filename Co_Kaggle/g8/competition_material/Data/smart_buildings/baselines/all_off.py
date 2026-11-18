from typing import Any

import numpy as np


class AllOffPolicy:
    """Policy that turns off all HVAC equipment."""

    def __init__(
        self,
        building_meta_data,
        action_spec,
    ):
        self.metadata = building_meta_data
        self.action_spec = action_spec
        self.n_discrete = building_meta_data["n_discrete_action"]
        self.n_continuous = building_meta_data["n_continuous_action"]
        self.temp_indices = building_meta_data["temp_indices"]
        self.pressure_indices = building_meta_data["pressure_indices"]
        self.water_indices = building_meta_data["water_indices"]
        self.recirculation_indices = building_meta_data["recirculation_indices"]
        self.zone_action_indices = building_meta_data["zone_action_indices"]

    @property
    def name(self) -> str:
        return "all_off"

    def _make_action(
        self,
        discrete_value: int,
        temp_value: float,
        pressure_value: float,
        water_value: float,
    ) -> dict[str, Any]:
        continuous = np.zeros(self.n_continuous, dtype=np.float32)

        for i in self.temp_indices:
            continuous[i] = temp_value
        for i in self.pressure_indices:
            continuous[i] = pressure_value
        for i in self.water_indices:
            continuous[i] = water_value
        # The zone actions have to be written even to say "off": an untouched
        # slot is normalized 0.0, which is the middle of the native range, not
        # the bottom of it. Damper follows the fan and reheat follows the hot
        # water, each pair being the same decision taken at the plant and again
        # at the zone.
        for indices in self.zone_action_indices.values():
            damper_idx = indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = pressure_value
            reheat_idx = indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = water_value
        # No economizer in a baseline this dumb: recirculate as much as the
        # ventilation cap allows, whatever the outside air happens to be doing.
        for i in self.recirculation_indices:
            continuous[i] = 1.0

        return {
            "continuous_action": continuous,
            "discrete_action": np.full(
                self.n_discrete,
                discrete_value,
                dtype=np.int32,
            ),
        }

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        return self._make_action(
            discrete_value=0, temp_value=-1, pressure_value=-1, water_value=-1
        )
