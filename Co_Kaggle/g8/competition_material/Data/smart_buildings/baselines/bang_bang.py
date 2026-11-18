import re
from typing import Any

import numpy as np


class BangBangPolicy:
    """Per-AHU bang-bang controller.

    Each AHU independently inspects the zones it serves and switches on when
    the mean zone temperature leaves the active comfort/eco band, and off once
    it returns inside. Each AHU is commanded independently, so this controller can heat one
    AHU while keeping another off, or cool one while the other is idle.

    Setpoints are fixed at the band edges (full hot/cold blast), same as
    bang_bang.  The water pump follows aggregate AHU demand.
    """

    def __init__(self, building_meta_data, action_spec):
        self.metadata = building_meta_data
        self.action_spec = action_spec

        self.comfort_temp_window = building_meta_data["comfort_temp_window"]
        self.eco_temp_window = building_meta_data["eco_temp_window"]
        self.morning_start_hour = building_meta_data["morning_start_hour"]
        self.evening_start_hour = building_meta_data["evening_start_hour"]

        self.n_discrete = building_meta_data["n_discrete_action"]
        self.n_continuous = building_meta_data["n_continuous_action"]
        self.temp_indices = list(building_meta_data["temp_indices"])
        self.pressure_indices = list(building_meta_data["pressure_indices"])
        self.water_indices = list(building_meta_data["water_indices"])
        self.recirculation_indices = list(building_meta_data["recirculation_indices"])
        self.zone_action_indices = building_meta_data["zone_action_indices"]
        self.field_names = building_meta_data["observation_field_names"]
        self.ahu_zone_assignments = building_meta_data["ahu_zone_assignments"]

        norm = building_meta_data["observation_normalization_metadata"]
        self._zone_mean = norm["zone_air_temperature_sensor"]["sample_mean"]
        self._zone_std = norm["zone_air_temperature_sensor"]["sample_std"]

        self.hod_sin_index = self.field_names.index("hod_sin_000")
        self.hod_cos_index = self.field_names.index("hod_cos_000")
        self.dow_sin_index = self.field_names.index("dow_sin_000")
        self.dow_cos_index = self.field_names.index("dow_cos_000")

        self._build_ahu_mapping()

    def _build_ahu_mapping(self) -> None:
        zone_field_index = {
            name: i
            for i, name in enumerate(self.field_names)
            if "zone_air_temperature" in name
        }

        ahu_ids = sorted(self.ahu_zone_assignments.keys(), key=self._numeric_suffix)
        self.n_ahu = len(self.temp_indices)
        if len(ahu_ids) != self.n_ahu:
            self.n_ahu = min(self.n_ahu, len(ahu_ids))
            ahu_ids = ahu_ids[: self.n_ahu]

        self.ahu_controllers: list[dict[str, Any]] = []
        for i, ahu_id in enumerate(ahu_ids):
            rooms = self.ahu_zone_assignments[ahu_id]
            zone_obs_indices: list[int] = []
            for room in rooms:
                field = room.replace("room_", "VAV_") + "_zone_air_temperature_sensor"
                if field in zone_field_index:
                    zone_obs_indices.append(zone_field_index[field])

            self.ahu_controllers.append(
                {
                    "ahu_id": ahu_id,
                    "temp_idx": self.temp_indices[i],
                    "pressure_idx": self.pressure_indices[i]
                    if i < len(self.pressure_indices)
                    else None,
                    "run_discrete_idx": i,
                    "zone_obs_indices": zone_obs_indices,
                    # Every VAV under this AHU, so the AHU's one decision can be
                    # copied down to its boxes. This controller stays deliberately
                    # plant-only: it never reads an individual zone to decide an
                    # individual damper, which is what separates it from pid.
                    "zone_action_indices": [
                        self.zone_action_indices[room.replace("room_", "VAV_")]
                        for room in rooms
                        if room.replace("room_", "VAV_") in self.zone_action_indices
                    ],
                }
            )

        self.water_temp_idx = self.water_indices[0] if self.water_indices else None
        self.water_pressure_idx = (
            self.pressure_indices[-1]
            if len(self.pressure_indices) > self.n_ahu
            else None
        )
        self.water_discrete_idx = self.n_discrete - 1

    @staticmethod
    def _numeric_suffix(name: str) -> int:
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0

    @property
    def name(self) -> str:
        return "bang_bang"

    def _cyclic_to_index(self, sin_v: float, cos_v: float, period: int) -> int:
        angle = np.arctan2(sin_v, cos_v)
        if angle < 0:
            angle += 2 * np.pi
        return int(np.floor(period * angle / (2 * np.pi)))

    def _denorm(self, v: float) -> float:
        return v * self._zone_std + self._zone_mean

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        hour = self._cyclic_to_index(obs[self.hod_sin_index], obs[self.hod_cos_index], 24)
        day = self._cyclic_to_index(obs[self.dow_sin_index], obs[self.dow_cos_index], 7)
        is_work = (day < 5) and (self.morning_start_hour <= hour < self.evening_start_hour)
        low_temp, high_temp = self.comfort_temp_window if is_work else self.eco_temp_window

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        any_heating = False
        any_active = False

        for ctrl in self.ahu_controllers:
            if not ctrl["zone_obs_indices"]:
                continue

            avg_temp = float(np.mean([self._denorm(obs[i]) for i in ctrl["zone_obs_indices"]]))

            if avg_temp < low_temp:
                continuous[ctrl["temp_idx"]] = 1.0   # max heat
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = 1.0
                discrete[ctrl["run_discrete_idx"]] = 1
                damper, reheat = 1.0, 1.0
                any_heating = True
                any_active = True

            elif avg_temp > high_temp:
                continuous[ctrl["temp_idx"]] = -1.0  # max cool
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = 1.0
                discrete[ctrl["run_discrete_idx"]] = 1
                damper, reheat = 1.0, -1.0
                any_active = True

            else:
                continuous[ctrl["temp_idx"]] = -1.0
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = -1.0
                discrete[ctrl["run_discrete_idx"]] = 0
                damper, reheat = -1.0, -1.0

            # Every box under this AHU takes the position the AHU just took. The
            # zone slots have to be written even to say "shut": an untouched slot
            # is normalized 0.0, which is mid-range natively, not closed.
            for indices in ctrl["zone_action_indices"]:
                damper_idx = indices.get("supply_air_damper_percentage_command")
                if damper_idx is not None:
                    continuous[damper_idx] = damper
                reheat_idx = indices.get("reheat_valve_percentage_command")
                if reheat_idx is not None:
                    continuous[reheat_idx] = reheat

        # No economizer: recirculate as much as the ventilation cap allows.
        for i in self.recirculation_indices:
            continuous[i] = 1.0

        # Water system follows aggregate AHU demand.
        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = 1.0 if any_heating else -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if any_active else -1.0
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 1 if any_active else 0

        return {"continuous_action": continuous, "discrete_action": discrete}
