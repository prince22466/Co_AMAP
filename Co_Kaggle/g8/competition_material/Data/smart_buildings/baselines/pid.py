import re
from typing import Any

import numpy as np


class PIDPolicy:
    """Distributed per-AHU PID controller with proportional fan pressure.

    Each AHU runs an independent PID loop on the average temperature of the
    zones it serves, modulating that AHU's supply-air-temperature setpoint in
    proportion to how far those zones sit outside the active comfort/eco band.
    An AHU idles whenever its zones are inside the band, which is the same
    rule BangBangPolicy uses.

    Fan pressure scales with the control effort |u| rather than being pinned
    at maximum whenever the AHU is active: |u| = 0 maps to pressure -1
    (minimum airflow) and |u| = 1 to pressure +1 (full blast), linearly in
    between. So a zone barely outside the band gets a gentle push instead of
    the full fan.

    That last point is the main energy lever, and it is worth stating why.
    Fan power is flow * static_pressure / efficiency, so holding pressure at
    maximum is expensive for as long as it is held. Pinning it while a zone
    drifts back into band is particularly costly here: because a PID corrects
    gently, that recovery window is long, and a smoothly modulating controller
    can end up drawing more energy than a blunter on/off one. Scaling pressure
    with control effort removes that penalty.

    The water loop is deliberately left binary: hot water on if any air handler
    is heating, pump on if any is active. Making it proportional to demand
    saves very little gas, because the heat is genuinely needed, and what it
    does save tends to be cancelled by extra fan electricity from slower zone
    recovery.

    Underneath the AHU loop, each VAV gets its own damper and reheat valve
    command from its own zone's temperature, so a zone that is comfortable is
    throttled to its ventilation minimum while its neighbours on the same AHU
    are still being conditioned. The AHU loop can only pick one supply air
    temperature for all of them, which is why this second layer is worth having:
    it is the only place a per-zone decision can be made at all.
    """

    def __init__(
        self,
        building_meta_data,
        action_spec,
        kp: float = 0.5,
        ki: float = 0.02,
        kd: float = 0.1,
        integral_limit: float = 10.0,
        # Excursion in K at which a zone's damper is commanded fully open. The
        # AHU loop reaches full effort at 1/kp = 2 K, so this matches it.
        zone_damper_full_k: float = 2.0,
    ):
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

        self.observation_normalization_metadata = building_meta_data[
            "observation_normalization_metadata"
        ]
        self.zone_air_temperature_sensor_mean = self.observation_normalization_metadata[
            "zone_air_temperature_sensor"
        ]["sample_mean"]
        self.zone_air_temperature_sensor_std = self.observation_normalization_metadata[
            "zone_air_temperature_sensor"
        ]["sample_std"]

        self.hod_sin_index = self.field_names.index("hod_sin_000")
        self.hod_cos_index = self.field_names.index("hod_cos_000")
        self.dow_sin_index = self.field_names.index("dow_sin_000")
        self.dow_cos_index = self.field_names.index("dow_cos_000")

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.zone_damper_full_k = zone_damper_full_k

        self._build_ahu_mapping()
        self.reset()

    def _build_ahu_mapping(self) -> None:
        zone_temp_field_index = {
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
            # (vav id, observation index, action indices) for each zone this AHU
            # serves that the building actually reports. A zone whose sensor is
            # missing is skipped entirely rather than being commanded blind.
            zones: list[tuple[str, int, dict[str, int]]] = []
            for room in rooms:
                vav_id = room.replace("room_", "VAV_")
                vav_field = vav_id + "_zone_air_temperature_sensor"
                if vav_field in zone_temp_field_index:
                    zones.append(
                        (
                            vav_id,
                            zone_temp_field_index[vav_field],
                            self.zone_action_indices.get(vav_id, {}),
                        )
                    )

            self.ahu_controllers.append(
                {
                    "ahu_id": ahu_id,
                    "temp_idx": self.temp_indices[i],
                    "pressure_idx": self.pressure_indices[i]
                    if i < len(self.pressure_indices)
                    else None,
                    "run_discrete_idx": i,
                    "zones": zones,
                    "zone_obs_indices": [obs_idx for _, obs_idx, _ in zones],
                }
            )

        self.water_temp_idx = self.water_indices[0] if self.water_indices else None
        self.water_pressure_idx = (
            self.pressure_indices[-1] if len(self.pressure_indices) > self.n_ahu else None
        )
        self.water_discrete_idx = self.n_discrete - 1

    @staticmethod
    def _numeric_suffix(name: str) -> int:
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0

    @property
    def name(self) -> str:
        return "pid"

    def reset(self) -> None:
        self.pid_state = {i: {"integral": 0.0, "prev_error": 0.0} for i in range(self.n_ahu)}

    def _cyclic_feature_to_index(self, sin_value: float, cos_value: float, period: int) -> int:
        angle = np.arctan2(sin_value, cos_value)
        if angle < 0:
            angle += 2 * np.pi
        return int(np.floor(period * angle / (2 * np.pi)))

    def _get_hour_and_day(self, obs: np.ndarray):
        hour = self._cyclic_feature_to_index(obs[self.hod_sin_index], obs[self.hod_cos_index], period=24)
        day = self._cyclic_feature_to_index(obs[self.dow_sin_index], obs[self.dow_cos_index], period=7)
        return hour, day

    def _denormalize_zone_temp(self, normalized_value: float) -> float:
        return normalized_value * self.zone_air_temperature_sensor_std + self.zone_air_temperature_sensor_mean

    def _deadband_error(self, avg_temp: float, low: float, high: float) -> float:
        if avg_temp < low:
            return low - avg_temp
        if avg_temp > high:
            return high - avg_temp
        return 0.0

    def _command_zones(
        self,
        continuous: np.ndarray,
        zones: list[tuple[str, int, dict[str, int]]],
        obs: np.ndarray,
        low: float,
        high: float,
    ) -> None:
        """Sets each VAV's damper and reheat valve from its own zone temperature.

        Proportional in the zone's own excursion, so a zone that is in band is
        throttled to its ventilation floor while a neighbour on the same air
        handler stays wide open. The reheat valve is binary because the supply
        air is either too cold for this zone or it is not; there is no third
        thing for the valve to do.
        """
        for _, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            zone_error = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), low, high
            )
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)

            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 1.0 if zone_error > 0 else -1.0

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        hour, day = self._get_hour_and_day(obs)
        is_weekday = day < 5
        is_work_hour = self.morning_start_hour <= hour < self.evening_start_hour

        if is_weekday and is_work_hour:
            low_temp, high_temp = self.comfort_temp_window
        else:
            low_temp, high_temp = self.eco_temp_window

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        any_heating = False
        any_active = False

        for i, ctrl in enumerate(self.ahu_controllers):
            zone_obs_indices = ctrl["zone_obs_indices"]
            if not zone_obs_indices:
                continue

            normalized_temps = obs[zone_obs_indices]
            avg_temp = float(np.mean([self._denormalize_zone_temp(v) for v in normalized_temps]))

            self._command_zones(continuous, ctrl["zones"], obs, low_temp, high_temp)

            error = self._deadband_error(avg_temp, low_temp, high_temp)

            state = self.pid_state[i]
            if error == 0.0:
                # Inside the band: idle, same as bang_bang.
                state["integral"] *= 0.5
                state["prev_error"] = 0.0
                continuous[ctrl["temp_idx"]] = -1.0
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = -1.0
                discrete[ctrl["run_discrete_idx"]] = 0
                continue

            state["integral"] = float(
                np.clip(state["integral"] + error, -self.integral_limit, self.integral_limit)
            )
            derivative = error - state["prev_error"]
            state["prev_error"] = error

            u = self.kp * error + self.ki * state["integral"] + self.kd * derivative
            u = float(np.clip(u, -1.0, 1.0))

            # Pressure scales with control effort: |u|=0 -> pressure=-1 (min),
            # |u|=1 -> pressure=+1 (max). No longer pinned to 1.0.
            pressure = float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))

            continuous[ctrl["temp_idx"]] = u
            if ctrl["pressure_idx"] is not None:
                continuous[ctrl["pressure_idx"]] = pressure
            discrete[ctrl["run_discrete_idx"]] = 1

            any_active = True
            if error > 0:
                any_heating = True

        # Recirculate as much as the ventilation cap allows. This is the no-
        # economizer choice, and it is deliberately left as headroom: deciding
        # when outside air is cheaper than recirculated air needs the weather,
        # which this controller does not look at.
        for i in self.recirculation_indices:
            continuous[i] = 1.0

        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = 1.0 if any_heating else -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if any_active else -1.0
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 1 if any_active else 0

        return {"continuous_action": continuous, "discrete_action": discrete}
