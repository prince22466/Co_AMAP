"""Pure, dependency-light P0.1 reward, comfort and action diagnostics.

Read-only instrumentation: all measurements are taken from reward information and
the command sent to the environment. Nothing here changes the simulator reward.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from .metrics import weighted_losses

THRESHOLDS_K = (1, 3, 5, 10)
NEAR_BOUND = 0.9  # Within the outer 5% of the normalized [-1, 1] range.


def action_groups(metadata: dict) -> dict[str, tuple[str, list[int]]]:
    """Map participant-facing action indices to equipment types."""
    zone_actions = metadata["zone_action_indices"]

    def zone_indices(field: str) -> list[int]:
        return sorted({int(fields[field]) for fields in zone_actions.values()
                       if field in fields})

    ahu_count = len(metadata["temp_indices"])
    discrete_count = int(metadata["n_discrete_action"])
    if discrete_count not in (ahu_count, ahu_count + 1):
        raise ValueError("Unexpected discrete action layout")
    pressure = list(map(int, metadata["pressure_indices"]))
    return {
        "ahu_supply_temperature": ("continuous", list(map(int, metadata["temp_indices"]))),
        "ahu_static_pressure": ("continuous", pressure[:ahu_count]),
        "ahu_recirculation": ("continuous", list(map(int, metadata["recirculation_indices"]))),
        "water_supply_temperature": ("continuous", list(map(int, metadata["water_indices"]))),
        "water_pressure": ("continuous", pressure[ahu_count:]),
        "vav_damper": ("continuous", zone_indices("supply_air_damper_percentage_command")),
        "vav_reheat": ("continuous", zone_indices("reheat_valve_percentage_command")),
        "ahu_run": ("discrete", list(range(ahu_count))),
        "water_run": ("discrete", list(range(ahu_count, discrete_count))),
    }


def zone_excursion(zone: Any) -> float:
    """Kelvin outside the active heating/cooling setpoint band."""
    t = float(zone.zone_air_temperature)
    low = float(zone.heating_setpoint_temperature)
    high = float(zone.cooling_setpoint_temperature)
    if not all(map(math.isfinite, (t, low, high))) or low > high:
        raise ValueError("Invalid zone reward temperature or setpoints")
    return max(0.0, low - t, t - high)


class DiagnosticAccumulator:
    """Accumulate small summaries while the caller streams CSV rows to disk."""

    def __init__(self, metadata: dict):
        self.groups = action_groups(metadata)
        self.groups_stats = {
            name: {"samples": 0, "sum": 0.0, "low": 0, "high": 0,
                   "active": 0, "changes": 0, "sum_abs_delta": 0.0,
                   "transition_samples": 0}
            for name in self.groups
        }
        self.previous = {}
        self.count = 0
        self.loss = defaultdict(float)
        self.by_hour: dict[int, dict] = {}
        self.by_occupancy: dict[str, dict] = {}
        self.by_schedule: dict[str, dict] = {}
        self.zones = 0
        self.occupied_zones = 0
        self.occupant_samples = 0.0
        self.max_excursion_k = 0.0
        self.violations = {
            str(k): {"all": 0, "occupied_zones": 0, "occupant_samples": 0.0}
            for k in THRESHOLDS_K
        }

    @staticmethod
    def _update_bucket(buckets: dict, key: Any, parts: dict, occupied: bool) -> None:
        bucket = buckets.setdefault(key, {"steps": 0, "comfort": 0.0,
                                          "energy": 0.0, "carbon": 0.0,
                                          "occupied_steps": 0})
        bucket["steps"] += 1
        bucket["occupied_steps"] += int(occupied)
        for name, value in parts.items():
            bucket[name] += value

    @staticmethod
    def _finish_buckets(buckets: dict) -> dict:
        return {
            str(key): {
                "steps": value["steps"],
                "occupied_steps": value["occupied_steps"],
                "mean_loss": sum(value[k] for k in ("comfort", "energy", "carbon"))
                / value["steps"],
                "mean_components": {
                    k: value[k] / value["steps"]
                    for k in ("comfort", "energy", "carbon")
                },
            }
            for key, value in sorted(buckets.items())
        }

    def _record_actions(self, action: dict) -> None:
        for group, (kind, indices) in self.groups.items():
            arr = action[kind + "_action"]
            stat = self.groups_stats[group]
            for index in indices:
                value = float(arr[index])
                if not math.isfinite(value):
                    raise ValueError("Nonfinite command")
                stat["samples"] += 1
                stat["sum"] += value
                if kind == "continuous":
                    stat["low"] += int(value <= -NEAR_BOUND)
                    stat["high"] += int(value >= NEAR_BOUND)
                else:
                    stat["active"] += int(value == 1)
                key = (kind, index)
                if key in self.previous:
                    delta = abs(value - self.previous[key])
                    stat["sum_abs_delta"] += delta
                    stat["transition_samples"] += 1
                    if kind == "discrete":
                        stat["changes"] += int(delta > 0)
                self.previous[key] = value

    def record(self, *, timestamp: Any, comfort_mode: bool, action: dict,
               reward_info: Any, response: Any, reward: float, weights: dict) -> dict:
        """Produce one CSV row from the *actual* post-action reward response."""
        parts = weighted_losses(response.normalized_productivity_regret,
                                response.normalized_energy_cost,
                                response.normalized_carbon_emission, weights)
        if not math.isclose(-float(reward), sum(parts.values()),
                            abs_tol=1e-8, rel_tol=1e-7):
            raise ValueError("Reward decomposition is not gateway-equivalent")
        occupancy = float(response.total_occupancy)
        if not math.isfinite(occupancy) or occupancy < 0:
            raise ValueError("Invalid occupancy")
        occupied = occupancy > 0
        hour = int(timestamp.hour)
        self.count += 1
        for name, value in parts.items():
            self.loss[name] += value
        self._update_bucket(self.by_hour, hour, parts, occupied)
        self._update_bucket(self.by_occupancy,
                            "occupied" if occupied else "unoccupied", parts, occupied)
        self._update_bucket(self.by_schedule,
                            "comfort_mode" if comfort_mode else "eco_mode", parts, occupied)
        self._record_actions(action)

        zones = list(reward_info.zone_reward_infos.values())
        exceed = {k: 0 for k in THRESHOLDS_K}
        occupied_exceed = {k: 0 for k in THRESHOLDS_K}
        max_excursion = 0.0
        zone_occupied = 0
        zone_total_occupancy = 0.0
        for zone in zones:
            deviation = zone_excursion(zone)
            persons = float(zone.average_occupancy)
            if not math.isfinite(persons) or persons < 0:
                raise ValueError("Invalid zone occupancy")
            zone_occupied += int(persons > 0)
            zone_total_occupancy += persons
            max_excursion = max(max_excursion, deviation)
            for threshold in THRESHOLDS_K:
                if deviation > threshold:
                    exceed[threshold] += 1
                    occupied_exceed[threshold] += int(persons > 0)
                    self.violations[str(threshold)]["occupant_samples"] += persons

        if not math.isclose(zone_total_occupancy, occupancy, abs_tol=1e-4, rel_tol=1e-5):
            raise ValueError("Zone occupancy disagrees with reward response")
        self.zones += len(zones)
        self.occupied_zones += zone_occupied
        self.occupant_samples += zone_total_occupancy
        self.max_excursion_k = max(self.max_excursion_k, max_excursion)
        row = {
            "timestamp": timestamp.isoformat(),
            "hour": hour, "comfort_mode": int(comfort_mode),
            "occupancy": occupancy,
            "reward": float(reward),
            "comfort_loss": parts["comfort"],
            "energy_loss": parts["energy"],
            "carbon_loss": parts["carbon"],
            "electricity_cost": float(response.electricity_energy_cost),
            "natural_gas_cost": float(response.natural_gas_energy_cost),
            "carbon_emitted_kg": float(response.carbon_emitted),
            "zone_samples": len(zones),
            "occupied_zone_samples": zone_occupied,
            "max_zone_excursion_k": max_excursion,
        }
        for threshold in THRESHOLDS_K:
            self.violations[str(threshold)]["all"] += exceed[threshold]
            self.violations[str(threshold)]["occupied_zones"] += occupied_exceed[threshold]
            row["exceed_%dk_zones" % threshold] = exceed[threshold]
            row["exceed_%dk_occupied_zones" % threshold] = occupied_exceed[threshold]
        return row

    def summary(self) -> dict:
        if self.count == 0:
            raise ValueError("No diagnostic steps collected")
        action_summary = {}
        for name, (kind, _) in self.groups.items():
            stat = self.groups_stats[name]
            n = stat["samples"]
            m = stat["transition_samples"]
            action_summary[name] = {
                "samples": n,
                "mean_normalized_command": stat["sum"] / n if n else None,
                "mean_abs_command_change": stat["sum_abs_delta"] / m if m else None,
            }
            if kind == "continuous":
                action_summary[name].update({
                    "near_lower_bound_fraction": stat["low"] / n if n else None,
                    "near_upper_bound_fraction": stat["high"] / n if n else None,
                })
            else:
                action_summary[name].update({
                    "on_fraction": stat["active"] / n if n else None,
                    "switch_fraction_per_transition": stat["changes"] / m if m else None,
                })
        return {
            "steps": self.count,
            "mean_loss": sum(self.loss.values()) / self.count,
            "loss_components": {name: value / self.count
                                for name, value in self.loss.items()},
            "by_hour": self._finish_buckets(self.by_hour),
            "by_occupancy": self._finish_buckets(self.by_occupancy),
            "by_schedule": self._finish_buckets(self.by_schedule),
            "comfort_excursions": {
                "zone_samples": self.zones,
                "occupied_zone_samples": self.occupied_zones,
                "occupant_samples": self.occupant_samples,
                "max_excursion_k": self.max_excursion_k,
                "thresholds_k": {
                    key: {
                        "all_zone_fraction": v["all"] / self.zones if self.zones else None,
                        "occupied_zone_fraction": v["occupied_zones"] / self.occupied_zones
                        if self.occupied_zones else None,
                        "occupant_weighted_fraction": v["occupant_samples"]
                        / self.occupant_samples if self.occupant_samples else None,
                    } for key, v in self.violations.items()
                },
            },
            "action_utilization": action_summary,
        }
