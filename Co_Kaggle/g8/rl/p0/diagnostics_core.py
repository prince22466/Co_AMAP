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
SEVERITY_BANDS = ("0_to_1", "1_to_3", "3_to_5", "5_to_10", "over_10")


def action_groups(metadata: dict) -> dict[str, tuple[str, list[int]]]:
    """Classify actual simulator action fields in hybrid vector order."""
    groups = {
        name: (kind, [])
        for name, kind in (
            ("ahu_supply_temperature", "continuous"),
            ("ahu_static_pressure", "continuous"),
            ("ahu_recirculation", "continuous"),
            ("water_supply_temperature", "continuous"),
            ("water_pressure", "continuous"),
            ("vav_damper", "continuous"),
            ("vav_reheat", "continuous"),
            ("ahu_run", "discrete"),
            ("water_run", "discrete"),
        )
    }
    fields = {
        "supply_water_temperature_setpoint": "water_supply_temperature",
        "differential_pressure_setpoint": "water_pressure",
        "supervisor_run_command": "water_run",
        "supply_air_damper_percentage_command": "vav_damper",
        "reheat_valve_percentage_command": "vav_reheat",
    }
    ahu_fields = {
        "_supply_air_temperature_setpoint": "ahu_supply_temperature",
        "_supply_air_static_pressure_setpoint": "ahu_static_pressure",
        "_recirculation_percentage_command": "ahu_recirculation",
        "_supervisor_run_command": "ahu_run",
    }
    counts = {"continuous": 0, "discrete": 0}
    for device, field in metadata["diagnostic_action_fields"]:
        group = fields.get(field)
        if field.startswith("ahu_"):
            group = next((g for suffix, g in ahu_fields.items()
                          if field.endswith(suffix)), None)
        if group is None:
            raise ValueError(f"Unknown diagnostic action: {device}/{field}")
        kind, indices = groups[group]
        indices.append(counts[kind])
        counts[kind] += 1
    for kind, count in counts.items():
        if count != int(metadata["n_" + kind + "_action"]):
            raise ValueError(f"Incomplete {kind} action mapping")
    return groups


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

    def __init__(self, metadata: dict, deviation_ref: float = 5.0):
        if not math.isfinite(deviation_ref) or deviation_ref <= 0:
            raise ValueError("Invalid productivity deviation reference")
        self.deviation_ref = deviation_ref
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
        self.by_phase: dict[str, dict] = {}
        self.severity_loss = dict.fromkeys(SEVERITY_BANDS, 0.0)
        self.occupied_max_excursion = 0.0
        self.flow_samples = self.flow_near_design = 0
        self.discomfort_flow_samples = self.discomfort_flow_near_design = 0
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
                "total_loss": sum(value[k] for k in ("comfort", "energy", "carbon")),
                "mean_loss": sum(value[k] for k in ("comfort", "energy", "carbon"))
                / value["steps"],
                "mean_components": {
                    k: value[k] / value["steps"]
                    for k in ("comfort", "energy", "carbon")
                },
            }
            for key, value in sorted(buckets.items())
        }

    def _record_actions(self, action: dict) -> dict:
        row = {}
        for group, (kind, indices) in self.groups.items():
            arr = action[kind + "_action"]
            stat = self.groups_stats[group]
            values = [float(arr[index]) for index in indices]
            n = len(values)
            if kind == "continuous":
                if any(not math.isfinite(v) or abs(v) > 1.00001 for v in values):
                    raise ValueError(f"Invalid continuous command in {group}")
                row[group + "_mean_command"] = sum(values) / n if n else None
                row[group + "_near_low_fraction"] = sum(v <= -NEAR_BOUND for v in values) / n if n else None
                row[group + "_near_high_fraction"] = sum(v >= NEAR_BOUND for v in values) / n if n else None
            else:
                if any(v not in (0, 1) for v in values):
                    raise ValueError(f"Invalid discrete command in {group}")
                row[group + "_on_fraction"] = sum(values) / n if n else None
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
        return row

    def record(self, *, timestamp: Any, comfort_mode: bool, action: dict,
               reward_info: Any, response: Any, reward: float, weights: dict,
               comfort_soon: bool = False) -> dict:
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
        phase = ("occupied" if occupied else "unoccupied_comfort" if comfort_mode
                 else "preconditioning_window" if comfort_soon else "unoccupied_eco")
        self._update_bucket(self.by_phase, phase, parts, occupied)
        action_row = self._record_actions(action)

        zones = list(reward_info.zone_reward_infos.values())
        exceed = {k: 0 for k in THRESHOLDS_K}
        occupied_exceed = {k: 0 for k in THRESHOLDS_K}
        max_excursion = 0.0
        occupied_max_excursion = 0.0
        severity = dict.fromkeys(SEVERITY_BANDS, 0.0)
        flow_n = flow_high = discomfort_flow_n = discomfort_flow_high = 0
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
            if persons > 0:
                occupied_max_excursion = max(occupied_max_excursion, deviation)
                ratio = deviation / self.deviation_ref
                shortfall = ratio if ratio <= 1.0 else 1.0 + (ratio - 1.0) ** 2
                band = SEVERITY_BANDS[sum(deviation > k for k in THRESHOLDS_K)]
                severity[band] += persons * shortfall
                # RewardInfo reports delivered flow and design maximum.
                design = float(zone.air_flow_rate_setpoint)
                flow = float(zone.air_flow_rate)
                if math.isfinite(design) and design > 0 and math.isfinite(flow):
                    near_design = flow >= 0.95 * design
                    flow_n += 1
                    flow_high += int(near_design)
                    if deviation > 1.0:
                        discomfort_flow_n += 1
                        discomfort_flow_high += int(near_design)
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
        self.occupied_max_excursion = max(self.occupied_max_excursion, occupied_max_excursion)
        self.flow_samples += flow_n
        self.flow_near_design += flow_high
        self.discomfort_flow_samples += discomfort_flow_n
        self.discomfort_flow_near_design += discomfort_flow_high
        # Allocate the actual comfort loss by each band's share of the
        # occupancy-weighted shortfall, normalized separately at each step.
        total_shortfall = sum(severity.values())
        if total_shortfall == 0 and abs(parts["comfort"]) > 1e-8:
            raise ValueError("Comfort loss without occupied temperature shortfall")
        severity_row = {}
        for band, shortfall in severity.items():
            loss = parts["comfort"] * shortfall / total_shortfall if total_shortfall else 0.0
            self.severity_loss[band] += loss
            severity_row["comfort_loss_" + band + "k"] = loss
        row = {
            "timestamp": timestamp.isoformat(),
            "hour": hour, "comfort_mode": int(comfort_mode),
            "comfort_mode_soon": int(comfort_soon), "phase": phase,
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
            "max_occupied_zone_excursion_k": occupied_max_excursion if zone_occupied else None,
            "occupied_zones_with_design_flow": flow_n,
            "occupied_zones_near_design_flow": flow_high,
            "discomfort_zones_with_design_flow": discomfort_flow_n,
            "discomfort_zones_near_design_flow": discomfort_flow_high,
            **severity_row,
            **action_row,
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
            "by_phase": self._finish_buckets(self.by_phase),
            "comfort_loss_by_severity_k": {k: v / self.count for k, v in self.severity_loss.items()},
            "comfort_excursions": {
                "zone_samples": self.zones,
                "occupied_zone_samples": self.occupied_zones,
                "occupant_samples": self.occupant_samples,
                "max_excursion_k": self.max_excursion_k,
                "max_occupied_excursion_k": self.occupied_max_excursion if self.occupied_zones else None,
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
            "zone_flow_utilization": {
                "occupied_samples": self.flow_samples,
                "occupied_near_design_fraction": self.flow_near_design / self.flow_samples if self.flow_samples else None,
                "occupied_discomfort_gt_1k_samples": self.discomfort_flow_samples,
                "occupied_discomfort_gt_1k_near_design_fraction": self.discomfort_flow_near_design / self.discomfort_flow_samples if self.discomfort_flow_samples else None,
            },
        }
