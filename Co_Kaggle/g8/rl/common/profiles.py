"""V23 operating profiles and gateway-only observation features."""
from __future__ import annotations

import numpy as np

# Each profile changes only V23's soft planning parameters, not its physical
# action bounds or device topology. Profile zero reproduces original V23.
PARAMETERS = (
    "ACTIVE_THRESHOLD_K", "COMFORT_RESPONSE", "AIR_CAPACITY_FIRST",
    "AIR_CAPACITY_ADDITIONAL", "WATER_CAPACITY_FIRST",
    "WATER_CAPACITY_ADDITIONAL", "COIL_COST", "FAN_COST", "REHEAT_COST",
)
PROFILES = (
    ("baseline", (1., 1., 1., 1., 1., 1., 1., 1., 1.)),
    ("comfort", (.5, 1.5, 1.25, 1.25, 1.25, 1.25, .6, .6, .6)),
    ("economy", (2., .7, .85, .85, .85, .85, 1.8, 1.8, 1.8)),
    ("air_capacity", (.7, 1.2, 1.5, 1.5, 1., 1., .8, .6, 1.)),
    ("heating", (.7, 1.3, 1.1, 1.1, 1.6, 1.6, 1., 1., .45)),
    ("cooling", (.7, 1.3, 1.3, 1.3, 1., 1., .35, .7, 1.4)),
    ("gentle", (1., .5, 1., 1., 1., 1., 1., 1., 1.)),
)
N_ACTIONS = len(PROFILES)
N_FEATURES = 18 + N_ACTIONS


class ProfileController:
    """V23 remains responsible for every 5-minute HVAC action."""

    def __init__(self, baseline):
        self.baseline = baseline
        self.original = {name: getattr(baseline, name) for name in PARAMETERS}
        self.profile = 0

    def select(self, profile: int):
        if not isinstance(profile, (int, np.integer)) or not 0 <= profile < N_ACTIONS:
            raise ValueError("Invalid profile")
        for name, factor in zip(PARAMETERS, PROFILES[profile][1]):
            # Always start from immutable original values, never compound.
            setattr(self.baseline, name, self.original[name] * factor)
        self.profile = profile

    def features(self, observation):
        """Only gateway observations/metadata; no simulator reward internals."""
        b = self.baseline
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        fallback = b._schedule_fallback(obs)
        zones, outdoor = [], []
        for ahu in b.ahus:
            for zone in ahu["zones"]:
                state = b._read_zone(obs, zone, fallback)
                if state is not None:
                    zones.append(state)
            temperature = b._outdoor_temperature(obs, ahu["outdoor"])
            if np.isfinite(temperature):
                outdoor.append(temperature)
        heat = np.array([z["heating_error_k"] for z in zones] or [0.])
        cool = np.array([z["cooling_error_k"] for z in zones] or [0.])
        temp = np.array([z["temperature_k"] for z in zones] or [295.])
        clock = [b._value(obs, b.fields.get(name)) for name in
                 ("hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000")]
        weather = np.mean(outdoor) if outdoor else 295.
        values = [
            *clock, (weather - 295.) / 30., float(bool(outdoor)),
            (temp.mean() - 295.) / 15., temp.std() / 10.,
            heat.mean() / 5., heat.max() / 10., np.mean(heat > 1.),
            cool.mean() / 5., cool.max() / 10., np.mean(cool > 1.),
            len(zones) / max(1, b.zone_count),
            float(b.water_run is not None),
            (fallback[0] - 295.) / 15., (fallback[1] - 295.) / 15.,
            *np.eye(N_ACTIONS, dtype=np.float32)[self.profile],
        ]
        features = np.asarray(values, dtype=np.float32)
        if features.shape != (N_FEATURES,):
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        return np.clip(np.nan_to_num(features, nan=0., posinf=5., neginf=-5.), -5., 5.)

    def get_action(self, observation):
        return self.baseline.get_action(observation)
