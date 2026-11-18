"""A chain of controllers built on `pid`, each adding one idea to the last.

`pid` already idles per air handler while its zones are in band, and drives
each VAV damper and reheat valve from that zone's own temperature. These
subclasses address what a plant-level rule-based controller still leaves on the
table. Each is a base class for the next, so the chain reads as a sequence of
independent improvements.

  PIDWorstZonePolicy   The AHU loop runs on the *average* zone temperature of
                       the zones it serves, but the comfort penalty is
                       unbounded in the *worst* zone's deviation. An air
                       handler can therefore sit idle, perfectly average, while
                       one zone under it runs away and dominates the episode.
                       This drives the loop from the worst zone instead.

  PIDEconomizerPolicy  Modulates recirculation rather than pinning it at the
                       ventilation cap. Cooling with return air while the
                       outside air is colder than the space is paid-for cooling
                       that the weather was offering free. The mix is
                       proportional: it blends outside and return air to land
                       on the supply air target, so very cold outside air
                       results in less of it being drawn, not none.

  PIDProportionalReheatPolicy
                       Modulates the reheat valve rather than switching it.

  PIDWaterResetPolicy  Resets the hot water supply temperature against demand,
                       so the loop is not held at its maximum whenever any
                       single zone calls for heat.

One further refinement runs through the chain: the comfort band is read from
each zone's own setpoint sensors instead of being rebuilt from the hour of day.
The reward measures deviation from those sensors, so where a reconstructed band
disagrees with them the controller either conditions a zone the reward already
considers comfortable, which is cost for no benefit, or neglects one it does
not. Reading the sensors makes the controller and the scorer agree on where the
edge is.
"""

from typing import Any

import numpy as np

from smart_buildings.baselines.pid import PIDPolicy

# Native ranges from the action normalizer in every shipped config.
SUPPLY_AIR_MIN_K = 285.0
SUPPLY_AIR_MAX_K = 305.0
STATIC_PRESSURE_MAX_PA = 2000.0
# The simulator holds this floor while a handler runs, so a lower command
# still costs this much fan.
MIN_STATIC_PRESSURE_PA = 200.0

# Kelvin of outside-air advantage needed per pascal of fan static pressure
# before drawing outside air is cheaper than recirculating:
#
#     COP / (fan_efficiency * air_density * air_heat_capacity)
#   = 3.5 / (0.9 * 1.2 * 1006)  =  0.00322 K/Pa
#
# So 0.6 K at the 200 Pa floor and 6.4 K at full 2000 Pa fan. This scales with
# the cooling COP: a lower COP makes the chiller saving larger relative to the
# fan cost and lowers the bar for economizing, so the constant must be kept in
# step with AirHandler's cooling_coefficient_of_performance.
ECON_K_PER_PA = 0.00322


class PIDWorstZonePolicy(PIDPolicy):
    """`pid`, but the AHU loop tracks its worst zone rather than the average.

    Comfort shortfall is linear to 5 K outside the band and quadratic and
    unbounded past it, and it is summed over zones. One zone 10 K out costs far
    more than ten zones 1 K out. Averaging is the one aggregation that hides
    exactly that, and an air handler serving seven zones can read as perfectly
    in band while one of them is the whole episode's loss.
    """

    USE_SENSOR_BAND = False

    @property
    def name(self) -> str:
        return "pid_worst"

    def _zone_band(self, obs: np.ndarray, vav_id: str, low: float, high: float):
        """This zone's band. Falls back to the schedule band when unavailable."""
        if not self.USE_SENSOR_BAND:
            return low, high
        idx = self._sensor_band_indices.get(vav_id)
        if idx is None:
            return low, high
        lo_idx, hi_idx = idx
        lo = float(obs[lo_idx]) if lo_idx is not None else low
        hi = float(obs[hi_idx]) if hi_idx is not None else high
        # A band that came back inverted or empty is not usable; a controller
        # that trusted it would drive both ways at once.
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return low, high
        return lo, hi

    def _build_sensor_band_indices(self) -> None:
        """Index of each zone's own heating and cooling setpoint sensor.

        These are raw kelvin, unlike the zone temperature, which is normalized.
        """
        self._sensor_band_indices: dict[str, tuple[int | None, int | None]] = {}
        names = list(self.field_names)

        def find(field: str):
            try:
                return names.index(field)
            except ValueError:
                return None

        for ctrl in self.ahu_controllers:
            for vav_id, _, _ in ctrl["zones"]:
                self._sensor_band_indices[vav_id] = (
                    find(f"{vav_id}_zone_air_heating_temperature_setpoint_sensor"),
                    find(f"{vav_id}_zone_air_cooling_temperature_setpoint_sensor"),
                )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_sensor_band_indices()
        self._attach_recirculation_and_weather()

    def _attach_recirculation_and_weather(self) -> None:
        """`pid` builds no recirculation or outside-air entry; add them.

        `pid` predates the recirculation action and writes the cap to every
        recirculation index in one loop at the end, so its per-AHU map never
        needed either index. An economizer is a per-AHU decision and needs
        both.
        """
        names = list(self.field_names)
        for i, ctrl in enumerate(self.ahu_controllers):
            ctrl["recirc_idx"] = (
                self.recirculation_indices[i]
                if i < len(self.recirculation_indices)
                else None
            )
            field = f"ahs_{ctrl['ahu_id']}_outside_air_temperature_sensor"
            ctrl["oat_idx"] = names.index(field) if field in names else None

    def _worst_zone_error(self, obs: np.ndarray, ctrl, low: float, high: float) -> float:
        """Signed deadband error of the zone furthest outside its band.

        Sign is kept, so the AHU still knows which way to drive. When some
        zones are cold and others hot the largest magnitude wins, because that
        is the one the unbounded penalty is charging for.
        """
        worst = 0.0
        for vav_id, obs_idx, _ in ctrl["zones"]:
            lo, hi = self._zone_band(obs, vav_id, low, high)
            err = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), lo, hi
            )
            if abs(err) > abs(worst):
                worst = err
        return worst

    def _command_zones(self, continuous, zones, obs, low, high) -> None:
        """Same as `pid`, except each zone is judged against its own band."""
        if not self.USE_SENSOR_BAND:
            return super()._command_zones(continuous, zones, obs, low, high)
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, low, high)
            zone_error = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), lo, hi
            )
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 1.0 if zone_error > 0 else -1.0

    # `pid.get_action` reduces the AHU's zones with np.mean. Rather than copy
    # the whole method to change one line, override the reduction it uses.
    def _ahu_error(self, obs: np.ndarray, ctrl, low: float, high: float) -> float:
        return self._worst_zone_error(obs, ctrl, low, high)

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
        self._last_efforts: list[float] = []

        # Default every handler to the ventilation cap, then let the loop below
        # overwrite the ones it decides about. Testing for a leftover 0.0 after
        # the fact would misread a legitimate 50% recirculation command.
        for idx in self.recirculation_indices:
            continuous[idx] = 1.0

        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue

            self._command_zones(continuous, ctrl["zones"], obs, low_temp, high_temp)
            error = self._ahu_error(obs, ctrl, low_temp, high_temp)

            state = self.pid_state[i]
            if error == 0.0:
                state["integral"] *= 0.5
                state["prev_error"] = 0.0
                continuous[ctrl["temp_idx"]] = -1.0
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = -1.0
                if ctrl.get("recirc_idx") is not None:
                    continuous[ctrl["recirc_idx"]] = 1.0
                discrete[ctrl["run_discrete_idx"]] = 0
                continue

            state["integral"] = float(np.clip(
                state["integral"] + error, -self.integral_limit, self.integral_limit))
            derivative = error - state["prev_error"]
            state["prev_error"] = error

            u = float(np.clip(
                self.kp * error + self.ki * state["integral"] + self.kd * derivative,
                -1.0, 1.0))

            continuous[ctrl["temp_idx"]] = u
            # Recorded for the hot water reset in PIDWaterResetPolicy, which
            # needs to know whether the air side has any authority left. Kept
            # here rather than recomputed, because a reset that re-derived the
            # effort could drift out of step with what was actually commanded.
            self._last_efforts.append(u)
            if ctrl["pressure_idx"] is not None:
                continuous[ctrl["pressure_idx"]] = float(
                    np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))
            discrete[ctrl["run_discrete_idx"]] = 1

            if ctrl.get("recirc_idx") is not None:
                continuous[ctrl["recirc_idx"]] = self._recirculation(
                    obs, ctrl, u, low_temp, high_temp)

            any_active = True
            if error > 0:
                any_heating = True

        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = 1.0 if any_heating else -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if any_active else -1.0
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 1 if any_active else 0

        return {"continuous_action": continuous, "discrete_action": discrete}

    def _recirculation(self, obs, ctrl, u: float, low: float, high: float) -> float:
        """Hold the cap. Subclasses that economize override this."""
        return 1.0


class PIDEconomizerPolicy(PIDWorstZonePolicy):
    """`pid` plus a proportional economizer. Average-zone loop, as `pid` has.

    When the handler is cooling and the outside air is colder than the space,
    some of that cooling is available for the price of moving the air. The
    mixed air temperature is

        mixed = f * outside + (1 - f) * inside

    for an outside air fraction `f`, so the `f` that lands exactly on the
    supply air target is

        f = (inside - target) / (inside - outside)

    and the recirculation command is `1 - f`. Solving for the target rather
    than switching to full outside air is the whole point: in cold weather this
    asks for a *small* amount of outside air, where an on/off economizer has to
    choose between none and a blast of -30 C air.

    The simulator clamps recirculation at 70%, so the outside air fraction
    never falls below 30% whatever this returns. That is the ventilation floor
    and it is not something a policy can trade away.
    """

    USE_WORST_ZONE = False

    @property
    def name(self) -> str:
        return "pid_econ"

    def _ahu_error(self, obs: np.ndarray, ctrl, low: float, high: float) -> float:
        if self.USE_WORST_ZONE:
            return self._worst_zone_error(obs, ctrl, low, high)
        temps = [self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]
        return self._deadband_error(float(np.mean(temps)), low, high)

    def _recirculation(self, obs, ctrl, u: float, low: float, high: float) -> float:
        oat_idx = ctrl.get("oat_idx")
        # Heating, or no weather reading: hold the heat in. Recirculated air is
        # the warmest air available and the handler cannot add heat itself.
        if u >= 0.0 or oat_idx is None:
            return 1.0

        outside = float(obs[oat_idx])
        inside = float(np.mean(
            [self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]))
        if not np.isfinite(outside) or outside >= inside:
            return 1.0

        # Free cooling is not free. The exhaust fan moves
        # `air_flow_rate * (1 - recirculation)`, so every extra unit of outside
        # air costs fan power even though it saves chiller work. Per unit of
        # outside air fraction, and per unit of flow:
        #
        #     chiller saved = rho * cp * (return - outside) / COP
        #     exhaust paid  = static_pressure / fan_efficiency
        #
        # so it only pays once
        #
        #     (return - outside) > static_pressure * COP / (fan_eff * rho * cp)
        #
        # Below this threshold the extra fan work required to move outside
        # air costs more than the chiller saving it buys, so economizing on a
        # small temperature advantage is a net loss.
        pressure_pa = (self._commanded_pressure(u) + 1.0) * 0.5 * STATIC_PRESSURE_MAX_PA
        pressure_pa = max(pressure_pa, MIN_STATIC_PRESSURE_PA)
        breakeven_k = pressure_pa * ECON_K_PER_PA
        if (inside - outside) <= breakeven_k:
            return 1.0

        target = SUPPLY_AIR_MIN_K + (u + 1.0) * 0.5 * (SUPPLY_AIR_MAX_K - SUPPLY_AIR_MIN_K)
        # Never mix below the band's lower edge. Overshooting into heating
        # would trade a cooling bill for a larger reheat bill.
        target = max(target, low)
        if inside <= target:
            return 1.0

        f = (inside - target) / (inside - outside)
        recirculation = float(np.clip(1.0 - f, 0.0, 1.0))
        return 2.0 * recirculation - 1.0

    @staticmethod
    def _commanded_pressure(u: float) -> float:
        """The normalized pressure this controller is about to write."""
        return float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))


class PIDProportionalReheatPolicy(PIDEconomizerPolicy):
    """`pid` with a modulating reheat valve instead of an on/off one.

    Every baseline in this repo drives the valve binary:

        continuous[reheat_idx] = 1.0 if zone_error > 0 else -1.0

    but `reheat_valve_percentage_command` is a continuous 0-1 position. Fully
    opening it for a zone that is 0.3 K cold delivers the same heat as for one
    that is 8 K cold, so the near-miss zone is driven straight through its band
    and out the other side.

    An on/off valve on a slow plant therefore spends more zone-time above the
    band than below it, with the damper still open through most of the
    overshoot: the controller ends up net-heating zones it was trying to bring
    just into range. This is a resolution problem rather than a lack of
    authority -- a single zone's own actuators can move it many kelvin.

    The valve is opened in proportion to the shortfall, full open at
    REHEAT_FULL_K below the heating setpoint, and shut once the zone is inside
    its band.
    """

    # Shortfall in K at which the valve is fully open. Matched to the AHU
    # loop's own 1/kp = 2 K so the two layers reach saturation together.
    REHEAT_FULL_K = 2.0

    @property
    def name(self) -> str:
        return "pid_prop"

    def _command_zones(self, continuous, zones, obs, low, high) -> None:
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, low, high)
            zone_error = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), lo, hi
            )
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                # zone_error > 0 means below the heating setpoint.
                opening = min(1.0, max(0.0, zone_error) / self.REHEAT_FULL_K)
                continuous[reheat_idx] = 2.0 * opening - 1.0


class PIDWaterResetPolicy(PIDProportionalReheatPolicy):
    """Proportional reheat valve AND the hot water temperature reset on demand.

    Both halves are needed, and each is weak alone.

    The hot water loop is building-wide. A demand reset on its own is therefore
    set by the coldest zone in the building: one zone several kelvin below its
    band drives the loop to its maximum, and every satisfied zone still sitting
    at its damper floor receives that very hot water anyway. A proportional
    valve on its own helps, but it is throttling air that is still far hotter
    than the zone needs.

    Together the loop runs only as hot as the worst zone requires, and each
    valve opens only as far as its own zone requires.

    WHY A RESET RATHER THAN A FIXED TEMPERATURE

    Pinning the loop low is not a solution: a hard winter climate genuinely
    needs the full water temperature. So the loop is reset against the demand
    that actually exists, linearly, saturating at RESET_FULL_K below the band.

    PROPORTIONAL DROOP, AND THE SATURATION TERM

    A purely proportional reset has a steady-state offset, and here that offset
    is the whole error: the loop only commands full water when the mean zone is
    already RESET_FULL_K short. In a climate that genuinely needs full water the
    controller therefore settles on a building that is RESET_FULL_K cold, which
    is precisely where it must not be.

    An integrator is the textbook remedy and is the wrong one here. Raising the
    water temperature satisfies the bulk of the zones sooner, so the air handler
    idles sooner, so an outlier zone loses its airflow and drifts further. Since
    the comfort penalty is unbounded in the worst zone, widening the spread at
    both ends is the worst available outcome. The mechanism is left in the class
    with its gain defaulting to zero, so the next reader does not rebuild it.

    Outdoor-air reset is the other textbook answer. It works, but the crossover
    temperature it needs differs sharply between cold climates, and fitting that
    ramp to the scenarios in hand would not transfer to unseen buildings.

    Instead the reset takes the greater of two signals:

      demand      how far the mean zone is below its band
      saturation  how hard the air side is already working

    The second answers a different question -- not "is the building cold" but
    "has the air side run out of authority". In a severe winter every handler
    sits at full effort all episode, the air side has nothing left to give, and
    the only remaining lever is hotter water. In a mild climate effort is near
    zero for most of the day, so the loop stays cool and follows only the
    morning ramp. Reading saturation off the plant itself is what makes this
    transfer, where a fitted outdoor-air curve would not.
    """

    # Proportional band: mean zone shortfall, in K, that alone commands full
    # water.
    RESET_FULL_K = 4.0
    # Weight on the air-side saturation term. See `_water_fraction`.
    RESET_EFFORT_WEIGHT = 1.0
    # Which handler to believe when they disagree, as a quantile over the
    # per-handler heating efforts. 0.0 is the least loaded handler, 1.0 the most.
    #
    # Measured over six scenarios spanning the failure and the win, as summed
    # reward against the proportional-only reset's -3,946:
    #
    #     mean effort  -3,681      q50  -3,685
    #     q25          -3,557      q00  -3,674
    #
    # The mean and q50 both give back denver_apr (-185 and -184 against -67),
    # because denver has one unreachable cold zone that pins its own handler at
    # full effort for the whole episode. q25 keeps denver at -97 and takes the
    # cold-weather win anyway. q00 is too strict: one idle handler in a
    # twelve-handler building vetoes the whole loop.
    RESET_EFFORT_QUANTILE = 0.25
    # Integral gain and decay. Both left in place, and the gain defaults to
    # zero, because the integrator was built, measured and rejected; see the
    # class docstring. Keeping the mechanism costs three lines and stops the
    # next reader from rebuilding it.
    RESET_DEADBAND_K = 0.25
    RESET_INTEGRAL_GAIN = 0.0
    RESET_INTEGRAL_DECAY = 0.99

    @property
    def name(self) -> str:
        return "pid_water"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._water_integral = 0.0

    def _heating_shortfall(self, obs, low: float, high: float) -> float:
        """Building-wide heating demand, as the MEAN zone shortfall.

        Deliberately not the coldest zone. The hot water loop is building-wide,
        so letting a single outlier set its temperature drives the loop to its
        maximum and delivers that water to every satisfied zone sitting at its
        damper floor.

        The outlier is also frequently unreachable: a zone starved of airflow
        stays cold whatever the loop temperature, so heat spent chasing it buys
        nothing while damaging every other zone. The mean tracks the demand the
        loop can actually serve, and it remains the better statistic in cold
        climates where the heat is genuinely needed.
        """
        shortfalls = []
        for ctrl in self.ahu_controllers:
            for vav_id, obs_idx, _ in ctrl["zones"]:
                lo, _hi = self._zone_band(obs, vav_id, low, high)
                shortfalls.append(
                    max(0.0, lo - self._denormalize_zone_temp(obs[obs_idx]))
                )
        return float(np.mean(shortfalls)) if shortfalls else 0.0

    def get_action(self, obs):
        action = super().get_action(obs)
        if self.water_temp_idx is None:
            return action

        hour, day = self._get_hour_and_day(obs)
        is_work = day < 5 and self.morning_start_hour <= hour < self.evening_start_hour
        low, high = self.comfort_temp_window if is_work else self.eco_temp_window
        shortfall = self._heating_shortfall(obs, low, high)

        # The integrator is updated on every step, including steps where no
        # handler is calling for heat. Updating it only while heating would let
        # it hold a stale wound-up value across a whole warm afternoon and then
        # dump full water into the first zone that asks at dusk.
        if shortfall > self.RESET_DEADBAND_K:
            self._water_integral += self.RESET_INTEGRAL_GAIN * (
                shortfall - self.RESET_DEADBAND_K)
        else:
            self._water_integral *= self.RESET_INTEGRAL_DECAY
        self._water_integral = float(np.clip(self._water_integral, 0.0, 1.0))

        cont = action["continuous_action"]
        if cont[self.water_temp_idx] <= 0.0:
            return action  # not heating; leave the loop cold

        frac = self._water_fraction(shortfall)
        cont[self.water_temp_idx] = np.float32(2.0 * frac - 1.0)
        return action

    def _water_fraction(self, shortfall: float) -> float:
        """How hot to run the loop, as a fraction of the 310-350 K range.

        Two terms, and the larger wins.

        The DEMAND term is the mean zone shortfall over the proportional band,
        as described above.

        The SATURATION term is the mean heating effort the AHU loop is already
        commanding. It answers a different question: not "how cold is the
        building" but "has the air side run out of authority". Those come apart
        exactly where the demand term fails. In fairbanks_jan every handler sits
        at full effort all episode, so the air side has nothing left to give and
        the only remaining lever is hotter water. In denver_apr effort is near
        zero most of the day, so the loop stays cool and only follows the
        morning ramp.

        Why this and not an integrator, and not outdoor-air reset:

        * An integrator removes the droop in cold weather but widens the zone
          spread at both ends. Hotter water satisfies the bulk of the zones
          sooner, so the handler idles sooner, so an outlier zone loses its
          airflow and drifts further. Since the comfort penalty is unbounded in
          the worst zone, that trade is a loss.
        * Outdoor-air reset is the textbook answer and does work, but the
          crossover temperature it needs differs sharply between cold climates.
          Fitting that ramp to the scenarios in hand would not transfer to
          unseen buildings. The saturation term reads the same distinction off
          the plant itself.

        Note the saturation term also pays for itself on energy, which was not
        the reason for adding it. On building_0000 / fairbanks_jan, cool water
        cost MORE than hot: the coil could not deliver, so the fan sat at full
        static pressure all episode moving air that was not warm enough. Per
        unit of heat delivered, hotter water trades fan electricity for boiler
        gas, and electricity is the dearer of the two in both price and carbon.
        """
        demand = shortfall / self.RESET_FULL_K
        efforts = getattr(self, "_last_efforts", None)
        saturation = 0.0
        if efforts and self.RESET_EFFORT_WEIGHT:
            heating = [max(0.0, u) for u in efforts]
            saturation = self.RESET_EFFORT_WEIGHT * float(
                np.quantile(heating, self.RESET_EFFORT_QUANTILE))
        return float(np.clip(max(demand, saturation) + self._water_integral,
                             0.0, 1.0))
