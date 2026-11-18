"""Models a Variable Air Volume device for the simulation."""

import math
import uuid

import pandas as pd

from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.simulator import air_handler as air_handler_py

# simulator.constants is the canonical source of thermal properties: it is the
# only one that defines AIR_DENSITY, and air_handler.py, which imports this
# module, reads from it too. Both must agree for the air-side energy balance to
# close.
from smart_buildings.smart_control.simulator import constants, smart_device, thermostat
from smart_buildings.smart_control.simulator import (
  hot_water_system as hot_water_system_py,
)


class Vav(smart_device.SmartDevice):
  """Models a Variable Air Volume device with damper and reheat.

  Attributes:
    max_air_flow_rate: Air flow rate when damper is fully open.
    reheat_max_water_flow_factor: Water flow factor when valve is fully open
      (m^3/h).
    reheat_valve_setting: Proportion of water the valve is allowing through [0,
      1].
    damper_setting: Proportion of air the damper is allowing through [0, 1].
    damper_command: Damper position an agent has asked for, or None to leave the
      local thermostat loop in charge.
    reheat_valve_command: Reheat valve position an agent has asked for, or None
      to leave the local thermostat loop in charge.
    thermostat: Thermostat which controls VAV.
    hot_water_system: Hot water system supplying hot water to the VAV.
    flow_rate_demand: the flow rate demand
    reheat_flow_factor: the reheat demand
    zone_air_temperature: the average temperature in the zone
    max_air_flow_static_pressure: The minimum  pressure at which the design max
      air flow rate can be reached.
    min_damper_setting: Damper position the box never closes below while its air
      handler is running, so that an occupied zone keeps receiving outside air.
  """

  def __init__(
      self,
      max_air_flow_rate: float,
      reheat_max_water_flow_factor: float,
      therm: thermostat.Thermostat,
      hot_water_system: hot_water_system_py.HotWaterSystem,
      air_handler: air_handler_py.AirHandler,
      device_id: str | None = None,
      zone_id: str | None = None,
      # A VAV duct system reaches design airflow at roughly 1 to 2 kPa.
      # This is the denominator of sqrt(P / P_design) in the flow demand,
      # so it must move together with the static-pressure action range.
      max_air_flow_static_pressure: float = 2000.0,
      initial_zone_air_heating_temperature_setpoint: float = 294.0,
      initial_zone_air_cooling_temperature_setpoint: float = 297.0,
      # A VAV box serving an occupied zone has to keep delivering outside air,
      # so it is not free to shut. Without a floor here the cheapest thing a
      # controller can do to any zone it is not actively conditioning is close
      # it completely, which is both the optimum and a building-code violation.
      min_damper_setting: float = 0.1,
      # Highest air temperature the box may discharge into the zone, in K.
      # 308.15 K is 35 C, the usual design limit on a VAV reheat box.
      #
      # The limit matters most at low airflow. `_reheat_effectiveness` rises as
      # airflow falls, and at the damper floor it saturates at 1.0, which by the
      # coil model's definition means the air leaves at the water temperature.
      # Since the hot water loop may be commanded as high as 350 K, a box held
      # at its minimum damper position could otherwise deliver 77 C air to a
      # zone that is only slightly below its setpoint.
      #
      # Real boxes bound this with a discharge-air limit and a reheat /
      # minimum-airflow interlock. This implements the first. Set it above the
      # hot water range to disable.
      max_discharge_air_temperature: float = 308.15,
  ):
    observable_fields = {
        'supply_air_damper_percentage_command': smart_device.AttributeInfo(
            'damper_setting', float
        ),
        'reheat_valve_percentage_command': smart_device.AttributeInfo(
            'reheat_valve_setting', float
        ),
        'supply_air_flowrate_setpoint': smart_device.AttributeInfo(
            'max_air_flow_rate', float
        ),
        'zone_air_temperature_sensor': smart_device.AttributeInfo(
            'zone_air_temperature', float
        ),
        'zone_air_heating_temperature_setpoint_sensor': (
            smart_device.AttributeInfo(
                'zone_air_heating_temperature_setpoint', float
            )
        ),
        'zone_air_cooling_temperature_setpoint_sensor': (
            smart_device.AttributeInfo(
                'zone_air_cooling_temperature_setpoint', float
            )
        ),
    }
    # The actions are commands, not positions. They are held until they are
    # overwritten, and update_settings turns them into positions subject to the
    # ventilation floor; the identically named observables report the position
    # the box actually took. Writing straight to damper_setting here would
    # accomplish nothing, because update_settings runs after the actions have
    # been applied and used to overwrite them unconditionally.
    action_fields = {
        'supply_air_damper_percentage_command': smart_device.AttributeInfo(
            'damper_command', float
        ),
        'reheat_valve_percentage_command': smart_device.AttributeInfo(
            'reheat_valve_command', float
        ),
    }

    if device_id is None:
      device_id = f'vav_id_{uuid.uuid4()}'
    if zone_id is None:
      zone_id = f'zone_id_{uuid.uuid4()}'

    super().__init__(
        observable_fields,
        action_fields,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.VAV,
        device_id=device_id,
        zone_id=zone_id,
    )

    self._init_max_air_flow_rate = max_air_flow_rate
    self._reheat_max_water_flow_factor = reheat_max_water_flow_factor
    self._init_reheat_valve_setting = 0.0
    if min_damper_setting < 0.0 or min_damper_setting > 1.0:
      raise ValueError('min_damper_setting must be in [0, 1]')
    self._min_damper_setting = min_damper_setting
    self._max_discharge_air_temperature = max_discharge_air_temperature
    self._init_damper_setting = min_damper_setting
    self._init_thermostat = therm
    self._init_zone_air_temperature = 0
    self._zone_air_heating_temperature_setpoint = (
        initial_zone_air_heating_temperature_setpoint
    )
    self._zone_air_cooling_temperature_setpoint = (
        initial_zone_air_cooling_temperature_setpoint
    )
    self.reset()
    self._hot_water_system = hot_water_system
    self._air_handler = air_handler
    self._max_air_flow_static_pressure = max_air_flow_static_pressure

  def reset(self):
    self._max_air_flow_rate = self._init_max_air_flow_rate
    self._supply_air_flow_ratio = 1.0
    self._reheat_valve_setting = self._init_reheat_valve_setting
    self._damper_setting = self._init_damper_setting
    self._damper_command = None
    self._reheat_valve_command = None
    self._thermostat = self._init_thermostat
    self._zone_air_temperature = self._init_zone_air_temperature

  @property
  def min_damper_setting(self) -> float:
    return self._min_damper_setting

  @property
  def zone_air_heating_temperature_setpoint(self) -> float:
    return self._zone_air_heating_temperature_setpoint

  @property
  def zone_air_cooling_temperature_setpoint(self) -> float:
    return self._zone_air_cooling_temperature_setpoint

  @property
  def thermostat(self) -> thermostat.Thermostat:
    return self._thermostat

  @property
  def hot_water_system(self) -> hot_water_system_py.HotWaterSystem:
    return self._hot_water_system

  @property
  def reheat_valve_setting(self) -> float:
    return self._reheat_valve_setting

  @reheat_valve_setting.setter
  def reheat_valve_setting(self, value: float):
    if value < 0 or value > 1:
      raise ValueError('reheat_valve_setting must be in [0 ,1]')
    self._reheat_valve_setting = value

  @property
  def max_air_flow_rate(self) -> float:
    return self._max_air_flow_rate

  @max_air_flow_rate.setter
  def max_air_flow_rate(self, value: float):
    assert value > 0
    self._max_air_flow_rate = value

  @property
  def damper_setting(self) -> float:
    return self._damper_setting

  @damper_setting.setter
  def damper_setting(self, value: float):
    if value < 0 or value > 1:
      raise ValueError('damper_setting must be in [0 ,1]')
    self._damper_setting = value

  @property
  def damper_command(self) -> float | None:
    return self._damper_command

  @damper_command.setter
  def damper_command(self, value: float | None):
    # Clamped rather than rejected, unlike damper_setting. This is where agent
    # output enters, and BoundedActionNormalizer lets a normalized action sit a
    # tolerance outside its declared range, so a policy that saturates its own
    # action spec can land marginally outside [0, 1] through no fault of its
    # own. request_action swallows the resulting ValueError into a logged
    # rejection, which would silently leave the previous command in place.
    self._damper_command = None if value is None else min(max(value, 0.0), 1.0)

  @property
  def reheat_valve_command(self) -> float | None:
    return self._reheat_valve_command

  @reheat_valve_command.setter
  def reheat_valve_command(self, value: float | None):
    self._reheat_valve_command = (
        None if value is None else min(max(value, 0.0), 1.0)
    )

  @property
  def unrestricted_flow_rate_demand(self) -> float:
    """Returns the flow this VAV would draw if the handler were unlimited."""
    return self._compute_flow_rate_demand()

  @property
  def supply_air_flow_ratio(self) -> float:
    """Returns the share of demanded flow the air handler can actually move."""
    return self._supply_air_flow_ratio

  @supply_air_flow_ratio.setter
  def supply_air_flow_ratio(self, value: float) -> None:
    if value <= 0.0 or value > 1.0:
      raise ValueError('supply_air_flow_ratio must be in (0, 1]')
    self._supply_air_flow_ratio = value

  @property
  def flow_rate_demand(self) -> float:
    """Returns the flow the VAV actually receives.

    The air handler caps its total flow, so when the zones collectively demand
    more than it can move, every consumer of this value must see the reduced
    figure. Otherwise the fan is billed for the capped flow while the zones are
    heated for the uncapped flow, which creates energy from nothing in the
    agent's favour.
    """
    return self._compute_flow_rate_demand() * self._supply_air_flow_ratio

  def _compute_flow_rate_demand(self) -> float:
    """Returns the flow rate demand of the VAV.

      This assumes that the flow rate of each VAV is not impacted by the flow
    from other VAVs. This allows us to not consider the damper positions of the
    other VAVs, which simplifies things considerably.

    Thus, we can compute the flow rate demand as: first getting the ratio of
    static pressure to design static pressure. Then using the ratio to adjust
    the max air flow rate, so that when the static pressure is at the design
    value, the max air flow rate is reached. The sqrt is used because pressure
    and airflow have a quadratic relationship.
    """

    flow_rate_demand = (
        self._damper_setting
        * self._max_air_flow_rate
        * math.sqrt(
            self._air_handler.supply_air_static_pressure_sensor
            / self._max_air_flow_static_pressure
        )
    )

    # we will assume that there is always a minimal flow rate demand, even when
    # the damper is closed. This is to avoid dividing by zero.
    return max(flow_rate_demand, 0.00001)

  @property
  def air_handler(self) -> air_handler_py.AirHandler:
    return self._air_handler

  @property
  def reheat_flow_factor(self) -> float:
    return self._reheat_valve_setting * self._reheat_max_water_flow_factor

  @property
  def zone_air_temperature(self) -> float:
    return self._zone_air_temperature

  @zone_air_temperature.setter
  def zone_air_temperature(self, value: float) -> None:
    self._zone_air_temperature = value

  def _reheat_flow_rate(self) -> float:
    """Returns this VAV's share of the hot water flow."""
    if self._hot_water_system.flow_factor_sum == 0:
      return 0.0
    return self._hot_water_system.total_flow_rate * (
        self.reheat_flow_factor / self._hot_water_system.flow_factor_sum
    )

  def _reheat_effectiveness(
      self,
      supply_air_temp: float | None = None,
      input_water_temp: float | None = None,
  ) -> float:
    """Returns the share of the water-to-air temperature gap the coil closes.

    The supply temperature is `T_air + effectiveness * (T_water - T_air)`, so an
    effectiveness of 1 means the air leaves at the water temperature. A coil
    cannot do better than that, which is why this is capped at 1.

    Uncapped, the ratio grows without bound as airflow falls, because the water
    side of the balance does not shrink with it. The supply temperature then runs
    past the water temperature, which is a second-law violation, and the zone
    receives a fixed amount of heat while the fan bill falls to zero. That is
    reachable from the action space by commanding zero static pressure or by
    switching the air handler off.

    A second cap applies when both temperatures are supplied: the box may not
    discharge air hotter than `max_discharge_air_temperature`. Solving
    `T_air + e * (T_water - T_air) <= T_max` for `e` gives the limit below.
    Every caller passes both, so the three quantities that describe one coil --
    the discharge temperature, the heat into the zone, and the heat charged to
    the boiler -- are all derived from this same number and cannot disagree.
    Keeping them consistent is covered by the physics regression suite.

    Args:
      supply_air_temp: Temperature in K of the air entering the coil.
      input_water_temp: Temperature in K of the water entering the coil.
    """
    air_capacity_rate = (
        constants.AIR_HEAT_CAPACITY
        * constants.AIR_DENSITY
        * self.flow_rate_demand
    )
    if air_capacity_rate <= 0.0:
      return 0.0
    water_capacity_rate = (
        constants.WATER_HEAT_CAPACITY
        * constants.WATER_DENSITY
        * self._reheat_flow_rate()
    )
    effectiveness = min(1.0, water_capacity_rate / air_capacity_rate)

    if supply_air_temp is None or input_water_temp is None:
      return effectiveness

    # Only bites when the coil is actually heating. When the water is colder
    # than the air the term is negative and this is not a reheat coil doing
    # reheat, so the discharge limit has nothing to say about it.
    gap = input_water_temp - supply_air_temp
    if gap <= 0.0:
      return effectiveness
    headroom = self._max_discharge_air_temperature - supply_air_temp
    if headroom <= 0.0:
      # Air already at or above the limit. A reheat coil cannot cool it.
      return 0.0
    return min(effectiveness, headroom / gap)

  def compute_reheat_energy_rate(
      self, supply_air_temp: float, input_water_temp: float
  ) -> float:
    """Returns energy consumption in W due to heating the air.

    Below saturation this equals the previous `flow_water * c_water * dT`, since
    the effectiveness is then exactly `c_water * flow_water / (c_air * flow_air)`
    and the air capacity rate cancels. Above saturation the coil can only deliver
    what the air is able to carry away.

    Args:
      supply_air_temp: Temperature in K of input air.
      input_water_temp: Temperature in K of input water.
    """
    return (
        self._reheat_effectiveness(supply_air_temp, input_water_temp)
        * constants.AIR_HEAT_CAPACITY
        * constants.AIR_DENSITY
        * self.flow_rate_demand
        * (input_water_temp - supply_air_temp)
    )

  def compute_zone_supply_temp(
      self, supply_air_temp: float, input_water_temp: float
  ) -> float:
    """Returns temperature of air output from the VAV, supplied to the zone.

    Temperatures are measured in Kelvin.

    Args:
      supply_air_temp: Temperature in K of input air.
      input_water_temp: Temperature in K of input water.
    """
    assert self.damper_setting > 0
    assert self._max_air_flow_rate > 0

    return supply_air_temp + self._reheat_effectiveness(
        supply_air_temp, input_water_temp
    ) * (input_water_temp - supply_air_temp)

  def compute_energy_applied_to_zone(
      self, zone_temp: float, supply_air_temp: float, input_water_temp: float
  ) -> float:
    """Returns thermal energy in W to apply to the zone.

    Args:
      zone_temp: Current temperature in K of the zone.
      supply_air_temp: Temperature in K of input air.
      input_water_temp: Temperature in K of input water.
    """
    if self.damper_setting == 0 or self._max_air_flow_rate == 0:
      return 0
    air_flow_rate = self.flow_rate_demand
    # Two independent heat sources reach the zone, and only one of them is air:
    #
    #   1. the supply air stream. Volumetric flow [m3/s] must be converted to
    #      mass flow [kg/s] with AIR_DENSITY so the result is in W, matching
    #      air_handler.compute_thermal_energy_rate (which does include the
    #      density) and the W expected by building.apply_thermal_power_zone.
    #      Omitting it made the AHU charge for 1.2x what the zone received.
    #   2. the reheat coil, which is already an energy rate in W.
    #
    # They are summed separately rather than folded into zone_supply_temp,
    # because compute_zone_supply_temp divides the reheat energy by the
    # density-free group (AIR_HEAT_CAPACITY * air_flow_rate). Multiplying that
    # combined expression by AIR_DENSITY would scale the water-side heat by 1.2
    # as well as the air -- inventing energy the hot water loop is never
    # charged for.
    air_energy_rate = (
        air_flow_rate
        * constants.AIR_DENSITY
        * constants.AIR_HEAT_CAPACITY
        * (supply_air_temp - zone_temp)
    )
    reheat_energy_rate = self.compute_reheat_energy_rate(
        supply_air_temp, input_water_temp
    )
    return air_energy_rate + reheat_energy_rate

  def update_settings(
      self, zone_temp: float, current_timestamp: pd.Timestamp
  ) -> None:
    """Adjusts the VAV configuration from any command, else thermostat mode.

    A command wins over the local thermostat loop. The two are latched
    separately, so commanding only the damper still leaves the reheat valve on
    the thermostat. Whatever the source, the damper is floored at
    min_damper_setting: the box may be told to stop conditioning a zone, but not
    to stop ventilating it.

    Args:
      zone_temp: Current temperature in K of zone.
      current_timestamp: Pandas timestamp representing current time.
    """
    self._zone_air_temperature = zone_temp

    temperature_window = (
        self._thermostat.get_setpoint_schedule().get_temperature_window(
            current_timestamp
        )
    )
    self._zone_air_heating_temperature_setpoint = temperature_window[0]
    self._zone_air_cooling_temperature_setpoint = temperature_window[1]
    mode = self._thermostat.update(zone_temp, current_timestamp)
    if mode == thermostat.Thermostat.Mode.HEAT:
      damper_setting, reheat_valve_setting = 1.0, 1.0
    elif mode == thermostat.Thermostat.Mode.COOL:
      damper_setting, reheat_valve_setting = 1.0, 0.0
    elif mode == thermostat.Thermostat.Mode.OFF or mode == thermostat.Thermostat.Mode.PASSIVE_COOL:
      damper_setting = self._min_damper_setting  # Allow for ventilation
      reheat_valve_setting = 0.0
    else:  # Do nothing - keep existing configuration
      damper_setting = self._damper_setting
      reheat_valve_setting = self._reheat_valve_setting

    if self._damper_command is not None:
      damper_setting = self._damper_command
    if self._reheat_valve_command is not None:
      reheat_valve_setting = self._reheat_valve_command

    self.damper_setting = max(damper_setting, self._min_damper_setting)
    self.reheat_valve_setting = reheat_valve_setting

  def output(
      self, zone_temp: float, supply_air_temp: float
  ) -> tuple[float, float]:
    """Returns values corresponding to current output.

    Args:
      zone_temp: Current temperature in K of zone.
      supply_air_temp: Temperature in K of air being supplied to VAV.

    Returns:
      Tuple containing energy to apply to zone and temperature applied to zone.
    """
    self._zone_air_temperature = zone_temp
    q_zone = self.compute_energy_applied_to_zone(
        zone_temp,
        supply_air_temp,
        self.hot_water_system.supply_water_temperature_sensor,
    )
    temp_vav_supply = self.compute_zone_supply_temp(
        supply_air_temp, self.hot_water_system.supply_water_temperature_sensor
    )
    return q_zone, temp_vav_supply

  def update(
      self,
      zone_temp: float,
      current_timestamp: pd.Timestamp,
      supply_air_temp: float,
  ) -> tuple[float, float]:
    """Returns values corresponding to current output.

    Adjusts the VAV configuration based on thermostat mode.

    Args:
      zone_temp: Current temperature in K of zone.
      current_timestamp: Pandas timestamp representing current time.
      supply_air_temp: Temperature in K of air being supplied to VAV.

    Returns:
      Tuple containing energy to apply to zone, temperature applied to zone, and
      flow rate demand.
    """
    self.update_settings(zone_temp, current_timestamp)
    return self.output(zone_temp, supply_air_temp)
