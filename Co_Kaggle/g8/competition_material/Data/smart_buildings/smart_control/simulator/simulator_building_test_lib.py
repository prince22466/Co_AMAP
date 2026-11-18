"""Provides a base class for testing variants of Simulator Building."""

import pandas as pd
from absl.testing import parameterized

from smart_buildings.smart_control.proto import smart_control_building_pb2
from smart_buildings.smart_control.simulator import air_handler as air_handler_py
from smart_buildings.smart_control.simulator import building as building_py
from smart_buildings.smart_control.simulator import (
  hot_water_system as hot_water_system_py,
)
from smart_buildings.smart_control.simulator import hvac as hvac_py
from smart_buildings.smart_control.simulator import (
  setpoint_schedule,
  step_function_occupancy,
)
from smart_buildings.smart_control.simulator import simulator as simulator_py
from smart_buildings.smart_control.simulator import simulator_building as sb_py
from smart_buildings.smart_control.simulator import (
  weather_controller as weather_controller_py,
)

_ACTION_RESPONSE_TYPE = (
    smart_control_building_pb2.SingleActionResponse.ActionResponseType
)


class SimulatorBuildingTestBase(parameterized.TestCase):
  """Base class for testing variants of Simulator Building."""

  occupancy = step_function_occupancy.StepFunctionOccupancy(
      work_start_time=pd.Timedelta(9, unit='h'),
      work_end_time=pd.Timedelta(17, unit='h'),
      work_occupancy=10,
      nonwork_occupancy=0.1,
  )

  def setUp(self):
    super().setUp()
    self.building = self.get_sim_building()

  def _create_small_building(self, initial_temp):
    """Returns building with specified initial temperature.

    The building returned will have a matrix size of: 21 x 10, this should be
    left as a comment in tests when relevant. Additionally initial_temp is
    added as a parameter for clarity in the tests.

    Args:
      initial_temp: Initial temperature of all CVs in building.
    """
    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (8, 6)
    building_shape = (2, 1)
    # Volumetric heat capacity (density x heat_capacity) is what the solver
    # actually sees, and for air it has to be large enough that a control volume
    # cannot be flung across the setpoint band by one timestep of VAV output:
    # the HVAC reads last step's temperature, so an air CV that is too light
    # oscillates divergently no matter how exact the conduction solve is. These
    # are the values the shipped scenarios use.
    inside_air_properties = building_py.MaterialProperties(
        conductivity=50.0, heat_capacity=70000.0, density=0.1
    )
    inside_wall_properties = building_py.MaterialProperties(
        conductivity=2.0, heat_capacity=500.0, density=1800.0
    )
    # A near-zero exterior conductivity seals the building off from the weather
    # entirely, which leaves the HVAC as the only forcing and lets the zones
    # wander off to the supply air temperature and stay there.
    building_exterior_properties = building_py.MaterialProperties(
        conductivity=5.5, heat_capacity=500.0, density=3000.0
    )

    building = building_py.Building(
        cv_size_cm,
        floor_height_cm,
        room_shape,
        building_shape,
        initial_temp,
        inside_air_properties,
        inside_wall_properties,
        building_exterior_properties,
    )
    return building

  def _create_small_hvac(self):
    """Returns hvac matching zones for small test building."""
    supply_water_temperature_setpoint = 360
    water_pump_differential_head = 3
    water_pump_efficiency = 0.6
    hot_water_system = hot_water_system_py.construct_hot_water_system(
        supply_water_temperature_setpoint=supply_water_temperature_setpoint,
        water_pump_differential_head=water_pump_differential_head,
        water_pump_efficiency=water_pump_efficiency,
        device_id='hws_id',
    )

    # AirHandler used to take separate heating and cooling setpoints; these are
    # keyword arguments so that they cannot silently slide along again.
    air_handler = air_handler_py.AirHandler(
        recirculation=0.3,
        supply_air_temperature_setpoint=288,
        fan_static_pressure=1000.0,
        fan_efficiency=0.8,
        max_air_flow_rate=0.8,
        device_id='air_handler_id',
    )

    morning_start_hour = 9
    evening_start_hour = 18
    comfort_temp_window = (292, 295)
    eco_temp_window = (290, 297)
    holidays = set([7, 223, 245])

    schedule = setpoint_schedule.SetpointSchedule(
        morning_start_hour,
        evening_start_hour,
        comfort_temp_window,
        eco_temp_window,
        holidays,
    )

    zone_coordinates = [(0, 0), (1, 0)]

    # The shipped scenarios give a VAV 2.0 kg/s to serve roughly 33 control
    # volumes; each zone here is only four, so the flow scales down to match.
    hvac = hvac_py.Hvac(
        zone_coordinates,
        air_handler,
        hot_water_system,
        schedule,
        vav_max_air_flow_rate=0.25,
        vav_reheat_max_water_flow_rate=0.015,
    )
    return hvac

  def _create_small_simulator(self):
    """Creats a small simulator for test."""
    # Set up simulation parameters
    weather_controller = weather_controller_py.WeatherController(296.0, 296.0)
    time_step_sec = 300.0
    hvac = self._create_small_hvac()
    start_timestamp = pd.Timestamp('2012-12-21')

    # Control Volume matrix shape: (21, 10)
    building = self._create_small_building(initial_temp=292.0)

    return simulator_py.Simulator(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        start_timestamp,
    )

  def get_sim_building(
      self,
      initial_rejection_count: int = 0,
      zones=None,
      simulator=None,
  ) -> sb_py.SimulatorBuilding:
    raise NotImplementedError()  # pragma: nocover

  def test_devices(self):
    simulator_building = self.building

    devices = simulator_building.devices

    self.assertLen(devices, 4)

  # OBSERVATIONS

  @parameterized.named_parameters(
      (
          'obs_supply_water_temperature_setpoint',
          'supply_water_temperature_setpoint',
          360,
      ),
      (
          'obs_supply_water_temperature_sensor',
          'supply_water_temperature_sensor',
          360,
      ),
      ('obs_heating_request_count', 'heating_request_count', 0),
  )
  def test_request_observation_single_success(
      self, measurement_name, expected_value
  ):
    """Tests request observations."""
    simulator_building = self.building

    observation_request = smart_control_building_pb2.ObservationRequest()
    single_field_request = smart_control_building_pb2.SingleObservationRequest(
        device_id='hws_id', measurement_name=measurement_name
    )

    observation_request.single_observation_requests.append(single_field_request)

    observation_response = simulator_building.request_observations(
        observation_request
    )

    self.assertEqual(observation_response.request, observation_request)
    self.assertEqual(
        observation_response.single_observation_responses[
            0
        ].single_observation_request,
        single_field_request,
    )
    self.assertEqual(
        observation_response.single_observation_responses[0].continuous_value,
        expected_value,
    )
    self.assertTrue(
        observation_response.single_observation_responses[0].observation_valid
    )

  def test_request_observation_multiple_success(self):
    """Tests request multiple observations."""
    simulator_building = self.building

    observation_request = smart_control_building_pb2.ObservationRequest()

    single_field_request_1 = (
        smart_control_building_pb2.SingleObservationRequest(
            device_id='hws_id',
            measurement_name='supply_water_temperature_setpoint',
        )
    )
    observation_request.single_observation_requests.append(
        single_field_request_1
    )

    single_field_request_2 = (
        smart_control_building_pb2.SingleObservationRequest(
            device_id='hws_id', measurement_name='heating_request_count'
        )
    )
    observation_request.single_observation_requests.append(
        single_field_request_2
    )

    observation_response = simulator_building.request_observations(
        observation_request
    )

    self.assertEqual(observation_response.request, observation_request)

    self.assertEqual(
        observation_response.single_observation_responses[
            0
        ].single_observation_request,
        single_field_request_1,
    )
    self.assertEqual(
        observation_response.single_observation_responses[0].continuous_value,
        360,
    )

    self.assertEqual(
        observation_response.single_observation_responses[
            1
        ].single_observation_request,
        single_field_request_2,
    )
    self.assertEqual(
        observation_response.single_observation_responses[1].continuous_value, 0
    )

  def test_request_observation_incorrect_device(self):
    """Tests when an observation is requested on a nonexistent device."""
    simulator_building = self.building

    observation_request = smart_control_building_pb2.ObservationRequest()
    single_field_request = smart_control_building_pb2.SingleObservationRequest(
        device_id='wrong_device', measurement_name='measurement_name'
    )

    observation_request.single_observation_requests.append(single_field_request)

    observation_response = simulator_building.request_observations(
        observation_request
    )

    self.assertFalse(
        observation_response.single_observation_responses[0].observation_valid
    )

  def test_request_observation_incorrect_measurement(self):
    """Tests when an observation is requested for a nonexistnt measurement."""
    simulator_building = self.building

    observation_request = smart_control_building_pb2.ObservationRequest()
    single_field_request = smart_control_building_pb2.SingleObservationRequest(
        device_id='hws_id', measurement_name='incorrect_measurement'
    )

    observation_request.single_observation_requests.append(single_field_request)

    observation_response = simulator_building.request_observations(
        observation_request
    )

    self.assertFalse(
        observation_response.single_observation_responses[0].observation_valid
    )

  # ACTIONS

  @parameterized.named_parameters(
      ('act_supply_water_setpoint', 'supply_water_temperature_setpoint', 360),
  )
  def test_request_action_single_success(self, setpoint_name, set_value):
    """Tests request single action with success."""
    simulator_building = self.building

    action_request = smart_control_building_pb2.ActionRequest()
    single_field_request = smart_control_building_pb2.SingleActionRequest(
        device_id='hws_id',
        setpoint_name=setpoint_name,
        continuous_value=set_value,
    )

    action_request.single_action_requests.append(single_field_request)

    action_response = simulator_building.request_action(action_request)

    self.assertEqual(action_response.request, action_request)
    self.assertEqual(
        action_response.single_action_responses[0].request, single_field_request
    )

    self.assertEqual(
        action_response.single_action_responses[0].response_type,
        _ACTION_RESPONSE_TYPE.ACCEPTED,
    )

  def test_request_action_incorrect_device(self):
    """Tests when an action is sent to a nonexistent device."""
    simulator_building = self.building

    action_request = smart_control_building_pb2.ActionRequest()
    single_field_request = smart_control_building_pb2.SingleActionRequest(
        device_id='wrong_device', setpoint_name='setpoint_name'
    )

    action_request.single_action_requests.append(single_field_request)

    action_response = simulator_building.request_action(action_request)

    self.assertEqual(
        action_response.single_action_responses[0].response_type,
        _ACTION_RESPONSE_TYPE.REJECTED_INVALID_DEVICE,
    )

  def test_request_action_incorrect_setpoint(self):
    """Tests when an action is sent to a nonexistent setpoint."""
    simulator_building = self.building

    action_request = smart_control_building_pb2.ActionRequest()
    single_field_request = smart_control_building_pb2.SingleActionRequest(
        device_id='hws_id', setpoint_name='incorrect_setpoint'
    )

    action_request.single_action_requests.append(single_field_request)

    action_response = simulator_building.request_action(action_request)

    self.assertEqual(
        action_response.single_action_responses[0].response_type,
        _ACTION_RESPONSE_TYPE.REJECTED_NOT_ENABLED_OR_AVAILABLE,
    )

  # ZONES

  def test_init_uses_hvac_zones_by_default(self):
    self.assertEqual(
        list(self.building.zones),
        list(self.building.simulator.hvac.zone_infos.values()),
    )

  # THERMODYNAMICS

  def test_a_day_of_stepping_stays_between_the_two_reservoirs(self):
    """The fixture must be a building, not just a plumbing harness.

    Everything above tests device wiring and would pass just as happily against
    a fixture whose temperatures diverge to 1e50. Ambient and the supply air are
    the only two reservoirs the toy building is attached to, so nothing in it
    can leave the interval they span; if some parameter here drifts back out of
    range - too little thermal mass in the air, too much VAV authority for four
    control volumes, an envelope too well sealed to feel the weather - this is
    what notices.
    """
    simulator = self.building.simulator

    for _ in range(288):  # A day at 300 s per step.
      simulator.step_sim()

    self.assertBetween(simulator.building.temp.min(), 288.0, 296.0)
    self.assertBetween(simulator.building.temp.max(), 288.0, 296.0)


