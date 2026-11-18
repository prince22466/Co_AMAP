"""Tests for setpoint_energy_carbon_regret.

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
from absl.testing import absltest, parameterized

from smart_buildings.smart_control.reward import (
  electricity_energy_cost,
  natural_gas_energy_cost,
  setpoint_energy_carbon_regret,
)
from smart_buildings.smart_control.utils import test_utils

_MAX_ELECTRICITY_RATE = 160000.0
_MAX_NATURAL_GAS_RATE = 400000.0


def _get_reward_function() -> (
    setpoint_energy_carbon_regret.SetpointEnergyCarbonRegretFunction
):
  """Returns a regret function configured like the training scenarios."""
  return setpoint_energy_carbon_regret.SetpointEnergyCarbonRegretFunction(
      max_productivity_personhour_usd=300.0,
      min_productivity_personhour_usd=100.0,
      max_electricity_rate=_MAX_ELECTRICITY_RATE,
      max_natural_gas_rate=_MAX_NATURAL_GAS_RATE,
      productivity_deviation_ref=5.0,
      electricity_energy_cost=electricity_energy_cost.ElectricityEnergyCost(),
      natural_gas_energy_cost=natural_gas_energy_cost.NaturalGasEnergyCost(),
      productivity_weight=0.6,
      energy_cost_weight=0.2,
      carbon_emission_weight=0.2,
  )


def _get_reward_info(pump_electrical_energy_rate: float):
  """Returns a RewardInfo differing only in the boiler pump electrical rate."""
  return test_utils.get_test_reward_info(
      zone_temp_occupancies=[('z0', 295.0, 8.0), ('z1', 292.0, 3.0)],
      air_handler_energies=[('ac0', 23.0, 15.0)],
      boiler_energies=[('b0', 200.1, pump_electrical_energy_rate)],
      start_timestamp=pd.Timestamp('2022-03-13 12:00:00'),
      end_timestamp=pd.Timestamp('2022-03-13 12:05:00'),
  )


class SetpointEnergyCarbonRegretTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Both rates are above max_electricity_rate, so before the cap was
      # removed these two infos produced an identical reward.
      ('above_cap', 200000.0, 400000.0),
      ('below_cap', 10000.0, 20000.0),
  )
  def test_higher_electricity_rate_increases_normalized_energy_cost(
      self, lower_rate, higher_rate
  ):
    reward_function = _get_reward_function()

    lower_response = reward_function.compute_reward(
        _get_reward_info(lower_rate)
    )
    higher_response = reward_function.compute_reward(
        _get_reward_info(higher_rate)
    )

    self.assertGreater(
        higher_response.normalized_energy_cost,
        lower_response.normalized_energy_cost,
    )

  def test_above_cap_higher_electricity_rate_lowers_agent_reward(self):
    reward_function = _get_reward_function()

    lower_response = reward_function.compute_reward(_get_reward_info(200000.0))
    higher_response = reward_function.compute_reward(_get_reward_info(400000.0))

    self.assertLess(
        higher_response.agent_reward_value,
        lower_response.agent_reward_value,
    )

  def test_energy_rate_above_max_exceeds_normalized_bounds(self):
    reward_function = _get_reward_function()

    # Electrical consumption well above max_electricity_rate is no longer
    # clipped, so normalized energy cost is allowed to exceed 1. The carbon
    # term is not checked here: it is normalized against the combined
    # electricity and natural gas maxima, and this info's small gas rate keeps
    # the combined ratio below 1.
    response = reward_function.compute_reward(_get_reward_info(400000.0))

    self.assertGreater(response.normalized_energy_cost, 1.0)


if __name__ == '__main__':
  absltest.main()
