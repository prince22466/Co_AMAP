"""Offline synthetic contract tests for submission-nb - Version 5.ipynb.

Run: python Co_Kaggle/g8/submissions/test_version5.py
This tests policy shape/logic; it does not replace a public simulator rollout.
"""
import json
import math
from pathlib import Path
import sys
import types

import numpy as np

root = Path(__file__).parent
notebook = json.loads((root / 'submission-nb - Version 5.ipynb').read_text())
code = ''.join(notebook['cells'][1]['source'])
compile(code, 'submission-v5', 'exec')

# Use a mock Kaggle gateway to exercise the exact notebook code offline.
package = types.ModuleType('kaggle_evaluation')
gateway = types.ModuleType('smart_buildings_inference_server')
gateway.Policy = type('Policy', (), {'__init__': lambda self, m, a: None})
package.smart_buildings_inference_server = gateway
sys.modules['kaggle_evaluation'] = package
sys.modules['kaggle_evaluation.smart_buildings_inference_server'] = gateway
namespace = {}
exec(code, namespace)
Policy = namespace['TransferPolicy']


def setup(n_ahu=2, zones_per_ahu=2):
    fields = ['hod_sin_000', 'hod_cos_000', 'dow_sin_000', 'dow_cos_000']
    assignments = {}
    action_map = {}
    temp_actions, press_actions, recirc_actions = [], [], []
    zone_ids = []
    for a in range(n_ahu):
        assignments[f'ahu_{a}'] = []
        temp_actions.append(3*a)
        press_actions.append(3*a+1)
        recirc_actions.append(3*a+2)
        fields.append(f'ahs_ahu_{a}_outside_air_temperature_sensor')
        for z in range(zones_per_ahu):
            i = a*zones_per_ahu+z
            zone_ids.append(i)
            assignments[f'ahu_{a}'].append(f'room_{i}')
            base = f'VAV_{i}_zone_air_'
            fields += [base+'temperature_sensor', base+'heating_temperature_setpoint_sensor',
                       base+'cooling_temperature_setpoint_sensor']
            start = 3*n_ahu + 2 + 2*i
            action_map[f'VAV_{i}'] = {'supply_air_damper_percentage_command': start,
                                      'reheat_valve_percentage_command': start+1}
    m = dict(n_continuous_action=3*n_ahu+2+2*len(zone_ids),
             n_discrete_action=n_ahu+1, temp_indices=temp_actions,
             pressure_indices=press_actions+[3*n_ahu+1],
             recirculation_indices=recirc_actions, water_indices=[3*n_ahu],
             observation_field_names=fields, ahu_zone_assignments=assignments,
             zone_action_indices=action_map, comfort_temp_window=[293, 297],
             eco_temp_window=[289, 301], morning_start_hour=8, evening_start_hour=18,
             observation_normalization_metadata={'zone_air_temperature_sensor':
                                                  {'sample_mean': 295, 'sample_std': 5}})
    obs = np.zeros(len(fields), dtype=np.float64)
    def clock(hour, day):
        obs[0], obs[1] = math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24)
        obs[2], obs[3] = math.sin(2*math.pi*day/7), math.cos(2*math.pi*day/7)
    clock(12, 2)
    for a in range(n_ahu):
        obs[fields.index(f'ahs_ahu_{a}_outside_air_temperature_sensor')] = 300
    for i in zone_ids:
        base = f'VAV_{i}_zone_air_'
        obs[fields.index(base+'temperature_sensor')] = 0
        obs[fields.index(base+'heating_temperature_setpoint_sensor')] = 293
        obs[fields.index(base+'cooling_temperature_setpoint_sensor')] = 297
    def temp(zone, value):
        obs[fields.index(f'VAV_{zone}_zone_air_temperature_sensor')] = (value-295)/5
    return m, obs, clock, temp


def check(policy, obs):
    a = policy.get_action(obs)
    con, dis = a['continuous_action'], a['discrete_action']
    assert con.shape == (policy.n_cont,) and con.dtype == np.float32
    assert dis.shape == (policy.n_disc,) and dis.dtype == np.int32
    assert np.all(np.isfinite(con)) and np.all(np.abs(con) <= 1)
    assert set(np.unique(dis)).issubset({0, 1})
    return con, dis


m, obs, clock, temp = setup()
p = Policy(m, None)
con, dis = check(p, obs)
assert list(dis) == [0, 0, 0], 'in-band should idle all equipment'
assert np.all(con[[0, 3]] == 1), 'idle AHU must avoid cooling demand'

# Independent AHUs; only first is hot.
temp(0, 299.0)
con, dis = check(p, obs)
assert list(dis) == [1, 0, 0]
assert con[0] < 1 and con[1] > -1 and con[2] == 1
assert con[m['zone_action_indices']['VAV_0']['supply_air_damper_percentage_command']] > -1
assert con[m['zone_action_indices']['VAV_2']['supply_air_damper_percentage_command']] == -1

# Severe cold zone requires reheat AND hot water; no AHU heating is possible.
temp(0, 290.0)
con, dis = check(p, obs)
assert list(dis) == [1, 0, 1]
assert con[0] == 1, 'heat mode must not demand AHU cooling'
assert con[m['zone_action_indices']['VAV_0']['reheat_valve_percentage_command']] > -1
assert con[6] > -1, 'water supply temperature must be driven'

# Mixed hot/cold zones: first AHU stays on and opens cold-zone reheat.
temp(0, 303)
temp(1, 291)
con, dis = check(p, obs)
assert list(dis) == [1, 0, 1] and con[0] < 1
assert con[m['zone_action_indices']['VAV_1']['reheat_valve_percentage_command']] > -1

# Economizer responds to favorable outdoor conditions only in cool mode.
obs[m['observation_field_names'].index('ahs_ahu_0_outside_air_temperature_sensor')] = 285
con, dis = check(p, obs)
assert con[2] < 1, 'cooling with favorable outdoor air should economize'

# Risk weighting must grow beyond a 5K excursion, not saturate.
assert p._risk(10.0) > 2*p._risk(5.0)

# 6-step same-mode persistence increases gain if stubborn violation persists.
p.reset()
temp(0, 303)
temp(1, 295)
obs[m['observation_field_names'].index('ahs_ahu_0_outside_air_temperature_sensor')] = 300
for _ in range(7):
    check(p, obs)
assert 1.0 < p.strength[0] <= 1.40, p.strength
p.reset()
assert np.all(p.strength == 1.0) and p.mode == ['off']*len(p.ahus)

# Time-based fallback and metadata variability: missing setpoint fields.
m2, o2, clk2, tmp2 = setup(n_ahu=1,zones_per_ahu=3)
for i in range(3):
    m2['observation_field_names'][m2['observation_field_names'].index(
        f'VAV_{i}_zone_air_heating_temperature_setpoint_sensor')] = f'unavailable_lo_{i}'
    m2['observation_field_names'][m2['observation_field_names'].index(
        f'VAV_{i}_zone_air_cooling_temperature_setpoint_sensor')] = f'unavailable_hi_{i}'
p2 = Policy(m2, None)
clk2(2, 6)  # weekend: wide eco band
con, dis = check(p2, o2)
assert list(dis) == [0, 0]
tmp2(2, 303)
con, dis = check(p2, o2)
assert list(dis) == [1, 0]

# Out-of-range temperatures ignored, rather than emitted as NaN commands.
tmp2(2, 500)
con, dis = check(p2, o2)
assert list(dis) == [0, 0]
print('PASS: syntax, types, ranges, indexing, hot, cold, mixed, economizer, calibration, reset, variable metadata')
