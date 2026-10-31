"""Offline smoke tests for submission-nb - Version 4.ipynb.

Only checks action contract and specific controller transitions. Does NOT run
the 8,640-step simulator and DOES NOT establish leaderboard performance.
Run: python Co_Kaggle/g8/submissions/test_submission_v4.py
"""
import json
import math
from pathlib import Path
import sys
import types

import numpy as np

server = types.ModuleType('kaggle_evaluation.smart_buildings_inference_server')
server.Policy = type('Policy', (), {'__init__': lambda self, metadata, spec: None})
server.SmartBuildingsInferenceServer = type('SmartBuildingsInferenceServer', (),
    {'__init__': lambda self, policy: None, 'run': lambda self: None})
package = types.ModuleType('kaggle_evaluation')
package.smart_buildings_inference_server = server
sys.modules['kaggle_evaluation'] = package
sys.modules['kaggle_evaluation.smart_buildings_inference_server'] = server
notebook = json.loads((Path(__file__).parent / 'submission-nb - Version 4.ipynb').read_text())
code = ''.join(notebook['cells'][1]['source'])
namespace = {}
exec(compile(code, 'submission-v4', 'exec'), namespace)
Policy = namespace['TransferPolicy']

fields = ['hod_sin_000', 'hod_cos_000', 'dow_sin_000', 'dow_cos_000']
for vav in ('VAV_2', 'VAV_7', 'VAV_10'):
    fields += [vav + '_zone_air_temperature_sensor',
               vav + '_zone_air_heating_temperature_setpoint_sensor',
               vav + '_zone_air_cooling_temperature_setpoint_sensor']
fields += ['ahs_ahu_2_outside_air_temperature_sensor',
           'ahs_ahu_10_outside_air_temperature_sensor']
idx = {v: i for i, v in enumerate(fields)}
metadata = {
    'n_continuous_action': 14, 'n_discrete_action': 3,
    'observation_field_names': fields,
    'ahu_zone_assignments': {
        'ahu_10': ['room_7', 'room_10'], 'ahu_2': ['room_2']},
    'zone_action_indices': {
        v: {'supply_air_damper_percentage_command': 8 + 2 * i,
            'reheat_valve_percentage_command': 9 + 2 * i}
        for i, v in enumerate(('VAV_2', 'VAV_7', 'VAV_10'))},
    'temp_indices': [0, 3], 'pressure_indices': [1, 4, 7],
    'recirculation_indices': [2, 5], 'water_indices': [6],
    'comfort_temp_window': [294., 297.],
    'eco_temp_window': [290., 301.],
    'morning_start_hour': 9., 'evening_start_hour': 17.,
    'observation_normalization_metadata': {
        'zone_air_temperature_sensor': {'sample_mean': 295., 'sample_std': 5.}}
}
p = Policy(metadata, {})

def obs(temps=(295., 295., 295.), lo=294., hi=297.,
        hour=11., day=2., outside=(290., 290.)):
    x = np.zeros(len(fields), dtype=np.float64)
    for prefix, period, value in (('hod', 24, hour), ('dow', 7, day)):
        x[idx[prefix + '_sin_000']] = math.sin(2 * math.pi * value / period)
        x[idx[prefix + '_cos_000']] = math.cos(2 * math.pi * value / period)
    for v, t in zip(('VAV_2', 'VAV_7', 'VAV_10'), temps):
        x[idx[v + '_zone_air_temperature_sensor']] = (t - 295.) / 5.
        x[idx[v + '_zone_air_heating_temperature_setpoint_sensor']] = lo
        x[idx[v + '_zone_air_cooling_temperature_setpoint_sensor']] = hi
    for ahu, t in zip(('ahu_2', 'ahu_10'), outside):
        x[idx['ahs_' + ahu + '_outside_air_temperature_sensor']] = t
    return x

def valid(action):
    c, d = action['continuous_action'], action['discrete_action']
    assert c.dtype == np.float32 and c.shape == (14,), c
    assert np.isfinite(c).all() and np.all((c >= -1) & (c <= 1)), c
    assert d.dtype == np.int32 and d.shape == (3,), d
    assert np.isin(d, [0, 1]).all(), d

a = p.get_action(obs()); valid(a)
assert list(a['discrete_action']) == [0, 0, 0]
assert a['continuous_action'][9] == -1
p.reset()
a = p.get_action(obs((300., 295., 295.))); valid(a)
assert list(a['discrete_action']) == [1, 0, 0], a
assert a['continuous_action'][0] < 1 and a['continuous_action'][9] == -1
p.reset()
a = p.get_action(obs((289., 295., 295.))); valid(a)
assert list(a['discrete_action']) == [1, 0, 1], a
assert a['continuous_action'][0] == 1 and a['continuous_action'][9] > -1
p.reset()
a = p.get_action(obs((295., 300., 291.))); valid(a)
assert list(a['discrete_action']) == [0, 1, 1], a
assert a['continuous_action'][3] < 1 and a['continuous_action'][13] > -1
p.reset()
a = p.get_action(obs()); valid(a)
assert a['discrete_action'].sum() == 0
p.reset()
x = obs()
x[idx['VAV_2_zone_air_temperature_sensor']] = np.nan
x[idx['VAV_7_zone_air_heating_temperature_setpoint_sensor']] = np.nan
valid(p.get_action(x))
p.reset()
for t in (296.10, 296.39, 296.68):
    p.get_action(obs((t, 295., 295.)))
a = p.get_action(obs((296.97, 295., 295.))); valid(a)
assert a['discrete_action'][0] == 1, a
p.reset()
a = p.get_action(obs((291., 295., 295.), lo=290., hi=301., hour=8.7))
valid(a)
assert a['discrete_action'][0] == 1, a
rng = np.random.default_rng(42)
for _ in range(200):
    valid(p.get_action(obs(rng.uniform(282, 310, 3),
        hour=float(rng.uniform(0, 24)), day=float(rng.integers(0, 7)),
        outside=tuple(rng.uniform(260, 315, 2)))))
print('PASS: V4 offline interface and control-path smoke tests')
