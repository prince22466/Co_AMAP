"""Synthetic contract, learning and decision tests for Version 6.

Use the notebook's policy cell as-is; do not exercise a copied implementation.
These tests do NOT measure the official simulator / Kaggle leaderboard score.
"""
import json
import math
from pathlib import Path
import sys
import time
import types
import numpy as np

nb = json.loads((Path(__file__).parent / 'submission-nb - Version 6.ipynb').read_text())
source = ''.join(nb['cells'][1]['source'])
compile(source, 'submission-v6', 'exec')
package = types.ModuleType('kaggle_evaluation')
gateway = types.ModuleType('smart_buildings_inference_server')
gateway.Policy = type('Policy', (), {'__init__': lambda self, metadata, spec: None})
package.smart_buildings_inference_server = gateway
sys.modules['kaggle_evaluation'] = package
sys.modules['kaggle_evaluation.smart_buildings_inference_server'] = gateway
ns = {}
exec(source, ns)
Policy, Model = ns['TransferPolicy'], ns['ZoneResponseModel']


def scenario(num_ahus=2, per_ahu=3, temperature=295., outdoor=297.):
    fields = ['hod_sin_000','hod_cos_000','dow_sin_000','dow_cos_000']
    assignments, action_index = {}, {}
    for ahu in range(num_ahus):
        assignments[f'ahu_{ahu}'] = []
        fields.append(f'ahs_ahu_{ahu}_outside_air_temperature_sensor')
        for local in range(per_ahu):
            i = ahu*per_ahu+local
            assignments[f'ahu_{ahu}'].append(f'room_{i}')
            names = [f'VAV_{i}_zone_air_temperature_sensor',
                     f'VAV_{i}_zone_air_heating_temperature_setpoint_sensor',
                     f'VAV_{i}_zone_air_cooling_temperature_setpoint_sensor']
            fields.extend(names)
            action_index[f'VAV_{i}'] = {
                'supply_air_damper_percentage_command': 3*num_ahus+2+2*i,
                'reheat_valve_percentage_command': 3*num_ahus+2+2*i+1}
    m = {'observation_field_names':fields,
         'n_continuous_action':3*num_ahus+2+2*num_ahus*per_ahu,
         'n_discrete_action':num_ahus+1,
         'temp_indices':[3*i for i in range(num_ahus)],
         'pressure_indices':[3*i+1 for i in range(num_ahus)]+[3*num_ahus+1],
         'recirculation_indices':[3*i+2 for i in range(num_ahus)],
         'water_indices':[3*num_ahus],
         'ahu_zone_assignments':assignments,'zone_action_indices':action_index,
         'comfort_temp_window':[293.,297.], 'eco_temp_window':[288.,302.],
         'morning_start_hour':8.,'evening_start_hour':18.,
         'observation_normalization_metadata':{
            'zone_air_temperature_sensor':{'sample_mean':295.,'sample_std':5.}}
         }
    x = np.zeros(len(fields), dtype=np.float64)
    x[0] = math.sin(2*math.pi*12/24)
    x[1] = math.cos(2*math.pi*12/24)
    x[2] = math.sin(2*math.pi*2/7)
    x[3] = math.cos(2*math.pi*2/7)
    for i,n in enumerate(fields):
        if n.endswith('zone_air_temperature_sensor'):x[i]=(temperature-295.)/5.
        elif n.endswith('_heating_temperature_setpoint_sensor'):x[i]=293.
        elif n.endswith('_cooling_temperature_setpoint_sensor'):x[i]=297.
        elif n.endswith('outside_air_temperature_sensor'):x[i]=outdoor
    def set_temp(zone, new):
        x[fields.index(f'VAV_{zone}_zone_air_temperature_sensor')] = (new-295.)/5.
    return m, x, set_temp


def check(policy, x):
    a = policy.get_action(x)
    c,d = a['continuous_action'],a['discrete_action']
    assert c.shape == (policy.continuous_count,) and c.dtype == np.float32
    assert d.shape == (policy.discrete_count,) and d.dtype == np.int32
    assert np.all(np.isfinite(c)) and np.all(np.abs(c) <= 1.000001)
    assert set(d).issubset({0,1})
    return c,d


for shape in [(1,1),(2,3),(4,7)]:
    meta,obs,put = scenario(*shape)
    p=Policy(meta,None)
    c,d=check(p,obs)
    assert d[-1] == 0, 'No reheat should be requested in the comfort band'
    # Force a very severe hot excursion; optimizer must not remain off.
    put(0,307.)
    c,d=check(p,obs)
    assert d[0] == 1 and c[0] < 1, 'Severe hot zone needs cooling'
    # Force a very severe cold excursion; optimizer must call hot water.
    put(0,285.)
    c,d=check(p,obs)
    assert d[0] == 1 and d[-1] == 1, 'Severe cold requires reheat and pump'
    assert c[0] == 1, 'AHU must not try to heat using its cooling coil'
    # Cold+hot on same AHU permits the mixed configuration.
    if shape[1] >= 2:
        put(0,307.)
        put(1,284.)
        c,d=check(p,obs)
        assert d[0] == 1 and d[-1] == 1
    # Episode restart must clear the prior and pending features.
    p.reset()
    for ahu in p.ahus:
        for zone in ahu['zones']:
            assert zone['model'].samples == 0
            assert zone['model'].last_temperature is None

# Model-specific behavior: learn sign-correct heating from observed trajectory.
model=Model()
initial=model.weights.copy()
for index in range(50):
    features=model.features(290.,280.,1,305.,1.,1.,1,348.)
    model.remember(290.,features)
    model.observe(290.24)
assert model.samples==50 and model.weights[2]>0
assert model.predict_delta(features) > 0
model.clear_pending()
assert model.last_features is None

# Missing setpoint and bad readings must fail gracefully.
meta,obs,put=scenario()
meta['observation_field_names'][meta['observation_field_names'].index(
    'VAV_0_zone_air_heating_temperature_setpoint_sensor')] = 'missing_low'
p=Policy(meta,None)
put(0,float('nan'))
check(p,obs)
put(0,309.)
check(p,obs)

# Reasonable latency: constant-size finite set of candidates, no optimizer lib.
meta,obs,put=scenario(2,7)
p=Policy(meta,None)
start=time.perf_counter()
for step in range(16):
    put(0, 305. - 0.08*step)
    check(p,obs)
elapsed=time.perf_counter()-start
assert elapsed < 15.0, f'Excessive latency: {elapsed:.2f}s for 16 steps'
print(f'PASS: notebook syntax, variable layouts, typed bounds, cold/hot/mixed safety, reset, RLS, missing sensors, latency ({elapsed:.2f}s/16)')
