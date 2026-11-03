"""Synthetic contract, learning and decision tests for Version 12.

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

nb = json.loads((Path(__file__).parent / 'submission-nb - Version 12.ipynb').read_text())
source = ''.join(nb['cells'][1]['source'])
compile(source, 'submission-v12', 'exec')
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
for step in range(128):
    put(0, 305. - 0.08*step)
    check(p,obs)
elapsed=time.perf_counter()-start
assert elapsed < 12.0, f'Excessive latency: {elapsed:.2f}s for 128 steps'

# Search-size budget is a property of the exact submitted policy.
assert len(Policy.OPERATING_POINTS) == 4
assert Policy.PLAN_STEPS == (4, 10)
meta, obs, set_temp = scenario(1, 3)
p = Policy(meta, None)
zone = p.ahus[0]['zones'][0]
profile_count = sum(len(p._profiles(config, zone))
                    for config in p.OPERATING_POINTS)
assert profile_count == 10, f'Unexpected search cardinality: {profile_count}'

# The fast scalar scorer must exactly preserve the V6 action-conditioned
# feature model for each *retained* action: no unintentional math drift.
for t in (283.0, 295.0, 304.0):
    for outdoor in (270.0, 299.0, float('nan')):
        state = {'zone': zone, 'temp': t,
                 'low': 293.0, 'high': 297.0, 'outdoor': outdoor}
        for config in p.OPERATING_POINTS:
            _, running, supply, pressure, water_k, water_on = config
            exhaustive = []
            for damper, valve in p._profiles(config, zone):
                features = Model.features(t, outdoor, float(running),
                                          supply, damper, valve,
                                          float(water_on), water_k)
                delta = zone['model'].predict_delta(features)
                discomfort = np.mean([
                    p._violation(t + delta * step, 293.0, 297.0)
                    for step in p.PLAN_STEPS
                ])
                energy = ((0.012 * damper if running else 0.0) +
                          (0.022 * damper * valve if water_on else 0.0))
                cost = (p.DISCOMFORT_WEIGHT * discomfort +
                        (p.ENERGY_WEIGHT * energy if p.OPTIMIZE_ENERGY else 0.0))
                exhaustive.append((cost, damper, valve))
            expected = min(exhaustive)
            actual = p._best_vav(state, config)
            assert np.allclose(actual, expected, atol=1e-12), (
                config[0], t, outdoor, actual, expected)

# Early cutoff must choose the same point as exhaustive evaluation over
# V12's compact space. It relies on nonnegative surrogate terms.
for (hot, cold) in ((295., 295.), (304., 295.), (295., 285.),
                    (305., 284.), (299., 291.), (311., 281.)):
    states = [
        {'zone': p.ahus[0]['zones'][i], 'temp': t,
         'low': 293., 'high': 297., 'outdoor': 297.}
        for i, t in enumerate((hot, cold, 295.))
    ]
    worst_hot = max(z['temp'] - z['high'] for z in states)
    worst_cold = max(z['low'] - z['temp'] for z in states)
    candidates = []
    for config in p.OPERATING_POINTS:
        if not p._candidate_allowed(config, worst_hot, worst_cold):
            continue
        scores = [p._best_vav(z, config) for z in states]
        total = (sum(choice[0] for choice in scores) / len(states) +
                 (p.ENERGY_WEIGHT * p._plant_cost(config)
                  if p.OPTIMIZE_ENERGY else 0.0))
        candidates.append((total, config, scores))
    full_best = min(candidates, key=lambda item: item[0])
    selected_mode, selected_zones = p._select_operating_point(states)
    assert selected_mode == full_best[1]
    assert selected_zones == full_best[2]

# Never request a shared hot-water plant for a missing reheat actuator.
meta, obs, set_temp = scenario(1, 1)
meta['zone_action_indices']['VAV_0']['reheat_valve_percentage_command'] = None
set_temp(0, 280.0)
p = Policy(meta, None)
_, switches = check(p, obs)
assert switches[-1] == 0, 'No mapped reheat valve: boiler must remain off'

# A moderate mixed demand should still deliver separate cooling and reheat.
meta, obs, set_temp = scenario(1, 3)
set_temp(0, 309.0)
set_temp(1, 281.0)
p = Policy(meta, None)
actions, switches = check(p, obs)
assert switches[0] == 1 and switches[-1] == 1
valve_idx = meta['zone_action_indices']['VAV_1']['reheat_valve_percentage_command']
assert actions[valve_idx] > -1.0, 'Cold room needs local reheat'

# The last notebook cell is the official submission entry point.
assert 'SmartBuildingsInferenceServer(TransferPolicy).run()' in (
    ''.join(nb['cells'][2]['source']))

print(f'PASS: notebook syntax, variable layouts, typed bounds, cold/hot/mixed safety, reset, RLS, missing sensors, latency ({elapsed:.2f}s/128)')
