"""Synthetic contract and QP regression tests for Version 14.

This is NOT an official simulator or leaderboard score. Load the notebook's
exact code cell, not a separately maintained copy of the controller.
"""
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np

nb = json.loads((Path(__file__).parent / 'submission-nb - Version 14.ipynb').read_text())
source = ''.join(nb['cells'][1]['source'])
compile(source, 'submission-v14', 'exec')
package = types.ModuleType('kaggle_evaluation')
gateway = types.ModuleType('smart_buildings_inference_server')
gateway.Policy = type('Policy', (), {'__init__': lambda self, metadata, spec: None})
package.smart_buildings_inference_server = gateway
sys.modules['kaggle_evaluation'] = package
sys.modules['kaggle_evaluation.smart_buildings_inference_server'] = gateway
namespace = {}
exec(source, namespace)
Policy = namespace['TransferPolicy']
allocate = namespace['allocate_qp']


def scenario(n_ahu=2, zones_per_ahu=3, outdoor=295.0):
    fields = ['hod_sin_000', 'hod_cos_000', 'dow_sin_000', 'dow_cos_000']
    assignments, actions = {}, {}
    for i in range(n_ahu):
        name = f'ahu_{i}'
        assignments[name] = []
        fields.append(f'ahs_{name}_outside_air_temperature_sensor')
        for j in range(zones_per_ahu):
            zone_id = i*zones_per_ahu+j
            vav = f'VAV_{zone_id}'
            assignments[name].append(f'room_{zone_id}')
            fields.extend([f'{vav}_zone_air_temperature_sensor',
                           f'{vav}_zone_air_heating_temperature_setpoint_sensor',
                           f'{vav}_zone_air_cooling_temperature_setpoint_sensor'])
            idx = 3*n_ahu+2+2*zone_id
            actions[vav] = {
                'supply_air_damper_percentage_command': idx,
                'reheat_valve_percentage_command': idx+1,
            }
    metadata = {
        'n_continuous_action': 3*n_ahu+2+2*n_ahu*zones_per_ahu,
        'n_discrete_action': n_ahu+1,
        'observation_field_names': fields,
        'temp_indices': [3*i for i in range(n_ahu)],
        'pressure_indices': [3*i+1 for i in range(n_ahu)]+[3*n_ahu+1],
        'recirculation_indices': [3*i+2 for i in range(n_ahu)],
        'water_indices': [3*n_ahu],
        'ahu_zone_assignments': assignments,
        'zone_action_indices': actions,
        'comfort_temp_window': [293.0, 297.0],
        'eco_temp_window': [288.0, 302.0],
        'morning_start_hour': 8.0, 'evening_start_hour': 18.0,
        'observation_normalization_metadata': {
            'zone_air_temperature_sensor': {'sample_mean': 295.0, 'sample_std': 5.0},
            'outside_air_temperature_sensor': {'sample_mean': 295.0, 'sample_std': 10.0},
        },
    }
    obs = np.zeros(len(fields), dtype=np.float64)
    obs[0], obs[1] = math.sin(math.pi), math.cos(math.pi)  # 12:00
    obs[2], obs[3] = math.sin(4*math.pi/7), math.cos(4*math.pi/7)
    positions = {name: k for k, name in enumerate(fields)}
    for name, index in positions.items():
        if name.endswith('zone_air_temperature_sensor'):
            obs[index] = 0.0
        elif name.endswith('heating_temperature_setpoint_sensor'):
            obs[index] = 293.0
        elif name.endswith('cooling_temperature_setpoint_sensor'):
            obs[index] = 297.0
        elif name.endswith('outside_air_temperature_sensor'):
            obs[index] = (outdoor-295.0)/10.0

    def set_zone(zone_id, kelvin):
        obs[positions[f'VAV_{zone_id}_zone_air_temperature_sensor']] = (kelvin-295.0)/5.0
    return metadata, obs, set_zone


def check(policy, obs):
    result = policy.get_action(obs)
    assert set(result) == {'continuous_action', 'discrete_action'}
    c, d = result['continuous_action'], result['discrete_action']
    assert c.dtype == np.float32 and d.dtype == np.int32
    assert c.shape == (policy.continuous_count,) and d.shape == (policy.discrete_count,)
    assert np.isfinite(c).all() and np.all(np.abs(c) <= 1.000001)
    assert np.isin(d, [0, 1]).all()
    return c, d


# Mathematical KKT conditions, binding capacity, bounds, and extreme corners.
for desired, weights, upper, cap in [
    ([.8, .1, .5], [10., 1., 2.], [1., 1., 1.], .55),
    ([.8, .1, .5], [10., 1., 2.], [1., 1., 1.], 5.),
    ([1.8, .5], [5., 1.], [.8, .5], .7),
    ([1.8, .5], [5., 1.], [.8, .5], 0.),
    ([], [], [], 1.),
]:
    x = allocate(desired, weights, upper, cap)
    desired = np.array(desired)
    weights = np.array(weights)
    upper = np.array(upper)
    assert np.all(x >= -1e-10) and np.all(x <= upper+1e-10)
    assert sum(x) <= cap + 1e-7
    if len(x) and cap > 0:
        marginal = weights*(desired-x)-.08*x
        interiors = (x > 1e-6) & (x < upper-1e-6)
        if np.any(interiors):
            price = float(np.mean(marginal[interiors]))
            assert np.max(np.abs(marginal[interiors]-price)) < 1e-5
            assert price >= -1e-6
        if sum(x) < cap-1e-5:
            assert np.all(marginal[interiors] < 1e-5)

for n_ahu, per in [(1, 1), (2, 3), (4, 7), (12, 1)]:
    metadata, obs, set_zone = scenario(n_ahu, per)
    policy = Policy(metadata, None)
    c, d = check(policy, obs)
    assert np.all(d == 0), 'Comfort-band rooms should not force plant activation'
    set_zone(0, 307.0)
    c, d = check(policy, obs)
    assert d[0] == 1 and d[-1] == 0
    assert c[metadata['temp_indices'][0]] < 1.0
    set_zone(0, 283.0)
    c, d = check(policy, obs)
    assert d[0] == 1 and d[-1] == 1
    if per > 1:
        set_zone(1, 309.0)
        c, d = check(policy, obs)
        assert d[0] == 1 and d[-1] == 1
        assert c[metadata['zone_action_indices']['VAV_0']['reheat_valve_percentage_command']] > -1.0
    policy.reset()
    check(policy, obs)

# Airflow QP must actually constrain an AHU under simultaneous extreme cooling.
metadata, obs, set_zone = scenario(1, 8)
for i in range(8):
    set_zone(i, 325.)
policy = Policy(metadata, None)
c, d = check(policy, obs)
openings = [0.5*(c[metadata['zone_action_indices'][f'VAV_{i}']['supply_air_damper_percentage_command']]+1)
            for i in range(8)]
assert sum(opening-0.1 for opening in openings) <= .88+.61*7+1e-5
assert max(openings) < .98

# Heating QP must cap aggregate damper*valve load, even with multiple AHUs.
metadata, obs, set_zone = scenario(2, 8, outdoor=270.)
for i in range(16):
    set_zone(i, 272.)
policy = Policy(metadata, None)
c, d = check(policy, obs)
heat = 0.0
for i in range(16):
    mapping = metadata['zone_action_indices'][f'VAV_{i}']
    damper = (c[mapping['supply_air_damper_percentage_command']]+1)*.5
    valve = (c[mapping['reheat_valve_percentage_command']]+1)*.5
    heat += damper*valve
assert d[-1] == 1
assert heat <= .85+.6*15+1e-5

# Missing reheat actuator cannot turn on a boiler; absent outside metadata
# never interprets z-scores as Kelvin (conservative recirculation default).
metadata, obs, set_zone = scenario(1, 2, outdoor=260.)
metadata['zone_action_indices']['VAV_0']['reheat_valve_percentage_command'] = None
set_zone(0, 280.)
p = Policy(metadata, None)
_, switches = check(p, obs)
assert switches[-1] == 0
set_zone(1, 310.)
c, switches = check(p, obs)
assert switches[0] == 1
assert c[metadata['recirculation_indices'][0]] <= 1.0
metadata['observation_normalization_metadata'].pop('outside_air_temperature_sensor')
p = Policy(metadata, None)
c, _ = check(p, obs)
assert abs(c[metadata['recirculation_indices'][0]] - .4) < 1e-5

# Invalid zone data do not propagate NaN; identifier ordering is numeric.
metadata, obs, set_zone = scenario(12, 1)
set_zone(10, 310.)
p = Policy(metadata, None)
_, d = check(p, obs)
assert d[10] == 1 and d[2] == 0
set_zone(10, np.nan)
check(p, obs)

# Approximate upper bound on decision latency for normal building topology.
metadata, obs, set_zone = scenario(2, 7)
p = Policy(metadata, None)
start = time.perf_counter()
for step in range(256):
    set_zone(0, 304.0 - .02*step)
    check(p, obs)
seconds = time.perf_counter()-start
assert seconds < 8.0, f'Excessive latency: {seconds:.2f}s for 256 steps'
assert 'SmartBuildingsInferenceServer(TransferPolicy).run()' in ''.join(nb['cells'][2]['source'])
print(f'PASS V14: QP feasibility/KKT, variable topology, mixed heat/cool, '
      f'normalization, missing sensors/actuators and latency ({seconds:.2f}s/256)')
