"""Version 20 notebook smoke tests with an offline mocked gateway."""
import ast, json, sys, types
from pathlib import Path
import numpy as np
from unittest.mock import patch
path = Path(__file__).with_name("submission-nb - Version 20.ipynb")
source = "".join(json.loads(path.read_text())["cells"][1]["source"])
ast.parse(source)
api = types.ModuleType("kaggle_evaluation")
server = types.ModuleType("smart_buildings_inference_server")
server.Policy = type("Policy", (), {"__init__": lambda self, *args: None})
api.smart_buildings_inference_server = server
with patch.dict(sys.modules, {"kaggle_evaluation": api}):
    env = {}
    exec(compile(source, str(path), "exec"), env)
Policy = env["TransferPolicy"]
metadata = dict(
    n_continuous_action=5, n_discrete_action=2,
    observation_field_names=[
        "VAV_0_zone_air_temperature_sensor",
        "VAV_0_zone_air_heating_temperature_setpoint_sensor",
        "VAV_0_zone_air_cooling_temperature_setpoint_sensor",
        "ahs_0_outside_air_temperature_sensor"],
    temp_indices=[0], pressure_indices=[1], recirculation_indices=[2],
    water_indices=[3], comfort_temp_window=[293, 297], eco_temp_window=[289, 301],
    morning_start_hour=8, evening_start_hour=18,
    ahu_zone_assignments={"0": ["room_0"]},
    zone_action_indices={"VAV_0": {
        "supply_air_damper_percentage_command": 4,
        "reheat_valve_percentage_command": 3}},
    observation_normalization_metadata={
        "zone_air_temperature_sensor": {"sample_mean": 295, "sample_std": 2},
        "outside_air_temperature_sensor": {"sample_mean": 290, "sample_std": 10}})
p = Policy(metadata, None)
for temp in [295, 296, 296.7, 297.2, 299, 291]:
    a = p.get_action(np.array([(temp-295)/2, 293, 297, 0]))
    assert a["continuous_action"].shape == (5,)
    assert a["discrete_action"].shape == (2,)
    assert a["continuous_action"].dtype == np.float32
    assert a["discrete_action"].dtype == np.int32
    assert np.isfinite(a["continuous_action"]).all()
    assert np.all(np.abs(a["continuous_action"]) <= 1)
assert p._outside_temperature(np.array([0, 0, 0, 0]), 3) == 290
p.reset()
assert not p._history
base = {"actions": {"temperature": 0}, "temp": 296.8, "low": 293, "high": 297}
assert p._sliding_status(dict(base))["smc_cooling"] == 0
p._history[0] = (296.0, 0)
assert p._sliding_status(dict(base))["smc_cooling"] > 0
p.reset()
assert p._sliding_status({**base, "temp": 300})["smc_cooling"] == 1
p.reset()
assert p._sliding_status({**base, "temp": 290})["smc_heating"] == 1
print("V20 synthetic contract and sliding surface checks passed")
