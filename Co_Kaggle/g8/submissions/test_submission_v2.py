"""Offline contract checks for submission_v2_adaptive_hvac.ipynb.

Run: python Co_Kaggle/g8/submissions/test_submission_v2.py
No Kaggle API, simulator, or network required. These checks DO NOT establish score.
"""
import json
import math
import sys
import types
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
nb = json.loads((root / "submission_v2_adaptive_hvac.ipynb").read_text())
assert nb["nbformat"] == 4
assert len(nb["cells"]) == 3
source = "".join(nb["cells"][1]["source"])
assert "SmartBuildingsInferenceServer" not in source  # gateway is separate final cell
assert "class TransferPolicy(" in source
assert "from kaggle_evaluation import smart_buildings_inference_server" in source
assert "SmartBuildingsInferenceServer(TransferPolicy).run()" in "".join(nb["cells"][2]["source"])

gateway_pkg = types.ModuleType("kaggle_evaluation")
gateway_mod = types.ModuleType("smart_buildings_inference_server")
class Policy:
    def __init__(self, building_meta_data, action_spec):
        pass
gateway_mod.Policy = Policy
gateway_pkg.smart_buildings_inference_server = gateway_mod
sys.modules["kaggle_evaluation"] = gateway_pkg
namespace = {}
exec(compile(source, "submission_v2_adaptive_hvac.ipynb", "exec"), namespace)
TransferPolicy = namespace["TransferPolicy"]


def test_case(num_ahus, temperature, outside):
    zones = num_ahus
    names = ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000"]
    names += [f"VAV_{i}_zone_air_temperature_sensor" for i in range(1, zones + 1)]
    names += [f"VAV_{i}_zone_air_heating_temperature_setpoint_sensor" for i in range(1, zones + 1)]
    names += [f"VAV_{i}_zone_air_cooling_temperature_setpoint_sensor" for i in range(1, zones + 1)]
    names += [f"ahs_ahu_{i}_outside_air_temperature_sensor" for i in range(1, num_ahus + 1)]
    obs = np.zeros(len(names), dtype=np.float64)
    obs[1] = 1.0
    obs[3] = 1.0
    obs[4:4 + zones] = (temperature - 295) / 10
    obs[4 + zones:4 + 2*zones] = 294
    obs[4 + 2*zones:4 + 3*zones] = 298
    obs[-num_ahus:] = outside
    m = {
        "n_discrete_action": num_ahus + 1,
        "n_continuous_action": 3*num_ahus + 2 + 2*zones,
        "temp_indices": [3*i for i in range(num_ahus)],
        "pressure_indices": [3*i+1 for i in range(num_ahus)] + [3*num_ahus+1],
        "recirculation_indices": [3*i+2 for i in range(num_ahus)],
        "water_indices": [3*num_ahus],
        "zone_action_indices": {
            f"VAV_{i}": {
                "supply_air_damper_percentage_command": 3*num_ahus+2+2*(i-1),
                "reheat_valve_percentage_command": 3*num_ahus+3+2*(i-1)
            } for i in range(1, zones+1)
        },
        "ahu_zone_assignments": {
            f"ahu_{i}": [f"room_{i}"] for i in range(1, num_ahus+1)
        },
        "observation_field_names": names,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 295.0, "sample_std": 10.0}
        },
        "comfort_temp_window": (294.0, 298.0),
        "eco_temp_window": (289.0, 303.0),
        "morning_start_hour": 8,
        "evening_start_hour": 18,
    }
    controller = TransferPolicy(m, None)
    for _ in range(3):
        action = controller.get_action(obs)
        c, d = action["continuous_action"], action["discrete_action"]
        assert c.shape == (m["n_continuous_action"],)
        assert d.shape == (m["n_discrete_action"],)
        assert c.dtype == np.float32 and d.dtype == np.int32
        assert np.all(np.isfinite(c)) and np.all((-1 <= c) & (c <= 1))
        assert np.all((d == 0) | (d == 1))
        if temperature < 294 or temperature > 298:
            assert np.all(d == 1), (num_ahus, temperature, d)
        else:
            assert np.all(d == 0), (num_ahus, temperature, d)
    return action


for n in (1, 2, 9):
    for t, outside in ((280.0, 275.0), (305.0, 290.0), (296.0, 296.0)):
        test_case(n, t, outside)
print("PASS: notebook JSON, gateway contract, 9 synthetic multi-step control cases")
print("NOTE: Kaggle simulator rollout/leaderboard performance not tested here.")
