"""Offline regression for the exact V28 submission cell, using the real P3 weights."""
import base64
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Co_Kaggle.g8.rl.common.profiles import ProfileController
from Co_Kaggle.g8.rl.p0.controllers import LocalPolicy, load_controller, validate_action
from Co_Kaggle.g8.rl.p3.model import QPolicy, signature

G8 = ROOT / "Co_Kaggle/g8"
CHECKPOINT = G8 / "rl/p3/results/run1/best.pt"
NOTEBOOK = Path(__file__).with_name("submission-nb - Version 28.ipynb")


def execute_notebook():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert [c["cell_type"] for c in notebook["cells"]] == ["markdown", "code"]
    code = "".join(notebook["cells"][1]["source"])
    calls = []

    class Gateway:
        def __init__(self, policy):
            self.policy = policy

        def run(self):
            calls.append(self.policy)

    package = ModuleType("kaggle_evaluation")
    package.smart_buildings_inference_server = SimpleNamespace(
        Policy=LocalPolicy, SmartBuildingsInferenceServer=Gateway)
    namespace = {"__name__": "v28_submission"}
    with patch.dict(sys.modules, {"kaggle_evaluation": package}):
        exec(compile(code, str(NOTEBOOK), "exec"), namespace)
    assert calls == [namespace["TransferPolicy"]]
    return namespace


def fixture():
    fields = [
        "hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000",
        "VAV_1_zone_air_temperature_sensor",
        "VAV_1_zone_air_heating_temperature_setpoint_sensor",
        "VAV_1_zone_air_cooling_temperature_setpoint_sensor",
        "ahs_1_outside_air_temperature_sensor",
    ]
    meta = {
        "n_continuous_action": 7, "n_discrete_action": 2,
        "observation_field_names": fields,
        "temp_indices": [0], "pressure_indices": [1, 6],
        "recirculation_indices": [2], "water_indices": [5],
        "comfort_temp_window": [294., 298.], "eco_temp_window": [288., 303.],
        "morning_start_hour": 8., "evening_start_hour": 18.,
        "ahu_zone_assignments": {"1": ["room_1"]},
        "zone_action_indices": {"VAV_1": {
            "supply_air_damper_percentage_command": 3,
            "reheat_valve_percentage_command": 4}},
    }
    spec = {"continuous_action": SimpleNamespace(shape=(7,)),
            "discrete_action": SimpleNamespace(shape=(2,))}
    return meta, spec


def observations():
    rng = np.random.default_rng(42)
    for t in range(72):
        angle = 2 * np.pi * t / 288
        yield np.array([np.sin(angle), np.cos(angle), 0., 1.,
                        rng.uniform(288., 305.), 294., 298.,
                        rng.uniform(270., 310.)], dtype=np.float32)


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = execute_notebook()
        cls.state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)

    def test_embedded_checkpoint_and_architecture(self):
        self.assertEqual(base64.b64decode(self.nb["CHECKPOINT_BASE64"]),
                         CHECKPOINT.read_bytes())
        self.assertEqual(self.state["signature"], signature())
        self.assertEqual(self.nb["signature"](), signature())
        model, interval = self.nb["load_p3_checkpoint"]()
        self.assertEqual(interval, self.state["interval"])
        self.assertEqual(interval, 12)
        self.assertEqual(float(model.epsilon), 0.)
        self.assertFalse(model.training)
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, self.state["model"][k], rtol=0, atol=0)
        reference = QPolicy()
        reference.load_state_dict(self.state["model"])
        x = torch.randn(64, 25, generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(model(x)[0], reference(x)[0], rtol=0, atol=0)

    def test_action_parity_and_reset(self):
        meta, spec = fixture()
        deployed = self.nb["TransferPolicy"](meta, spec)
        ref = ProfileController(load_controller(23)(meta, spec))
        model = QPolicy()
        model.load_state_dict(self.state["model"])
        model.eval()
        model.epsilon.zero_()
        interval = self.state["interval"]
        for episode in range(2):
            deployed.reset()
            ref.baseline.reset()
            ref.select(0)
            self.assertEqual(deployed.step, 0)
            self.assertEqual(deployed.controller.profile, 0)
            for t, obs in enumerate(observations()):
                if t % interval == 0:
                    expected_features = ref.features(obs)
                    np.testing.assert_array_equal(deployed.controller.features(obs),
                                                  expected_features)
                    with torch.no_grad():
                        logits, _ = model(torch.from_numpy(expected_features))
                    ref.select(int(logits.argmax()))
                expected = validate_action(ref.get_action(obs), spec)
                actual = validate_action(deployed.get_action(obs), spec)
                self.assertEqual(deployed.controller.profile, ref.profile)
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])
                self.assertEqual(deployed.step, t + 1)

    def test_macro_interval_and_reset_with_forced_choices(self):
        meta, spec = fixture()
        policy = self.nb["TransferPolicy"](meta, spec)
        obs = next(observations())
        # Force different choices at successive decisions to exercise timing,
        # independently of whether these synthetic observations activate P3.
        outputs = iter([5, 2, 0])
        calls = []

        def select(features):
            calls.append(features.clone())
            scores = torch.zeros(7)
            scores[next(outputs)] = 1.
            return scores, torch.tensor(0.)

        with patch.object(policy.model, "forward", side_effect=select):
            for t in range(24):
                validate_action(policy.get_action(obs), spec)
                self.assertEqual(policy.controller.profile, 5 if t < 12 else 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(float(calls[0][-7]), 1.)
            self.assertEqual(float(calls[1][-2]), 1.)
            policy.reset()
            self.assertEqual(policy.controller.profile, 0)
            policy.get_action(obs)
            self.assertEqual(len(calls), 3)
            self.assertEqual(float(calls[2][-7]), 1.)


if __name__ == "__main__":
    unittest.main()
