"""Regression checks for V34's fixed residual and single-cell layout."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent

def notebook(version):
    return json.loads((ROOT / f"submission-nb - Version {version}.ipynb").read_text(encoding="utf-8"))

class TestVersion34(unittest.TestCase):
    def test_layout_and_baseline(self):
        nb = notebook(34)
        self.assertEqual([c["cell_type"] for c in nb["cells"]], ["markdown", "code"])
        source = "".join(nb["cells"][1]["source"])
        tree = ast.parse(source)
        baseline = ast.parse("".join(notebook(23)["cells"][1]["source"]))
        # Only the module docstring changes in the inherited baseline.
        self.assertEqual(
            [ast.dump(x) for x in tree.body[1:len(baseline.body)]],
            [ast.dump(x) for x in baseline.body[1:]],
        )
        self.assertEqual(ast.unparse(tree.body[-1]),
                         "smart_buildings_inference_server.SmartBuildingsInferenceServer(ResidualPolicy).run()")

    def test_residual_matches_pilot(self):
        tree = ast.parse("".join(notebook(34)["cells"][1]["source"]))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ResidualPolicy")
        original = np.array([0.2, -0.95, 0.6, 0.3, -0.2], dtype=np.float32)
        discrete = np.array([1, 1, 0, 1], dtype=np.int32)
        class TransferPolicy:
            def get_action(self, obs):
                return {"continuous_action": original, "discrete_action": discrete}
        scope = {"TransferPolicy": TransferPolicy, "np": np}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "<residual>", "exec"), scope)
        policy = scope["ResidualPolicy"]()
        policy.ahus = [
            {"run": 0, "supply": 0},
            {"run": 1, "supply": 1},
            {"run": 2, "supply": 2},
            {"run": 3, "supply": None},
        ]
        result = policy.get_action(None)
        np.testing.assert_allclose(result["continuous_action"], [0.1, -1, 0.6, 0.3, -0.2])
        np.testing.assert_allclose(original, [0.2, -0.95, 0.6, 0.3, -0.2])
        np.testing.assert_array_equal(result["discrete_action"], discrete)
        self.assertEqual(result["continuous_action"].dtype, np.float32)
        self.assertEqual(result["discrete_action"].dtype, np.int32)

if __name__ == "__main__":
    unittest.main()
