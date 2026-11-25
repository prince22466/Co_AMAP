"""Fast shared-infrastructure tests (no simulator assets required)."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from . import evaluation, profiles, rollout


def echo_task(task):
    return task[1], task[-1], float(task[4]["weight"][0, 0])


class CommonTests(unittest.TestCase):
    def test_dispatch_order_seed_and_snapshot(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.)
        rows = rollout.run_jobs(None, ".", ["b", "a"], 12, 3, model, "eval", 42,
                                worker=echo_task)
        self.assertEqual(rows, [("b", 42, 2.), ("a", 43, 2.)])
        saved = rollout.snapshot(model)
        with torch.no_grad():
            model.weight.zero_()
        self.assertEqual(float(saved["weight"][0, 0]), 2.)

    def test_profile_restores_baseline(self):
        baseline = SimpleNamespace(**dict.fromkeys(profiles.PARAMETERS, 2.))
        controller = profiles.ProfileController(baseline)
        for i in range(profiles.N_ACTIONS):
            controller.select(i)
            controller.select(i)
            for name, factor in zip(profiles.PARAMETERS, profiles.PROFILES[i][1]):
                self.assertAlmostEqual(getattr(baseline, name), 2. * factor)
        controller.select(0)
        self.assertTrue(all(getattr(baseline, name) == 2. for name in profiles.PARAMETERS))

    def test_split_and_pair_order(self):
        scenarios = [f"building_train_{i}/weather_{j}"
                     for i in (1, 2, 4, 5, 7, 8) for j in range(5)]
        groups = evaluation.split_scenarios(scenarios)
        self.assertEqual([len(groups[k]) for k in evaluation.SPLIT], [20, 5, 5])
        self.assertEqual(len(set().union(*map(set, groups.values()))), 30)
        rows = [{"scenario": s, "steps": 12, "full_episode": False, "mean_loss": .01}
                for s in groups["validation"]]
        self.assertAlmostEqual(evaluation.comparison(rows, rows)["delta"], 0.)
        with self.assertRaises(ValueError):
            evaluation.comparison(rows, list(reversed(rows)))

    def test_no_reverse_model_dependencies(self):
        root = Path(__file__).parent
        for path in [root / n for n in ("profiles.py", "rollout.py", "evaluation.py")] + [
                root.parent / "p2" / n for n in ("model.py", "train.py")]:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertNotIn("p1", module.split("."), str(path))
                    if path.parent == root:
                        self.assertNotIn("p2", module.split("."), str(path))


if __name__ == "__main__":
    unittest.main()
