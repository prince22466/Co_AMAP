"""Offline Version 15 gain optimization against the organizer's real simulator.

Use in an environment with the official smart-buildings starter kit installed.
No part of this module is imported by the Kaggle submission notebook.

  pip install cma scikit-optimize
  python tune_version15.py --starter-root /path/to/starter-kit --method cmaes
  python tune_version15.py --starter-root /path/to/starter-kit --method bayes

Optimization/selection use equal-weight scenario mean per-step loss. Short
episodes are SEARCH PROXIES, not comparable to official 8,640-step scores.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "submission-nb - Version 15.ipynb"
# Hold-out buildings are never used to fit the optimizer.
DEFAULT_TRAIN = (
    "building_train_1/atlanta_2",
    "building_train_2/denver_8",
    "building_train_4/fairbanks_4",
    "building_train_5/phoenix_1",
)
DEFAULT_VALIDATION = (
    "building_train_7/singapore_6",
    "building_train_8/fairbanks_8",
)


def load_notebook_policy(notebook_path: Path = NOTEBOOK):
    """Execute the EXACT inference cell with a stub Kaggle base class.

    The official local simulator expects __init__(meta, spec) and get_action
    but does not provide Kaggle's inference server package.
    """
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if len(notebook["cells"]) < 3:
        raise ValueError("Expected markdown, policy code, and gateway cells")
    source = "".join(notebook["cells"][1]["source"])
    package = types.ModuleType("kaggle_evaluation")
    gateway = types.ModuleType("smart_buildings_inference_server")
    gateway.Policy = type("Policy", (), {
        "__init__": lambda self, metadata, action_spec: None
    })
    package.smart_buildings_inference_server = gateway
    sys.modules["kaggle_evaluation"] = package
    sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = gateway
    namespace = {"__name__": "_version15_notebook_policy"}
    exec(compile(source, str(notebook_path), "exec"), namespace)
    return notebook, namespace["TransferPolicy"], namespace["PARAMETER_BOUNDS"], namespace["DEFAULT_PARAMETERS"]


def normalized_vector(params, bounds):
    """Convert a physical parameter dictionary to a unit-cube search point."""
    return [(float(params[name]) - lo) / (hi - lo)
            for name, (lo, hi) in bounds.items()]


def physical_parameters(point, bounds):
    """Project an optimizer suggestion to safe physical controller bounds."""
    if len(point) != len(bounds):
        raise ValueError("Optimizer parameter count differs from controller")
    result = {}
    for (name, (lo, hi)), component in zip(bounds.items(), point):
        if not math.isfinite(float(component)):
            raise ValueError("Non-finite candidate in optimizer")
        result[name] = float(lo + (hi - lo) * np.clip(component, 0.0, 1.0))
    return result


def scenario_paths(starter_root: Path, names):
    """Only explicit public scenario names; reject missing or escaped paths."""
    root = (starter_root / "smart_buildings" / "training_scenarios").resolve()
    found = []
    for name in names:
        parts = name.split("/")
        if (len(parts) != 2 or
                any(not re.fullmatch(r"[A-Za-z0-9_]+", p) for p in parts)):
            raise ValueError("Invalid public scenario id: " + name)
        candidate = root.joinpath(*parts).resolve()
        if not candidate.is_relative_to(root) or not (candidate / "config.gin").is_file():
            raise FileNotFoundError("Missing generated scenario: " + str(candidate))
        found.append(candidate)
    if len(set(found)) != len(found):
        raise ValueError("Duplicate scenarios are not allowed")
    if not found:
        raise ValueError("At least one scenario is required")
    return found


def check_split(train, validation):
    """Avoid leakage: a training building may not appear in validation."""
    train_buildings = {p.parent.name for p in train}
    valid_buildings = {p.parent.name for p in validation}
    overlap = train_buildings & valid_buildings
    if overlap:
        raise ValueError("Training/validation building overlap: " + str(overlap))


def simulate_episode(policy_class, parameters, config_dir: Path, steps: int):
    """Run real local organizer dynamics without plotting or video files."""
    from smart_buildings.evaluate import (
        create_building_meta_data, load_hybrid_environment, validate_action,
    )
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker

    env = load_hybrid_environment(config_dir / "config.gin")
    # Match organizer's evaluate_policy_for_config() normalization and options.
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    metadata = create_building_meta_data(env, config_dir)
    candidate_class = type("TunedVersion15", (policy_class,),
                           {"PARAMETERS": dict(parameters)})
    policy = candidate_class(metadata, env.action_spec())

    tracker.reset()
    observation = tracker.last_obs
    for _ in range(steps):
        action = validate_action(policy.get_action(observation), env.action_spec())
        observation, _, terminated, truncated, _ = tracker.step(action)
        if terminated or truncated:
            break
    rewards = np.asarray(tracker.rewards, dtype=float)
    if len(rewards) != steps or not np.isfinite(rewards).all():
        raise RuntimeError(f"Invalid/incomplete episode: {config_dir} "
                           f"got {len(rewards)}/{steps} steps")
    return -float(np.mean(rewards))


class ScenarioObjective:
    """Cached equal-weight mean per-scenario loss with an append-only audit."""

    def __init__(self, policy_class, bounds, scenarios, steps, trace_path):
        self.policy_class = policy_class
        self.bounds = bounds
        self.scenarios = tuple(scenarios)
        self.steps = steps
        self.trace_path = trace_path
        self.cache = {}
        self.trials = []

    def evaluate(self, point):
        params = physical_parameters(point, self.bounds)
        key = tuple(round(params[name], 9) for name in self.bounds)
        if key in self.cache:
            return self.cache[key]
        losses = {}
        for config in self.scenarios:
            name = config.parent.name + "/" + config.name
            losses[name] = simulate_episode(
                self.policy_class, params, config, self.steps)
        score = float(np.mean(list(losses.values())))
        if not math.isfinite(score):
            raise RuntimeError("Non-finite training objective")
        trial = {"trial": len(self.trials), "loss": score,
                 "parameters": params, "scenario_losses": losses}
        self.trials.append(trial)
        self.cache[key] = score
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trial, sort_keys=True) + "\n")
        print(f"trial={trial['trial']:03d} mean_loss={score:.7f}", flush=True)
        return score


def run_search(method: str, evaluations: int, seed: int, initial, objective):
    """Reproducible optimizer on [0,1]^dimensions; baseline counts as trial 1."""
    if evaluations < 2:
        raise ValueError("Require at least two evaluations including baseline")
    initial = [float(x) for x in initial]
    dimension = len(initial)
    baseline_score = objective.evaluate(initial)

    if method == "cmaes":
        try:
            import cma
        except ImportError as error:
            raise RuntimeError("CMA-ES requires pip install cma") from error
        population = min(8, max(4, int(4 + 2 * np.log(dimension))))
        optimizer = cma.CMAEvolutionStrategy(
            initial, 0.20,
            {"bounds": [0.0, 1.0], "seed": seed,
             "popsize": population, "verbose": -9, "verb_disp": 0})
        completed = 1
        while completed < evaluations:
            candidates = optimizer.ask()
            chosen = candidates[:evaluations - completed]
            losses = [objective.evaluate(point) for point in chosen]
            completed += len(chosen)
            # CMA-ES update requires a complete population; never lie to tell.
            if len(chosen) == len(candidates):
                optimizer.tell(candidates, losses)
    elif method == "bayes":
        try:
            from skopt import Optimizer
            from skopt.space import Real
        except ImportError as error:
            raise RuntimeError(
                "Bayesian optimization requires pip install scikit-optimize") from error
        optimizer = Optimizer(
            dimensions=[Real(0.0, 1.0) for _ in range(dimension)],
            base_estimator="GP", acq_func="EI",
            n_initial_points=min(8, max(3, evaluations // 3)),
            random_state=seed)
        optimizer.tell(initial, baseline_score)
        for _ in range(evaluations - 1):
            proposal = optimizer.ask()
            score = objective.evaluate(proposal)
            optimizer.tell(proposal, score)
    else:
        raise ValueError("method must be cmaes or bayes")

    best_trial = min(objective.trials, key=lambda item: item["loss"])
    return baseline_score, best_trial


def validate(policy_class, candidates, scenarios, steps):
    """Evaluate candidates on held-out buildings, never used by the search."""
    outcomes = []
    for params in candidates:
        individual = {
            config.parent.name + "/" + config.name:
                simulate_episode(policy_class, params, config, steps)
            for config in scenarios
        }
        outcomes.append({"loss": float(np.mean(list(individual.values()))),
                         "scenario_losses": individual})
    return outcomes


def export_notebook(notebook, params, output: Path):
    """Embed chosen coefficients: Kaggle inference never reads tuning files."""
    updated = json.loads(json.dumps(notebook))
    source = "".join(updated["cells"][1]["source"])
    replacement = "DEFAULT_PARAMETERS = " + repr(dict(params))
    source, count = re.subn(r"(?m)^DEFAULT_PARAMETERS = [^\n]*$",
                            lambda _: replacement, source)
    if count != 1:
        raise ValueError("Missing or duplicated DEFAULT_PARAMETERS export line")
    updated["cells"][1]["source"] = source.splitlines(keepends=True)
    updated["cells"][0]["source"] = (
        ["> Exported by tune_version15.py after separate-building validation. "
         "Training-set/holdout proxy losses do not guarantee leaderboard improvement.\n\n"]
        + updated["cells"][0]["source"])
    for cell in updated["cells"]:
        cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs", [])
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(updated, indent=1) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter-root", required=True, type=Path,
                        help="Root of installed official smart-buildings starter kit")
    parser.add_argument("--method", required=True, choices=["cmaes", "bayes"])
    parser.add_argument("--evaluations", type=int, default=25,
                        help="Total training evaluations, including the default")
    parser.add_argument("--steps", type=int, default=2016,
                        help="Steps per scenario (8640 for score-comparable full episodes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", nargs="+", default=list(DEFAULT_TRAIN),
                        metavar="BUILDING/WEATHER")
    parser.add_argument("--validation", nargs="+", default=list(DEFAULT_VALIDATION),
                        metavar="BUILDING/WEATHER")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "version15_tuning")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 1 <= args.steps <= 8640:
        raise ValueError("steps must be in 1..8640")
    # The public simulator Python package lives in the starter-kit repository.
    sys.path.insert(0, str(args.starter_root.resolve()))
    train = scenario_paths(args.starter_root, args.train)
    heldout = scenario_paths(args.starter_root, args.validation)
    check_split(train, heldout)
    notebook, policy, bounds, defaults = load_notebook_policy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.out_dir / "trials.jsonl"
    if trace_path.exists():
        raise FileExistsError("Refusing to overwrite previous tuning trace: " +
                              str(trace_path))
    objective = ScenarioObjective(policy, bounds, train, args.steps, trace_path)
    baseline, best = run_search(
        args.method, args.evaluations, args.seed,
        normalized_vector(defaults, bounds), objective)

    # Validation is a gate, not an optimizer input. Fall back if it worsens.
    base_valid, candidate_valid = validate(
        policy, [defaults, best["parameters"]], heldout, args.steps)
    improvement = candidate_valid["loss"] < base_valid["loss"] - 1e-9
    selected = best["parameters"] if improvement else defaults

    report = {
        "method": args.method, "seed": args.seed, "evaluations": args.evaluations,
        "steps": args.steps, "score_comparable_to_official": args.steps == 8640,
        "train": list(args.train), "validation": list(args.validation),
        "baseline_train_loss": baseline, "best_train_trial": best,
        "baseline_validation": base_valid, "candidate_validation": candidate_valid,
        "accepted_on_holdout": improvement,
        "selected_parameters": selected,
        "note": "No official hidden score; validate full 8640-step/all-scenario before submission.",
    }
    (args.out_dir / "best_params.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    exported = args.out_dir / "submission-nb - Version 15.tuned.ipynb"
    export_notebook(notebook, selected, exported)
    print(f"Holdout improvement accepted: {improvement}")
    print(f"Training default={baseline:.7f}, best={best['loss']:.7f}")
    print(f"Validation default={base_valid['loss']:.7f}, "
          f"candidate={candidate_valid['loss']:.7f}")
    print(f"Exported notebook: {exported}")


if __name__ == "__main__":
    main()
