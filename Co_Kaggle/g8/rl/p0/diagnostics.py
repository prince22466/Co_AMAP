"""P0.1 targeted diagnostics; never changes the controller or scoring.

Run from repo root:
  python -m Co_Kaggle.g8.rl.p0.diagnostics --steps 288
  python -m Co_Kaggle.g8.rl.p0.diagnostics --scenarios building_train_1/fairbanks_1 --versions 23
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from .benchmark import default_data_root, discover_scenarios
from .controllers import VERSIONS, load_controller, validate_action
from .diagnostics_core import DiagnosticAccumulator
from .runner import simulator_working_directory

DEFAULT_SCENARIOS = (
    "building_train_1/fairbanks_1",  # High-loss, heating-dominated.
    "building_train_2/phoenix_6",    # Hot-weather case where V23 loses to V7.
    "building_train_7/singapore_6",  # Humid-weather case where V25 is competitive.
)


def run_diagnostic(data_root: Path, scenario: str, version: int,
                   steps: int | None, out: Path) -> dict:
    """Collect exact post-step reward responses with no full-grid snapshots."""
    config = data_root / "smart_buildings" / "training_scenarios" / scenario / "config.gin"
    if not config.is_file():
        raise FileNotFoundError(config)

    cache = Path(tempfile.gettempdir()) / "g8_rl_p01_cache" / str(os.getpid())
    for key, leaf in (("NUMBA_CACHE_DIR", "numba"),
                      ("MPLCONFIGDIR", "matplotlib"),
                      ("XDG_CACHE_HOME", "xdg")):
        folder = cache / leaf
        folder.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(folder))
    if str(data_root) not in sys.path:
        sys.path.insert(0, str(data_root))

    with simulator_working_directory(data_root):
        from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
        env = load_hybrid_environment(config)
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0  # Gateway parity, including typo.
        meta = create_building_meta_data(env, config.parent)
        field_names = list(meta["observation_field_names"])
        comfort_index = field_names.index("comfort_mode_now")
        spec = env.action_spec()
        policy = load_controller(version)(meta, spec)
        obs, _ = env.reset()
        tracker = DiagnosticAccumulator(meta)
        weights = env.reward_function.weights
        configured = int(env._num_timesteps_in_episode)
        if steps is None and configured != 8640:
            raise ValueError("Unexpected full episode length")
        limit = configured + 2 if steps is None else steps
        terminated = False

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = None
            for i in range(limit):
                # Timestamp and comfort flag correspond to the PRE-action state.
                timestamp = env.current_simulation_timestamp
                comfort_mode = bool(float(obs[comfort_index]) >= 0.5)
                action = validate_action(policy.get_action(obs), spec)
                obs, reward, term, trunc, _ = env.step(action)
                if not math.isfinite(float(reward)):
                    raise ValueError("Rejected or nonfinite reward")
                reward_info, response = env.get_reward_info_and_response()
                row = tracker.record(
                    timestamp=timestamp, comfort_mode=comfort_mode,
                    action=action, reward_info=reward_info, response=response,
                    reward=reward, weights=weights,
                )
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if term or trunc:
                    terminated = True
                    if steps is not None and i + 1 != steps:
                        raise RuntimeError("Episode terminated before requested short trajectory")
                    break

        if steps is None and not terminated:
            raise RuntimeError("Full episode did not terminate")
        summary = tracker.summary()
        summary.update({
            "scenario": scenario, "version": version,
            "episode_terminated": terminated,
            "full_30_day_episode": steps is None and terminated,
            "configured_episode_steps": configured,
            "csv": str(out),
        })
        summary_path = out.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS),
                        help="Scenario IDs as building/weather, from the training manifest")
    parser.add_argument("--versions", nargs="+", type=int, choices=VERSIONS,
                        default=list(VERSIONS))
    parser.add_argument("--steps", type=int, default=288,
                        help="Short-trajectory step limit; default is one day (288)")
    parser.add_argument("--full", action="store_true",
                        help="Run to simulator termination (slow); overrides --steps")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "diagnostics")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    available = set(discover_scenarios(data_root))
    unknown = set(args.scenarios) - available
    if unknown:
        parser.error("Unknown training scenario(s): " + ", ".join(sorted(unknown)))
    if not args.full and not 1 <= args.steps <= 8640:
        parser.error("--steps must be between 1 and 8640; use --full for completion")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Duplicate scenario")
    if len(set(args.versions)) != len(args.versions):
        parser.error("Duplicate controller version")

    results = []
    for scenario in args.scenarios:
        for version in args.versions:
            path = args.out.resolve() / ("v%d" % version) / (
                scenario.replace("/", "__") + ".csv"
            )
            print("P0.1 V%d %s" % (version, scenario), flush=True)
            summary = run_diagnostic(
                data_root, scenario, version, None if args.full else args.steps, path
            )
            results.append({
                "scenario": scenario, "version": version,
                "steps": summary["steps"], "mean_loss": summary["mean_loss"],
                "full_30_day_episode": summary["full_30_day_episode"],
                "summary_path": str(path.with_suffix(".json")),
            })
            print("  loss=%.6f (%s)" % (
                summary["mean_loss"],
                "completed training episode" if args.full else "partial diagnostic, NOT leaderboard",
            ), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(
        json.dumps({"runs": results}, indent=2) + "\n", encoding="utf-8"
    )
    print("Saved diagnostic CSV/JSON files to", args.out.resolve())


if __name__ == "__main__":
    main()
