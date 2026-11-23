"""P0 baseline benchmark and throughput profiler.

Run from repository root:
  python -m Co_Kaggle.g8.rl.p0.benchmark --list
  python -m Co_Kaggle.g8.rl.p0.benchmark --steps 20
  python -m Co_Kaggle.g8.rl.p0.benchmark --all --steps 20 --workers 4
  python -m Co_Kaggle.g8.rl.p0.benchmark --full --workers 4

Only --full runs all 30 scenarios to actual simulator termination.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

from .controllers import VERSIONS
from .runner import run_scenario


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "competition_material" / "Data"


def discover_scenarios(data_root: Path) -> list[str]:
    scenarios_dir = data_root / "smart_buildings" / "training_scenarios"
    # as_posix(): manifest uses "building/weather"; Windows would yield "\".
    return sorted(path.parent.relative_to(scenarios_dir).as_posix()
                  for path in scenarios_dir.glob("building_*/*/config.gin"))


def verify_training_manifest(data_root: Path, scenarios: list[str]) -> None:
    """Reject missing, substituted, or additional shipped training scenarios."""
    import yaml  # Installed in the organizer's pinned simulator environment.

    manifest_path = data_root / "smart_buildings" / "training_set.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    pairs = manifest["scenarios"]
    expected = [
        f"{pair['building']}/{pair['weather']}" for pair in pairs
    ]
    if (len(expected) != 30 or len(set(expected)) != len(expected)
            or sorted(scenarios) != sorted(expected)):
        raise ValueError(
            "Training scenarios do not match the 30 unique "
            "training_set.yaml building/weather pairs"
        )


def aggregate(results: list[dict], *, full: bool) -> dict:
    grouped: dict[int, list[dict]] = {}
    for result in results:
        grouped.setdefault(result["version"], []).append(result)
    aggregate_by_version = {}
    for version, rows in sorted(grouped.items()):
        n = len(rows)
        aggregate_by_version[str(version)] = {
            "scenarios": n,
            "transitions": sum(row["steps"] for row in rows),
            "mean_loss": sum(row["mean_loss"] for row in rows) / n,
            "loss_components": {
                key: sum(row["loss_components"][key] for row in rows) / n
                for key in ("comfort", "energy", "carbon")
            },
            "training_set_score_comparable": (
                full and n == 30
                and all(row["full_30_day_episode"] for row in rows)
            ),
        }
    return aggregate_by_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--versions", nargs="+", type=int, choices=VERSIONS,
                        default=list(VERSIONS))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=1,
                        help="Number of training scenarios; defaults to 1")
    parser.add_argument("--all", action="store_true",
                        help="Use all shipped training scenarios")
    parser.add_argument("--full", action="store_true",
                        help="Run all 30 shipped scenarios to actual simulator termination")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent processes (never threads)")
    parser.add_argument("--list", action="store_true",
                        help="Print all available scenarios and exit")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "results",
                        help="Output directory for reports (default: %(default)s)")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    scenarios = discover_scenarios(data_root)
    if not scenarios:
        parser.error(f"No generated training scenarios found under {data_root}")
    if args.list:
        for item in scenarios:
            print(item)
        return
    if args.workers < 1 or args.steps < 1 or args.limit < 1:
        parser.error("--workers, --steps and --limit must be positive")
    if args.steps > 8640 and not args.full:
        parser.error("--steps cannot exceed 8640")
    if args.full:
        try:
            verify_training_manifest(data_root, scenarios)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            parser.error(f"Invalid training scenario set: {exc}")
    # None means run until official simulator termination, not 8640 steps.
    steps = None if args.full else args.steps
    selected = scenarios if args.all or args.full else scenarios[:args.limit]
    tasks = [(str(data_root), scenario, version, steps)
             for version in args.versions for scenario in selected]

    step_description = "until environment termination" if steps is None else f"{steps} steps/job"
    print(f"Running {len(tasks)} jobs, {step_description}, "
          f"{args.workers} worker process(es)", flush=True)
    started = perf_counter()
    results: list[dict] = []
    if args.workers == 1:
        for task in tasks:
            row = run_scenario(task)
            results.append(row)
            print(f"V{row['version']} {row['scenario']}: "
                  f"{row['mean_loss']:.6f}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=get_context("spawn")
        ) as pool:
            futures = [pool.submit(run_scenario, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                print(f"V{row['version']} {row['scenario']}: "
                      f"{row['mean_loss']:.6f}", flush=True)

    wall_seconds = perf_counter() - started
    results.sort(key=lambda row: (row["version"], row["scenario"]))
    aggregates = aggregate(results, full=args.full)
    report = {
        "data_root": str(data_root),
        "full": args.full,
        "requested_steps": steps,
        "workers": args.workers,
        "wall_seconds": wall_seconds,
        "aggregate_steps_per_second": sum(row["steps"] for row in results)
        / max(wall_seconds, 1e-9),
        "aggregates": aggregates,
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out / "per_scenario.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=[
            "version", "scenario", "steps", "configured_episode_steps",
            "episode_terminated", "mean_reward", "mean_loss",
            "comfort_loss", "energy_loss", "carbon_loss",
            "policy_ms_per_step", "simulator_ms_per_step",
            "diagnostics_ms_per_step",
        ])
        writer.writeheader()
        for row in results:
            writer.writerow({
                **{key: row[key] for key in (
                    "version", "scenario", "steps", "configured_episode_steps",
                    "episode_terminated", "mean_reward", "mean_loss",
                    "policy_ms_per_step", "simulator_ms_per_step",
                    "diagnostics_ms_per_step"
                )},
                **{f"{key}_loss": row["loss_components"][key]
                   for key in ("comfort", "energy", "carbon")},
            })
    for version, data in aggregates.items():
        label = ("30-day TRAINING score" if data["training_set_score_comparable"]
                 else "partial diagnostic, NOT leaderboard score")
        print(f"V{version}: {data['mean_loss']:.6f} [{label}]")
        print(f"  weighted components: {data['loss_components']}")
    print(f"Aggregate throughput: {report['aggregate_steps_per_second']:.2f} "
          "steps/s across workers")
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
