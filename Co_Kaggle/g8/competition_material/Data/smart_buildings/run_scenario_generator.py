#!/usr/bin/env python3
"""Turn building + weather configs into runnable simulation scenarios.

A "scenario" pairs one building YAML (``smart_buildings/buildings/*.yaml``)
with one weather preset (``smart_buildings/weather/*.yaml``) and materializes
everything the environment needs to run: a ``config.gin`` plus the
``floor_plan.npy`` and ``zone_map.npy`` arrays. Scenarios are written to
``smart_buildings/training_scenarios/<building>/<weather>/``.

The 30 shipped training scenarios are already generated there. They are a
PAIRING listed in ``smart_buildings/training_set.yaml``, not a cross product:
each of the 6 buildings in the set is paired with one weather strip per
station, and no strip is used twice. Running with no arguments rebuilds exactly
those 30.

Naming ``--buildings`` or ``--weathers`` switches to a cross product, which is
what you want for your own scenarios. Note the difference in size: the shipped
pairing is 30 scenarios, while all 8 shipped building definitions against all
48 shipped presets is 384.

Usage:
    # Rebuild the shipped training set (30 scenarios)
    python -m smart_buildings.run_scenario_generator

    # Your own cross product (2 x 2 = 4 scenarios)
    python -m smart_buildings.run_scenario_generator \
        --buildings building_train_1 building_train_2 \
        --weathers fairbanks_1 phoenix_5

    # Plot the floor plan + zone map of one already-generated scenario
    python -m smart_buildings.run_scenario_generator \
        --plot building_train_1 fairbanks_1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings.smart_control.utils.scenario_generator import (
    generate_scenario_from_config,
    list_weather_presets,
    load_scenario_from_parts,
)

DEFAULT_BUILDINGS_DIR = Path("smart_buildings/buildings")
DEFAULT_WEATHER_DIR = Path("smart_buildings/weather")
DEFAULT_SCENARIOS_DIR = Path("smart_buildings/training_scenarios")
# Defines the shipped set as an explicit pairing; see training_set_pairs().
TRAINING_SET_MANIFEST = Path("smart_buildings/training_set.yaml")


def resolve_buildings(patterns: list[str] | None) -> list[str]:
    """Return building YAML paths for the requested stems (default: all)."""
    if not patterns:
        return [str(p) for p in sorted(DEFAULT_BUILDINGS_DIR.glob("*.yaml"))]
    paths: list[str] = []
    for pat in patterns:
        candidate = Path(pat)
        if candidate.exists():
            paths.append(str(candidate))
        else:
            paths.append(str(DEFAULT_BUILDINGS_DIR / f"{Path(pat).stem}.yaml"))
    return paths


def resolve_weathers(names: list[str] | None) -> list[str]:
    """Return weather YAML paths for the requested presets (default: all)."""
    if not names:
        return [str(DEFAULT_WEATHER_DIR / f"{n}.yaml") for n in list_weather_presets()]
    paths: list[str] = []
    for name in names:
        candidate = Path(name)
        paths.append(str(candidate) if candidate.exists()
                     else str(DEFAULT_WEATHER_DIR / f"{Path(name).stem}.yaml"))
    return paths


def plot_scenario(building: str, weather: str, scenarios_dir: Path) -> None:
    """Show the floor plan and zone map of an already-generated scenario."""
    import matplotlib.pyplot as plt

    scenario_dir = scenarios_dir / building / weather
    floor_plan = np.load(scenario_dir / "floor_plan.npy")
    zone_map = np.load(scenario_dir / "zone_map.npy")
    print(f"Scenario: {scenario_dir}")
    print("Floor plan shape:", floor_plan.shape)
    print("Zone map shape:", zone_map.shape)

    for array, title in ((floor_plan, "Floor Plan"), (zone_map, "Zone Map")):
        plt.figure()
        plt.imshow(array)
        plt.title(f"{building} / {weather}: {title}")
        plt.colorbar()
    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--buildings",
        nargs="*",
        default=None,
        help="Building stems or YAML paths (default: all in buildings/).",
    )
    parser.add_argument(
        "--weathers",
        nargs="*",
        default=None,
        help=f"Weather presets (default: all): {', '.join(list_weather_presets())}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SCENARIOS_DIR,
        help="Base directory for generated scenarios.",
    )
    parser.add_argument(
        "--plot",
        nargs=2,
        metavar=("BUILDING", "WEATHER"),
        default=None,
        help="Plot the floor plan + zone map of one generated scenario and exit.",
    )
    return parser


def training_set_pairs() -> list[tuple[str, str]] | None:
    """The shipped (building, weather) pairs, or None if the manifest is absent.

    The training set is a pairing, not a cross product: each building sees one
    weather strip per station and no strip is reused, so the 6 buildings in
    the set give 30 scenarios rather than 384. Taking the cross product by
    default would generate many times the set and put each building through
    every station's whole year.
    """
    if not TRAINING_SET_MANIFEST.exists():
        return None
    with open(TRAINING_SET_MANIFEST) as f:
        manifest = yaml.safe_load(f) or {}
    return [(s["building"], s["weather"]) for s in manifest.get("scenarios", [])]


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.plot:
        plot_scenario(args.plot[0], args.plot[1], args.output_dir)
        return

    # No arguments means "rebuild the shipped set", which is the manifest.
    # Naming either axis explicitly means "I want my own scenarios", and that
    # is a cross product.
    explicit = args.buildings or args.weathers
    pairs: list[tuple[str, str]] | None = None
    if not explicit:
        pairs = training_set_pairs()

    if pairs is not None:
        jobs = [
            (str(DEFAULT_BUILDINGS_DIR / f"{b}.yaml"),
             str(DEFAULT_WEATHER_DIR / f"{w}.yaml"))
            for b, w in pairs
        ]
        print(f"Regenerating the shipped training set: {len(jobs)} scenario(s) "
              f"from {TRAINING_SET_MANIFEST} into {args.output_dir}")
    else:
        buildings = resolve_buildings(args.buildings)
        weathers = resolve_weathers(args.weathers)
        jobs = [(b, w) for b in buildings for w in weathers]
        print(f"Generating {len(buildings)} building(s) x {len(weathers)} "
              f"weather(s) = {len(jobs)} scenario(s) into {args.output_dir}")

    generated: list[str] = []
    for building, weather in jobs:
        config = load_scenario_from_parts(
            building, weather, output_base_dir=str(args.output_dir)
        )
        result = generate_scenario_from_config(config)
        out = result["output_dir"]
        generated.append(out)
        print(f"  {Path(building).stem} + {Path(weather).stem} -> {out}")

    print(f"\nDone. Generated {len(generated)} scenario(s).")


if __name__ == "__main__":
    os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "true")
    main()
