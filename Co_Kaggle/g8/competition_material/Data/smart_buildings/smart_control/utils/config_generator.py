"""Config generator for creating varied simulation scenarios."""

import os
import random
from dataclasses import dataclass, field

import yaml

BUILDING_SHAPES = [
    "rectangle",
    "trapezoid",
    "h_shape",
    "t_shape",
    "pentagon",
    "oval",
    "u_shape",
    "parallelogram",
    "semicircle",
    "triangle",
    "composite",
]

COMPOSITE_STYLES = ["architectural", "random"]

# Climate presets: (low_temp_K, high_temp_K)
CLIMATES = {
    "hot_summer": (305.0, 309.0),  # 32C-36C
    "mild_summer": (300.0, 305.0),  # 27C-32C
    "cold_winter": (280.0, 285.0),  # 7C-12C
    "mild_winter": (285.0, 290.0),  # 12C-17C
    "tropical": (297.0, 308.0),  # 24C-35C
    "temperate": (283.0, 298.0),  # 10C-25C
}


@dataclass
class ConfigRange:
    """Defines a range for random sampling."""

    min_val: float
    max_val: float

    def sample(self) -> float:
        return random.uniform(self.min_val, self.max_val)

    def sample_int(self) -> int:
        return random.randint(int(self.min_val), int(self.max_val))


@dataclass
class DatasetConfig:
    """Configuration for generating a dataset of scenarios."""

    # Output
    output_dir: str = "dataset"
    num_configs: int = 100

    # Building variations
    shapes: list[str] = field(default_factory=lambda: BUILDING_SHAPES)
    width_range: tuple[int, int] = (200, 800)
    height_range: tuple[int, int] = (150, 500)
    num_floors_range: tuple[int, int] = (1, 5)
    min_room_size_range: tuple[int, int] = (30, 60)
    shape_ratio_range: tuple[float, float] = (0.28, 0.40)  # For H/T/U shapes
    composite_coverage_range: tuple[float, float] = (0.65, 0.85)  # For composite
    composite_styles: list[str] = field(default_factory=lambda: ["architectural"])

    # AHU variations
    num_ahus_range: tuple[int, int] = (1, 6)

    # Occupancy variations
    occupants_per_zone_range: tuple[int, int] = (1, 10)

    # Simulation
    num_days_range: tuple[int, int] = (30, 30)

    # Seeds
    base_seed: int = 0


def _sample(val, rng):
    """Sample from value - if tuple, treat as range; if list, choose random; else return as-is."""
    if isinstance(val, tuple) and len(val) == 2:
        if isinstance(val[0], int) and isinstance(val[1], int):
            return rng.randint(val[0], val[1])
        return rng.uniform(val[0], val[1])
    elif isinstance(val, list):
        return rng.choice(val)
    return val


def generate_config(
    name: str,
    seed: int,
    # Building - can be value, tuple (min, max), or list [choices]
    building_shape="rectangle",
    width=400,
    height=300,
    num_floors=2,
    min_room_size=40,
    # max_room_size is min_room_size + this, so the two cannot cross however
    # they are sampled.
    room_size_span=20,
    # Shape parameters (for h_shape, t_shape, u_shape, composite)
    shape_ratio=0.33,  # Controls thick vs thin parts (0.1-0.5)
    composite_coverage=0.6,  # For composite shape: min canvas coverage (0.0-1.0)
    composite_style="architectural",  # "architectural" or "random"
    # AHU
    num_ahus=2,
    # Occupancy
    occupants_per_zone=1,
    # Simulation
    num_days_in_episode=30,
    start_timestamp: str = "2023-07-10 06:00",
) -> dict:
    """Generate a building config dict (no weather - use weather presets separately).

    All numeric params can be:
        - A single value: width=400
        - A tuple range: width=(300, 600) -> samples random int
        - A list of choices: building_shape=["rectangle", "h_shape"] -> picks one

    Shape parameters:
        - shape_ratio: For H/T/U shapes, controls thick vs thin parts (0.1-0.5)
        - composite_coverage: For composite shape, minimum canvas coverage (0.0-1.0)
        - composite_style: For composite shape, "architectural" or "random"
    """
    rng = random.Random(seed)

    # Sample all values
    building_shape = _sample(building_shape, rng)
    width = _sample(width, rng)
    height = _sample(height, rng)
    num_floors = _sample(num_floors, rng)
    min_room_size = _sample(min_room_size, rng)
    room_size_span = _sample(room_size_span, rng)
    shape_ratio = _sample(shape_ratio, rng)
    composite_coverage = _sample(composite_coverage, rng)
    composite_style = _sample(composite_style, rng)
    num_ahus = _sample(num_ahus, rng)
    occupants_per_zone = _sample(occupants_per_zone, rng)
    num_days_in_episode = _sample(num_days_in_episode, rng)

    config = {
        "name": name,
        "seed": seed,
        "floor_plan": {
            "width": width,
            "height": height,
            "num_floors": num_floors,
            "min_room_size": min_room_size,
            "max_room_size": min_room_size + room_size_span,
            "building_shape": building_shape,
            "shape_ratio": shape_ratio,
            "composite_coverage": composite_coverage,
            "composite_style": composite_style,
        },
        "ahu": {
            "num_ahus": num_ahus,
        },
        "simulation": {
            "start_timestamp": start_timestamp,
            "num_days_in_episode": num_days_in_episode,
            "occupants_per_zone": occupants_per_zone,
        },
    }

    return config


def generate_random_config(
    index: int,
    dataset_config: DatasetConfig,
) -> dict:
    """Generate a random building config based on dataset config ranges.

    Note: Weather is handled separately via weather presets in weather/ directory.
    """
    seed = dataset_config.base_seed + index
    random.seed(seed)

    # Sample random values
    shape = random.choice(dataset_config.shapes)
    width = random.randint(*dataset_config.width_range)
    height = random.randint(*dataset_config.height_range)
    num_floors = random.randint(*dataset_config.num_floors_range)
    min_room_size = random.randint(*dataset_config.min_room_size_range)
    shape_ratio = random.uniform(*dataset_config.shape_ratio_range)
    composite_coverage = random.uniform(*dataset_config.composite_coverage_range)
    composite_style = random.choice(dataset_config.composite_styles)
    num_ahus = random.randint(*dataset_config.num_ahus_range)
    occupants = random.randint(*dataset_config.occupants_per_zone_range)
    num_days = random.randint(*dataset_config.num_days_range)

    name = f"env_{index:04d}_{shape}"

    return generate_config(
        name=name,
        seed=seed,
        building_shape=shape,
        width=width,
        height=height,
        num_floors=num_floors,
        min_room_size=min_room_size,
        shape_ratio=shape_ratio,
        composite_coverage=composite_coverage,
        composite_style=composite_style,
        num_ahus=num_ahus,
        occupants_per_zone=occupants,
        num_days_in_episode=num_days,
    )


def generate_dataset_configs(
    dataset_config: DatasetConfig | None = None, **kwargs
) -> list[dict]:
    """Generate a list of varied configs for dataset creation.

    Args:
        dataset_config: DatasetConfig instance, or None to use defaults
        **kwargs: Override any DatasetConfig fields

    Returns:
        List of config dicts
    """
    if dataset_config is None:
        dataset_config = DatasetConfig(**kwargs)

    configs = []
    for i in range(dataset_config.num_configs):
        config = generate_random_config(i, dataset_config)
        configs.append(config)

    return configs


def save_configs(
    configs: list[dict],
    output_dir: str = "buildings",
) -> list[str]:
    """Save building configs to YAML files.

    Note: Weather is handled separately via weather presets in weather/ directory.

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for config in configs:
        name = config["name"]
        path = os.path.join(output_dir, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        paths.append(path)

    return paths


def generate_and_save_dataset(
    output_dir: str = "buildings", num_configs: int = 100, **kwargs
) -> list[str]:
    """Generate and save a dataset of building config files.

    Note: Weather is handled separately via weather presets in weather/ directory.
    Use run_baselines.py --building <path> to run against all weather presets.

    Args:
        output_dir: Directory to save configs (default: buildings/)
        num_configs: Number of configs to generate
        **kwargs: Additional DatasetConfig parameters

    Returns:
        List of saved config file paths
    """
    dataset_config = DatasetConfig(
        output_dir=output_dir, num_configs=num_configs, **kwargs
    )

    configs = generate_dataset_configs(dataset_config)
    paths = save_configs(configs, output_dir)

    print(f"Generated {len(paths)} building configs in {output_dir}")
    return paths


# Convenience function for quick generation
def quick_dataset(n: int = 10, output_dir: str = "buildings") -> list[str]:
    """Quick way to generate a small dataset of building configs.

    Example:
        paths = quick_dataset(10)
        for path in paths:
            # Use with weather presets
            config = load_scenario_from_parts(path, "hot_summer")
            result = generate_scenario_from_config(config)
            env = get_env(result)
    """
    return generate_and_save_dataset(output_dir=output_dir, num_configs=n)


def visualize_config(
    config: dict, save_path: str | None = None, show: bool = True
):
    """Generate scenario from config dict and visualize AHU zones.

    Uses a temp directory to avoid creating permanent output files.

    Args:
        config: Config dict from generate_config()
        save_path: Optional path to save the visualization as a PDF
        show: Whether to display the plot interactively
    """
    import shutil
    import tempfile

    from smart_buildings.smart_control.utils.floor_generator import visualize_ahu_zones
    from smart_buildings.smart_control.utils.scenario_generator import (
        generate_scenario_from_config,
        scenario_config_from_dict,
    )

    # Use temp directory to avoid polluting scenarios/
    temp_dir = tempfile.mkdtemp()
    try:
        config_with_output = config.copy()
        config_with_output["output_base_dir"] = temp_dir

        result = generate_scenario_from_config(
            scenario_config_from_dict(config_with_output)
        )

        print(f"Generated {result['num_rooms']} rooms with {result['num_ahus']} AHUs")
        print(f"Shape: {config['floor_plan']['building_shape']}")

        visualize_ahu_zones(
            result["bsp"],
            result["ahu_assignments"],
            title=f"{config['name']} - AHU Zones",
            save_path=save_path,
            show=show,
        )
    finally:
        # Clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
