"""Sampling envelope material properties for a generated building.

A building's construction is described by six numbers, drawn per building.
They are written into the building YAML as a ``materials:`` block and read by
the scenario generator, which puts them into ``config.gin``.

WHY THE PROPERTIES ARE DRAWN INDEPENDENTLY
------------------------------------------
The alternative is to pick from a handful of named constructions. That turns
out to be nearly one-dimensional: across a set of named constructions the
pairwise correlations sit close to +/-1, so choosing one really just chooses
"how heavy is this building". Every scenario then lies on a single line through
material space, and a controller that works out where it sits on that line has
learned everything the construction can tell it.

Drawing the six properties independently turns that line into a volume. All
four (U, C) quadrants are real buildings -- a leaky heavy one is an uninsulated
masonry warehouse, a tight light one is a modern panel build -- so independence
is physically honest here, not just statistically convenient.

``exterior_cv_*`` is an effective control volume standing in for a whole facade
assembly, not a handbook material. That is what makes an independent
(k, rho, cp) draw meaningful rather than nonsense.

The interior AIR node is deliberately never drawn. Room air is room air.
"""

from __future__ import annotations

import math
import random

# name -> (low, high, sampling scale)
#
# "log" for anything spanning more than about 2x, so the draw is uniform in
# ratio rather than in absolute value. A uniform draw on 350..2400 would put
# three quarters of its mass above 900 and make light buildings rare.
#
# Ordered by how much each property actually moves the building.
PROPERTIES: dict[str, tuple[float, float, str]] = {
    # 1. Sets facade U through 1/U = 0.1/k + 0.299. The dominant property.
    #    Note the fixed film term: it caps the achievable U spread at about 2x
    #    across this whole conductivity range, so widening the range buys less
    #    than it looks like it should.
    "exterior_cv_conductivity": (0.15, 0.55, "log"),
    # 2. With cp, sets envelope heat capacity, so the thermal time constant.
    "exterior_cv_density": (350.0, 2400.0, "log"),
    # 3. The slab and furniture node. Controls overnight coasting.
    "interior_mass_cv_density": (10.0, 40.0, "log"),
    # 4. How fast heat crosses partitions, so how much zones drag each other.
    "interior_wall_cv_conductivity": (0.45, 0.60, "lin"),
    # 5. Partition mass.
    "interior_wall_cv_density": (110.0, 700.0, "log"),
    # 6. Narrow range, included for completeness rather than leverage.
    "exterior_cv_heat_capacity": (840.0, 1000.0, "lin"),
}


def draw_materials(rng: random.Random, n: int) -> list[dict[str, float]]:
    """Latin hypercube: n draws that are independent AND cover every range.

    Plain independent sampling is the obvious way to draw these, but at small n
    it clusters badly: with eight buildings it can easily put six of them into
    a quarter of a property's range and leave the rest of it represented once.
    That is luck rather than design, and it makes both a worse set to train on
    and a worse set to test against.

    A Latin hypercube fixes it without giving up independence. Each property's
    range is cut into n equal-probability bins, one value is drawn inside each
    bin, and the bin order is then shuffled SEPARATELY for each property. So:

      - every property is guaranteed one building per bin, hence full coverage
      - which bin a building lands in for property A says nothing about which
        bin it lands in for property B, because the shuffles are independent

    That is stratification, not correlation.

    Bins are equal-probability in the sampling scale, so a "log" property gets
    bins equal in ratio and a "lin" property gets bins equal in width.

    Because the draw is stratified across the batch, a building's materials
    depend on how many buildings were generated alongside it. The same
    invocation always reproduces; a different batch size redraws.
    """
    if n <= 0:
        return []

    cols: dict[str, list[float]] = {}
    for name, (lo, hi, scale) in PROPERTIES.items():
        lo_s, hi_s = (math.log(lo), math.log(hi)) if scale == "log" else (lo, hi)
        width = (hi_s - lo_s) / n
        vals = [lo_s + (k + rng.random()) * width for k in range(n)]
        rng.shuffle(vals)
        cols[name] = [math.exp(v) if scale == "log" else v for v in vals]

    return [
        {name: round(cols[name][i], 4) for name in PROPERTIES}
        for i in range(n)
    ]


def facade_u(materials: dict[str, float]) -> float:
    """Facade U value in W/m2K, from the drawn conductivity.

    The 0.299 is the combined inside and outside air film resistance, which is
    fixed regardless of construction, and 0.1 m is the control volume depth.
    """
    return 1.0 / (0.1 / materials["exterior_cv_conductivity"] + 0.299)


def envelope_capacity(materials: dict[str, float]) -> float:
    """Envelope volumetric heat capacity in J/m3K, i.e. rho * cp."""
    return (
        materials["exterior_cv_density"] * materials["exterior_cv_heat_capacity"]
    )
