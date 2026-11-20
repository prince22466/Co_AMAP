"""CPU-only numerical references and configuration helpers for Group11.

Do not use GPU float64 for reference calculations on BI-V150.
"""
from __future__ import annotations
import math
import numpy as np


def base_arrays(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(.2, .8, size=n).astype(np.float32)
    a = rng.uniform(.998, 1.001, size=n).astype(np.float32)
    b = rng.uniform(.0001, .0003, size=n).astype(np.float32)
    return x, a, b


def fma_reference(x, a, b, iters):
    acc = x.astype(np.float64).copy()
    aa, bb = a.astype(np.float64), b.astype(np.float64)
    for _ in range(iters):
        acc = acc * aa + bb
    return acc.astype(np.float32)


def live_range_reference(x, a, b, iters):
    if x.ndim != 2 or x.shape[0] != 8:
        raise ValueError('Expected eight input streams')
    z = fma_reference(x[0], a, b, iters).astype(np.float64)
    for i in range(8):
        z = z + x[i].astype(np.float64)
    return z.astype(np.float32)


def licm_reference(x, a, b, iters):
    z = x.astype(np.float64).copy()
    bb = b.astype(np.float64)
    expa = np.exp(a.astype(np.float64))
    for _ in range(iters):
        z = z * bb + expa
    return z.astype(np.float32)


def validate_tiling(br, bc, warps):
    if min(br, bc, warps) <= 0 or any(v & (v-1) for v in (br,bc,warps)):
        raise ValueError('Tile sizes and warps must be positive powers of two')
    if br*bc > 16384:
        raise ValueError('Tile too large for this teaching lab')


def parse_tile_pairs(value):
    pairs = []
    for part in value.split(','):
        try:
            a,b = (int(v) for v in part.lower().split('x'))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid tile {part!r}; use e.g. 16x64,32x32') from exc
        validate_tiling(a,b,4)
        if (a,b) not in pairs:
            pairs.append((a,b))
    if not pairs:
        raise ValueError('No tiles supplied')
    return pairs


def assert_gpu_warp_size(value):
    # Do not hardcode NVIDIA 32-lane warps on BI-V150.
    if not isinstance(value, int) or value < 16 or value > 128 or value & (value-1):
        raise ValueError('Cannot determine valid hardware warp size')
    return value
