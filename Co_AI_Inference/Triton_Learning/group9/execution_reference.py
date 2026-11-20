"""Pure-CPU reference math and scheduling helpers for Group9.

No torch, Triton, or accelerator is required to import this module.
"""
from __future__ import annotations

import numpy as np


def parse_positive_list(spec: str, name: str, *, powers_of_two: bool = False) -> list[int]:
    values: list[int] = []
    for entry in spec.split(','):
        try:
            val = int(entry.strip())
        except ValueError as exc:
            raise ValueError(f'{name}: invalid value {entry!r}') from exc
        if val < 1 or (powers_of_two and (val & (val - 1))):
            raise ValueError(f'{name}: value {val} must be a positive '
                             f'{"power of two" if powers_of_two else "integer"}')
        if val not in values:
            values.append(val)
    if not values:
        raise ValueError(f'{name}: empty list')
    return values


def cpu_launch(x):
    return (np.asarray(x, dtype=np.float64) + 0.25).astype(np.float32)


def cpu_fusion(x):
    x = np.asarray(x, dtype=np.float32)
    tmp = (x * np.float32(1.125) + np.float32(0.25)).astype(np.float32)
    return (tmp * tmp + x).astype(np.float32)


def cpu_affine_sequence(x, steps: int):
    if steps < 1:
        raise ValueError('steps must be positive')
    acc = np.asarray(x, dtype=np.float32).copy()
    for _ in range(steps):
        acc = (acc * np.float32(0.987) + np.float32(0.002)).astype(np.float32)
    return acc


def cpu_persistent(x, iterations: int):
    if iterations < 1:
        raise ValueError('iterations must be positive')
    acc = np.asarray(x, dtype=np.float64).copy()
    for _ in range(iterations):
        acc = acc * 0.999 + 0.0001
    return acc.astype(np.float32)


def persistent_tile_assignment(n: int, block: int, workers: int):
    if min(n, block, workers) <= 0:
        raise ValueError('positive problem size, block and workers required')
    tiles = (n + block - 1) // block
    w = min(workers, tiles)
    return [[tile for tile in range(worker, tiles, w)] for worker in range(w)]
