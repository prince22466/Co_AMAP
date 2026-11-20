"""CPU-only reference/input construction for Group10; independent of GPU execution.

Requires NumPy. All references are computed on the CPU to avoid device FP64
issues previously observed in the BI-V150 Group8 experiments.
"""
from __future__ import annotations

import numpy as np


def parse_int_csv(value: str, *, minimum: int = 1) -> list[int]:
    out = []
    for field in value.split(','):
        field = field.strip()
        if not field:
            raise ValueError(f'Empty comma-separated item in {value!r}')
        val = int(field)
        if val < minimum:
            raise ValueError(f'Expected integers >= {minimum}, got {val}')
        if val not in out:
            out.append(val)
    if not out:
        raise ValueError('No values provided')
    return out


def parse_floats(value: str) -> list[float]:
    out = []
    for item in value.split(','):
        if not item.strip():
            raise ValueError('Empty fill-rate item')
        val = float(item)
        if not (0 < val <= 1):
            raise ValueError('Fill rates must satisfy 0 < rate <= 1')
        if val not in out:
            out.append(val)
    return out


def gather_inputs(n: int, source_size: int, seed: int, pattern: str) -> dict:
    if n <= 0 or source_size <= 0:
        raise ValueError('n and source_size must be positive')
    if pattern not in ('uniform', 'hotset'):
        raise ValueError(f'Unknown gather pattern {pattern}')
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(source_size).astype(np.float32)
    ceiling = source_size if pattern == 'uniform' else min(source_size, 4096)
    indices = rng.integers(0, ceiling, size=n, dtype=np.int32)
    order = np.argsort(indices, kind='stable').astype(np.int32)
    sorted_indices = indices[order]
    expected = source[indices]
    return dict(source=source, indices=indices, sorted_indices=sorted_indices,
                original_positions=order, reference=expected)


def scatter_inputs(n: int, seed: int, pattern: str) -> dict:
    if n <= 0:
        raise ValueError('n must be positive')
    if pattern not in ('random', 'block_local'):
        raise ValueError(f'Unknown scatter pattern {pattern}')
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n).astype(np.float32)
    if pattern == 'random':
        destinations = rng.permutation(n).astype(np.int32)
    else:
        destinations = np.arange(n, dtype=np.int32)
        # Intra-block shuffles: destinations are still globally unique.
        for start in range(0, n, 256):
            end = min(n, start + 256)
            destinations[start:end] = rng.permutation(destinations[start:end])
    order = np.argsort(destinations, kind='stable').astype(np.int32)
    sorted_destinations = destinations[order]
    expected = np.empty(n, dtype=np.float32)
    expected[destinations] = values
    return dict(values=values, destinations=destinations,
                sorted_destinations=sorted_destinations,
                source_positions=order, reference=expected)


def tail_reference(n: int, seed: int) -> dict:
    if n <= 0:
        raise ValueError('n must be positive')
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    # Use float64 for a CPU reference, then round to target output dtype.
    ref = (x.astype(np.float64) * 1.125 + 0.375).astype(np.float32)
    return dict(x=x, reference=ref)


def ragged_inputs(rows: int, max_len: int, fill_rate: float, seed: int) -> dict:
    if rows <= 0 or max_len <= 0 or not 0 < fill_rate <= 1:
        raise ValueError('Invalid ragged geometry/fill-rate')
    rng = np.random.default_rng(seed)
    # Different row lengths, including zero and maximum-length segments.
    lengths = rng.binomial(max_len, fill_rate, size=rows).astype(np.int32)
    if rows >= 4:
        lengths[0] = 0
        lengths[1] = max_len
        lengths[2] = 1
        lengths[3] = max_len // 2
    offsets = np.zeros(rows + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64).astype(np.int32)
    packed = rng.standard_normal(int(offsets[-1])).astype(np.float32)
    padded = np.zeros((rows, max_len), dtype=np.float32)
    expected = np.empty(rows, dtype=np.float32)
    for r in range(rows):
        a, b = int(offsets[r]), int(offsets[r+1])
        if b > a:
            padded[r, :b-a] = packed[a:b]
        expected[r] = np.float32(packed[a:b].astype(np.float64).sum())
    return dict(lengths=lengths, offsets=offsets, packed=packed,
                padded=padded, reference=expected,
                actual_fill=float(offsets[-1]) / (rows * max_len))
