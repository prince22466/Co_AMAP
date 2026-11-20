"""Pure-Python reduction reference routines for CPU-only checks (no GPU needed)."""
from __future__ import annotations

import math


def sequential_sum(values):
    accum = 0.0
    for value in values:
        accum += value
    return accum


def chunked_sum(values, width):
    if width < 1:
        raise ValueError('width must be positive')
    partials = [sequential_sum(values[i:i + width])
                for i in range(0, len(values), width)]
    return sequential_sum(partials)


def stable_softmax(values):
    if not values:
        raise ValueError('non-empty row needed')
    m = max(values)
    exps = [math.exp(x - m) for x in values]
    s = sum(exps)
    return [e / s for e in exps]


def online_stats(values, chunk):
    if not values or chunk < 1:
        raise ValueError('nonempty values and positive chunk required')
    m = -math.inf
    s = 0.0
    for k in range(0, len(values), chunk):
        block = values[k:k + chunk]
        new_m = max(m, max(block))
        s = s * math.exp(m - new_m) + sum(math.exp(x - new_m) for x in block)
        m = new_m
    return m, s


def online_softmax(values, chunk):
    m, s = online_stats(values, chunk)
    return [math.exp(x - m) / s for x in values]
