from __future__ import annotations
"""Branchless arithmetic helpers."""

import array as _array


def min_int(a: int, b: int) -> int:
    return a if a < b else b


def max_int(a: int, b: int) -> int:
    return a if a > b else b


def abs_int(x: int) -> int:
    return x if x >= 0 else -x


def clamp(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def batch_filter_gt(values: _array.array, threshold: object, out: _array.array) -> int:
    """Write indices where values[i] > threshold into *out*. Return count."""
    count = 0
    for i, v in enumerate(values):
        if v > threshold:  # type: ignore[operator]
            out.append(i)
            count += 1
    return count
