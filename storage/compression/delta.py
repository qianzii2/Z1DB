from __future__ import annotations
"""Delta Encoding — best for sorted/incrementing integers (IDs, timestamps).
Often combined with BitPacking for maximum compression."""
from typing import List, Tuple


def delta_encode(values: list) -> Tuple[int, list]:
    """Encode as (base, deltas). base = first value, deltas = differences."""
    if not values:
        return 0, []
    base = values[0]
    deltas = [0] * len(values)
    deltas[0] = 0
    for i in range(1, len(values)):
        deltas[i] = values[i] - values[i - 1]
    return base, deltas


def delta_decode(base: int, deltas: list) -> list:
    if not deltas:
        return []
    result = [base + deltas[0]]
    for i in range(1, len(deltas)):
        result.append(result[-1] + deltas[i])
    return result


def delta_of_delta_encode(values: list) -> Tuple[int, int, list]:
    """Delta-of-delta: best for near-constant-step sequences.
    Example: timestamps with fixed interval → dod ≈ 0."""
    if len(values) < 2:
        return values[0] if values else 0, 0, []
    base = values[0]
    first_delta = values[1] - values[0]
    dod = [0] * len(values)
    prev_delta = first_delta
    for i in range(2, len(values)):
        curr_delta = values[i] - values[i - 1]
        dod[i] = curr_delta - prev_delta
        prev_delta = curr_delta
    return base, first_delta, dod


def delta_of_delta_decode(base: int, first_delta: int, dod: list) -> list:
    if not dod:
        return [base]
    result = [base, base + first_delta]
    delta = first_delta
    for i in range(2, len(dod)):
        delta += dod[i]
        result.append(result[-1] + delta)
    return result


def max_delta(deltas: list) -> int:
    """Maximum absolute delta — determines bit width needed."""
    if not deltas:
        return 0
    return max(abs(d) for d in deltas)


def bits_needed(max_val: int) -> int:
    """Minimum bits to represent a value."""
    if max_val <= 0:
        return 1
    return max_val.bit_length() + 1  # +1 for sign
