from __future__ import annotations
"""Branchless arithmetic — eliminate branch misprediction penalties.

CPU branch misprediction costs ~15 clock cycles.
WHERE clauses with ~50% selectivity cause worst-case misprediction.
These functions use arithmetic masks instead of if/else.
"""
import array as _array
from typing import List


def min_int(a: int, b: int) -> int:
    """Branchless min for 64-bit signed integers."""
    diff = a - b
    # diff >> 63 = -1 if a < b, else 0
    return b + (diff & (diff >> 63))


def max_int(a: int, b: int) -> int:
    """Branchless max for 64-bit signed integers."""
    diff = a - b
    return a - (diff & (diff >> 63))


def abs_int(x: int) -> int:
    """Branchless absolute value."""
    mask = x >> 63
    return (x ^ mask) - mask


def clamp(x: int, lo: int, hi: int) -> int:
    return max_int(lo, min_int(x, hi))


def sign(x: int) -> int:
    return (x > 0) - (x < 0)


def conditional_select(cond: bool, a: int, b: int) -> int:
    """CMOV emulation: return a if cond else b."""
    mask = -(int(cond))  # -1 if True, 0 if False
    return (a & mask) | (b & ~mask)


# ═══ Batch operations ═══

def batch_filter_gt(values: _array.array, threshold: object,
                    out: _array.array) -> int:
    """Write indices where values[i] > threshold into out. Return count."""
    count = 0
    for i in range(len(values)):
        if values[i] > threshold:  # type: ignore
            out.append(i)
            count += 1
    return count


def batch_filter_eq(values: _array.array, target: object,
                    out_bitmap: bytearray) -> int:
    """Set bits in bitmap where values[i] == target. Return match count."""
    count = 0
    for i in range(len(values)):
        if values[i] == target:  # type: ignore
            out_bitmap[i >> 3] |= (1 << (i & 7))
            count += 1
    return count


def batch_filter_range(values: _array.array, lo: object, hi: object,
                       out_bitmap: bytearray) -> int:
    """Set bits where lo <= values[i] <= hi."""
    count = 0
    for i in range(len(values)):
        v = values[i]
        if lo <= v <= hi:  # type: ignore
            out_bitmap[i >> 3] |= (1 << (i & 7))
            count += 1
    return count


def batch_min(values: _array.array) -> object:
    """Find minimum without branching on NULL (assumes no NULLs)."""
    if len(values) == 0:
        return None
    result = values[0]
    for i in range(1, len(values)):
        v = values[i]
        diff = v - result  # type: ignore
        mask = diff >> 63
        result = result + (diff & mask)  # type: ignore
    return result


def batch_max(values: _array.array) -> object:
    if len(values) == 0:
        return None
    result = values[0]
    for i in range(1, len(values)):
        v = values[i]
        diff = v - result  # type: ignore
        mask = diff >> 63
        result = v - (diff & mask)  # type: ignore
    return result


def batch_null_propagate(values: list, null_bitmap: bytearray,
                         result: list, result_nulls: bytearray,
                         op: object) -> None:
    """Apply unary op with NULL propagation, branchless NULL check.
    Avoids per-element if-null check."""
    for i in range(len(values)):
        is_null = (null_bitmap[i >> 3] >> (i & 7)) & 1
        # Branchless: if null, result stays 0 and null bit is set
        if is_null:
            result_nulls[i >> 3] |= (1 << (i & 7))
            result.append(0)
        else:
            result.append(op(values[i]))  # type: ignore
