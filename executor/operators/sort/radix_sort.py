from __future__ import annotations
"""Radix Sort — O(n*w) for integers. Faster than comparison sort for n > 10K.
LSD (Least Significant Digit) variant, 8 bits per pass, 8 passes for int64."""
import array as _array
from typing import List, Optional, Tuple


def radix_sort_int64(values: list, indices: Optional[list] = None,
                     descending: bool = False) -> Tuple[list, list]:
    """Sort int64 values using LSD Radix Sort.
    Returns (sorted_values, sorted_indices).
    8 passes × 256 buckets = handles full int64 range."""
    n = len(values)
    if n <= 1:
        idx = indices if indices is not None else list(range(n))
        return list(values), list(idx)

    if indices is None:
        indices = list(range(n))

    # Convert to unsigned for correct sort order
    # Flip sign bit: signed → unsigned that sorts correctly
    uvals = _array.array('Q', [0] * n)
    for i in range(n):
        uvals[i] = values[i] ^ (1 << 63)  # flip sign bit

    src_vals = uvals
    src_idx = _array.array('Q', indices)
    dst_vals = _array.array('Q', [0] * n)
    dst_idx = _array.array('Q', [0] * n)

    for byte_num in range(8):
        shift = byte_num * 8
        counts = [0] * 256

        # Count
        for i in range(n):
            b = (src_vals[i] >> shift) & 0xFF
            counts[b] += 1

        # Prefix sum
        if descending:
            # Reverse order
            total = 0
            for i in range(255, -1, -1):
                old = counts[i]
                counts[i] = total
                total += old
        else:
            total = 0
            for i in range(256):
                old = counts[i]
                counts[i] = total
                total += old

        # Scatter
        for i in range(n):
            b = (src_vals[i] >> shift) & 0xFF
            pos = counts[b]
            dst_vals[pos] = src_vals[i]
            dst_idx[pos] = src_idx[i]
            counts[b] += 1

        src_vals, dst_vals = dst_vals, src_vals
        src_idx, dst_idx = dst_idx, src_idx

    # Convert back to signed
    result_vals = [int(v) ^ (1 << 63) for v in src_vals]
    # Fix sign: the XOR trick may produce Python ints > 2^63
    result_vals = [v - (1 << 64) if v >= (1 << 63) else v for v in result_vals]
    result_idx = [int(v) for v in src_idx]

    return result_vals, result_idx


def radix_sort_uint(values: list, bit_width: int = 32) -> list:
    """Radix sort for unsigned integers of given bit width."""
    n = len(values)
    if n <= 1:
        return list(values)

    src = list(values)
    dst = [0] * n
    passes = (bit_width + 7) // 8

    for byte_num in range(passes):
        shift = byte_num * 8
        counts = [0] * 256
        for v in src:
            counts[(v >> shift) & 0xFF] += 1
        total = 0
        for i in range(256):
            old = counts[i]
            counts[i] = total
            total += old
        for v in src:
            b = (v >> shift) & 0xFF
            dst[counts[b]] = v
            counts[b] += 1
        src, dst = dst, src
    return src
