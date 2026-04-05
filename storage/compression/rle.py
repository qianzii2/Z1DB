from __future__ import annotations
"""Run-Length Encoding — best for sorted low-cardinality columns.
Compression ratio: 10-1000x for sorted data."""
from typing import Any, List, Tuple


def rle_encode(values: list) -> Tuple[list, list]:
    """Encode values into (run_values, run_lengths).
    Example: [1,1,1,2,2,3] → ([1,2,3], [3,2,1])"""
    if not values:
        return [], []
    run_vals: list = [values[0]]
    run_lens: list = [1]
    for i in range(1, len(values)):
        if values[i] == run_vals[-1]:
            run_lens[-1] += 1
        else:
            run_vals.append(values[i])
            run_lens.append(1)
    return run_vals, run_lens


def rle_decode(run_vals: list, run_lens: list) -> list:
    """Decode RLE back to flat list."""
    result: list = []
    for v, n in zip(run_vals, run_lens):
        result.extend([v] * n)
    return result


def rle_aggregate_sum(run_vals: list, run_lens: list) -> Any:
    """SUM directly on RLE without decompressing. O(runs) not O(n)."""
    total = 0
    for v, n in zip(run_vals, run_lens):
        if v is not None:
            total += v * n
    return total


def rle_aggregate_count(run_lens: list) -> int:
    return sum(run_lens)


def rle_filter_eq(run_vals: list, run_lens: list, target: Any) -> list:
    """Return indices where value == target, without decompressing."""
    indices: list = []
    offset = 0
    for v, n in zip(run_vals, run_lens):
        if v == target:
            indices.extend(range(offset, offset + n))
        offset += n
    return indices


def rle_compression_ratio(values: list) -> float:
    """Estimate compression ratio (lower = better compression)."""
    if not values:
        return 1.0
    _, run_lens = rle_encode(values)
    # Each run = 1 value + 1 length vs original n values
    return len(run_lens) * 2 / len(values)
