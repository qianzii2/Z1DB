from __future__ import annotations

"""Compressed execution — operate directly on compressed data.
Paper: Abadi, Madden, Ferreira, 2006
"Integrating Compression and Execution in Column-Oriented Database Systems"

Key insight: don't decompress → operate → recompress.
Instead, operate ON the compressed representation directly."""
from typing import Any, Dict, List, Optional, Tuple
from metal.bitmap import Bitmap


def rle_aggregate_direct(run_vals: list, run_lens: list,
                         agg_type: str) -> Any:
    """Aggregate directly on RLE-encoded data without decompression.

    SUM: Σ(val × length) — O(runs) not O(n)
    COUNT: Σ(length)
    MIN: min(run_vals)
    MAX: max(run_vals)
    AVG: SUM / COUNT
    """
    if not run_vals:
        return None
    if agg_type == 'COUNT':
        return sum(run_lens)
    if agg_type == 'SUM':
        total = 0
        for v, l in zip(run_vals, run_lens):
            if v is not None:
                total += v * l
        return total
    if agg_type == 'MIN':
        vals = [v for v in run_vals if v is not None]
        return min(vals) if vals else None
    if agg_type == 'MAX':
        vals = [v for v in run_vals if v is not None]
        return max(vals) if vals else None
    if agg_type == 'AVG':
        total = 0
        count = 0
        for v, l in zip(run_vals, run_lens):
            if v is not None:
                total += v * l
                count += l
        return total / count if count > 0 else None
    return None


def rle_filter_eq_direct(run_vals: list, run_lens: list,
                         target: Any) -> Bitmap:
    """Filter RLE-encoded data for equality without decompression.
    Returns a bitmap of matching positions."""
    total = sum(run_lens)
    bm = Bitmap(total)
    offset = 0
    for v, l in zip(run_vals, run_lens):
        if v == target:
            for i in range(offset, offset + l):
                bm.set_bit(i)
        offset += l
    return bm


def dict_filter_eq(codes: Any, target_code: int, n: int) -> Bitmap:
    """Filter dictionary-encoded column by code (integer comparison).

    Instead of comparing strings:
      WHERE city = 'Beijing'
    We do:
      code = dict.lookup('Beijing')  → 5
      WHERE codes[i] == 5  (int comparison, 10-100x faster)
    """
    bm = Bitmap(n)
    if hasattr(codes, '__getitem__'):
        for i in range(n):
            if codes[i] == target_code:
                bm.set_bit(i)
    return bm


def dict_group_aggregate(codes: Any, values: Any, n: int,
                         ndv: int, agg_type: str) -> Dict[int, Any]:
    """Group-by aggregation directly on dictionary codes.

    Instead of hashing string keys:
      GROUP BY city → hash('Beijing'), hash('Shanghai'), ...
    We do:
      GROUP BY code → array[code] += value  (O(1) per row, zero hashing)

    Only works when NDV < 65536 (uint16 codes).
    """
    if agg_type == 'COUNT':
        counts = [0] * ndv
        for i in range(n):
            counts[codes[i]] += 1
        return {code: cnt for code, cnt in enumerate(counts) if cnt > 0}
    if agg_type == 'SUM':
        sums = [0] * ndv
        has = [False] * ndv
        for i in range(n):
            c = codes[i]
            if values is not None and values[i] is not None:
                sums[c] += values[i]
                has[c] = True
        return {c: s for c, s in enumerate(sums) if has[c]}
    if agg_type == 'MIN':
        mins: list = [None] * ndv
        for i in range(n):
            c = codes[i]
            v = values[i] if values is not None else None
            if v is not None and (mins[c] is None or v < mins[c]):
                mins[c] = v
        return {c: m for c, m in enumerate(mins) if m is not None}
    if agg_type == 'MAX':
        maxs: list = [None] * ndv
        for i in range(n):
            c = codes[i]
            v = values[i] if values is not None else None
            if v is not None and (maxs[c] is None or v > maxs[c]):
                maxs[c] = v
        return {c: m for c, m in enumerate(maxs) if m is not None}
    return {}


def can_use_dict_execution(vec: Any) -> bool:
    """Check if a DataVector has dictionary encoding available."""
    return hasattr(vec, '_dict_encoded') and vec._dict_encoded is not None


def can_use_rle_execution(vec: Any) -> bool:
    """Check if a DataVector has RLE encoding available."""
    return hasattr(vec, '_rle_encoded') and vec._rle_encoded is not None
