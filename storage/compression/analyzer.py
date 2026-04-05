from __future__ import annotations
"""Compression analyzer — samples data and selects optimal codec per column."""
from typing import Any, List, Optional
from storage.types import DataType


def analyze_and_choose(values: list, dtype: DataType,
                       sample_size: int = 1024) -> str:
    """Sample values and choose the best compression method.
    Returns codec name: 'NONE', 'RLE', 'DICT', 'DELTA', 'FOR', 'GORILLA'."""
    if not values:
        return 'NONE'

    sample = values[:sample_size]
    non_null = [v for v in sample if v is not None]
    if not non_null:
        return 'NONE'

    # String columns
    if dtype in (DataType.VARCHAR, DataType.TEXT):
        ndv = len(set(non_null))
        if ndv < min(len(non_null), 65536):
            return 'DICT'
        return 'NONE'

    # Boolean
    if dtype == DataType.BOOLEAN:
        return 'NONE'  # Already compact as bitmap

    # Float columns
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        ratio = _try_gorilla(non_null)
        if ratio < 0.5:
            return 'GORILLA'
        return 'NONE'

    # Integer columns
    if dtype in (DataType.INT, DataType.BIGINT, DataType.DATE, DataType.TIMESTAMP):
        # Check if sorted (good for delta)
        sorted_check = all(non_null[i] <= non_null[i + 1] for i in range(len(non_null) - 1))
        if sorted_check:
            return 'DELTA'

        # Check RLE ratio
        rle_ratio = _try_rle(non_null)
        if rle_ratio < 0.3:
            return 'RLE'

        # Check FOR ratio
        for_ratio = _try_for(non_null)
        if for_ratio < 0.5:
            return 'FOR'

        # Check NDV for dict
        ndv = len(set(non_null))
        if ndv < min(len(non_null) // 2, 65536):
            return 'DICT'

        return 'NONE'

    return 'NONE'


def _try_rle(values: list) -> float:
    if not values:
        return 1.0
    runs = 1
    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            runs += 1
    return runs / len(values)


def _try_for(values: list) -> float:
    if not values:
        return 1.0
    min_v = min(values)
    max_v = max(values)
    range_v = max_v - min_v
    if range_v == 0:
        return 1.0 / 64  # 1 bit per value
    bit_width = max(1, range_v.bit_length())
    return bit_width / 64


def _try_gorilla(values: list) -> float:
    if len(values) < 2:
        return 1.0
    try:
        from storage.compression.gorilla import gorilla_compression_ratio
        return gorilla_compression_ratio(values)
    except Exception:
        return 1.0
