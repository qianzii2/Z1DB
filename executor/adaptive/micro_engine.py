from __future__ import annotations
"""Micro-Adaptive Engine — selects optimal strategy based on data size.

Five tiers:
  NANO    (<64):    List traversal, no overhead
  MICRO   (<1K):    Python dict/sorted, minimal setup
  STD     (<100K):  Vectorized batch + JIT
  TURBO   (<10M):   JIT + parallel scan
  NUCLEAR (>10M):   Multiprocessing + external sort
"""
from typing import Any, Dict, List, Optional
from metal.config import (NANO_THRESHOLD, MICRO_THRESHOLD,
                           STANDARD_THRESHOLD, TURBO_THRESHOLD)
from storage.types import DataType


class AdaptiveStrategy:
    """Container for strategy decisions."""
    __slots__ = ('scan', 'join', 'agg', 'sort', 'window', 'use_jit', 'parallel')

    def __init__(self) -> None:
        self.scan = 'SEQ_SCAN'
        self.join = 'HASH_DICT'
        self.agg = 'HASH_DICT'
        self.sort = 'PYTHON_SORTED'
        self.window = 'BRUTE_FORCE'
        self.use_jit = False
        self.parallel = False


class MicroAdaptiveEngine:
    """Select optimal execution strategy based on row count and data characteristics."""

    @staticmethod
    def select_strategy(row_count: int,
                        has_where: bool = False,
                        has_join: bool = False,
                        join_right_rows: int = 0,
                        has_group_by: bool = False,
                        group_ndv: int = 0,
                        has_order_by: bool = False,
                        has_limit: bool = False,
                        limit_n: int = 0,
                        sort_dtype: Optional[DataType] = None,
                        has_window: bool = False,
                        window_frame_size: int = 0,
                        has_zone_maps: bool = False) -> AdaptiveStrategy:
        s = AdaptiveStrategy()

        # ═══ Scan ═══
        if has_where and has_zone_maps and row_count > MICRO_THRESHOLD:
            s.scan = 'ZONE_MAP_SCAN'
        elif row_count > STANDARD_THRESHOLD:
            s.scan = 'PARALLEL_SCAN'
        else:
            s.scan = 'SEQ_SCAN'

        # ═══ JOIN ═══
        if has_join:
            if join_right_rows < NANO_THRESHOLD:
                s.join = 'NESTED_LOOP'
            elif join_right_rows < MICRO_THRESHOLD:
                s.join = 'HASH_DICT'
            elif join_right_rows < STANDARD_THRESHOLD:
                s.join = 'HASH_ROBIN_HOOD'
            else:
                s.join = 'HASH_ROBIN_HOOD'  # RADIX/GRACE in Phase 12

        # ═══ Aggregation ═══
        if has_group_by:
            if group_ndv < 16:
                s.agg = 'ARRAY_AGG'
            elif row_count < MICRO_THRESHOLD:
                s.agg = 'HASH_DICT'
            elif group_ndv < 256:
                s.agg = 'PERFECT_HASH'
            else:
                s.agg = 'HASH_ROBIN_HOOD'

        # ═══ Sort ═══
        if has_order_by:
            if has_limit and limit_n > 0 and limit_n < row_count // 10:
                s.sort = 'TOP_N'
            elif row_count < NANO_THRESHOLD:
                s.sort = 'INSERTION'
            elif row_count < MICRO_THRESHOLD:
                s.sort = 'PYTHON_SORTED'
            elif sort_dtype in (DataType.INT, DataType.BIGINT) and row_count > 10000:
                s.sort = 'RADIX'
            else:
                s.sort = 'PDQ'

        # ═══ Window ═══
        if has_window:
            if window_frame_size <= 8:
                s.window = 'BRUTE_FORCE'
            elif window_frame_size > 0:
                s.window = 'SEGMENT_TREE'
            else:
                s.window = 'MOS_ALGORITHM'

        # ═══ JIT ═══
        s.use_jit = row_count >= MICRO_THRESHOLD

        # ═══ Parallel ═══
        s.parallel = row_count >= STANDARD_THRESHOLD

        return s

    @staticmethod
    def tier_name(row_count: int) -> str:
        if row_count < NANO_THRESHOLD:
            return 'NANO'
        if row_count < MICRO_THRESHOLD:
            return 'MICRO'
        if row_count < STANDARD_THRESHOLD:
            return 'STD'
        if row_count < TURBO_THRESHOLD:
            return 'TURBO'
        return 'NUCLEAR'
