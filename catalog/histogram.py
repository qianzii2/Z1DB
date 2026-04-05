from __future__ import annotations
"""Equi-height histogram + Most Common Values for cardinality estimation.
Paper: Poosala, Ioannidis, et al., 1996 "Improved Histograms for Selectivity Estimation"

ANALYZE builds:
  1. MCV list: top-K most frequent values with exact frequencies
  2. Equi-height histogram: remaining values split into B equal-count buckets
  3. Column correlation: Pearson coefficient between column pairs"""
import math
from typing import Any, Dict, List, Optional, Tuple


class Histogram:
    """Equi-height histogram for a single column."""

    __slots__ = ('_buckets', '_mcv', '_mcv_freq', '_ndv', '_null_frac',
                 '_row_count', '_min_val', '_max_val')

    def __init__(self) -> None:
        self._buckets: List[Tuple[Any, Any, int]] = []  # (lo, hi, count)
        self._mcv: List[Any] = []
        self._mcv_freq: List[float] = []
        self._ndv = 0
        self._null_frac = 0.0
        self._row_count = 0
        self._min_val: Any = None
        self._max_val: Any = None

    @staticmethod
    def build(values: list, num_buckets: int = 100,
              mcv_count: int = 10) -> Histogram:
        """Build histogram from a list of values."""
        h = Histogram()
        h._row_count = len(values)
        non_null = [v for v in values if v is not None]
        null_count = len(values) - len(non_null)
        h._null_frac = null_count / len(values) if values else 0.0

        if not non_null:
            return h

        h._ndv = len(set(non_null))
        h._min_val = min(non_null)
        h._max_val = max(non_null)

        # MCV: most common values
        freq: Dict[Any, int] = {}
        for v in non_null:
            freq[v] = freq.get(v, 0) + 1
        sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
        top = sorted_freq[:mcv_count]
        h._mcv = [v for v, _ in top]
        h._mcv_freq = [c / len(non_null) for _, c in top]

        # Remaining values (not in MCV) for histogram
        mcv_set = set(h._mcv)
        remaining = sorted(v for v in non_null if v not in mcv_set)

        if not remaining:
            return h

        # Build equi-height buckets
        bucket_size = max(1, len(remaining) // num_buckets)
        for i in range(0, len(remaining), bucket_size):
            chunk = remaining[i:i + bucket_size]
            if chunk:
                h._buckets.append((chunk[0], chunk[-1], len(chunk)))

        return h

    def estimate_eq(self, value: Any) -> float:
        """Estimate selectivity of col = value."""
        if self._row_count == 0:
            return 0.0
        # Check MCV
        for i, mv in enumerate(self._mcv):
            if mv == value:
                return self._mcv_freq[i] * (1 - self._null_frac)
        # Not in MCV — estimate from histogram
        if self._ndv <= len(self._mcv):
            return 0.0  # All values are in MCV and this one isn't
        remaining_ndv = self._ndv - len(self._mcv)
        if remaining_ndv <= 0:
            return 0.0
        remaining_frac = 1.0 - sum(self._mcv_freq) - self._null_frac
        return max(0.0, remaining_frac / remaining_ndv)

    def estimate_range(self, lo: Any, hi: Any) -> float:
        """Estimate selectivity of lo <= col <= hi."""
        if self._row_count == 0:
            return 0.0
        if self._min_val is None:
            return 0.33
        total_in_range = 0
        total_count = 0
        for blo, bhi, bcount in self._buckets:
            total_count += bcount
            try:
                if bhi < lo or blo > hi:
                    continue
                if blo >= lo and bhi <= hi:
                    total_in_range += bcount
                else:
                    # Partial overlap — linear interpolation
                    bucket_range = bhi - blo if bhi != blo else 1
                    overlap_lo = max(lo, blo)
                    overlap_hi = min(hi, bhi)
                    try:
                        frac = (overlap_hi - overlap_lo) / bucket_range
                    except (TypeError, ZeroDivisionError):
                        frac = 0.5
                    total_in_range += int(bcount * max(0, min(1, frac)))
            except TypeError:
                total_in_range += bcount // 2

        # Add MCV contributions
        for i, mv in enumerate(self._mcv):
            try:
                if lo <= mv <= hi:
                    total_in_range += int(self._mcv_freq[i] * self._row_count)
            except TypeError:
                pass

        non_null_count = self._row_count * (1 - self._null_frac)
        return total_in_range / non_null_count if non_null_count > 0 else 0.0

    def estimate_lt(self, value: Any) -> float:
        """Estimate col < value."""
        if self._min_val is None:
            return 0.33
        try:
            return self.estimate_range(self._min_val, value) - self.estimate_eq(value)
        except TypeError:
            return 0.33

    @property
    def ndv(self) -> int:
        return self._ndv

    @property
    def null_fraction(self) -> float:
        return self._null_frac

    @property
    def mcv_list(self) -> List[Tuple[Any, float]]:
        return list(zip(self._mcv, self._mcv_freq))
