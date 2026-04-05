from __future__ import annotations
"""Runtime adaptive optimizer — re-optimizes mid-query based on actual cardinalities.

Problem: estimated cardinality can be 10-1000x wrong.
Solution: after materializing a pipeline breaker, compare actual vs estimated.
If off by >10x, re-plan the remaining operators.

Paper inspiration: Adaptive Query Processing (various, 2000s)
"""
import time
from typing import Any, Dict, List, Optional
from planner.cost_model import CostEstimate


class RuntimeStats:
    """Collected during execution for adaptive re-optimization."""
    __slots__ = ('estimated_rows', 'actual_rows', 'elapsed_ms',
                 'operator_name', 'children_stats')

    def __init__(self, operator_name: str = '',
                 estimated_rows: float = 0,
                 actual_rows: int = 0,
                 elapsed_ms: float = 0) -> None:
        self.operator_name = operator_name
        self.estimated_rows = estimated_rows
        self.actual_rows = actual_rows
        self.elapsed_ms = elapsed_ms
        self.children_stats: List[RuntimeStats] = []

    @property
    def estimation_error(self) -> float:
        """Ratio of actual/estimated. 1.0 = perfect estimate."""
        if self.estimated_rows <= 0: return 1.0
        return self.actual_rows / self.estimated_rows

    @property
    def is_off(self) -> bool:
        """True if estimation error > 10x in either direction."""
        e = self.estimation_error
        return e > 10 or e < 0.1

    def explain(self, indent: int = 0) -> str:
        prefix = '  ' * indent
        status = '⚠️' if self.is_off else '✓'
        line = (f"{prefix}{status} {self.operator_name}: "
                f"est={self.estimated_rows:.0f} actual={self.actual_rows} "
                f"ratio={self.estimation_error:.2f}x "
                f"({self.elapsed_ms:.1f}ms)")
        lines = [line]
        for child in self.children_stats:
            lines.append(child.explain(indent + 1))
        return '\n'.join(lines)


class RuntimeOptimizer:
    """Monitors execution and suggests re-optimization."""

    def __init__(self) -> None:
        self._stats: List[RuntimeStats] = []
        self._reoptimize_threshold = 10.0  # Re-optimize if error > 10x

    def record(self, operator_name: str, estimated_rows: float,
               actual_rows: int, elapsed_ms: float) -> RuntimeStats:
        """Record actual execution statistics for an operator."""
        rs = RuntimeStats(
            operator_name=operator_name,
            estimated_rows=estimated_rows,
            actual_rows=actual_rows,
            elapsed_ms=elapsed_ms)
        self._stats.append(rs)
        return rs

    def should_reoptimize(self) -> bool:
        """Should we re-plan the remaining operators?"""
        for s in self._stats:
            if s.is_off:
                return True
        return False

    def suggest_strategy_change(self, current_strategy: str,
                                actual_rows: int) -> Optional[str]:
        """Suggest a different strategy based on actual data."""
        if current_strategy == 'HASH_JOIN' and actual_rows < 64:
            return 'NESTED_LOOP'
        if current_strategy == 'NESTED_LOOP' and actual_rows > 1000:
            return 'HASH_JOIN'
        if current_strategy == 'FULL_SORT' and actual_rows > 1000000:
            return 'EXTERNAL_SORT'
        if current_strategy == 'HASH_AGG' and actual_rows < 16:
            return 'ARRAY_AGG'
        return None

    def get_correction_factor(self, table: str) -> float:
        """How much to adjust future estimates for this table."""
        for s in self._stats:
            if table in s.operator_name:
                return s.estimation_error
        return 1.0

    def report(self) -> str:
        """Generate execution report with estimation accuracy."""
        if not self._stats:
            return "No runtime stats collected."
        lines = ["Runtime Optimization Report:"]
        total_time = sum(s.elapsed_ms for s in self._stats)
        lines.append(f"  Total time: {total_time:.1f}ms")
        for s in self._stats:
            lines.append(s.explain(1))
        if self.should_reoptimize():
            lines.append("  ⚠️ Cardinality estimation was >10x off. "
                         "Consider ANALYZE to update statistics.")
        else:
            lines.append("  ✓ All estimates within 10x of actual.")
        return '\n'.join(lines)
