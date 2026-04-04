from __future__ import annotations
"""Table and column statistics for query optimization."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from storage.types import DataType


@dataclass
class ColumnStatistics:
    """Statistics for a single column."""
    column_name: str = ''
    row_count: int = 0
    null_count: int = 0
    ndv: int = 0            # number of distinct values
    min_val: Any = None
    max_val: Any = None
    avg_len: float = 0.0    # average length for strings

    def selectivity_estimate(self, op: str, value: Any) -> float:
        """Estimate selectivity of a predicate."""
        if self.row_count == 0:
            return 0.0
        if op == '=':
            if self.ndv == 0:
                return 0.0
            return 1.0 / self.ndv
        if op == '!=':
            if self.ndv == 0:
                return 1.0
            return 1.0 - 1.0 / self.ndv
        if op in ('<', '<=', '>', '>='):
            if self.min_val is None or self.max_val is None:
                return 0.33
            if self.min_val == self.max_val:
                return 0.5
            try:
                range_val = self.max_val - self.min_val
                if range_val == 0:
                    return 0.5
                if op in ('<', '<='):
                    return min(max((value - self.min_val) / range_val, 0.01), 0.99)
                else:
                    return min(max((self.max_val - value) / range_val, 0.01), 0.99)
            except (TypeError, ZeroDivisionError):
                return 0.33
        if op == 'IS NULL':
            return self.null_count / max(self.row_count, 1)
        if op == 'IS NOT NULL':
            return 1.0 - self.null_count / max(self.row_count, 1)
        return 0.33  # default


@dataclass
class TableStatistics:
    """Statistics for an entire table."""
    table_name: str = ''
    row_count: int = 0
    column_stats: Dict[str, ColumnStatistics] = field(default_factory=dict)

    @staticmethod
    def compute(table_name: str, store: Any, schema: Any) -> TableStatistics:
        """Compute statistics by scanning all rows."""
        from storage.table_store import TableStore
        assert isinstance(store, TableStore)
        ts = TableStatistics(table_name=table_name, row_count=store.row_count)
        all_rows = store.read_all_rows()
        for ci, col in enumerate(schema.columns):
            cs = ColumnStatistics(column_name=col.name, row_count=store.row_count)
            distinct: set = set()
            total_len = 0
            for row in all_rows:
                val = row[ci]
                if val is None:
                    cs.null_count += 1
                    continue
                distinct.add(val)
                if cs.min_val is None or val < cs.min_val:
                    cs.min_val = val
                if cs.max_val is None or val > cs.max_val:
                    cs.max_val = val
                if isinstance(val, str):
                    total_len += len(val)
            cs.ndv = len(distinct)
            non_null = store.row_count - cs.null_count
            cs.avg_len = total_len / non_null if non_null > 0 else 0
            ts.column_stats[col.name] = cs
        return ts
