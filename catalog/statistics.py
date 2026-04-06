from __future__ import annotations
"""表和列统计信息 — 查询优化用。
集成 histogram：ANALYZE 时自动构建等高直方图。"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from storage.types import DataType

try:
    from catalog.histogram import Histogram
    _HAS_HISTOGRAM = True
except ImportError:
    _HAS_HISTOGRAM = False


@dataclass
class ColumnStatistics:
    """单列统计。"""
    column_name: str = ''
    row_count: int = 0
    null_count: int = 0
    ndv: int = 0
    min_val: Any = None
    max_val: Any = None
    avg_len: float = 0.0
    histogram: Any = None

    def selectivity_estimate(self, op: str, value: Any) -> float:
        """估算谓词选择率。优先使用直方图。"""
        if self.histogram is not None and _HAS_HISTOGRAM:
            if op == '=':
                return self.histogram.estimate_eq(value)
            if op == '!=':
                return 1.0 - self.histogram.estimate_eq(value)
            if op == '<':
                return self.histogram.estimate_lt(value)
            if op == '<=':
                return (self.histogram.estimate_range(self.min_val, value)
                        if self.min_val is not None else 0.33)
            if op == '>':
                le = (self.histogram.estimate_range(self.min_val, value)
                      if self.min_val is not None else 0.33)
                return max(0.0, 1.0 - le)
            if op == '>=':
                lt = self.histogram.estimate_lt(value)
                return max(0.0, 1.0 - lt)
        # 回退到简单估算
        if self.row_count == 0:
            return 0.0
        if op == '=':
            return 1.0 / self.ndv if self.ndv > 0 else 0.0
        if op == '!=':
            return 1.0 - 1.0 / self.ndv if self.ndv > 0 else 1.0
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
                    return min(max(
                        (value - self.min_val) / range_val, 0.01), 0.99)
                else:
                    return min(max(
                        (self.max_val - value) / range_val, 0.01), 0.99)
            except (TypeError, ZeroDivisionError):
                return 0.33
        if op == 'IS NULL':
            return self.null_count / max(self.row_count, 1)
        if op == 'IS NOT NULL':
            return 1.0 - self.null_count / max(self.row_count, 1)
        return 0.33


@dataclass
class TableStatistics:
    """全表统计。"""
    table_name: str = ''
    row_count: int = 0
    column_stats: Dict[str, ColumnStatistics] = field(
        default_factory=dict)

    @staticmethod
    def compute(table_name: str, store: Any,
                schema: Any) -> 'TableStatistics':
        """[FIX-B12] 扫描所有行计算统计信息。
        不再 assert TableStore，使用鸭子类型兼容 LSMStore。"""
        ts = TableStatistics(
            table_name=table_name,
            row_count=store.row_count)
        all_rows = store.read_all_rows()

        for ci, col in enumerate(schema.columns):
            cs = ColumnStatistics(
                column_name=col.name,
                row_count=store.row_count)
            distinct: set = set()
            total_len = 0
            col_values: list = []
            for row in all_rows:
                val = row[ci] if ci < len(row) else None
                col_values.append(val)
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
            # 构建直方图
            if _HAS_HISTOGRAM and non_null > 0:
                try:
                    cs.histogram = Histogram.build(col_values)
                except Exception:
                    pass
            ts.column_stats[col.name] = cs
        return ts
