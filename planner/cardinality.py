from __future__ import annotations
"""Cardinality estimation — predicts output row counts for plan nodes.
Uses column statistics when available, falls back to heuristics.

Key formulas:
  col = val       → 1/NDV
  col > val       → (max - val) / (max - min)  [uniform assumption]
  col IS NULL     → null_count / row_count
  A AND B         → sel(A) * sel(B)  [independence assumption]
  A OR B          → sel(A) + sel(B) - sel(A)*sel(B)
  NOT A           → 1 - sel(A)
  JOIN(A,B)       → |A| * |B| / max(NDV_A, NDV_B)  [foreign key assumption]
"""
import dataclasses
import math
from typing import Any, Dict, Optional
from parser.ast import (BinaryExpr, ColumnRef, InExpr, IsNullExpr, LikeExpr,
                         BetweenExpr, Literal, UnaryExpr, AliasExpr)
from storage.types import DataType

# Default selectivity when we have no statistics
_DEFAULT_EQ = 0.1
_DEFAULT_RANGE = 0.33
_DEFAULT_LIKE = 0.1
_DEFAULT_IN = 0.3
_DEFAULT_IS_NULL = 0.05


class CardinalityEstimator:
    """Estimates selectivity and output cardinality."""

    def __init__(self, stats: Optional[Dict] = None) -> None:
        self._stats = stats or {}  # table_name → TableStatistics

    def estimate_selectivity(self, predicate: Any,
                             table: Optional[str] = None) -> float:
        """Estimate fraction of rows surviving a predicate."""
        if predicate is None:
            return 1.0
        if isinstance(predicate, Literal):
            if predicate.value is True: return 1.0
            if predicate.value is False: return 0.0
            if predicate.value is None: return 0.0
            return 1.0

        if isinstance(predicate, BinaryExpr):
            if predicate.op == 'AND':
                # Independence assumption
                left = self.estimate_selectivity(predicate.left, table)
                right = self.estimate_selectivity(predicate.right, table)
                return left * right
            if predicate.op == 'OR':
                left = self.estimate_selectivity(predicate.left, table)
                right = self.estimate_selectivity(predicate.right, table)
                return min(1.0, left + right - left * right)
            if predicate.op in ('=', '!=', '<', '>', '<=', '>='):
                return self._estimate_comparison(predicate, table)

        if isinstance(predicate, UnaryExpr) and predicate.op == 'NOT':
            inner = self.estimate_selectivity(predicate.operand, table)
            return 1.0 - inner

        if isinstance(predicate, IsNullExpr):
            col = self._get_column_name(predicate.expr)
            if col and table:
                cs = self._get_col_stats(table, col)
                if cs:
                    sel = cs.null_count / max(cs.row_count, 1)
                    return (1.0 - sel) if predicate.negated else sel
            return (1.0 - _DEFAULT_IS_NULL) if predicate.negated else _DEFAULT_IS_NULL

        if isinstance(predicate, InExpr):
            n_values = len(predicate.values)
            col = self._get_column_name(predicate.expr)
            if col and table:
                cs = self._get_col_stats(table, col)
                if cs and cs.ndv > 0:
                    sel = min(1.0, n_values / cs.ndv)
                    return (1.0 - sel) if predicate.negated else sel
            sel = min(1.0, n_values * _DEFAULT_EQ)
            return (1.0 - sel) if predicate.negated else sel

        if isinstance(predicate, BetweenExpr):
            # Estimate as two range conditions ANDed
            return _DEFAULT_RANGE * _DEFAULT_RANGE

        if isinstance(predicate, LikeExpr):
            return _DEFAULT_LIKE

        if isinstance(predicate, AliasExpr):
            return self.estimate_selectivity(predicate.expr, table)

        return _DEFAULT_RANGE

    def estimate_join_cardinality(self, left_rows: float, right_rows: float,
                                  left_table: Optional[str] = None,
                                  right_table: Optional[str] = None,
                                  join_col_left: Optional[str] = None,
                                  join_col_right: Optional[str] = None) -> float:
        """Estimate output rows for an equi-join."""
        # Foreign key heuristic: output ≈ max(left, right)
        # General: output ≈ left * right / max(NDV_left, NDV_right)
        ndv_left = self._get_ndv(left_table, join_col_left)
        ndv_right = self._get_ndv(right_table, join_col_right)
        if ndv_left > 0 and ndv_right > 0:
            return left_rows * right_rows / max(ndv_left, ndv_right)
        # Fallback: 10% of cross product
        return max(1, left_rows * right_rows * 0.1)

    def estimate_group_ndv(self, table: Optional[str],
                           group_cols: list) -> int:
        """Estimate number of distinct groups."""
        if not group_cols:
            return 1
        # Product of NDVs (capped at row count)
        ndv = 1
        for col in group_cols:
            col_name = self._get_column_name(col)
            if col_name and table:
                cn = self._get_ndv(table, col_name)
                if cn > 0:
                    ndv *= cn
                    continue
            ndv *= 10  # default
        # Cap at estimated row count
        if table and table in self._stats:
            ndv = min(ndv, self._stats[table].row_count)
        return max(1, ndv)

    def _estimate_comparison(self, expr: BinaryExpr,
                             table: Optional[str]) -> float:
        col = self._get_column_name(expr.left) or self._get_column_name(expr.right)
        val = self._get_literal_value(expr.right) or self._get_literal_value(expr.left)

        if col and table:
            cs = self._get_col_stats(table, col)
            if cs:
                return cs.selectivity_estimate(expr.op, val)

        if expr.op == '=': return _DEFAULT_EQ
        if expr.op == '!=': return 1.0 - _DEFAULT_EQ
        return _DEFAULT_RANGE

    def _get_col_stats(self, table: str, col: str) -> Any:
        if table in self._stats:
            return self._stats[table].column_stats.get(col)
        return None

    def _get_ndv(self, table: Optional[str], col: Optional[str]) -> int:
        if table and col:
            cs = self._get_col_stats(table, col)
            if cs: return cs.ndv
        return 0

    @staticmethod
    def _get_column_name(expr: Any) -> Optional[str]:
        if isinstance(expr, ColumnRef): return expr.column
        if isinstance(expr, AliasExpr): return CardinalityEstimator._get_column_name(expr.expr)
        return None

    @staticmethod
    def _get_literal_value(expr: Any) -> Any:
        if isinstance(expr, Literal): return expr.value
        return None
