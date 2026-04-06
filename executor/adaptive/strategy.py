from __future__ import annotations
"""策略选择器 — 分析 AST + 统计信息选择执行策略。"""
from typing import Any, Dict, List, Optional, Set, Tuple
from catalog.catalog import Catalog
from executor.adaptive.micro_engine import (
    AdaptiveStrategy, MicroAdaptiveEngine)
from parser.ast import ColumnRef, SelectStmt, SortKey
from storage.types import DataType


class StrategySelector:
    """分析 SELECT 并推荐执行策略。"""

    def __init__(self, catalog: Catalog,
                 stats: Optional[Dict] = None) -> None:
        self._catalog = catalog
        self._stats = stats or {}

    def analyze_select(self,
                       ast: SelectStmt) -> AdaptiveStrategy:
        """分析 SELECT 返回策略决策。"""
        row_count = 0
        has_where = ast.where is not None
        has_join = bool(ast.from_clause and ast.from_clause.joins)
        join_right_rows = 0
        has_group_by = ast.group_by is not None
        group_ndv = 0
        has_order_by = bool(ast.order_by)
        has_limit = ast.limit is not None
        limit_n = 0
        sort_dtype = None
        has_zone_maps = False

        if ast.from_clause and ast.from_clause.table:
            tname = ast.from_clause.table.name
            if self._catalog.table_exists(tname):
                store = self._catalog.get_store(tname)
                row_count = store.row_count
                # [FIX-C01] 安全检查 zone_map 方法是否存在
                if row_count > 0:
                    has_zone_maps = self._has_zone_maps(tname)

        if has_join:
            for jc in ast.from_clause.joins:
                if (jc.table
                        and self._catalog.table_exists(jc.table.name)):
                    join_right_rows = self._catalog.get_store(
                        jc.table.name).row_count

        if has_group_by and ast.group_by.keys:
            first_key = ast.group_by.keys[0]
            if (isinstance(first_key, ColumnRef)
                    and ast.from_clause):
                tname = ast.from_clause.table.name
                if tname in self._stats:
                    col_stats = self._stats[tname] \
                        .column_stats.get(first_key.column)
                    if col_stats:
                        group_ndv = col_stats.ndv
                    else:
                        group_ndv = max(row_count // 10, 1)
                else:
                    group_ndv = max(row_count // 10, 1)

        if has_order_by and ast.order_by:
            first_sort = ast.order_by[0]
            if (isinstance(first_sort.expr, ColumnRef)
                    and ast.from_clause):
                tname = ast.from_clause.table.name
                if self._catalog.table_exists(tname):
                    schema = self._catalog.get_table(tname)
                    for c in schema.columns:
                        if c.name == first_sort.expr.column:
                            sort_dtype = c.dtype
                            break

        if has_limit:
            try:
                if (hasattr(ast.limit, 'value')
                        and isinstance(ast.limit.value, int)):
                    limit_n = ast.limit.value
            except Exception:
                pass

        return MicroAdaptiveEngine.select_strategy(
            row_count=row_count,
            has_where=has_where,
            has_join=has_join,
            join_right_rows=join_right_rows,
            has_group_by=has_group_by,
            group_ndv=group_ndv,
            has_order_by=has_order_by,
            has_limit=has_limit,
            limit_n=limit_n,
            sort_dtype=sort_dtype,
            has_zone_maps=has_zone_maps,
        )

    def recommend_scan(self, table_name: str,
                       has_where: bool = False) -> str:
        if not self._catalog.table_exists(table_name):
            return 'SEQ_SCAN'
        store = self._catalog.get_store(table_name)
        rc = store.row_count
        strategy = MicroAdaptiveEngine.select_strategy(
            row_count=rc, has_where=has_where,
            has_zone_maps=self._has_zone_maps(table_name))
        return strategy.scan

    def recommend_join(self, left_rows: int,
                       right_rows: int) -> str:
        strategy = MicroAdaptiveEngine.select_strategy(
            row_count=left_rows,
            has_join=True,
            join_right_rows=right_rows)
        return strategy.join

    def _has_zone_maps(self, table_name: str) -> bool:
        """[FIX-C01] 安全检测 zone_map 可用性。"""
        try:
            store = self._catalog.get_store(table_name)
            if store.row_count == 0:
                return False
            if not hasattr(store, 'get_column_chunks'):
                return False
            if not hasattr(store, 'schema') or not store.schema.columns:
                return False
            chunks = store.get_column_chunks(
                store.schema.columns[0].name)
            return any(
                hasattr(c, 'zone_map')
                and isinstance(c.zone_map, dict)
                and c.zone_map.get('min') is not None
                for c in chunks)
        except Exception:
            return False
