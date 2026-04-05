from __future__ import annotations
"""Strategy selector — bridges adaptive engine with planner.
Examines AST + catalog statistics to choose strategies."""
from typing import Any, Dict, List, Optional, Set, Tuple
from catalog.catalog import Catalog
from executor.adaptive.micro_engine import AdaptiveStrategy, MicroAdaptiveEngine
from parser.ast import ColumnRef, SelectStmt, SortKey
from storage.types import DataType


class StrategySelector:
    """Selects execution strategies based on table statistics and query shape."""

    def __init__(self, catalog: Catalog, stats: Optional[Dict] = None) -> None:
        self._catalog = catalog
        self._stats = stats or {}

    def analyze_select(self, ast: SelectStmt) -> AdaptiveStrategy:
        """Analyze a SELECT statement and return strategy decisions."""
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

        # Get row count from primary table
        if ast.from_clause and ast.from_clause.table:
            tname = ast.from_clause.table.name
            if self._catalog.table_exists(tname):
                store = self._catalog.get_store(tname)
                row_count = store.row_count
                # Check if zone maps exist
                if row_count > 0:
                    chunks = store.get_column_chunks(store.schema.columns[0].name)
                    has_zone_maps = any(
                        c.zone_map.get('min') is not None for c in chunks)

        # Join right side size
        if has_join:
            for jc in ast.from_clause.joins:
                if jc.table and self._catalog.table_exists(jc.table.name):
                    join_right_rows = self._catalog.get_store(jc.table.name).row_count

        # GROUP BY NDV estimate
        if has_group_by and ast.group_by.keys:
            first_key = ast.group_by.keys[0]
            if isinstance(first_key, ColumnRef) and ast.from_clause:
                tname = ast.from_clause.table.name
                if tname in self._stats:
                    col_stats = self._stats[tname].column_stats.get(first_key.column)
                    if col_stats:
                        group_ndv = col_stats.ndv
                    else:
                        group_ndv = max(row_count // 10, 1)
                else:
                    group_ndv = max(row_count // 10, 1)

        # Sort type
        if has_order_by and ast.order_by:
            first_sort = ast.order_by[0]
            if isinstance(first_sort.expr, ColumnRef) and ast.from_clause:
                tname = ast.from_clause.table.name
                if self._catalog.table_exists(tname):
                    schema = self._catalog.get_table(tname)
                    for c in schema.columns:
                        if c.name == first_sort.expr.column:
                            sort_dtype = c.dtype
                            break

        # Limit value
        if has_limit:
            try:
                if hasattr(ast.limit, 'value') and isinstance(ast.limit.value, int):
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
