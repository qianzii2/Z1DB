"""系统表投影算子 — [FIX] 修正导入，支持聚合。"""
from __future__ import annotations
from typing import Any, List, Tuple, Optional

from executor.core.operator import Operator
from executor.core.batch import VectorBatch
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.bitmap import Bitmap
from parser.ast import ColumnRef, StarExpr, AliasExpr, AggregateCall
from parser.ast_utils import contains_agg
from storage.types import DataType


class SystemTableProjectOperator(Operator):
    """系统表投影算子，处理 SELECT 列表的投影和表达式计算。"""

    def __init__(self, child: Operator, select_list: List[Any],
                 all_columns: List[str]):
        super().__init__()
        self.child = child
        self._select_list = select_list
        self._all_columns = all_columns
        self._evaluator = ExpressionEvaluator()
        self._projections: List[Tuple[str, Any]] = []
        self._output_schema_cache = None
        self._has_agg = any(contains_agg(item) for item in select_list)

    def _build_projections(self):
        self._projections = []
        if not self._select_list:
            for col in self._all_columns:
                self._projections.append(
                    (col, ColumnRef(table=None, column=col)))
        else:
            for item in self._select_list:
                if isinstance(item, StarExpr):
                    for col in self._all_columns:
                        self._projections.append(
                            (col, ColumnRef(table=None, column=col)))
                elif isinstance(item, AliasExpr):
                    self._projections.append((item.alias, item.expr))
                else:
                    col_alias = None
                    if isinstance(item, ColumnRef):
                        col_alias = item.column
                    else:
                        from parser.formatter import Formatter
                        col_alias = Formatter.expr_to_sql(item)
                    self._projections.append((col_alias, item))

    def output_schema(self):
        if self._output_schema_cache is None:
            self._build_projections()
            if self._has_agg:
                self._output_schema_cache = []
                for alias, expr in self._projections:
                    dtype = self._evaluator.infer_type(expr, {c: DataType.VARCHAR for c in self._all_columns})
                    self._output_schema_cache.append((alias, dtype))
            else:
                child_schema = dict(self.child.output_schema())
                self._output_schema_cache = []
                for alias, expr in self._projections:
                    dtype = self._evaluator.infer_type(expr, child_schema)
                    self._output_schema_cache.append((alias, dtype))
        return self._output_schema_cache

    def open(self):
        self.child.open()
        self._build_projections()

    def next_batch(self) -> Optional[VectorBatch]:
        if self._has_agg:
            return self._compute_aggregate()

        batch = self.child.next_batch()
        if batch is None:
            return None

        proj_columns = {}
        for alias, expr in self._projections:
            vec = self._evaluator.evaluate(expr, batch)
            proj_columns[alias] = vec

        col_names = [alias for alias, _ in self._projections]
        return VectorBatch(columns=proj_columns, _column_order=col_names)

    def _compute_aggregate(self) -> Optional[VectorBatch]:
        """Handle aggregate queries on system tables."""
        # Collect all rows first
        all_rows = []
        while True:
            batch = self.child.next_batch()
            if batch is None:
                break
            all_rows.extend(batch.to_rows())

        if not all_rows and not self._projections:
            return None

        # Build a single batch from all rows
        child_schema = self.child.output_schema()
        col_names = [n for n, _ in child_schema]
        col_types = [t for _, t in child_schema]

        if all_rows:
            full_batch = VectorBatch.from_rows(all_rows, col_names, col_types)
        else:
            full_batch = VectorBatch.empty(col_names, col_types)

        # Use SimplePlanner's scalar agg computation
        from executor.functions.registry import FunctionRegistry
        registry = FunctionRegistry()
        registry.register_defaults()

        results = {}
        out_names = []
        for alias, expr in self._projections:
            out_names.append(alias)
            if isinstance(expr, AggregateCall):
                func = registry.get_aggregate(expr.name.upper())
                state = func.init()
                if expr.args and isinstance(expr.args[0], StarExpr):
                    state = func.update(state, None, full_batch.row_count if all_rows else 0)
                elif expr.args:
                    vec = self._evaluator.evaluate(expr.args[0], full_batch) if all_rows else None
                    state = func.update(state, vec, full_batch.row_count if all_rows else 0)
                val = func.finalize(state)
                dt = func.return_type([])
                results[alias] = DataVector.from_scalar(val, dt)
            else:
                if all_rows:
                    results[alias] = self._evaluator.evaluate(expr, full_batch)
                else:
                    results[alias] = DataVector.from_scalar(None, DataType.UNKNOWN)

        self._has_agg = False  # Only return one batch
        return VectorBatch(columns=results, _column_order=out_names, _row_count=1)

    def close(self):
        self.child.close()
