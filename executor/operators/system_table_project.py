"""系统表投影算子 — [FIX] 修正导入。"""
from __future__ import annotations
from typing import Any, List, Tuple, Optional

from executor.core.operator import Operator
from executor.core.batch import VectorBatch
from executor.expression.evaluator import ExpressionEvaluator
from parser.ast import ColumnRef, StarExpr, AliasExpr


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
        batch = self.child.next_batch()
        if batch is None:
            return None

        proj_columns = {}
        for alias, expr in self._projections:
            vec = self._evaluator.evaluate(expr, batch)
            proj_columns[alias] = vec

        col_names = [alias for alias, _ in self._projections]
        return VectorBatch(columns=proj_columns, _column_order=col_names)

    def close(self):
        self.child.close()
