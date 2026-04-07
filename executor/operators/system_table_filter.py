### executor/operators/system_table_filter.py
from __future__ import annotations
"""系统表过滤算子 — [FIX] 修正导入，改进空结果处理。"""
from typing import Any, Optional

from executor.core.operator import Operator
from executor.core.batch import VectorBatch
from executor.expression.evaluator import ExpressionEvaluator


class SystemTableFilterOperator(Operator):
    """系统表过滤算子，执行 WHERE 条件过滤。"""

    def __init__(self, child: Operator, predicate: Any):
        super().__init__()
        self.child = child
        self._predicate = predicate
        self._evaluator = ExpressionEvaluator()

    def output_schema(self):
        return self.child.output_schema()

    def open(self):
        self.child.open()

    def next_batch(self) -> Optional[VectorBatch]:
        """[FIX] 空匹配返回 None 而非空 batch，与 FilterOperator 一致。"""
        while True:
            batch = self.child.next_batch()
            if batch is None:
                return None

            mask = self._evaluator.evaluate_predicate(
                self._predicate, batch)

            if mask.popcount() == 0:
                # [FIX] 继续读取下一个 batch 而非返回空 batch
                continue

            if mask.popcount() == batch.row_count:
                return batch

            return batch.filter_by_bitmap(mask)

    def close(self):
        self.child.close()
