"""系统表限制算子 — [FIX] 消除递归，改用循环。"""
from __future__ import annotations
from typing import Any, Optional

from executor.core.operator import Operator
from executor.core.batch import VectorBatch
from parser.ast import Literal


class SystemTableLimitOperator(Operator):
    """系统表限制算子，执行 LIMIT / OFFSET。"""

    def __init__(self, child: Operator, limit_expr: Any = None,
                 offset_expr: Any = None):
        super().__init__()
        self.child = child
        self._limit_expr = limit_expr
        self._offset_expr = offset_expr
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._current_row = 0
        self._finished = False

    def _extract_int(self, expr: Any) -> Optional[int]:
        """尝试从表达式提取整数常量。"""
        from executor.expression.evaluator import ExpressionEvaluator
        evaluator = ExpressionEvaluator()
        dummy = VectorBatch.single_row_no_columns()
        try:
            vec = evaluator.evaluate(expr, dummy)
            val = vec.get(0)
            if isinstance(val, int) and val >= 0:
                return val
        except Exception:
            pass
        return None

    def output_schema(self):
        return self.child.output_schema()

    def open(self):
        self.child.open()
        self._limit = (self._extract_int(self._limit_expr)
                       if self._limit_expr else None)
        self._offset = self._extract_int(self._offset_expr) or 0
        self._current_row = 0
        self._finished = False

    def next_batch(self) -> Optional[VectorBatch]:
        """[FIX] 使用 while 循环替代递归，避免栈溢出。"""
        while not self._finished:
            # 已达 LIMIT 上限
            if (self._limit is not None
                    and self._current_row >= self._offset + self._limit):
                self._finished = True
                return None

            batch = self.child.next_batch()
            if batch is None:
                self._finished = True
                return None

            batch_start = self._current_row
            batch_end = self._current_row + batch.row_count

            # 计算 skip 和 keep
            skip_rows = max(0, self._offset - batch_start)
            keep_start = skip_rows
            keep_end = batch.row_count

            # LIMIT 截断
            if self._limit is not None:
                rows_already_returned = max(0, batch_start - self._offset)
                rows_remaining = self._limit - rows_already_returned
                if rows_remaining <= 0:
                    self._current_row = batch_end
                    self._finished = True
                    return None
                keep_end = min(keep_end, keep_start + rows_remaining)

            self._current_row = batch_end

            if keep_start >= keep_end:
                # 本 batch 无有效行，继续下一个（循环而非递归）
                continue

            if keep_start > 0 or keep_end < batch.row_count:
                batch = batch.slice(keep_start, keep_end)

            # 检查是否已达 LIMIT
            if self._limit is not None:
                rows_returned = max(0, batch_start - self._offset) + (keep_end - keep_start)
                if rows_returned >= self._limit:
                    self._finished = True

            return batch

        return None

    def close(self):
        self.child.close()
