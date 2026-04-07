### executor/operators/system_table_sort.py
"""系统表排序算子 — [FIX] close() 安全处理子算子。"""
from __future__ import annotations
from typing import List, Optional
import functools

from executor.core.operator import Operator
from executor.core.batch import VectorBatch
from executor.expression.evaluator import ExpressionEvaluator
from parser.ast import SortKey


class SystemTableSortOperator(Operator):
    """系统表排序算子，执行 ORDER BY 排序。"""

    def __init__(self, child: Operator,
                 order_by_list: List[SortKey]):
        super().__init__()
        self.child = child
        self._order_by_list = order_by_list
        self._evaluator = ExpressionEvaluator()
        self._sorted_batch = None
        self._emitted = False
        self._child_closed = False                  # [FIX] 跟踪子算子状态

    def output_schema(self):
        return self.child.output_schema()

    def open(self):
        self.child.open()
        all_batches = []
        while True:
            batch = self.child.next_batch()
            if batch is None:
                break
            all_batches.append(batch)
        self.child.close()
        self._child_closed = True                   # [FIX]

        if not all_batches:
            self._sorted_batch = VectorBatch.empty(
                [name for name, _ in self.output_schema()],
                [dtype for _, dtype in self.output_schema()])
        else:
            self._sorted_batch = VectorBatch.merge(all_batches)
            self._sorted_batch = self._sort_batch(self._sorted_batch)
        self._emitted = False

    def _sort_batch(self, batch: VectorBatch) -> VectorBatch:
        if batch.row_count == 0:
            return batch

        sort_keys = []
        for order_by in self._order_by_list:
            vec = self._evaluator.evaluate(
                order_by.expr, batch).to_python_list()
            sort_keys.append({
                'values': vec,
                'direction': order_by.direction or 'ASC',
                'nulls': order_by.nulls or (
                    'NULLS_LAST' if order_by.direction == 'ASC'
                    else 'NULLS_FIRST')
            })

        indices = list(range(batch.row_count))

        def cmp_rows(i, j):
            for key_info in sort_keys:
                a = key_info['values'][i]
                b = key_info['values'][j]
                direction = key_info['direction']
                nulls = key_info['nulls']

                if a is None and b is None:
                    continue
                elif a is None:
                    return 1 if nulls == 'NULLS_LAST' else -1
                elif b is None:
                    return -1 if nulls == 'NULLS_LAST' else 1

                if a < b:
                    return -1 if direction == 'ASC' else 1
                elif a > b:
                    return 1 if direction == 'ASC' else -1
            return 0

        indices.sort(key=functools.cmp_to_key(cmp_rows))
        return batch.reorder_by_indices(indices)

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted or self._sorted_batch is None:
            return None
        self._emitted = True
        return self._sorted_batch

    def close(self):
        # [FIX] 安全关闭子算子
        if not self._child_closed:
            try:
                self.child.close()
            except Exception:
                pass
            self._child_closed = True
        self._sorted_batch = None
        self._emitted = False
