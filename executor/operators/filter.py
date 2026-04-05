from __future__ import annotations
"""WHERE过滤算子 — 支持JIT编译和延迟物化。"""
from typing import Any, List, Optional
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType

try:
    from executor.codegen.compiler import ExprCompiler
    _HAS_JIT = True
except ImportError:
    _HAS_JIT = False

try:
    from executor.core.lazy_batch import LazyBatch
    _HAS_LAZY = True
except ImportError:
    _HAS_LAZY = False


class FilterOperator(Operator):
    """按谓词过滤行。
    优化1: JIT编译谓词为Python callable
    优化2: 返回LazyBatch（位图引用）避免拷贝"""

    def __init__(self, child: Operator, predicate: Any,
                 enable_jit: bool = True, enable_lazy: bool = True) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._predicate = predicate
        self._evaluator = ExpressionEvaluator()
        self._closed = False
        self._enable_jit = enable_jit and _HAS_JIT
        self._enable_lazy = enable_lazy and _HAS_LAZY
        self._jit_fn: Optional[Any] = None
        self._jit_attempted = False

    def output_schema(self) -> List[tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self._closed = False
        self.child.open()
        if self._enable_jit and not self._jit_attempted:
            self._jit_attempted = True
            self._jit_fn = ExprCompiler.compile_predicate(self._predicate)

    def close(self) -> None:
        if not self._closed:
            self.child.close()
            self._closed = True

    def next_batch(self) -> Optional[Any]:
        while True:
            raw = self.child.next_batch()
            if raw is None:
                return None

            # 确保是具体batch
            batch = self._ensure_batch(raw)
            if batch is None:
                return None

            # 评估谓词
            if self._jit_fn is not None:
                mask = self._eval_jit(batch)
            else:
                mask = self._evaluator.evaluate_predicate(
                    self._predicate, batch)

            if mask.popcount() == 0:
                continue

            # 返回LazyBatch或过滤后的batch
            if self._enable_lazy and _HAS_LAZY:
                return LazyBatch(batch, mask)

            filtered = batch.filter_by_bitmap(mask)
            if filtered.row_count > 0:
                return filtered

    def _eval_jit(self, batch: VectorBatch) -> Any:
        """JIT路径：用编译后的函数逐行评估。"""
        from metal.bitmap import Bitmap
        n = batch.row_count
        col_names = batch.column_names
        mask = Bitmap(n)
        for i in range(n):
            row_dict = {cn: batch.columns[cn].get(i) for cn in col_names}
            if self._jit_fn(row_dict):
                mask.set_bit(i)
        return mask
