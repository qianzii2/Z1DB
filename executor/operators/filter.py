from __future__ import annotations
"""WHERE 过滤算子 — JIT 编译 + 编译缓存 + 延迟物化。
首个 batch 尝试 JIT 编译谓词，后续 batch 复用编译结果。
编译失败时回退到向量化 evaluate_predicate。"""
from typing import Any, List, Optional
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from metal.config import JIT_THRESHOLD
from storage.types import DataType

try:
    from executor.codegen.compiler import ExprCompiler
    from executor.codegen.cache import CompileCache
    _HAS_JIT = True
except ImportError:
    _HAS_JIT = False

try:
    from executor.core.lazy_batch import LazyBatch
    _HAS_LAZY = True
except ImportError:
    _HAS_LAZY = False

# 全局共享编译缓存
_GLOBAL_COMPILE_CACHE: Optional[CompileCache] = None


def _get_compile_cache() -> Optional[CompileCache]:
    global _GLOBAL_COMPILE_CACHE
    if _HAS_JIT and _GLOBAL_COMPILE_CACHE is None:
        _GLOBAL_COMPILE_CACHE = CompileCache(max_size=512)
    return _GLOBAL_COMPILE_CACHE


class FilterOperator(Operator):
    """按谓词过滤行。
    策略：JIT 编译 → 向量化评估 → 延迟物化。"""

    def __init__(self, child: Operator, predicate: Any,
                 enable_lazy: bool = True) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._predicate = predicate
        self._evaluator = ExpressionEvaluator()
        self._closed = False
        self._enable_lazy = enable_lazy and _HAS_LAZY
        self._jit_fn: Optional[Any] = None
        self._jit_attempted = False

    def output_schema(self):
        return self.child.output_schema()

    def open(self) -> None:
        self._closed = False
        self._jit_attempted = False
        self._jit_fn = None
        self.child.open()

    def close(self) -> None:
        if not self._closed:
            self.child.close()
            self._closed = True

    def next_batch(self) -> Optional[Any]:
        while True:
            batch = self._next_child_batch(self.child)
            if batch is None:
                return None
            # 首个 batch 尝试 JIT
            if (not self._jit_attempted and _HAS_JIT
                    and batch.row_count >= JIT_THRESHOLD):
                self._jit_attempted = True
                self._jit_fn = self._try_compile()

            # JIT 路径
            if self._jit_fn is not None:
                try:
                    jit_type, jit_fn = self._jit_fn
                    if jit_type == 'columnar':
                        from metal.bitmap import Bitmap
                        mask = jit_fn(
                            batch.columns, batch.row_count, Bitmap)
                    else:
                        from metal.bitmap import Bitmap
                        mask = Bitmap(batch.row_count)
                        col_names = batch.column_names
                        for i in range(batch.row_count):
                            row_dict = {
                                n: batch.columns[n].get(i)
                                for n in col_names}
                            if jit_fn(row_dict):
                                mask.set_bit(i)
                    if mask.popcount() == 0:
                        continue
                    if mask.popcount() == batch.row_count:
                        return batch
                    if self._enable_lazy and _HAS_LAZY:
                        return LazyBatch(batch, mask)
                    filtered = batch.filter_by_bitmap(mask)
                    if filtered.row_count > 0:
                        return filtered
                    continue
                except Exception:
                    self._jit_fn = None

            # 向量化路径
            mask = self._evaluator.evaluate_predicate(
                self._predicate, batch)
            if mask.popcount() == 0:
                continue
            if mask.popcount() == batch.row_count:
                return batch
            if self._enable_lazy and _HAS_LAZY:
                return LazyBatch(batch, mask)
            filtered = batch.filter_by_bitmap(mask)
            if filtered.row_count > 0:
                return filtered

    def _try_compile(self) -> Optional[tuple]:
        """编译谓词。返回 (type_str, callable) 或 None。"""
        cache = _get_compile_cache()
        if cache is None:
            return None
        expr_hash = cache.hash_expr(self._predicate)
        cached_fn = cache.get(expr_hash)
        if cached_fn is not None:
            if isinstance(cached_fn, tuple) and len(cached_fn) == 2:
                return cached_fn
            return ('row', cached_fn)
        # 优先列式编译
        col_fn = ExprCompiler.compile_columnar_predicate(
            self._predicate)
        if col_fn is not None:
            entry = ('columnar', col_fn)
            cache.put(expr_hash, entry)
            return entry
        # 回退行式
        fn = ExprCompiler.compile_predicate(self._predicate)
        if fn is not None:
            entry = ('row', fn)
            cache.put(expr_hash, entry)
            return entry
        return None
