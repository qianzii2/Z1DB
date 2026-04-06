from __future__ import annotations
"""WHERE 过滤算子 — JIT 编译 + 编译缓存 + 延迟物化。
[集成] ExprCompiler 编译谓词 → CompileCache 缓存 → 热查询零解释开销。
回退：编译失败时用向量化 evaluate_predicate。"""
from typing import Any, List, Optional
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
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

# 全局共享编译缓存（跨算子实例复用）
_GLOBAL_COMPILE_CACHE: Optional[CompileCache] = None


def _get_compile_cache() -> Optional[CompileCache]:
    global _GLOBAL_COMPILE_CACHE
    if _HAS_JIT and _GLOBAL_COMPILE_CACHE is None:
        _GLOBAL_COMPILE_CACHE = CompileCache(max_size=512)
    return _GLOBAL_COMPILE_CACHE


class FilterOperator(Operator):
    """按谓词过滤行。
    执行策略（自动选择）：
    1. JIT 编译路径：谓词编译为 Python 函数，批量评估
    2. 向量化路径：ExpressionEvaluator.evaluate_predicate
    3. 延迟物化：返回 LazyBatch 而非立即复制数据"""

    # JIT 编译只在行数超过此阈值时尝试
    _JIT_THRESHOLD = 512  # [R5] 提高阈值

    def __init__(self, child: Operator, predicate: Any,
                 enable_lazy: bool = True) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._predicate = predicate
        self._evaluator = ExpressionEvaluator()
        self._closed = False
        self._enable_lazy = enable_lazy and _HAS_LAZY
        # JIT 编译状态
        self._jit_fn: Optional[Any] = None
        self._jit_attempted = False

    def output_schema(self) -> List[tuple[str, DataType]]:
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
            raw = self.child.next_batch()
            if raw is None:
                return None
            batch = self._ensure_batch(raw)
            if batch is None:
                return None

            # 首个 batch 尝试 JIT
            if (not self._jit_attempted and _HAS_JIT
                    and batch.row_count >= self._JIT_THRESHOLD):
                self._jit_attempted = True
                self._jit_fn = self._try_compile()

            # JIT 路径
            if self._jit_fn is not None:
                jit_type, jit_fn = self._jit_fn
                if jit_type == 'columnar':
                    # [P12] 列式 JIT：直接在列向量上操作
                    from metal.bitmap import Bitmap
                    try:
                        mask = jit_fn(batch.columns, batch.row_count, Bitmap)
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
                        self._jit_fn = None  # 列式失败，禁用
                else:
                    # 行式 JIT
                    from metal.bitmap import Bitmap
                    mask = Bitmap(batch.row_count)
                    col_names = batch.column_names
                    hit = 0
                    for i in range(batch.row_count):
                        row_dict = {n: batch.columns[n].get(i) for n in col_names}
                        if jit_fn(row_dict):
                            mask.set_bit(i); hit += 1
                    if hit == 0: continue
                    if hit == batch.row_count: return batch
                    if self._enable_lazy and _HAS_LAZY:
                        return LazyBatch(batch, mask)
                    filtered = batch.filter_by_bitmap(mask)
                    if filtered.row_count > 0: return filtered
                    continue

            # 向量化评估路径
            mask = self._evaluator.evaluate_predicate(self._predicate, batch)
            if mask.popcount() == 0: continue
            if mask.popcount() == batch.row_count: return batch
            if self._enable_lazy and _HAS_LAZY:
                return LazyBatch(batch, mask)
            filtered = batch.filter_by_bitmap(mask)
            if filtered.row_count > 0: return filtered
            continue

    def _try_compile(self) -> Optional[Any]:
        """优先列式 JIT，回退行式 JIT。"""
        cache = _get_compile_cache()
        if cache is None:
            return None
        expr_hash = cache.hash_expr(self._predicate)
        cached_fn = cache.get(expr_hash)
        if cached_fn is not None:
            return cached_fn
        # [P12] 优先列式编译
        col_fn = ExprCompiler.compile_columnar_predicate(self._predicate)
        if col_fn is not None:
            # 包装为统一接口
            def wrapper(batch_or_dict):
                if isinstance(batch_or_dict, dict):
                    # 行式回退
                    return None
                return '__columnar__', col_fn
            cache.put(expr_hash, ('columnar', col_fn))
            return ('columnar', col_fn)
        # 回退行式编译
        fn = ExprCompiler.compile_predicate(self._predicate)
        if fn is not None:
            cache.put(expr_hash, ('row', fn))
            return ('row', fn)
        return None
