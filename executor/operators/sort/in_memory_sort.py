from __future__ import annotations
"""内存排序 — 自适应算法 + InlineString + 外部排序。
[P19] InlineStringStore 中 NULL 用哨兵标记，不与空字符串混淆。"""
import functools
from typing import Any, List, Optional, Set, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType
from metal.config import NANO_THRESHOLD, DEFAULT_MEMORY_LIMIT

try:
    from executor.operators.sort.pdqsort import pdqsort
    _HAS_PDQ = True
except ImportError: _HAS_PDQ = False
try:
    from executor.operators.sort.radix_sort import radix_sort_int64
    _HAS_RADIX = True
except ImportError: _HAS_RADIX = False
try:
    from executor.operators.sort.interleaved_key import encode_sort_key
    _HAS_INTERLEAVED = True
except ImportError: _HAS_INTERLEAVED = False
try:
    from executor.operators.sort.external_sort import ExternalSort
    _HAS_EXTERNAL = True
except ImportError: _HAS_EXTERNAL = False
try:
    from metal.inline_string import InlineStringStore
    _HAS_INLINE_STRING = True
except ImportError: _HAS_INLINE_STRING = False

_EST_ROW_BYTES = 200
# [P19] NULL 哨兵：InlineStringStore 中用此字符串表示 NULL
_NULL_SENTINEL = '\x00__NULL__\x00'


class SortOperator(Operator):
    def __init__(self, child: Operator,
                 sort_keys: List[Tuple[Any, str, Optional[str]]],
                 memory_limit: int = DEFAULT_MEMORY_LIMIT) -> None:
        super().__init__()
        self.child = child; self.children = [child]
        self._sort_keys = sort_keys
        self._memory_limit = memory_limit
        self._result: Optional[VectorBatch] = None
        self._emitted = False; self._child_closed = False

    def output_schema(self):
        return self.child.output_schema()

    def open(self) -> None:
        self.child.open()
        batches = []
        while True:
            b = self._ensure_batch(self.child.next_batch())
            if b is None: break
            batches.append(b)
        self.child.close(); self._child_closed = True
        if not batches:
            self._result = None; self._emitted = True; return

        merged = VectorBatch.merge(batches)
        evaluator = ExpressionEvaluator()
        n = merged.row_count
        key_columns = []
        key_dtypes = []
        for expr, direction, nulls_pos in self._sort_keys:
            if nulls_pos is None:
                nulls_pos = 'NULLS_LAST' if direction == 'ASC' else 'NULLS_FIRST'
            vec = evaluator.evaluate(expr, merged)
            key_columns.append((vec.to_python_list(), direction, nulls_pos))
            key_dtypes.append(vec.dtype)
        indices = list(range(n))

        est_mem = n * _EST_ROW_BYTES
        if _HAS_EXTERNAL and est_mem > self._memory_limit and n > 10000:
            self._result = self._external_sort(merged, indices, key_columns)
            self._emitted = False; return

        # [P19] 构建 InlineStringStore（NULL 用哨兵）
        inline_stores = self._build_inline_stores(key_columns, key_dtypes, n)

        if self._try_radix_sort(indices, key_columns, n): pass
        elif self._try_interleaved_sort(indices, key_columns, key_dtypes, n): pass
        elif _HAS_PDQ and n > NANO_THRESHOLD:
            cmp_fn = self._make_compare_fn(key_columns, inline_stores)
            pdqsort(indices, key=functools.cmp_to_key(cmp_fn))
        else:
            cmp_fn = self._make_compare_fn(key_columns, inline_stores)
            indices.sort(key=functools.cmp_to_key(cmp_fn))

        self._result = merged.reorder_by_indices(indices)
        self._emitted = False

    def _build_inline_stores(self, key_columns, key_dtypes, n):
        """构建 InlineStringStore（NULL 用集合跟踪）。
        0.2 修复后 InlineStringStore 可用。"""
        try:
            from metal.inline_string import InlineStringStore
        except ImportError:
            return {}

        stores = {}
        for ki, (values, _, _) in enumerate(key_columns):
            if key_dtypes[ki] not in (DataType.VARCHAR, DataType.TEXT):
                continue
            if n < 100:
                continue  # 太小不值得
            try:
                iss = InlineStringStore(capacity=n)
                null_set: set = set()
                for idx, val in enumerate(values):
                    if val is None:
                        iss.append('')  # 占位
                        null_set.add(idx)
                    else:
                        iss.append(str(val))
                stores[ki] = (iss, null_set)
            except Exception:
                # 构建失败回退到标准比较
                continue
        return stores

    def _make_compare_fn(self, key_columns, inline_stores):
        def compare(i, j):
            for ki, (values, direction, nulls_pos) in enumerate(key_columns):
                if ki in inline_stores:
                    iss, null_set = inline_stores[ki]
                    a_null = i in null_set
                    b_null = j in null_set
                    if a_null and b_null: continue
                    if a_null:
                        cmp = 1 if nulls_pos == 'NULLS_LAST' else -1
                        return -cmp if direction == 'DESC' else cmp
                    if b_null:
                        cmp = -1 if nulls_pos == 'NULLS_LAST' else 1
                        return -cmp if direction == 'DESC' else cmp
                    # [P19] 两个都非 NULL，用 InlineStringStore 前缀比较
                    try:
                        cmp = iss.compare(i, j)
                    except Exception:
                        cmp = SortOperator._compare_values(values[i], values[j], nulls_pos)
                    if cmp != 0:
                        return -cmp if direction == 'DESC' else cmp
                    continue
                cmp = SortOperator._compare_values(values[i], values[j], nulls_pos)
                if cmp != 0:
                    return -cmp if direction == 'DESC' else cmp
            return 0
        return compare

    def _external_sort(self, merged, indices, key_columns):
        rows_with_keys = [(tuple(kc[0][i] for kc in key_columns), i) for i in indices]
        ext = ExternalSort(memory_limit=self._memory_limit)
        sorted_pairs = ext.sort(rows_with_keys,
            key_fn=functools.cmp_to_key(
                lambda a, b: self._compare_key_tuples(a[0], b[0], key_columns)))
        return merged.reorder_by_indices([p[1] for p in sorted_pairs])

    def _try_radix_sort(self, indices, key_columns, n):
        if not _HAS_RADIX or n < 1000 or len(key_columns) != 1: return False
        values, direction, _ = key_columns[0]
        if any(v is None for v in values): return False
        if not all(isinstance(v, int) for v in values): return False
        _, sorted_idx = radix_sort_int64(values, indices, direction == 'DESC')
        indices[:] = sorted_idx; return True

    def _try_interleaved_sort(self, indices, key_columns, key_dtypes, n):
        if not _HAS_INTERLEAVED or len(key_columns) < 2 or n < 100: return False
        dirs = [d for _, d, _ in key_columns]
        nps = [np for _, _, np in key_columns]
        encoded = [encode_sort_key(
            [key_columns[c][0][i] for c in range(len(key_columns))],
            dirs, nps, key_dtypes) for i in range(n)]
        indices.sort(key=lambda i: encoded[i]); return True

    def next_batch(self):
        if self._emitted or self._result is None: return None
        self._emitted = True; return self._result

    def close(self):
        if not self._child_closed: self.child.close(); self._child_closed = True

    @staticmethod
    def _compare_values(a, b, nulls_pos):
        if a is None and b is None: return 0
        if a is None: return 1 if nulls_pos == 'NULLS_LAST' else -1
        if b is None: return -1 if nulls_pos == 'NULLS_LAST' else 1
        try:
            if a < b: return -1
            if a > b: return 1
            return 0
        except TypeError:
            return (str(a) > str(b)) - (str(a) < str(b))

    @staticmethod
    def _compare_key_tuples(a_keys, b_keys, key_columns):
        for idx, (_, direction, nulls_pos) in enumerate(key_columns):
            av = a_keys[idx] if idx < len(a_keys) else None
            bv = b_keys[idx] if idx < len(b_keys) else None
            cmp = SortOperator._compare_values(av, bv, nulls_pos)
            if cmp != 0: return -cmp if direction == 'DESC' else cmp
        return 0
