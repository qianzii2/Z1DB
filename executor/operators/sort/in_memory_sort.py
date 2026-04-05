from __future__ import annotations
"""内存排序 — 自适应算法选择 + interleaved key多列加速。"""
import functools
from typing import Any, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType
from metal.config import NANO_THRESHOLD

try:
    from executor.operators.sort.pdqsort import pdqsort
    _HAS_PDQ = True
except ImportError:
    _HAS_PDQ = False

try:
    from executor.operators.sort.radix_sort import radix_sort_int64
    _HAS_RADIX = True
except ImportError:
    _HAS_RADIX = False

try:
    from executor.operators.sort.interleaved_key import encode_sort_key
    _HAS_INTERLEAVED = True
except ImportError:
    _HAS_INTERLEAVED = False


class SortOperator(Operator):
    """物化全部输入，自适应排序，输出单个batch。

    算法选择:
    - 多列 + n > 100: interleaved key（单次memcmp替代多列比较）
    - 单INT列 + n > 1000: RadixSort O(n×8)
    - n > 64: PDQSort
    - 回退: Python sorted + cmp_to_key
    """

    def __init__(self, child: Operator,
                 sort_keys: List[Tuple[Any, str, Optional[str]]]) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._sort_keys = sort_keys
        self._result: Optional[VectorBatch] = None
        self._emitted = False
        self._child_closed = False

    def output_schema(self) -> List[tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self.child.open()
        batches: List[VectorBatch] = []
        while True:
            raw = self.child.next_batch()
            b = self._ensure_batch(raw)
            if b is None:
                break
            batches.append(b)
        self.child.close()
        self._child_closed = True

        if not batches:
            self._result = None
            self._emitted = True
            return

        merged = VectorBatch.merge(batches)
        evaluator = ExpressionEvaluator()
        n = merged.row_count

        # 预计算所有排序key的值列表
        key_columns: List[Tuple[list, str, str]] = []
        key_dtypes: List[DataType] = []
        for expr, direction, nulls_pos in self._sort_keys:
            if nulls_pos is None:
                nulls_pos = ('NULLS_LAST' if direction == 'ASC'
                             else 'NULLS_FIRST')
            vec = evaluator.evaluate(expr, merged)
            key_columns.append((vec.to_python_list(), direction, nulls_pos))
            key_dtypes.append(vec.dtype)

        indices = list(range(n))

        # 算法选择
        if self._try_radix_sort(indices, key_columns, n):
            pass
        elif (self._try_interleaved_sort(indices, key_columns,
                                         key_dtypes, n)):
            pass
        elif _HAS_PDQ and n > NANO_THRESHOLD:
            pdqsort(indices, key=functools.cmp_to_key(
                lambda i, j: self._compare_rows(i, j, key_columns)))
        else:
            indices.sort(key=functools.cmp_to_key(
                lambda i, j: self._compare_rows(i, j, key_columns)))

        self._result = merged.reorder_by_indices(indices)
        self._emitted = False

    def _try_radix_sort(self, indices: list, key_columns: list,
                        n: int) -> bool:
        """单INT列 + n > 1000时用基数排序。"""
        if not _HAS_RADIX or n < 1000 or len(key_columns) != 1:
            return False
        values, direction, nulls_pos = key_columns[0]
        if any(v is None for v in values):
            return False
        if not all(isinstance(v, int) for v in values):
            return False
        descending = direction == 'DESC'
        _, sorted_idx = radix_sort_int64(values, indices, descending)
        indices[:] = sorted_idx
        return True

    def _try_interleaved_sort(self, indices: list, key_columns: list,
                              key_dtypes: list, n: int) -> bool:
        """多列 + n > 100时用interleaved key排序。
        将多列排序key编码为单个可比较字节串。"""
        if not _HAS_INTERLEAVED or len(key_columns) < 2 or n < 100:
            return False

        directions = [d for _, d, _ in key_columns]
        null_positions = [np for _, _, np in key_columns]

        # 为每行编码interleaved key
        encoded_keys: List[bytes] = []
        for i in range(n):
            values = [key_columns[c][0][i]
                      for c in range(len(key_columns))]
            ek = encode_sort_key(values, directions, null_positions,
                                 key_dtypes)
            encoded_keys.append(ek)

        # 按编码key排序（单次字节比较替代多列比较）
        indices.sort(key=lambda i: encoded_keys[i])
        return True

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted or self._result is None:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        if not self._child_closed:
            self.child.close()
            self._child_closed = True

    @staticmethod
    def _compare_rows(i: int, j: int,
                      key_columns: List[Tuple[list, str, str]]) -> int:
        for values, direction, nulls_pos in key_columns:
            cmp = SortOperator._compare_values(
                values[i], values[j], nulls_pos)
            if cmp != 0:
                return -cmp if direction == 'DESC' else cmp
        return 0

    @staticmethod
    def _compare_values(a: Any, b: Any, nulls_pos: str) -> int:
        a_null = a is None
        b_null = b is None
        if a_null and b_null:
            return 0
        if a_null:
            return 1 if nulls_pos == 'NULLS_LAST' else -1
        if b_null:
            return -1 if nulls_pos == 'NULLS_LAST' else 1
        try:
            if a < b:
                return -1
            if a > b:
                return 1
            return 0
        except TypeError:
            ta, tb = type(a).__name__, type(b).__name__
            return -1 if ta < tb else (1 if ta > tb else 0)
