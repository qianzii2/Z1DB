from __future__ import annotations
"""Top-N算子 — ORDER BY ... LIMIT N 用堆实现。
O(n log N) 替代 O(n log n)，N << n 时大幅优化。
修复：提取时不再 reverse=True（sort_key 已编码方向）。"""
import heapq
from typing import Any, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType


class TopNOperator(Operator):
    """维护大小为N的堆，只物化top N行。"""

    def __init__(self, child: Operator,
                 sort_keys: List[Tuple[Any, str, Optional[str]]],
                 n: int) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._sort_keys = sort_keys
        self._n = n
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self.child.open()
        evaluator = ExpressionEvaluator()
        heap: list = []
        row_idx = 0

        while True:
            batch = self._ensure_batch(self.child.next_batch())
            if batch is None:
                break
            key_vecs = []
            for expr, direction, nulls_pos in self._sort_keys:
                key_vecs.append(
                    evaluator.evaluate(expr, batch).to_python_list())
            col_names = batch.column_names
            for i in range(batch.row_count):
                row = [batch.columns[n].get(i) for n in col_names]
                key_parts = []
                for ki, (_, direction, nulls_pos) in enumerate(
                        self._sort_keys):
                    val = key_vecs[ki][i]
                    key_parts.append(
                        _make_comparable(val, direction, nulls_pos))
                entry = _HeapEntry(tuple(key_parts), row_idx, row)
                if len(heap) < self._n:
                    heapq.heappush(heap, entry)
                else:
                    heapq.heappushpop(heap, entry)
                row_idx += 1

        self.child.close()

        if not heap:
            schema = self.output_schema()
            self._result = VectorBatch.empty(
                [n for n, _ in schema], [t for _, t in schema])
        else:
            # 修复：用 sort_key 直接排序，不 reverse
            # sort_key 已编码方向（DESC用负值），直接升序即正确顺序
            entries = sorted(heap, key=lambda e: e.sort_key)
            rows = [e.row for e in entries]
            schema = self.output_schema()
            self._result = VectorBatch.from_rows(
                rows,
                [n for n, _ in schema],
                [t for _, t in schema])
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        pass


class _HeapEntry:
    """堆条目。反转比较使 heapq 成为 max-heap（保留最小的N个）。"""
    __slots__ = ('sort_key', 'row_idx', 'row')

    def __init__(self, sort_key: tuple, row_idx: int, row: list) -> None:
        self.sort_key = sort_key
        self.row_idx = row_idx
        self.row = row

    def __lt__(self, other: _HeapEntry) -> bool:
        # 反转比较：max-heap行为（淘汰最大的 sort_key）
        if self.sort_key != other.sort_key:
            return self.sort_key > other.sort_key
        return self.row_idx > other.row_idx

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _HeapEntry):
            return NotImplemented
        return (self.sort_key == other.sort_key
                and self.row_idx == other.row_idx)

    def __le__(self, other: _HeapEntry) -> bool:
        return self.__lt__(other) or self.__eq__(other)


def _make_comparable(val: Any, direction: str,
                     nulls_pos: Optional[str]) -> tuple:
    """创建可比较的排序 key 元组。"""
    if nulls_pos is None:
        nulls_pos = ('NULLS_LAST' if direction == 'ASC'
                     else 'NULLS_FIRST')
    if val is None:
        null_priority = 1 if nulls_pos == 'NULLS_LAST' else -1
        return (null_priority, 0)
    if direction == 'DESC':
        if isinstance(val, (int, float)):
            return (0, -val)
        return (0, _Reversed(val))
    return (0, val)


class _Reversed:
    """包装器，反转非数值类型的比较顺序。"""
    __slots__ = ('val',)

    def __init__(self, val: Any) -> None:
        self.val = val

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, _Reversed):
            try:
                return self.val > other.val
            except TypeError:
                return str(self.val) > str(other.val)
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, _Reversed):
            try:
                return self.val < other.val
            except TypeError:
                return str(self.val) < str(other.val)
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _Reversed):
            return self.val == other.val
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        return self == other or self < other

    def __ge__(self, other: Any) -> bool:
        return self == other or self > other

    def __repr__(self) -> str:
        return f"Rev({self.val!r})"
