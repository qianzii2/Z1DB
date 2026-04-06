from __future__ import annotations
"""DISTINCT 算子 — 流式 hash-based 去重。
[P12] 不再全量物化后去重，改为逐 batch 流式去重。"""
from typing import List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from storage.types import DataType


class _NullSentinel:
    """NULL 哨兵 — 在 set 中使 NULL 可哈希且互相相等。"""
    __slots__ = ()
    def __hash__(self) -> int: return 0
    def __eq__(self, other) -> bool:
        return isinstance(other, _NullSentinel)
    def __repr__(self) -> str: return 'NULL'


NULL_SENTINEL = _NullSentinel()


def null_safe_key(val: object) -> object:
    return NULL_SENTINEL if val is None else val


def null_safe_row_key(row: tuple) -> tuple:
    return tuple(
        NULL_SENTINEL if v is None else v for v in row)


class DistinctOperator(Operator):
    """[P12] 流式去重。逐 batch 过滤已见过的行。"""

    def __init__(self, child: Operator) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._seen: set = set()
        self._closed = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self._seen = set()
        self._closed = False
        self.child.open()

    def next_batch(self) -> Optional[VectorBatch]:
        """[P12] 流式：逐 batch 去重，不全量物化。"""
        while True:
            batch = self._next_child_batch(self.child)
            if batch is None:
                return None

            col_names = batch.column_names
            col_types = [batch.columns[n].dtype
                         for n in col_names]
            unique_rows = []

            for i in range(batch.row_count):
                row = tuple(
                    batch.columns[n].get(i)
                    for n in col_names)
                key = null_safe_row_key(row)
                if key not in self._seen:
                    self._seen.add(key)
                    unique_rows.append(list(row))

            if unique_rows:
                return VectorBatch.from_rows(
                    unique_rows, col_names, col_types)
            # 本 batch 全部重复，继续读下一个

    def close(self) -> None:
        if not self._closed:
            self.child.close()
            self._closed = True
        self._seen.clear()
