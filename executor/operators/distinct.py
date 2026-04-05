from __future__ import annotations
"""DISTINCT算子 — 去重。使用统一的NullSentinel。"""
from typing import List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from storage.types import DataType


class _NullSentinel:
    """NULL哨兵 — 在set/dict中使NULL可哈希且互相相等。"""
    __slots__ = ()
    def __hash__(self) -> int:
        return 0
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _NullSentinel)
    def __repr__(self) -> str:
        return 'NULL'


NULL_SENTINEL = _NullSentinel()


def null_safe_key(val: object) -> object:
    """将None替换为哨兵，使其可哈希。"""
    return NULL_SENTINEL if val is None else val


def null_safe_row_key(row: tuple) -> tuple:
    """对整行做null安全转换。"""
    return tuple(NULL_SENTINEL if v is None else v for v in row)


class DistinctOperator(Operator):
    """管道阻断算子，用set去重。"""

    def __init__(self, child: Operator) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self.child.open()
        seen: set = set()
        unique_rows: list = []
        col_names = None
        col_types = None

        while True:
            batch = self._ensure_batch(self.child.next_batch())
            if batch is None:
                break
            if col_names is None:
                col_names = batch.column_names
                col_types = [batch.columns[n].dtype for n in col_names]
            for i in range(batch.row_count):
                row = tuple(batch.columns[n].get(i) for n in col_names)
                key = null_safe_row_key(row)
                if key not in seen:
                    seen.add(key)
                    unique_rows.append(list(row))

        self.child.close()

        if col_names and unique_rows:
            self._result = VectorBatch.from_rows(
                unique_rows, col_names, col_types)
        else:
            schema = self.output_schema()
            self._result = VectorBatch.empty(
                [n for n, _ in schema], [t for _, t in schema])
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted or self._result is None:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        pass
