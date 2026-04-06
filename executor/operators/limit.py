from __future__ import annotations
"""LIMIT / OFFSET 算子 — 跳过 offset 行后输出至多 limit 行。"""
from typing import List, Optional
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from storage.types import DataType


class LimitOperator(Operator):
    """流式 LIMIT/OFFSET，不物化全部数据。"""

    def __init__(self, child: Operator, limit: Optional[int],
                 offset: int = 0) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._limit = limit
        self._offset = offset
        self._rows_skipped = 0
        self._rows_emitted = 0
        self._closed = False

    def output_schema(self) -> List[tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self._closed = False
        self._rows_skipped = 0
        self._rows_emitted = 0
        self.child.open()

    def close(self) -> None:
        if not self._closed:
            self.child.close()
            self._closed = True

    def next_batch(self) -> Optional[VectorBatch]:
        # 已达 LIMIT 上限，直接结束
        if self._limit is not None and self._rows_emitted >= self._limit:
            return None

        while True:
            raw = self.child.next_batch()
            batch = self._ensure_batch(raw)
            if batch is None:
                return None
            n = batch.row_count

            # OFFSET：跳过前 offset 行
            if self._rows_skipped < self._offset:
                skip = self._offset - self._rows_skipped
                if n <= skip:
                    self._rows_skipped += n
                    continue
                batch = batch.slice(skip, n)
                self._rows_skipped = self._offset
                n = batch.row_count

            # LIMIT：截断多余行
            if self._limit is not None:
                remaining = self._limit - self._rows_emitted
                if remaining <= 0:
                    return None
                if n > remaining:
                    batch = batch.slice(0, remaining)
                    n = batch.row_count

            self._rows_emitted += n
            return batch
