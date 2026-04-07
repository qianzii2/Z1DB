from __future__ import annotations
"""UNION / INTERSECT / EXCEPT 算子。
[P13] UNION ALL 改为流式（不物化全部数据）。
[FIX] close() 避免重复关闭已在 open/next_batch 中关闭的子算子。"""
from typing import List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.operators.distinct import null_safe_row_key
from storage.types import DataType


def _drain_to_rows(op: Operator) -> list:
    """排空算子收集所有行。注意：不关闭算子，由调用方管理。"""
    rows = []
    while True:
        b = Operator._ensure_batch(op.next_batch())
        if b is None: break
        rows.extend(b.to_rows())
    return rows


class UnionOperator(Operator):
    """UNION [ALL]。[P13] ALL 模式流式输出，不物化。
    [FIX] 统一用 _left_closed/_right_closed 跟踪关闭状态，避免重复关闭。"""

    def __init__(self, left: Operator, right: Operator,
                 all_: bool = False) -> None:
        super().__init__()
        self.left = left; self.right = right
        self.children = [left, right]
        self._all = all_
        # 非 ALL 模式：物化去重
        self._result: Optional[VectorBatch] = None
        self._emitted = False
        # ALL 模式：流式
        self._left_done = False
        self._right_done = False
        # [FIX] 统一关闭状态跟踪
        self._left_closed = False
        self._right_closed = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema()

    def open(self) -> None:
        self.left.open()
        self.right.open()
        self._emitted = False
        self._left_closed = False
        self._right_closed = False
        if self._all:
            self._left_done = False
            self._right_done = False
        else:
            # 去重模式：全量物化
            left_rows = _drain_to_rows(self.left)
            right_rows = _drain_to_rows(self.right)
            # [FIX] 物化后立即关闭子算子并标记
            self.left.close(); self._left_closed = True
            self.right.close(); self._right_closed = True
            combined = left_rows + right_rows
            seen: set = set()
            unique = []
            for row in combined:
                key = null_safe_row_key(tuple(row))
                if key not in seen:
                    seen.add(key)
                    unique.append(row)
            schema = self.output_schema()
            names = [n for n, _ in schema]
            types = [t for _, t in schema]
            self._result = (
                VectorBatch.from_rows(unique, names, types)
                if unique
                else VectorBatch.empty(names, types))
            self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._all:
            # [P13] 流式：先左表再右表
            if not self._left_done:
                batch = self._ensure_batch(
                    self.left.next_batch())
                if batch is not None:
                    return batch
                self._left_done = True
                self.left.close(); self._left_closed = True
            if not self._right_done:
                batch = self._ensure_batch(
                    self.right.next_batch())
                if batch is not None:
                    return batch
                self._right_done = True
                self.right.close(); self._right_closed = True
            return None
        else:
            if self._emitted: return None
            self._emitted = True
            return self._result

    def close(self) -> None:
        # [FIX] 只关闭尚未关闭的子算子
        if not self._left_closed:
            try:
                self.left.close()
            except Exception:
                pass
            self._left_closed = True
        if not self._right_closed:
            try:
                self.right.close()
            except Exception:
                pass
            self._right_closed = True


class IntersectOperator(Operator):
    """INTERSECT [ALL]。"""

    def __init__(self, left: Operator, right: Operator,
                 all_: bool = False) -> None:
        super().__init__()
        self.left = left; self.right = right
        self.children = [left, right]
        self._all = all_
        self._result: Optional[VectorBatch] = None
        self._emitted = False
        self._closed = False

    def output_schema(self):
        return self.left.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        self._closed = False
        lr = _drain_to_rows(self.left)
        rr = _drain_to_rows(self.right)
        self.left.close(); self.right.close()
        self._closed = True
        right_set: dict = {}
        for row in rr:
            key = null_safe_row_key(tuple(row))
            right_set[key] = right_set.get(key, 0) + 1
        result = []
        for row in lr:
            key = null_safe_row_key(tuple(row))
            if key in right_set and right_set[key] > 0:
                result.append(row)
                if not self._all:
                    right_set[key] = 0
                else:
                    right_set[key] -= 1
        s = self.output_schema()
        names = [n for n, _ in s]
        types = [t for _, t in s]
        self._result = (
            VectorBatch.from_rows(result, names, types)
            if result
            else VectorBatch.empty(names, types))
        self._emitted = False

    def next_batch(self):
        if self._emitted: return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        if not self._closed:
            try:
                self.left.close()
            except Exception:
                pass
            try:
                self.right.close()
            except Exception:
                pass
            self._closed = True


class ExceptOperator(Operator):
    """EXCEPT [ALL]。"""

    def __init__(self, left: Operator, right: Operator,
                 all_: bool = False) -> None:
        super().__init__()
        self.left = left; self.right = right
        self.children = [left, right]
        self._all = all_
        self._result: Optional[VectorBatch] = None
        self._emitted = False
        self._closed = False

    def output_schema(self):
        return self.left.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        self._closed = False
        lr = _drain_to_rows(self.left)
        rr = _drain_to_rows(self.right)
        self.left.close(); self.right.close()
        self._closed = True
        right_set: dict = {}
        for row in rr:
            key = null_safe_row_key(tuple(row))
            right_set[key] = right_set.get(key, 0) + 1
        result = []
        for row in lr:
            key = null_safe_row_key(tuple(row))
            if key in right_set and right_set[key] > 0:
                if self._all:
                    right_set[key] -= 1
                else:
                    right_set[key] = 0
            else:
                result.append(row)
        s = self.output_schema()
        names = [n for n, _ in s]
        types = [t for _, t in s]
        self._result = (
            VectorBatch.from_rows(result, names, types)
            if result
            else VectorBatch.empty(names, types))
        self._emitted = False

    def next_batch(self):
        if self._emitted: return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        if not self._closed:
            try:
                self.left.close()
            except Exception:
                pass
            try:
                self.right.close()
            except Exception:
                pass
            self._closed = True
