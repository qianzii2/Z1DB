from __future__ import annotations
"""UNION / INTERSECT / EXCEPT 算子。
[P13] UNION ALL 改为流式（不物化全部数据）。"""
from typing import List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.operators.distinct import null_safe_row_key
from storage.types import DataType


def _drain_to_rows(op: Operator) -> list:
    """排空算子收集所有行。"""
    rows = []
    while True:
        b = Operator._ensure_batch(op.next_batch())
        if b is None: break
        rows.extend(b.to_rows())
    op.close()
    return rows


class UnionOperator(Operator):
    """UNION [ALL]。[P13] ALL 模式流式输出，不物化。"""

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

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema()

    def open(self) -> None:
        self.left.open()
        self.right.open()
        self._emitted = False  # [NB04] 两种模式都需要
        if self._all:
            self._left_done = False
            self._right_done = False
        else:
            # 去重模式：全量物化
            left_rows = _drain_to_rows(self.left)
            right_rows = _drain_to_rows(self.right)
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
                self.left.close()
            if not self._right_done:
                batch = self._ensure_batch(
                    self.right.next_batch())
                if batch is not None:
                    return batch
                self._right_done = True
                self.right.close()
            return None
        else:
            if self._emitted: return None
            self._emitted = True
            return self._result

    def close(self) -> None:
        if self._all:
            if not self._left_done:
                try: self.left.close()
                except Exception: pass
            if not self._right_done:
                try: self.right.close()
                except Exception: pass


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

    def output_schema(self):
        return self.left.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        lr = _drain_to_rows(self.left)
        rr = _drain_to_rows(self.right)
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

    def close(self): pass


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

    def output_schema(self):
        return self.left.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        lr = _drain_to_rows(self.left)
        rr = _drain_to_rows(self.right)
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

    def close(self): pass
