from __future__ import annotations
"""UNION / INTERSECT / EXCEPT operators."""
from typing import List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from storage.types import DataType


class UnionOperator(Operator):
    """UNION [ALL] of two operator trees."""

    def __init__(self, left: Operator, right: Operator, all_: bool = False) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.children = [left, right]
        self._all = all_
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema()

    def open(self) -> None:
        self.left.open()
        self.right.open()
        left_rows = []
        while True:
            b = self.left.next_batch()
            if b is None:
                break
            left_rows.extend(b.to_rows())
        self.left.close()
        right_rows = []
        while True:
            b = self.right.next_batch()
            if b is None:
                break
            right_rows.extend(b.to_rows())
        self.right.close()

        combined = left_rows + right_rows
        if not self._all:
            seen: set = set()
            unique = []
            for row in combined:
                key = tuple(_ns(v) for v in row)
                if key not in seen:
                    seen.add(key)
                    unique.append(row)
            combined = unique

        schema = self.output_schema()
        names = [n for n, _ in schema]
        types = [t for _, t in schema]
        self._result = VectorBatch.from_rows(combined, names, types) if combined else VectorBatch.empty(names, types)
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        pass


class IntersectOperator(Operator):
    """INTERSECT [ALL]."""

    def __init__(self, left: Operator, right: Operator, all_: bool = False) -> None:
        super().__init__()
        self.left = left; self.right = right; self.children = [left, right]
        self._all = all_
        self._result: Optional[VectorBatch] = None; self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        lr = []; rr = []
        while True:
            b = self.left.next_batch()
            if b is None: break
            lr.extend(b.to_rows())
        self.left.close()
        while True:
            b = self.right.next_batch()
            if b is None: break
            rr.extend(b.to_rows())
        self.right.close()
        right_set = {}
        for row in rr:
            key = tuple(_ns(v) for v in row)
            right_set[key] = right_set.get(key, 0) + 1
        result = []
        for row in lr:
            key = tuple(_ns(v) for v in row)
            if key in right_set and right_set[key] > 0:
                result.append(row)
                if not self._all:
                    right_set[key] = 0
                else:
                    right_set[key] -= 1
        s = self.output_schema()
        self._result = VectorBatch.from_rows(result, [n for n,_ in s], [t for _,t in s]) if result else VectorBatch.empty([n for n,_ in s], [t for _,t in s])
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted: return None
        self._emitted = True; return self._result

    def close(self) -> None: pass


class ExceptOperator(Operator):
    """EXCEPT [ALL]."""

    def __init__(self, left: Operator, right: Operator, all_: bool = False) -> None:
        super().__init__()
        self.left = left; self.right = right; self.children = [left, right]
        self._all = all_
        self._result: Optional[VectorBatch] = None; self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        lr = []; rr = []
        while True:
            b = self.left.next_batch()
            if b is None: break
            lr.extend(b.to_rows())
        self.left.close()
        while True:
            b = self.right.next_batch()
            if b is None: break
            rr.extend(b.to_rows())
        self.right.close()
        right_set = {}
        for row in rr:
            key = tuple(_ns(v) for v in row)
            right_set[key] = right_set.get(key, 0) + 1
        result = []
        for row in lr:
            key = tuple(_ns(v) for v in row)
            if key in right_set and right_set[key] > 0:
                if self._all:
                    right_set[key] -= 1
                else:
                    right_set[key] = 0
            else:
                result.append(row)
        s = self.output_schema()
        self._result = VectorBatch.from_rows(result, [n for n,_ in s], [t for _,t in s]) if result else VectorBatch.empty([n for n,_ in s], [t for _,t in s])
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted: return None
        self._emitted = True; return self._result

    def close(self) -> None: pass


class _NS:
    """Null-safe sentinel for hashing."""
    def __hash__(self) -> int: return 0
    def __eq__(self, o: object) -> bool: return isinstance(o, _NS)
_NSV = _NS()
def _ns(v: object) -> object:
    return _NSV if v is None else v
