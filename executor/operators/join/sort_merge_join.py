from __future__ import annotations
"""排序归并连接 — O(n+m) 归并。[P03] 排序 key 安全比较。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.operators.join.hash_join import extract_equi_keys
from storage.types import DataType


class SortMergeJoinOperator(Operator):
    def __init__(self, left: Operator, right: Operator,
                 join_type: str, on_expr: Any) -> None:
        super().__init__()
        self.left = left; self.right = right
        self.children = [left, right]
        self._join_type = join_type
        self._on_expr = on_expr
        self._result_rows: list = []
        self._out_names: list = []
        self._out_types: list = []
        self._emitted = False
        left_key, right_key = extract_equi_keys(on_expr)
        self._left_key = left_key or ''
        self._right_key = right_key or ''

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        schema = self.output_schema()
        self._out_names = [n for n, _ in schema]
        self._out_types = [t for _, t in schema]
        l_names = [n for n, _ in self.left.output_schema()]
        r_names = [n for n, _ in self.right.output_schema()]
        left_col_count = len(l_names)
        l_rows = self._collect_and_sort(self.left, l_names, self._left_key)
        r_rows = self._collect_and_sort(self.right, r_names, self._right_key)
        self.left.close(); self.right.close()

        self._result_rows = []
        li = ri = 0
        right_matched: set = set()

        while li < len(l_rows) and ri < len(r_rows):
            lk = l_rows[li][0]; rk = r_rows[ri][0]
            if lk is None:
                if self._join_type in ('LEFT', 'FULL'):
                    self._result_rows.append(l_rows[li][1] + [None] * len(r_names))
                li += 1; continue
            if rk is None:
                if self._join_type in ('RIGHT', 'FULL'):
                    self._result_rows.append([None] * left_col_count + r_rows[ri][1])
                    right_matched.add(ri)
                ri += 1; continue
            cmp = self._safe_cmp(lk, rk)
            if cmp < 0:
                if self._join_type in ('LEFT', 'FULL'):
                    self._result_rows.append(l_rows[li][1] + [None] * len(r_names))
                li += 1
            elif cmp > 0:
                if self._join_type in ('RIGHT', 'FULL'):
                    self._result_rows.append([None] * left_col_count + r_rows[ri][1])
                    right_matched.add(ri)
                ri += 1
            else:
                li_start = li
                while li < len(l_rows) and self._safe_cmp(l_rows[li][0], lk) == 0:
                    li += 1
                ri_start = ri
                while ri < len(r_rows) and self._safe_cmp(r_rows[ri][0], rk) == 0:
                    ri += 1
                for lj in range(li_start, li):
                    for rj in range(ri_start, ri):
                        self._result_rows.append(l_rows[lj][1] + r_rows[rj][1])
                        right_matched.add(rj)
        while li < len(l_rows):
            if self._join_type in ('LEFT', 'FULL'):
                self._result_rows.append(l_rows[li][1] + [None] * len(r_names))
            li += 1
        while ri < len(r_rows):
            if self._join_type in ('RIGHT', 'FULL'):
                if ri not in right_matched:
                    self._result_rows.append([None] * left_col_count + r_rows[ri][1])
                    right_matched.add(ri)
            ri += 1

        self._emitted = False

    def _collect_and_sort(self, op, names, key_col):
        rows = []
        while True:
            batch = self._ensure_batch(op.next_batch())
            if batch is None: break
            for i in range(batch.row_count):
                row = [batch.columns[n].get(i) for n in names]
                key_val = batch.columns[key_col].get(i) if key_col in batch.columns else None
                rows.append((key_val, row))
        # [P03] 安全排序：NULL 排最后，不同类型用 str() 回退
        rows.sort(key=lambda x: _SortableKey(x[0]))
        return rows

    @staticmethod
    def _safe_cmp(a, b) -> int:
        """[P03] 类型安全比较。"""
        if a is None and b is None: return 0
        if a is None: return 1
        if b is None: return -1
        try:
            return (a > b) - (a < b)
        except TypeError:
            return (str(a) > str(b)) - (str(a) < str(b))

    def next_batch(self):
        if self._emitted: return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(self._result_rows, self._out_names, self._out_types)

    def close(self):
        try:
            self.left.close()
        except Exception:
            pass
        try:
            self.right.close()
        except Exception:
            pass


class _SortableKey:
    """[P03] 包装任意类型为可安全排序的 key。NULL 排最后。"""
    __slots__ = ('val',)
    def __init__(self, val): self.val = val
    def __lt__(self, other):
        if self.val is None: return False
        if other.val is None: return True
        try: return self.val < other.val
        except TypeError: return str(self.val) < str(other.val)
    def __eq__(self, other):
        if self.val is None and other.val is None: return True
        if self.val is None or other.val is None: return False
        try: return self.val == other.val
        except TypeError: return str(self.val) == str(other.val)
    def __le__(self, other): return self.__lt__(other) or self.__eq__(other)
    def __gt__(self, other): return not self.__le__(other)
    def __ge__(self, other): return not self.__lt__(other)
