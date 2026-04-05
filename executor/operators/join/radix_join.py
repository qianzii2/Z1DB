from __future__ import annotations
"""Radix分区哈希连接 — 按 join key 哈希分区，分区内建哈希表探测。
修复：全局行ID不再用 p*100000+ri，改用连续递增ID。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.operators.join.hash_join import extract_equi_keys, _ensure
from metal.hash import murmur3_64
from storage.types import DataType


class RadixJoinOperator(Operator):
    """Radix分区连接，适合10万-1000万行。"""

    RADIX_BITS = 8
    NUM_PARTITIONS = 1 << RADIX_BITS

    def __init__(self, left: Operator, right: Operator,
                 join_type: str, on_expr: Any) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.children = [left, right]
        self._join_type = join_type
        self._on_expr = on_expr
        self._evaluator = ExpressionEvaluator()
        self._result_rows: list = []
        self._emitted = False
        self._out_names: list = []
        self._out_types: list = []
        self._left_key, self._right_key = extract_equi_keys(on_expr)

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open()
        self.right.open()
        schema = self.output_schema()
        self._out_names = [n for n, _ in schema]
        self._out_types = [t for _, t in schema]

        left_rows = self._collect(self.left)
        right_rows = self._collect(self.right)
        self.left.close()
        self.right.close()

        # 按 join key 哈希分区
        l_parts: List[list] = [[] for _ in range(self.NUM_PARTITIONS)]
        r_parts: List[List[Tuple[int, dict]]] = [
            [] for _ in range(self.NUM_PARTITIONS)]

        for row in left_rows:
            p = self._key_hash(row, self._left_key) & (
                self.NUM_PARTITIONS - 1)
            l_parts[p].append(row)

        # 修复：用连续递增的全局ID，不再用 p*100000+ri
        global_ri = 0
        for row in right_rows:
            p = self._key_hash(row, self._right_key) & (
                self.NUM_PARTITIONS - 1)
            r_parts[p].append((global_ri, row))
            global_ri += 1

        # 分区内连接
        self._result_rows = []
        right_matched_global: set = set()

        for p in range(self.NUM_PARTITIONS):
            if not l_parts[p] and self._join_type not in ('LEFT', 'FULL'):
                continue
            if not r_parts[p]:
                if self._join_type in ('LEFT', 'FULL'):
                    for lr in l_parts[p]:
                        self._result_rows.append(
                            [lr.get(n) for n in self._out_names])
                continue

            # 分区内建哈希表
            ht: Dict[Any, List[Tuple[int, dict]]] = {}
            for gri, rr in r_parts[p]:
                k = rr.get(self._right_key) if self._right_key else None
                ht.setdefault(k, []).append((gri, rr))

            for lr in l_parts[p]:
                lk = lr.get(self._left_key) if self._left_key else None
                found = False
                candidates = ht.get(lk, []) if self._left_key else [
                    item for bucket in ht.values() for item in bucket]
                for gri, rr in candidates:
                    combined = {**lr, **rr}
                    if self._on_expr is None or self._eval_cond(
                            combined, schema):
                        self._result_rows.append(
                            [combined.get(n) for n in self._out_names])
                        found = True
                        right_matched_global.add(gri)
                if not found and self._join_type in ('LEFT', 'FULL'):
                    self._result_rows.append(
                        [lr.get(n) for n in self._out_names])

        if self._join_type in ('RIGHT', 'FULL'):
            for p in range(self.NUM_PARTITIONS):
                for gri, rr in r_parts[p]:
                    if gri not in right_matched_global:
                        self._result_rows.append(
                            [rr.get(n) for n in self._out_names])
        self._emitted = False

    def _collect(self, op: Operator) -> List[Dict[str, Any]]:
        rows = []
        while True:
            b = op.next_batch()
            if b is None:
                break
            b = _ensure(b)
            for i in range(b.row_count):
                rows.append(
                    {n: b.columns[n].get(i) for n in b.column_names})
        return rows

    def _key_hash(self, row: Dict[str, Any],
                  key_name: Optional[str]) -> int:
        val = row.get(key_name) if key_name else None
        return murmur3_64(str(val).encode('utf-8'))

    def _eval_cond(self, combined: dict, schema: list) -> bool:
        cols = {}
        for cn, ct in schema:
            val = combined.get(cn)
            cols[cn] = DataVector.from_scalar(
                val, ct if val is not None else DataType.INT)
        batch = VectorBatch(
            columns=cols,
            _column_order=[n for n, _ in schema], _row_count=1)
        try:
            return self._evaluator.evaluate_predicate(
                self._on_expr, batch).get_bit(0)
        except Exception:
            return False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(
            self._result_rows, self._out_names, self._out_types)

    def close(self) -> None:
        pass
