from __future__ import annotations
"""Radix Hash Join — cache-friendly partitioned join.
Paper: Balkesen et al., 2013.
Partitions both sides by hash → each partition fits in L2 cache."""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.hash import murmur3_64
from storage.types import DataType


class RadixJoinOperator(Operator):
    """Radix-partitioned hash join. Best for 100K-10M rows."""

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

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open()
        self.right.open()
        schema = self.output_schema()
        self._out_names = [n for n, _ in schema]
        self._out_types = [t for _, t in schema]

        # Collect all rows
        left_rows = self._collect(self.left)
        right_rows = self._collect(self.right)
        self.left.close()
        self.right.close()

        # Phase 1: Partition both sides by hash
        l_parts: List[list] = [[] for _ in range(self.NUM_PARTITIONS)]
        r_parts: List[list] = [[] for _ in range(self.NUM_PARTITIONS)]

        for row in left_rows:
            h = self._row_hash(row) & (self.NUM_PARTITIONS - 1)
            l_parts[h].append(row)
        for row in right_rows:
            h = self._row_hash(row) & (self.NUM_PARTITIONS - 1)
            r_parts[h].append(row)

        # Phase 2: Join each partition pair
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

            # Build hash table on right partition
            ht: Dict[int, list] = {}
            for ri, rr in enumerate(r_parts[p]):
                rh = self._row_hash(rr)
                ht.setdefault(rh, []).append((p, ri, rr))

            for lr in l_parts[p]:
                lh = self._row_hash(lr)
                found = False
                if lh in ht:
                    for p_idx, ri, rr in ht[lh]:
                        combined = {**lr, **rr}
                        if self._on_expr is None or self._eval_cond(combined, schema):
                            self._result_rows.append(
                                [combined.get(n) for n in self._out_names])
                            found = True
                            right_matched_global.add((p_idx, ri))
                if not found and self._join_type in ('LEFT', 'FULL'):
                    self._result_rows.append(
                        [lr.get(n) for n in self._out_names])

        if self._join_type in ('RIGHT', 'FULL'):
            for p in range(self.NUM_PARTITIONS):
                for ri, rr in enumerate(r_parts[p]):
                    if (p, ri) not in right_matched_global:
                        self._result_rows.append(
                            [rr.get(n) for n in self._out_names])
        self._emitted = False

    def _collect(self, op: Operator) -> List[Dict[str, Any]]:
        rows = []
        while True:
            b = op.next_batch()
            if b is None:
                break
            for i in range(b.row_count):
                rows.append({n: b.columns[n].get(i) for n in b.column_names})
        return rows

    def _row_hash(self, row: Dict[str, Any]) -> int:
        parts = sorted(row.items())
        return murmur3_64(str(parts).encode('utf-8'))

    def _eval_cond(self, combined: dict, schema: list) -> bool:
        cols = {}
        for cn, ct in schema:
            val = combined.get(cn)
            cols[cn] = DataVector.from_scalar(val, ct if val is not None else DataType.INT)
        batch = VectorBatch(columns=cols, _column_order=[n for n, _ in schema], _row_count=1)
        try:
            return self._evaluator.evaluate_predicate(self._on_expr, batch).get_bit(0)
        except Exception:
            return False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(self._result_rows, self._out_names, self._out_types)

    def close(self) -> None:
        pass
