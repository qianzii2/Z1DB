from __future__ import annotations
"""Radix 分区哈希连接 + WriteCombiningBuffer。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.operators.join.hash_join import (
    extract_equi_keys, _ensure)
from metal.hash import z1hash64
from storage.types import DataType

try:
    from metal.advanced_hash import WriteCombiningBuffer
    _HAS_WCB = True
except ImportError:
    _HAS_WCB = False


class RadixJoinOperator(Operator):
    RADIX_BITS = 8
    NUM_PARTITIONS = 1 << RADIX_BITS

    def __init__(self, left, right, join_type, on_expr):
        super().__init__()
        self.left = left; self.right = right
        self.children = [left, right]
        self._join_type = join_type
        self._on_expr = on_expr
        self._evaluator = ExpressionEvaluator()
        self._result_rows: list = []
        self._emitted = False
        self._out_names: list = []
        self._out_types: list = []
        self._left_key, self._right_key = (
            extract_equi_keys(on_expr))

    def output_schema(self):
        return (self.left.output_schema()
                + self.right.output_schema())

    def open(self):
        self.left.open(); self.right.open()
        schema = self.output_schema()
        self._out_names = [n for n, _ in schema]
        self._out_types = [t for _, t in schema]
        left_rows = self._collect(self.left)
        right_rows = self._collect(self.right)
        self.left.close(); self.right.close()
        l_parts, r_parts = self._partition(
            left_rows, right_rows)
        self._result_rows = []
        right_matched_global: set = set()
        for p in range(self.NUM_PARTITIONS):
            lp = l_parts[p] if p < len(l_parts) else []
            rp = r_parts[p] if p < len(r_parts) else []
            if not lp and self._join_type not in (
                    'LEFT', 'FULL'): continue
            if not rp:
                if self._join_type in ('LEFT', 'FULL'):
                    for lr in lp:
                        self._result_rows.append(
                            [lr.get(n)
                             for n in self._out_names])
                continue
            ht: Dict[Any, List[Tuple[Tuple, dict]]] = {}
            for ri, rr in enumerate(rp):
                k = (rr.get(self._right_key)
                     if self._right_key else None)
                gid = (p, ri)
                ht.setdefault(k, []).append((gid, rr))
            for lr in lp:
                lk = (lr.get(self._left_key)
                      if self._left_key else None)
                found = False
                candidates = (
                    ht.get(lk, []) if self._left_key
                    else [item for bucket in ht.values()
                          for item in bucket])
                for gid, rr in candidates:
                    combined = {**lr, **rr}
                    if (self._on_expr is None
                            or self._eval_cond(
                                combined, schema)):
                        self._result_rows.append(
                            [combined.get(n)
                             for n in self._out_names])
                        found = True
                        right_matched_global.add(gid)
                if not found and self._join_type in (
                        'LEFT', 'FULL'):
                    self._result_rows.append(
                        [lr.get(n)
                         for n in self._out_names])
        if self._join_type in ('RIGHT', 'FULL'):
            for p in range(self.NUM_PARTITIONS):
                rp = r_parts[p] if p < len(r_parts) else []
                for ri, rr in enumerate(rp):
                    if (p, ri) not in right_matched_global:
                        self._result_rows.append(
                            [rr.get(n)
                             for n in self._out_names])
        self._emitted = False

    def _partition(self, left_rows, right_rows):
        if _HAS_WCB and len(left_rows) > 1000:
            return self._partition_wcb(
                left_rows, right_rows)
        return self._partition_simple(
            left_rows, right_rows)

    def _partition_wcb(self, left_rows, right_rows):
        l_buf = WriteCombiningBuffer(self.NUM_PARTITIONS)
        r_buf = WriteCombiningBuffer(self.NUM_PARTITIONS)
        for row in left_rows:
            p = (self._key_hash(row, self._left_key)
                 & (self.NUM_PARTITIONS - 1))
            l_buf.write(p, row)
        for row in right_rows:
            p = (self._key_hash(row, self._right_key)
                 & (self.NUM_PARTITIONS - 1))
            r_buf.write(p, row)
        l_buf.flush_all(); r_buf.flush_all()
        return ([l_buf.get_partition(p)
                 for p in range(self.NUM_PARTITIONS)],
                [r_buf.get_partition(p)
                 for p in range(self.NUM_PARTITIONS)])

    def _partition_simple(self, left_rows, right_rows):
        l_parts = [[] for _ in range(self.NUM_PARTITIONS)]
        r_parts = [[] for _ in range(self.NUM_PARTITIONS)]
        for row in left_rows:
            p = (self._key_hash(row, self._left_key)
                 & (self.NUM_PARTITIONS - 1))
            l_parts[p].append(row)
        for row in right_rows:
            p = (self._key_hash(row, self._right_key)
                 & (self.NUM_PARTITIONS - 1))
            r_parts[p].append(row)
        return l_parts, r_parts

    def _collect(self, op):
        rows = []
        while True:
            b = op.next_batch()
            if b is None: break
            b = _ensure(b)
            for i in range(b.row_count):
                rows.append(
                    {n: b.columns[n].get(i)
                     for n in b.column_names})
        return rows

    def _key_hash(self, row, key_name):
        val = row.get(key_name) if key_name else None
        return z1hash64(str(val).encode('utf-8'))

    def _eval_cond(self, combined, schema):
        cols = {}
        for cn, ct in schema:
            val = combined.get(cn)
            cols[cn] = DataVector.from_scalar(
                val, ct if val is not None else DataType.INT)
        batch = VectorBatch(
            columns=cols,
            _column_order=[n for n, _ in schema],
            _row_count=1)
        try:
            return self._evaluator.evaluate_predicate(
                self._on_expr, batch).get_bit(0)
        except Exception: return False

    def next_batch(self):
        if self._emitted: return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(
                self._out_names, self._out_types)
        return VectorBatch.from_rows(
            self._result_rows, self._out_names,
            self._out_types)

    def close(self): pass
