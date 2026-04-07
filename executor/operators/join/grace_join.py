from __future__ import annotations
"""Grace 哈希连接 — 按 join key 哈希分区到磁盘。"""
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.operators.join.hash_join import (
    extract_equi_keys, _ensure)
from metal.hash import z1hash64
from storage.types import DataType


class GraceHashJoinOperator(Operator):
    NUM_PARTITIONS = 64

    def __init__(self, left: Operator, right: Operator,
                 join_type: str, on_expr: Any,
                 memory_limit: int = 64 * 1024 * 1024
                 ) -> None:
        super().__init__()
        self.left = left; self.right = right
        self.children = [left, right]
        self._join_type = join_type
        self._on_expr = on_expr
        self._memory_limit = memory_limit
        self._evaluator = ExpressionEvaluator()
        self._result_rows: list = []
        self._emitted = False
        self._out_names: list = []
        self._out_types: list = []
        self._temp_dir = tempfile.mkdtemp(
            prefix='z1db_grace_')
        self._temp_files: list = []
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
        l_files = self._partition_to_disk(
            self.left, 'L', self._left_key)
        r_files = self._partition_to_disk(
            self.right, 'R', self._right_key)
        self.left.close(); self.right.close()
        self._result_rows = []
        for p in range(self.NUM_PARTITIONS):
            l_rows = self._read_partition(l_files[p])
            r_rows = self._read_partition(r_files[p])
            if not l_rows and not r_rows: continue
            self._join_partition(l_rows, r_rows, schema)
        self._cleanup(); self._emitted = False

    def _partition_to_disk(self, op, prefix, key_name):
        """[P08] 二进制分区写入。"""
        from executor.spill.temp_file import BinaryTempFile
        files = []
        handles = []
        try:
            for p in range(self.NUM_PARTITIONS):
                path = os.path.join(
                    self._temp_dir, f'{prefix}_{p}.bin')
                files.append(path)
                handles.append(BinaryTempFile(path))
            while True:
                batch = op.next_batch()
                if batch is None:
                    break
                batch = _ensure(batch)
                col_names = batch.column_names
                for i in range(batch.row_count):
                    row = {n: batch.columns[n].get(i)
                           for n in col_names}
                    val = row.get(key_name) if key_name else None
                    h = z1hash64(
                        str(val).encode('utf-8')
                    ) % self.NUM_PARTITIONS
                    handles[h].write_row(row)
        finally:
            for fh in handles:
                fh.close()
        self._temp_files.extend(files)
        return files

    def _read_partition(self, path):
        """[P08] 二进制分区读取。"""
        from executor.spill.temp_file import BinaryTempFile
        btf = BinaryTempFile.__new__(BinaryTempFile)
        btf._path = path
        btf._closed = True
        btf._count = 0
        return btf.read_all()

    def _join_partition(self, l_rows, r_rows, schema):
        if (not r_rows
                and self._join_type not in ('LEFT', 'FULL')):
            return
        ht: Dict[Any, List[Tuple[int, dict]]] = {}
        for ri, rr in enumerate(r_rows):
            k = (rr.get(self._right_key)
                 if self._right_key else None)
            ht.setdefault(k, []).append((ri, rr))
        right_matched: set = set()
        for lr in l_rows:
            lk = (lr.get(self._left_key)
                  if self._left_key else None)
            found = False
            candidates = (
                ht.get(lk, []) if self._left_key
                else [item for bucket in ht.values()
                      for item in bucket])
            for ri, rr in candidates:
                combined = {**lr, **rr}
                if (self._on_expr is None
                        or self._eval_cond(
                            combined, schema)):
                    self._result_rows.append(
                        [combined.get(n)
                         for n in self._out_names])
                    found = True; right_matched.add(ri)
            if not found and self._join_type in (
                    'LEFT', 'FULL'):
                self._result_rows.append(
                    [lr.get(n) for n in self._out_names])
        if self._join_type in ('RIGHT', 'FULL'):
            for ri, rr in enumerate(r_rows):
                if ri not in right_matched:
                    self._result_rows.append(
                        [rr.get(n)
                         for n in self._out_names])

    def _eval_cond(self, combined, schema):
        from executor.operators.join.join_utils import eval_join_condition
        return eval_join_condition(combined, schema, self._on_expr)

    def _cleanup(self):
        for path in self._temp_files:
            try: os.unlink(path)
            except OSError: pass
        try: os.rmdir(self._temp_dir)
        except OSError: pass
        self._temp_files = []

    def next_batch(self):
        if self._emitted: return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(
                self._out_names, self._out_types)
        return VectorBatch.from_rows(
            self._result_rows, self._out_names,
            self._out_types)

    def close(self):
        self._cleanup()
        try:
            self.left.close()
        except Exception:
            pass
        try:
            self.right.close()
        except Exception:
            pass

