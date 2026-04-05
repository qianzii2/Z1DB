from __future__ import annotations
"""Grace Hash Join — disk-spilling join for data exceeding memory.
Partitions both sides to temp files → joins partition by partition."""
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.hash import murmur3_64
from storage.types import DataType


class GraceHashJoinOperator(Operator):
    """Grace Hash Join — handles arbitrarily large datasets."""

    NUM_PARTITIONS = 64

    def __init__(self, left: Operator, right: Operator,
                 join_type: str, on_expr: Any,
                 memory_limit: int = 64 * 1024 * 1024) -> None:
        super().__init__()
        self.left = left; self.right = right; self.children = [left, right]
        self._join_type = join_type; self._on_expr = on_expr
        self._memory_limit = memory_limit
        self._evaluator = ExpressionEvaluator()
        self._result_rows: list = []; self._emitted = False
        self._out_names: list = []; self._out_types: list = []
        self._temp_dir = tempfile.mkdtemp(prefix='z1db_grace_')
        self._temp_files: list = []

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        schema = self.output_schema()
        self._out_names = [n for n, _ in schema]
        self._out_types = [t for _, t in schema]

        # Phase 1: Partition both sides to disk
        l_files = self._partition_to_disk(self.left, 'L')
        r_files = self._partition_to_disk(self.right, 'R')
        self.left.close(); self.right.close()

        # Phase 2: Join each partition pair in memory
        self._result_rows = []
        for p in range(self.NUM_PARTITIONS):
            l_rows = self._read_partition(l_files[p])
            r_rows = self._read_partition(r_files[p])
            if not l_rows and not r_rows: continue
            self._join_partition(l_rows, r_rows, schema)

        # Cleanup temp files
        self._cleanup()
        self._emitted = False

    def _partition_to_disk(self, op: Operator, prefix: str) -> List[str]:
        """Write rows to partition files."""
        files = []
        handles = []
        for p in range(self.NUM_PARTITIONS):
            path = os.path.join(self._temp_dir, f'{prefix}_{p}.json')
            files.append(path)
            handles.append(open(path, 'w'))
        col_names: Optional[list] = None
        while True:
            batch = op.next_batch()
            if batch is None: break
            if col_names is None:
                col_names = batch.column_names
            for i in range(batch.row_count):
                row = {n: batch.columns[n].get(i) for n in col_names}
                h = self._row_hash(row) % self.NUM_PARTITIONS
                handles[h].write(json.dumps(row, default=str) + '\n')
        for fh in handles:
            fh.close()
        self._temp_files.extend(files)
        return files

    def _read_partition(self, path: str) -> List[Dict[str, Any]]:
        rows = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line: rows.append(json.loads(line))
        except FileNotFoundError:
            pass
        return rows

    def _join_partition(self, l_rows: list, r_rows: list,
                        schema: list) -> None:
        """In-memory hash join for one partition."""
        if not r_rows and self._join_type not in ('LEFT', 'FULL'):
            return

        # Build HT on right
        ht: Dict[int, list] = {}
        for ri, rr in enumerate(r_rows):
            h = self._row_hash(rr)
            ht.setdefault(h, []).append((ri, rr))

        right_matched: set = set()
        for lr in l_rows:
            lh = self._row_hash(lr)
            found = False
            if lh in ht:
                for ri, rr in ht[lh]:
                    combined = {**lr, **rr}
                    if self._on_expr is None or self._eval_cond(combined, schema):
                        self._result_rows.append(
                            [combined.get(n) for n in self._out_names])
                        found = True; right_matched.add(ri)
            if not found and self._join_type in ('LEFT', 'FULL'):
                self._result_rows.append(
                    [lr.get(n) for n in self._out_names])

        if self._join_type in ('RIGHT', 'FULL'):
            for ri, rr in enumerate(r_rows):
                if ri not in right_matched:
                    self._result_rows.append(
                        [rr.get(n) for n in self._out_names])

    def _row_hash(self, row: dict) -> int:
        return murmur3_64(str(sorted(row.items())).encode('utf-8'))

    def _eval_cond(self, combined, schema):
        cols = {}
        for cn, ct in schema:
            val = combined.get(cn)
            cols[cn] = DataVector.from_scalar(val, ct if val is not None else DataType.INT)
        batch = VectorBatch(columns=cols, _column_order=[n for n,_ in schema], _row_count=1)
        try:
            return self._evaluator.evaluate_predicate(self._on_expr, batch).get_bit(0)
        except Exception: return False

    def _cleanup(self) -> None:
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
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(self._result_rows, self._out_names, self._out_types)

    def close(self): self._cleanup()
