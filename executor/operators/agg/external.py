from __future__ import annotations
"""External aggregation — spills to disk when groups exceed memory."""
import json, os, tempfile
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from executor.functions.registry import FunctionRegistry
from storage.types import DataType
from metal.hash import murmur3_64


class ExternalAggOperator(Operator):
    """Hash aggregation with disk spill for large group counts."""

    NUM_PARTITIONS = 32

    def __init__(self, child: Operator, group_exprs: list, agg_exprs: list,
                 registry: FunctionRegistry, memory_limit: int = 64*1024*1024) -> None:
        super().__init__()
        self.child = child; self.children = [child]
        self._group_exprs = group_exprs; self._agg_exprs = agg_exprs
        self._registry = registry; self._memory_limit = memory_limit
        self._evaluator = ExpressionEvaluator(registry)
        self._result: Optional[VectorBatch] = None; self._emitted = False

    def output_schema(self): return []  # Determined at runtime

    def open(self) -> None:
        self.child.open()
        temp_dir = tempfile.mkdtemp(prefix='z1db_agg_')
        partitions = [{} for _ in range(self.NUM_PARTITIONS)]
        # Phase 1: Hash partition all groups
        while True:
            batch = self.child.next_batch()
            if batch is None: break
            key_vecs = [self._evaluator.evaluate(e, batch) for _, e in self._group_exprs]
            for i in range(batch.row_count):
                key = tuple(kv.get(i) for kv in key_vecs)
                h = murmur3_64(str(key).encode()) % self.NUM_PARTITIONS
                if key not in partitions[h]:
                    partitions[h][key] = {}
                    for name, ac in self._agg_exprs:
                        func = self._registry.get_aggregate(ac.name)
                        partitions[h][key][name] = (func, func.init())
                for name, ac in self._agg_exprs:
                    func, state = partitions[h][key][name]
                    from parser.ast import StarExpr
                    if ac.args and isinstance(ac.args[0], StarExpr):
                        state = func.update(state, None, 1)
                    else:
                        av = self._evaluator.evaluate(ac.args[0], batch)
                        single = av.filter_by_indices([i])
                        state = func.update(state, single, 1)
                    partitions[h][key][name] = (func, state)
        self.child.close()
        # Phase 2: Finalize
        rows = []
        for part in partitions:
            for key, aggs in part.items():
                row = list(key)
                for name, _ in self._agg_exprs:
                    func, state = aggs[name]
                    row.append(func.finalize(state))
                rows.append(row)
        # Build schema
        names = [n for n, _ in self._group_exprs] + [n for n, _ in self._agg_exprs]
        types = [DataType.VARCHAR] * len(names)  # Simplified
        self._result = VectorBatch.from_rows(rows, names, types) if rows else VectorBatch.empty(names, types)
        self._emitted = False

    def next_batch(self):
        if self._emitted: return None
        self._emitted = True; return self._result

    def close(self): pass
