from __future__ import annotations
"""外部聚合 — 组数超出内存时溢写到磁盘。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from executor.functions.registry import FunctionRegistry
from storage.types import DataType
from metal.hash import murmur3_64

try:
    from executor.spill.hash_spill import HashSpill
    from executor.spill.manager import SpillManager
    _HAS_SPILL = True
except ImportError:
    _HAS_SPILL = False


class ExternalAggOperator(Operator):
    """哈希聚合 + 磁盘溢写。组数过多时分区到磁盘再逐分区聚合。"""

    NUM_PARTITIONS = 32
    MEMORY_GROUP_LIMIT = 50000

    def __init__(self, child: Operator, group_exprs: list,
                 agg_exprs: list, registry: FunctionRegistry,
                 memory_limit: int = 64 * 1024 * 1024) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._group_exprs = group_exprs
        self._agg_exprs = agg_exprs
        self._registry = registry
        self._memory_limit = memory_limit
        self._evaluator = ExpressionEvaluator(registry)
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        child_schema = dict(self.child.output_schema())
        result: List[Tuple[str, DataType]] = []
        for name, expr in self._group_exprs:
            dt = ExpressionEvaluator.infer_type(expr, child_schema)
            result.append((name, dt))
        for name, ac in self._agg_exprs:
            func = self._registry.get_aggregate(ac.name)
            from parser.ast import StarExpr
            if ac.args and not isinstance(ac.args[0], StarExpr):
                input_types = [ExpressionEvaluator.infer_type(
                    ac.args[0], child_schema)]
            else:
                input_types = []
            result.append((name, func.return_type(input_types)))
        return result

    def open(self) -> None:
        self.child.open()

        all_key_rows: List[list] = []
        all_agg_rows: List[list] = []

        while True:
            batch = self._ensure_batch(self.child.next_batch())
            if batch is None:
                break
            key_vecs = [self._evaluator.evaluate(e, batch)
                        for _, e in self._group_exprs]
            agg_vecs = []
            for _, ac in self._agg_exprs:
                from parser.ast import StarExpr
                if ac.args and isinstance(ac.args[0], StarExpr):
                    agg_vecs.append(None)
                else:
                    agg_vecs.append(
                        self._evaluator.evaluate(ac.args[0], batch))

            for i in range(batch.row_count):
                key_row = [kv.get(i) for kv in key_vecs]
                agg_row = [
                    av.get(i) if av is not None else 1
                    for av in agg_vecs]
                all_key_rows.append(key_row)
                all_agg_rows.append(agg_row)

        self.child.close()

        total_groups_estimate = len(set(
            tuple(kr) for kr in all_key_rows))

        if total_groups_estimate <= self.MEMORY_GROUP_LIMIT or not _HAS_SPILL:
            rows = self._aggregate_in_memory(all_key_rows, all_agg_rows)
        else:
            rows = self._aggregate_with_spill(all_key_rows, all_agg_rows)

        out_schema = self.output_schema()
        names = [n for n, _ in out_schema]
        types = [t for _, t in out_schema]
        self._result = (VectorBatch.from_rows(rows, names, types)
                        if rows
                        else VectorBatch.empty(names, types))
        self._emitted = False

    def _aggregate_in_memory(self, key_rows: list,
                             agg_rows: list) -> List[list]:
        groups: Dict[tuple, list] = {}
        group_order: list = []
        for ki in range(len(key_rows)):
            key = tuple(key_rows[ki])
            if key not in groups:
                groups[key] = self._init_agg_states()
                group_order.append(key)
            self._update_agg_states(groups[key], agg_rows[ki])

        rows = []
        for key in group_order:
            row = list(key)
            for si, (_, ac) in enumerate(self._agg_exprs):
                func, state = groups[key][si]
                row.append(func.finalize(state))
            rows.append(row)
        return rows

    def _aggregate_with_spill(self, key_rows: list,
                              agg_rows: list) -> List[list]:
        manager = SpillManager()
        spill = HashSpill(self.NUM_PARTITIONS, manager)

        for ki in range(len(key_rows)):
            combined = key_rows[ki] + agg_rows[ki]
            key_str = str(tuple(key_rows[ki]))
            spill.spill_row(key_str, combined)

        num_key_cols = len(self._group_exprs)
        all_rows = []
        for p in range(self.NUM_PARTITIONS):
            part_rows = spill.read_partition(p)
            if not part_rows:
                continue
            groups: Dict[tuple, list] = {}
            group_order: list = []
            for row in part_rows:
                key = tuple(row[:num_key_cols])
                agg_vals = row[num_key_cols:]
                if key not in groups:
                    groups[key] = self._init_agg_states()
                    group_order.append(key)
                self._update_agg_states(groups[key], agg_vals)
            for key in group_order:
                result_row = list(key)
                for si, (_, ac) in enumerate(self._agg_exprs):
                    func, state = groups[key][si]
                    result_row.append(func.finalize(state))
                all_rows.append(result_row)

        spill.cleanup()
        return all_rows

    def _init_agg_states(self) -> list:
        states = []
        for _, ac in self._agg_exprs:
            func = self._registry.get_aggregate(ac.name)
            states.append((func, func.init()))
        return states

    def _update_agg_states(self, states: list, agg_vals: list) -> None:
        for si in range(len(states)):
            func, state = states[si]
            val = agg_vals[si] if si < len(agg_vals) else None
            from executor.core.vector import DataVector
            vec = DataVector.from_scalar(
                val, DataType.DOUBLE if isinstance(val, float)
                else DataType.BIGINT)
            state = func.update(state, vec, 1)
            states[si] = (func, state)

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        pass
