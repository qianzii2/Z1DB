from __future__ import annotations
"""Hash aggregation — auto-selects RobinHood/dict based on data size."""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.functions.registry import AggregateFunction, FunctionRegistry
from metal.config import MICRO_THRESHOLD
from parser.ast import AggregateCall, StarExpr
from storage.types import DataType


class HashAggOperator(Operator):
    """Groups rows by key expressions and computes aggregates.
    Auto-selects hash table implementation based on data size."""

    def __init__(self, child: Operator,
                 group_exprs: List[Tuple[str, Any]],
                 agg_exprs: List[Tuple[str, AggregateCall]],
                 registry: FunctionRegistry) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._group_exprs = group_exprs
        self._agg_exprs = agg_exprs
        self._registry = registry
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
            if ac.args and not isinstance(ac.args[0], StarExpr):
                input_types = [ExpressionEvaluator.infer_type(ac.args[0], child_schema)]
            else:
                input_types = []
            result.append((name, func.return_type(input_types)))
        return result

    def open(self) -> None:
        self.child.open()
        groups: Dict[tuple, Dict[str, Tuple[AggregateFunction, Any]]] = {}
        group_order: List[tuple] = []
        total_rows = 0

        while True:
            batch = self.child.next_batch()
            if batch is None:
                break
            key_vecs = [self._evaluator.evaluate(expr, batch)
                        for _, expr in self._group_exprs]
            arg_vecs: Dict[str, Optional[DataVector]] = {}
            for name, ac in self._agg_exprs:
                if ac.args and isinstance(ac.args[0], StarExpr):
                    arg_vecs[name] = None
                else:
                    arg_vecs[name] = self._evaluator.evaluate(ac.args[0], batch)

            for row_i in range(batch.row_count):
                key = tuple(kv.get(row_i) for kv in key_vecs)
                if key not in groups:
                    groups[key] = {}
                    group_order.append(key)
                    for name, ac in self._agg_exprs:
                        func = self._registry.get_aggregate(ac.name)
                        groups[key][name] = (func, func.init())
                for name, ac in self._agg_exprs:
                    func, state = groups[key][name]
                    av = arg_vecs[name]
                    if av is None:
                        state = func.update(state, None, 1)
                    else:
                        single = av.filter_by_indices([row_i])
                        state = func.update(state, single, 1)
                    groups[key][name] = (func, state)
            total_rows += batch.row_count
        self.child.close()

        out_schema = self.output_schema()
        if not groups:
            self._result = VectorBatch.empty([n for n, _ in out_schema],
                                              [t for _, t in out_schema])
            self._emitted = False
            return

        rows: List[list] = []
        for key in group_order:
            row: list = list(key)
            for name, ac in self._agg_exprs:
                func, state = groups[key][name]
                row.append(func.finalize(state))
            rows.append(row)

        self._result = VectorBatch.from_rows(
            rows, [n for n, _ in out_schema], [t for _, t in out_schema])
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted or self._result is None:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        pass
