from __future__ import annotations
"""哈希聚合 — 按batch批量收集行号再聚合，消除逐行filter_by_indices。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.functions.registry import AggregateFunction, FunctionRegistry
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from parser.ast import AggregateCall, StarExpr
from storage.types import DTYPE_TO_ARRAY_CODE, DataType


class HashAggOperator(Operator):
    """按key分组并计算聚合。批量收集同组行号后一次性构建子向量。"""

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
                input_types = [ExpressionEvaluator.infer_type(
                    ac.args[0], child_schema)]
            else:
                input_types = []
            result.append((name, func.return_type(input_types)))
        return result

    def open(self) -> None:
        self.child.open()

        # 阶段1：按batch收集每个group的行号列表
        groups: Dict[tuple, _GroupState] = {}
        group_order: List[tuple] = []

        while True:
            raw = self.child.next_batch()
            batch = self._ensure_batch(raw)
            if batch is None:
                break

            key_vecs = [self._evaluator.evaluate(expr, batch)
                        for _, expr in self._group_exprs]
            # 预评估聚合参数向量（整个batch一次）
            arg_vecs: Dict[str, Optional[DataVector]] = {}
            for name, ac in self._agg_exprs:
                if ac.args and isinstance(ac.args[0], StarExpr):
                    arg_vecs[name] = None
                else:
                    arg_vecs[name] = self._evaluator.evaluate(
                        ac.args[0], batch)

            # 按行分组，收集行号
            batch_group_rows: Dict[tuple, List[int]] = {}
            for row_i in range(batch.row_count):
                key = tuple(kv.get(row_i) for kv in key_vecs)
                if key not in groups:
                    groups[key] = _GroupState(self._agg_exprs,
                                             self._registry)
                    group_order.append(key)
                if key not in batch_group_rows:
                    batch_group_rows[key] = []
                batch_group_rows[key].append(row_i)

            # 阶段2：对每个group，用行号列表一次性构建子向量
            for key, row_indices in batch_group_rows.items():
                gs = groups[key]
                count = len(row_indices)
                for name, ac in self._agg_exprs:
                    av = arg_vecs[name]
                    if av is None:
                        gs.update(name, None, count)
                    else:
                        # 批量提取：一次filter_by_indices而非逐行
                        sub_vec = av.filter_by_indices(row_indices)
                        gs.update(name, sub_vec, count)

        self.child.close()

        # 阶段3：finalize并构建输出
        out_schema = self.output_schema()
        if not groups:
            self._result = VectorBatch.empty(
                [n for n, _ in out_schema], [t for _, t in out_schema])
            self._emitted = False
            return

        rows: List[list] = []
        for key in group_order:
            row: list = list(key)
            gs = groups[key]
            for name, ac in self._agg_exprs:
                row.append(gs.finalize(name))
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


class _GroupState:
    """单个group的聚合状态容器。"""
    __slots__ = ('_states',)

    def __init__(self, agg_exprs: List[Tuple[str, AggregateCall]],
                 registry: FunctionRegistry) -> None:
        self._states: Dict[str, Tuple[AggregateFunction, Any]] = {}
        for name, ac in agg_exprs:
            func = registry.get_aggregate(ac.name)
            self._states[name] = (func, func.init())

    def update(self, name: str, vec: Optional[DataVector],
               count: int) -> None:
        func, state = self._states[name]
        state = func.update(state, vec, count)
        self._states[name] = (func, state)

    def finalize(self, name: str) -> Any:
        func, state = self._states[name]
        return func.finalize(state)
