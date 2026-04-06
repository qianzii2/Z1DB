from __future__ import annotations
"""哈希聚合 — 批量分组 + Robin Hood 批量聚合 + 溢写 + 压缩执行。
[B04] 溢写时用缓存 batch 回放，不丢失已消费数据。
[B05] NULL key 独立分组，不与整数 0 混淆。
[P02] ReplayOperator 不依赖 child 可重新 open。
[性能] Robin Hood 按 key 分桶后批量 filter_by_indices。
[集成] compressed_execution dict_group_aggregate 零哈希 GROUP BY。
[R8] AVG dict 路径直接操作 (sum,count) state。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.functions.registry import (
    AggregateFunction, FunctionRegistry)
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from parser.ast import AggregateCall, StarExpr
from storage.types import DTYPE_TO_ARRAY_CODE, DataType

try:
    from executor.operators.agg.external import ExternalAggOperator
    _HAS_EXTERNAL = True
except ImportError:
    _HAS_EXTERNAL = False

try:
    from executor.compressed_execution import (
        rle_aggregate_direct, can_use_rle_execution,
        can_use_dict_execution, dict_group_aggregate)
    _HAS_COMPRESSED = True
except ImportError:
    _HAS_COMPRESSED = False

try:
    from structures.robin_hood_ht import RobinHoodHashTable
    _HAS_ROBIN_HOOD = True
except ImportError:
    _HAS_ROBIN_HOOD = False

try:
    from executor.memory_budget import MemoryBudget
    _HAS_BUDGET = True
except ImportError:
    _HAS_BUDGET = False

_MAX_MEMORY_GROUPS = 100000
_BUDGET_CHECK_INTERVAL = 1000


class HashAggOperator(Operator):
    """按 key 分组并计算聚合。超预算时溢写到磁盘。"""

    def __init__(self, child: Operator,
                 group_exprs: List[Tuple[str, Any]],
                 agg_exprs: List[Tuple[str, AggregateCall]],
                 registry: FunctionRegistry,
                 budget: Optional[Any] = None) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._group_exprs = group_exprs
        self._agg_exprs = agg_exprs
        self._registry = registry
        self._evaluator = ExpressionEvaluator(registry)
        self._result: Optional[VectorBatch] = None
        self._emitted = False
        self._budget = budget

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

        if self._budget and _HAS_BUDGET:
            self._budget.request('HashAgg', 0)

        # Robin Hood 快速路径（单 key + 整数 key）
        if _HAS_ROBIN_HOOD and len(self._group_exprs) == 1:
            rh_result = self._try_robin_hood_agg()
            if rh_result is not None:
                self._result = rh_result
                self._emitted = False
                self._release_budget()
                return

        groups: Dict[tuple, _GroupState] = {}
        group_order: List[tuple] = []
        overflowed = False
        groups_since_check = 0
        # [B04] 缓存已消费的 batch，溢写时回放
        consumed_batches: List[VectorBatch] = []

        while True:
            raw = self.child.next_batch()
            batch = self._ensure_batch(raw)
            if batch is None:
                break

            consumed_batches.append(batch)

            key_vecs = [
                self._evaluator.evaluate(expr, batch)
                for _, expr in self._group_exprs]
            arg_vecs: Dict[str, Optional[DataVector]] = {}
            for name, ac in self._agg_exprs:
                if ac.args and isinstance(ac.args[0], StarExpr):
                    arg_vecs[name] = None
                else:
                    arg_vecs[name] = self._evaluator.evaluate(
                        ac.args[0], batch)

            # ═══ RLE 快速路径（无分组 + 单聚合）═══
            if (_HAS_COMPRESSED and not self._group_exprs
                    and len(self._agg_exprs) == 1):
                agg_name, agg_call = self._agg_exprs[0]
                av = arg_vecs[agg_name]
                rle_result = self._try_rle_agg(av, agg_call)
                if rle_result is not None:
                    if () not in groups:
                        groups[()] = _GroupState(
                            self._agg_exprs, self._registry)
                        group_order.append(())
                    gs = groups[()]
                    fn_obj, state = gs._states[agg_name]
                    vec = DataVector.from_scalar(
                        rle_result,
                        DataType.BIGINT if isinstance(rle_result, int)
                        else DataType.DOUBLE)
                    state = fn_obj.update(state, vec, 1)
                    gs._states[agg_name] = (fn_obj, state)
                    continue

            # ═══ 字典编码 GROUP BY 快速路径 ═══
            if (_HAS_COMPRESSED and len(self._group_exprs) == 1
                    and len(self._agg_exprs) == 1):
                _, key_expr = self._group_exprs[0]
                kv = self._evaluator.evaluate(key_expr, batch)
                agg_name, agg_call = self._agg_exprs[0]
                av = arg_vecs[agg_name]
                if (can_use_dict_execution(kv)
                        and agg_call.name.upper() in (
                            'COUNT', 'SUM', 'MIN', 'MAX', 'AVG')
                        and not agg_call.distinct):
                    if self._try_dict_group_agg(
                            kv, av, agg_call, groups, group_order):
                        continue

            # ═══ 通用分组路径 ═══
            key_lists = [kv.to_python_list() for kv in key_vecs]
            num_keys = len(key_lists)
            batch_group_rows: Dict[tuple, List[int]] = {}

            for row_i in range(batch.row_count):
                if num_keys == 1:
                    key = (key_lists[0][row_i],)
                elif num_keys == 2:
                    key = (key_lists[0][row_i],
                           key_lists[1][row_i])
                else:
                    key = tuple(kl[row_i] for kl in key_lists)

                if key not in groups:
                    groups_since_check += 1
                    if groups_since_check >= _BUDGET_CHECK_INTERVAL:
                        groups_since_check = 0
                        if self._should_spill(len(groups)):
                            overflowed = True
                            break
                    if (len(groups) >= _MAX_MEMORY_GROUPS
                            and _HAS_EXTERNAL):
                        overflowed = True
                        break
                    groups[key] = _GroupState(
                        self._agg_exprs, self._registry)
                    group_order.append(key)

                if key not in batch_group_rows:
                    batch_group_rows[key] = []
                batch_group_rows[key].append(row_i)

            if overflowed:
                break

            # 批量更新聚合状态
            for key, row_indices in batch_group_rows.items():
                gs = groups[key]
                count = len(row_indices)
                for name, ac in self._agg_exprs:
                    av = arg_vecs[name]
                    if av is None:
                        gs.update(name, None, count)
                    else:
                        sub_vec = av.filter_by_indices(row_indices)
                        gs.update(name, sub_vec, count)

        # ═══ 溢写处理 ═══
        if overflowed and _HAS_EXTERNAL:
            self.child.close()
            # [B04] 用缓存的 batch 回放
            replay_op = _ReplayOperator(consumed_batches)
            ext = ExternalAggOperator(
                replay_op, self._group_exprs,
                self._agg_exprs, self._registry)
            ext.open()
            self._result = ext.next_batch()
            ext.close()
            self._emitted = False
            self._release_budget()
            return

        self.child.close()

        # ═══ 构建结果 ═══
        out_schema = self.output_schema()
        if not groups:
            self._result = VectorBatch.empty(
                [n for n, _ in out_schema],
                [t for _, t in out_schema])
            self._emitted = False
            self._release_budget()
            return

        rows: List[list] = []
        for key in group_order:
            row: list = list(key)
            gs = groups[key]
            for name, ac in self._agg_exprs:
                row.append(gs.finalize(name))
            rows.append(row)

        self._result = VectorBatch.from_rows(
            rows,
            [n for n, _ in out_schema],
            [t for _, t in out_schema])
        self._emitted = False
        self._release_budget()

    # ═══ 溢写检查 ═══

    def _should_spill(self, num_groups: int) -> bool:
        if not self._budget or not _HAS_BUDGET:
            return False
        return self._budget.should_spill('HashAgg', num_groups * 200)

    def _release_budget(self) -> None:
        if self._budget and _HAS_BUDGET:
            self._budget.release('HashAgg')

    # ═══ RLE 快速路径 ═══

    def _try_rle_agg(self, vec, agg_call):
        if vec is None:
            return None
        if (agg_call.name.upper() == 'COUNT'
                and not agg_call.distinct
                and vec.dict_encoded is not None):
            return sum(1 for i in range(len(vec))
                       if not vec.is_null(i))
        return None

    # ═══ 字典编码 GROUP BY ═══

    def _try_dict_group_agg(self, key_vec, agg_vec, agg_call,
                            groups, group_order) -> bool:
        """字典编码 GROUP BY。AVG 路径加类型安全检查。"""
        try:
            de = key_vec.dict_encoded
            n = len(key_vec)
            agg_upper = agg_call.name.upper()
            if agg_upper not in ('COUNT', 'SUM', 'MIN', 'MAX', 'AVG'):
                return False

            if agg_vec is not None and hasattr(de, 'codes'):
                val_list = agg_vec.to_python_list()
                code_results = dict_group_aggregate(
                    de.codes, val_list, n, de.ndv, agg_upper)
            elif agg_upper == 'COUNT':
                code_results = dict_group_aggregate(
                    de.codes, None, n, de.ndv, 'COUNT')
            else:
                return False

            for code, agg_val in code_results.items():
                if code >= len(de.dictionary):
                    continue
                key = (de.dictionary[code],)
                if key not in groups:
                    groups[key] = _GroupState(
                        self._agg_exprs, self._registry)
                    group_order.append(key)
                gs = groups[key]

                for name, ac in self._agg_exprs:
                    if ac is not agg_call:
                        continue
                    func, state = gs._states[name]

                    if agg_upper == 'AVG' and isinstance(agg_val, tuple):
                        # 类型安全检查：确保 state 是 (sum, count) 格式
                        if (isinstance(state, tuple)
                                and len(state) == 2
                                and isinstance(state[0], (int, float))
                                and isinstance(state[1], (int,))):
                            partial_sum, partial_count = agg_val
                            old_sum, old_count = state
                            state = (old_sum + partial_sum,
                                     old_count + partial_count)
                        else:
                            # state 格式不兼容，回退到标准路径
                            vec = DataVector.from_scalar(
                                agg_val[0] / agg_val[1] if agg_val[1] > 0 else None,
                                DataType.DOUBLE)
                            state = func.update(state, vec, 1)
                    else:
                        vec = DataVector.from_scalar(
                            agg_val,
                            DataType.BIGINT if isinstance(agg_val, int)
                            else DataType.DOUBLE)
                        state = func.update(state, vec, 1)

                    gs._states[name] = (func, state)
                    break

            return True
        except Exception:
            return False

    # ═══ Robin Hood 批量聚合 ═══

    def _try_robin_hood_agg(self):
        """[B05] NULL 独立分组。[性能] 按 key 分桶批量 filter_by_indices。"""
        try:
            _, key_expr = self._group_exprs[0]
            rh = RobinHoodHashTable(capacity=1024)
            group_list: list = []  # [(key_tuple, _GroupState)]
            null_group: Optional[_GroupState] = None

            while True:
                raw = self.child.next_batch()
                batch = self._ensure_batch(raw)
                if batch is None:
                    break

                key_vec = self._evaluator.evaluate(key_expr, batch)
                arg_vecs: Dict[str, Optional[DataVector]] = {}
                for name, ac in self._agg_exprs:
                    if ac.args and isinstance(ac.args[0], StarExpr):
                        arg_vecs[name] = None
                    else:
                        arg_vecs[name] = self._evaluator.evaluate(
                            ac.args[0], batch)

                # 第一遍：按 key 分桶收集行索引
                buckets: Dict[int, List[int]] = {}
                null_bucket: List[int] = []

                for row_i in range(batch.row_count):
                    k_val = (None if key_vec.is_null(row_i)
                             else key_vec.get(row_i))

                    if k_val is None:
                        # [B05] NULL 独立分组
                        if null_group is None:
                            null_group = _GroupState(
                                self._agg_exprs, self._registry)
                            group_list.append(((None,), null_group))
                        null_bucket.append(row_i)
                    elif not isinstance(k_val, int):
                        # 非整数 key 放弃 Robin Hood
                        self.child.close()
                        return None
                    else:
                        found, existing_idx = rh.get(k_val)
                        if not found:
                            gs = _GroupState(
                                self._agg_exprs, self._registry)
                            idx = len(group_list)
                            group_list.append(((k_val,), gs))
                            rh.put(k_val, idx)
                            buckets[idx] = [row_i]
                        else:
                            if existing_idx not in buckets:
                                buckets[existing_idx] = []
                            buckets[existing_idx].append(row_i)

                # 第二遍：批量 filter_by_indices 聚合
                for group_idx, row_indices in buckets.items():
                    _, gs = group_list[group_idx]
                    count = len(row_indices)
                    for name, ac in self._agg_exprs:
                        av = arg_vecs[name]
                        if av is None:
                            gs.update(name, None, count)
                        else:
                            sub = av.filter_by_indices(row_indices)
                            gs.update(name, sub, count)

                # NULL 桶
                if null_bucket and null_group is not None:
                    count = len(null_bucket)
                    for name, ac in self._agg_exprs:
                        av = arg_vecs[name]
                        if av is None:
                            null_group.update(name, None, count)
                        else:
                            sub = av.filter_by_indices(null_bucket)
                            null_group.update(name, sub, count)

            self.child.close()

            if not group_list:
                out_schema = self.output_schema()
                return VectorBatch.empty(
                    [n for n, _ in out_schema],
                    [t for _, t in out_schema])

            out_schema = self.output_schema()
            rows = []
            for key, gs in group_list:
                row = list(key)
                for name, ac in self._agg_exprs:
                    row.append(gs.finalize(name))
                rows.append(row)
            return VectorBatch.from_rows(
                rows,
                [n for n, _ in out_schema],
                [t for _, t in out_schema])

        except Exception:
            try:
                self.child.close()
            except Exception:
                pass
            return None

    # ═══ Volcano 接口 ═══

    def next_batch(self):
        if self._emitted or self._result is None:
            return None
        self._emitted = True
        return self._result

    def close(self):
        self._release_budget()


class _GroupState:
    """单个分组的聚合状态。"""
    __slots__ = ('_states',)

    def __init__(self, agg_exprs, registry):
        self._states: Dict[str, Tuple[AggregateFunction, Any]] = {}
        for name, ac in agg_exprs:
            func = registry.get_aggregate(ac.name)
            self._states[name] = (func, func.init())

    def update(self, name, vec, count):
        func, state = self._states[name]
        state = func.update(state, vec, count)
        self._states[name] = (func, state)

    def finalize(self, name):
        func, state = self._states[name]
        return func.finalize(state)


class _ReplayOperator(Operator):
    """[B04][P02] 重放已缓存的 batch。
    不依赖 child 可重新 open — 所有数据都在缓存中。"""

    def __init__(self, cached_batches: List[VectorBatch]) -> None:
        super().__init__()
        self._cached = cached_batches
        self._idx = 0
        self._schema: Optional[List[Tuple[str, DataType]]] = None
        if cached_batches:
            first = cached_batches[0]
            self._schema = [
                (n, first.columns[n].dtype)
                for n in first.column_names]

    def output_schema(self):
        return self._schema or []

    def open(self) -> None:
        self._idx = 0

    def next_batch(self):
        if self._idx >= len(self._cached):
            return None
        b = self._cached[self._idx]
        self._idx += 1
        return b

    def close(self) -> None:
        pass
