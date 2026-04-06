from __future__ import annotations
"""集成规划器 — AdaptiveStrategy 驱动算子选择。
[集成] StrategySelector 分析查询特征 → 选择最优扫描/JOIN/聚合/排序算法。"""
import dataclasses, functools, time
from typing import Any, Dict, List, Optional, Set, Tuple
from catalog.catalog import Catalog
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.result import ExecutionResult
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.functions.registry import FunctionRegistry
from executor.simple_planner import SimplePlanner, _ScalarAggOperator
from executor.operators.agg.hash_agg import HashAggOperator
from executor.operators.distinct import DistinctOperator
from executor.operators.filter import FilterOperator
from executor.operators.join.cross_join import CrossJoinOperator
from executor.operators.join.hash_join import HashJoinOperator, extract_equi_keys
from executor.operators.limit import LimitOperator
from executor.operators.project import ProjectOperator
from executor.operators.scan.seq_scan import SeqScan
from executor.operators.scan.values_scan import DualScan
from executor.operators.set_ops import UnionOperator, IntersectOperator, ExceptOperator
from executor.operators.sort.in_memory_sort import SortOperator
from executor.operators.window.window_op import WindowOperator



try:
    from executor.operators.sort.top_n import TopNOperator
    _HAS_TOPN = True
except ImportError: _HAS_TOPN = False
try:
    from executor.operators.join.nested_loop_join import NestedLoopJoinOperator
    _HAS_NL = True
except ImportError: _HAS_NL = False
try:
    from executor.operators.join.radix_join import RadixJoinOperator
    _HAS_RADIX = True
except ImportError: _HAS_RADIX = False
try:
    from executor.operators.join.grace_join import GraceHashJoinOperator
    _HAS_GRACE = True
except ImportError: _HAS_GRACE = False
try:
    from executor.operators.join.sort_merge_join import SortMergeJoinOperator
    _HAS_SMJ = True
except ImportError: _HAS_SMJ = False
try:
    from executor.operators.scan.index_scan import IndexScanOperator
    _HAS_INDEX_SCAN = True
except ImportError: _HAS_INDEX_SCAN = False
try:
    from executor.operators.scan.parallel_scan import ParallelScanOperator
    _HAS_PARALLEL = True
except ImportError: _HAS_PARALLEL = False
try:
    from executor.operators.scan.zone_map_scan import ZoneMapScanOperator
    _HAS_ZONEMAP_SCAN = True
except ImportError: _HAS_ZONEMAP_SCAN = False
try:
    from executor.codegen.compiler import ExprCompiler
    from executor.codegen.cache import CompileCache
    _HAS_JIT = True
except ImportError: _HAS_JIT = False
try:
    from planner.cost_model import CostModel, CostEstimate
    from planner.cardinality import CardinalityEstimator
    from planner.rules import PredicateReorder, PredicatePushdown, TopNPushdown
    _HAS_PLANNER = True
except ImportError: _HAS_PLANNER = False
try:
    from planner.join_reorder import JoinGraph, DPccp
    _HAS_DPCCP = True
except ImportError: _HAS_DPCCP = False
try:
    from executor.adaptive.micro_engine import MicroAdaptiveEngine
    _HAS_ADAPTIVE = True
except ImportError: _HAS_ADAPTIVE = False
try:
    from executor.adaptive.strategy import StrategySelector
    _HAS_STRATEGY = True
except ImportError: _HAS_STRATEGY = False
try:
    from executor.pipeline.fuser import try_fuse
    _HAS_FUSER = True
except ImportError: _HAS_FUSER = False
try:
    from planner.runtime_optimizer import RuntimeOptimizer
    _HAS_RUNTIME_OPT = True
except ImportError: _HAS_RUNTIME_OPT = False

from parser.ast import *
from storage.types import DataType, resolve_type_name
from metal.config import NANO_THRESHOLD, MICRO_THRESHOLD, STANDARD_THRESHOLD
from utils.errors import ExecutionError

# JOIN 算法名 → 算子类映射
_JOIN_ALGO_MAP = {}
if _HAS_NL: _JOIN_ALGO_MAP['NESTED_LOOP'] = NestedLoopJoinOperator
if _HAS_SMJ: _JOIN_ALGO_MAP['SORT_MERGE'] = SortMergeJoinOperator
if _HAS_RADIX:
    _JOIN_ALGO_MAP['HASH_ROBIN_HOOD'] = RadixJoinOperator
    _JOIN_ALGO_MAP['HASH_DICT'] = HashJoinOperator
    _JOIN_ALGO_MAP['HASH_JOIN'] = HashJoinOperator
if _HAS_GRACE:
    from executor.operators.join.grace_join import GraceHashJoinOperator
    _JOIN_ALGO_MAP['GRACE'] = GraceHashJoinOperator

import threading
_STRATEGY_LOCK = threading.Lock()

class IntegratedPlanner:
    def __init__(self, function_registry: FunctionRegistry,
                 budget: Optional[Any] = None) -> None:
        self._registry = function_registry
        self._evaluator = ExpressionEvaluator(function_registry)
        self._simple = SimplePlanner(function_registry, budget=budget)
        self._compile_cache = CompileCache() if _HAS_JIT else None
        self._execution_count: Dict[str, int] = {}
        self._stats: Dict[str, Any] = {}
        self._rt_opt = RuntimeOptimizer() if _HAS_RUNTIME_OPT else None
        self._budget = budget
        # [集成] AdaptiveStrategy
        self._strategy_selector: Optional[StrategySelector] = None

    def _get_strategy_selector(self, catalog):
        if not _HAS_STRATEGY: return None
        with _STRATEGY_LOCK:
            if self._strategy_selector is None:
                self._strategy_selector = StrategySelector(catalog, self._stats)
        return self._strategy_selector

    def execute(self, ast, catalog):
        if not isinstance(ast, SelectStmt):
            return self._simple.execute(ast, catalog)
        row_count = self._estimate_total_rows(ast, catalog)
        tier = self._select_tier(row_count)
        if tier == 'NANO' and not self._has_complex_features(ast):
            return self._exec_nano(ast, catalog)
        if _HAS_PLANNER:
            ast = PredicatePushdown.apply(ast)
            ast = PredicateReorder.apply(ast)
        if tier == 'MICRO' and _HAS_FUSER:
            fused = self._try_fuse(ast, catalog)
            if fused is not None:
                return self._drain(fused)
        try:
            op = self._build_optimized_plan(ast, catalog, tier)
            return self._drain_with_monitoring(op, tier)
        except Exception:
            return self._simple.execute(ast, catalog)

    def _estimate_total_rows(self, ast, catalog):
        if ast.from_clause is None: return 1
        tname = ast.from_clause.table.name
        if catalog.table_exists(tname):
            return catalog.get_store(tname).row_count
        return 1000

    def _select_tier(self, row_count):
        if _HAS_ADAPTIVE:
            return MicroAdaptiveEngine.tier_name(row_count)
        if row_count < NANO_THRESHOLD: return 'NANO'
        if row_count < MICRO_THRESHOLD: return 'MICRO'
        if row_count < STANDARD_THRESHOLD: return 'STD'
        return 'TURBO'

    def _has_complex_features(self, ast):
        return (bool(ast.from_clause and ast.from_clause.joins)
                or any(self._simple._contains_agg(e) for e in ast.select_list)
                or any(self._simple._contains_window(e) for e in ast.select_list))

    def _exec_nano(self, ast, catalog):
        ast = self._simple._resolve_subqueries(ast, catalog)
        if ast.from_clause is None:
            return self._simple._exec_select(ast, catalog)
        store = catalog.get_store(ast.from_clause.table.name)
        schema = catalog.get_table(ast.from_clause.table.name)
        all_rows = store.read_all_rows()
        cn = schema.column_names; ct = [c.dtype for c in schema.columns]
        if ast.where:
            filtered = []
            for row in all_rows:
                b = VectorBatch.from_rows([row], cn, ct)
                if self._evaluator.evaluate_predicate(ast.where, b).get_bit(0):
                    filtered.append(row)
            all_rows = filtered
        if ast.order_by and all_rows:
            fb = VectorBatch.from_rows(all_rows, cn, ct)
            kc = []
            for sk in ast.order_by:
                np = sk.nulls or ('NULLS_LAST' if sk.direction == 'ASC' else 'NULLS_FIRST')
                kc.append((self._evaluator.evaluate(sk.expr, fb).to_python_list(), sk.direction, np))
            indices = list(range(len(all_rows)))
            indices.sort(key=functools.cmp_to_key(lambda i, j: _cmp(i, j, kc)))
            all_rows = [all_rows[i] for i in indices]
        pn, pe = [], []
        for expr in ast.select_list:
            pn.append(self._simple._out_name(expr))
            pe.append(expr.expr if isinstance(expr, AliasExpr) else expr)
        rr = []
        for row in all_rows:
            b = VectorBatch.from_rows([row], cn, ct)
            rr.append([self._evaluator.evaluate(e, b).get(0) for e in pe])
        if ast.distinct:
            seen, unique = set(), []
            for r in rr:
                k = tuple(r)
                if k not in seen: seen.add(k); unique.append(r)
            rr = unique
        off = self._simple._eval_const(ast.offset) or 0
        lim = self._simple._eval_const(ast.limit) if ast.limit else None
        if lim is not None: rr = rr[off:off + lim]
        elif off > 0: rr = rr[off:]
        ot = ([ExpressionEvaluator.infer_type(e, dict(zip(cn, ct))) for e in pe]
              if rr else [DataType.VARCHAR] * len(pn))
        return ExecutionResult(columns=pn, column_types=ot, rows=rr, row_count=len(rr))

    def _try_fuse(self, ast, catalog):
        if not _HAS_FUSER: return None
        if ast.from_clause is None or ast.from_clause.joins: return None
        if any(self._simple._contains_agg(e) for e in ast.select_list): return None
        if any(self._simple._contains_window(e) for e in ast.select_list): return None
        tref = ast.from_clause.table
        if tref.subquery: return None
        store = catalog.get_store(tref.name)
        proj = self._simple._build_proj(ast)
        cs = dict(self._simple._build_source(ast, catalog).output_schema())
        ot = [ExpressionEvaluator.infer_type(e, cs) for _, e in proj]
        lim = self._simple._eval_const(ast.limit)
        fused = try_fuse(None, ast.where, proj, lim, store, ot)
        if fused is None: return None
        op = fused
        if ast.order_by:
            op = SortOperator(op, [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by])
        if ast.distinct: op = DistinctOperator(op)
        return op

    def _build_optimized_plan(self, ast, catalog, tier):
        ast = self._simple._resolve_subqueries(ast, catalog)
        has_agg = any(self._simple._contains_agg(e) for e in ast.select_list)
        has_win = any(self._simple._contains_window(e) for e in ast.select_list)
        if has_agg or ast.group_by:
            return self._simple._plan_grouped(ast, catalog)
        if has_win:
            return self._simple._plan_windowed(ast, catalog)
        source = self._build_optimized_source(ast, catalog, tier)
        if ast.where: source = FilterOperator(source, ast.where)
        if ast.order_by:
            use_topn = (ast.limit is not None and _HAS_TOPN and _HAS_PLANNER
                        and TopNPushdown.should_use_top_n(ast))
            if use_topn:
                source = TopNOperator(source,
                    [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by],
                    TopNPushdown.get_limit_value(ast))
            else:
                source = SortOperator(source,
                    [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by])
        source = ProjectOperator(source, self._simple._build_proj(ast))
        if ast.distinct: source = DistinctOperator(source)
        has_topn = (ast.order_by and ast.limit is not None and _HAS_TOPN
                    and _HAS_PLANNER and TopNPushdown.should_use_top_n(ast))
        if not has_topn and (ast.limit is not None or ast.offset is not None):
            source = LimitOperator(source,
                self._simple._eval_const(ast.limit),
                self._simple._eval_const(ast.offset) or 0)
        return source

    def _build_optimized_source(self, ast, catalog, tier):
        """[集成] AdaptiveStrategy 驱动扫描选择。"""
        if ast.from_clause is None: return DualScan()
        fc = ast.from_clause; tref = fc.table
        if tref.subquery:
            return self._simple._plan_any(tref.subquery, catalog)
        if fc.joins:
            return self._build_optimized_joins(ast, catalog, tier)
        store = catalog.get_store(tref.name)
        needed = self._simple._collect_all_cols(ast)
        ordered = [c.name for c in store.schema.columns if c.name in needed]
        if not ordered: ordered = [store.schema.columns[0].name]
        rc = store.row_count

        # [集成] 使用 AdaptiveStrategy 推荐扫描类型
        strategy = None
        selector = self._get_strategy_selector(catalog)
        if selector and isinstance(ast, SelectStmt):
            try:
                strategy = selector.analyze_select(ast)
            except Exception:
                strategy = None

        # 索引扫描（优先级最高）
        if _HAS_INDEX_SCAN and ast.where and catalog.index_manager:
            idx_scan = self._try_index_scan(ast, catalog, store, ordered)
            if idx_scan is not None:
                return idx_scan

        # [集成] 策略驱动的扫描选择
        recommended_scan = strategy.scan if strategy else 'SEQ_SCAN'

        if recommended_scan == 'PARALLEL_SCAN' and _HAS_PARALLEL and rc > MICRO_THRESHOLD:
            if hasattr(store, 'get_column_chunks'):
                import os
                return ParallelScanOperator(tref.name, store, ordered,
                    min(os.cpu_count() or 2, 4))

        if recommended_scan == 'ZONE_MAP_SCAN' and _HAS_ZONEMAP_SCAN and ast.where:
            if hasattr(store, 'get_column_chunks'):
                return ZoneMapScanOperator(tref.name, store, ordered, ast.where)

        # TURBO 层级 morsel 并行（策略无覆盖时）
        if tier == 'TURBO' and rc > STANDARD_THRESHOLD:
            if hasattr(store, 'get_chunk_count') and hasattr(store, 'get_column_chunks'):
                try:
                    from executor.pipeline.morsel import MorselDriver
                    import os
                    driver = MorselDriver(min(os.cpu_count() or 2, 4))
                    total = store.get_chunk_count()
                    cis = []
                    for ci in range(total):
                        chunks = store.get_column_chunks(ordered[0])
                        if ci < len(chunks) and chunks[ci].row_count > 0:
                            cis.append(ci)
                    def scan_chunk(ci):
                        cols = {}
                        for name in ordered:
                            cl = store.get_column_chunks(name)
                            if ci < len(cl):
                                cols[name] = DataVector.from_column_chunk(cl[ci])
                        return VectorBatch(columns=cols, _column_order=list(ordered)) if cols else None
                    results = driver.execute(cis, scan_chunk)
                    batches = [r for r in results if r is not None]
                    if batches:
                        return _PrecomputedOperator(batches, ordered, store)
                except Exception: pass

        # 回退到 ZoneMap 扫描（未被策略选中但条件满足时）
        if _HAS_ZONEMAP_SCAN and ast.where and rc > MICRO_THRESHOLD:
            if hasattr(store, 'get_column_chunks'):
                return ZoneMapScanOperator(tref.name, store, ordered, ast.where)
        # 回退到并行扫描
        if _HAS_PARALLEL and rc > STANDARD_THRESHOLD:
            if hasattr(store, 'get_column_chunks'):
                import os
                return ParallelScanOperator(tref.name, store, ordered,
                    min(os.cpu_count() or 2, 4))
        return SeqScan(tref.name, store, ordered)

    def _try_index_scan(self, ast, catalog, store, columns):
        mgr = catalog.index_manager
        if mgr is None: return None
        pred = ast.where
        if not isinstance(pred, BinaryExpr) or pred.op != '=': return None
        col_name = val = None
        if isinstance(pred.left, ColumnRef) and isinstance(pred.right, Literal):
            col_name, val = pred.left.column, pred.right.value
        elif isinstance(pred.right, ColumnRef) and isinstance(pred.left, Literal):
            col_name, val = pred.right.column, pred.left.value
        if col_name is None or val is None: return None
        tname = ast.from_clause.table.name
        if hasattr(mgr, 'might_contain') and not mgr.might_contain(tname, col_name, val):
            return None
        index = mgr.get_index_for_column(tname, col_name)
        if index is None: return None
        return IndexScanOperator(tname, store, columns, index, scan_type='EQ', key_value=val)

    def _build_optimized_joins(self, ast, catalog, tier):
        fc = ast.from_clause
        if 1 + len(fc.joins) >= 3 and _HAS_DPCCP:
            result = self._try_dpccp(ast, catalog)
            if result is not None: return result
        tref = fc.table; lq = tref.alias or tref.name
        store = catalog.get_store(tref.name)
        lc = [c.name for c in store.schema.columns]; lr = store.row_count
        cur: Operator = ProjectOperator(SeqScan(tref.name, store, lc),
            [(f"{lq}.{c}", ColumnRef(table=None, column=c)) for c in lc])
        for jc in fc.joins:
            assert jc.table is not None
            rq = jc.table.alias or jc.table.name
            if jc.table.subquery:
                rop = self._simple._plan_any(jc.table.subquery, catalog)
                rs = rop.output_schema()
                rop = ProjectOperator(rop,
                    [(f"{rq}.{n}", ColumnRef(table=None, column=n)) for n, _ in rs])
                rr = 1000
            else:
                rs = catalog.get_store(jc.table.name)
                rc = [c.name for c in rs.schema.columns]
                rop = ProjectOperator(SeqScan(jc.table.name, rs, rc),
                    [(f"{rq}.{c}", ColumnRef(table=None, column=c)) for c in rc])
                rr = rs.row_count
            if jc.join_type == 'CROSS':
                cur = CrossJoinOperator(cur, rop)
            else:
                cur = self._select_join_op(cur, rop, jc.join_type, jc.on, lr, rr, tier)
            lr = max(lr, rr)
        return cur

    def _try_dpccp(self, ast, catalog):
        fc = ast.from_clause; graph = JoinGraph()
        base = fc.table.alias or fc.table.name
        graph.add_table(base, catalog.get_store(fc.table.name).row_count)
        for jc in fc.joins:
            if jc.table:
                t = jc.table.alias or jc.table.name
                if catalog.table_exists(jc.table.name):
                    graph.add_table(t, catalog.get_store(jc.table.name).row_count)
                else: graph.add_table(t, 1000)
                if jc.on:
                    lk, rk = extract_equi_keys(jc.on)
                    if lk and rk:
                        lt = lk.split('.')[0] if '.' in lk else base
                        rt = rk.split('.')[0] if '.' in rk else t
                        graph.add_edge(lt, rt, jc.on)
        try:
            est = CardinalityEstimator(self._stats) if _HAS_PLANNER else None
            optimal = DPccp(graph, est).optimize()
            return self._dpccp_plan_to_operator(optimal.plan, ast, catalog)
        except Exception: return None

    def _dpccp_plan_to_operator(self, plan, ast, catalog):
        if plan is None: return None
        if isinstance(plan, tuple):
            if plan[0] == 'SCAN':
                tname = plan[1]
                if not catalog.table_exists(tname): return None
                store = catalog.get_store(tname)
                cols = [c.name for c in store.schema.columns]
                return ProjectOperator(SeqScan(tname, store, cols),
                    [(f"{tname}.{c}", ColumnRef(table=None, column=c)) for c in cols])
            left_op = self._dpccp_plan_to_operator(plan[1], ast, catalog)
            right_op = self._dpccp_plan_to_operator(plan[2], ast, catalog)
            if left_op is None or right_op is None: return None
            on_expr = self._find_join_condition(ast, left_op, right_op)
            if plan[0] == 'CROSS_JOIN': return CrossJoinOperator(left_op, right_op)
            if plan[0] == 'SORT_MERGE' and _HAS_SMJ:
                return SortMergeJoinOperator(left_op, right_op, 'INNER', on_expr)
            return HashJoinOperator(left_op, right_op, 'INNER', on_expr)
        return None

    def _find_join_condition(self, ast, left_op, right_op):
        if not ast.from_clause: return None
        lt = self._collect_op_tables(left_op)
        rt = self._collect_op_tables(right_op)
        for jc in ast.from_clause.joins:
            if jc.on is None: continue
            ct = self._collect_expr_tables(jc.on)
            if ct & lt and ct & rt: return jc.on
        for jc in ast.from_clause.joins:
            if jc.on: return jc.on
        return None

    @staticmethod
    def _collect_op_tables(op):
        tables: Set[str] = set()
        try:
            for cn, _ in op.output_schema():
                if '.' in cn: tables.add(cn.split('.')[0])
        except Exception: pass
        return tables

    @staticmethod
    def _collect_expr_tables(expr):
        tables: Set[str] = set()
        if isinstance(expr, ColumnRef):
            if expr.table: tables.add(expr.table)
            return tables
        if isinstance(expr, BinaryExpr):
            tables |= IntegratedPlanner._collect_expr_tables(expr.left)
            tables |= IntegratedPlanner._collect_expr_tables(expr.right)
            return tables
        if dataclasses.is_dataclass(expr) and not isinstance(expr, type):
            for f in dataclasses.fields(expr):
                child = getattr(expr, f.name)
                if dataclasses.is_dataclass(child) and not isinstance(child, type):
                    tables |= IntegratedPlanner._collect_expr_tables(child)
                elif isinstance(child, list):
                    for item in child:
                        tables |= IntegratedPlanner._collect_expr_tables(item)
        return tables

    def _select_join_op(self, left, right, jt, on, lr, rr, tier):
        """[集成] AdaptiveStrategy 驱动 JOIN 算法选择。"""
        build = min(lr, rr)

        # [集成] 策略推荐
        selector = self._strategy_selector  # 可能已在 _build_optimized_source 中初始化
        recommended_join = None
        if selector and _HAS_STRATEGY:
            try:
                recommended_join = selector.recommend_join(lr, rr)
            except Exception:
                pass

        # 策略推荐的算子映射
        if recommended_join and recommended_join in _JOIN_ALGO_MAP:
            op_cls = _JOIN_ALGO_MAP[recommended_join]
            return op_cls(left, right, jt, on)

        # CostModel 选择（回退路径）
        if _HAS_PLANNER:
            lc = CostEstimate(rows=lr, width=100)
            rc = CostEstimate(rows=rr, width=100)
            algo, _ = CostModel.select_join_algorithm(lc, rc, 0.1)
            if algo == 'NESTED_LOOP' and _HAS_NL:
                return NestedLoopJoinOperator(left, right, jt, on)
            if algo == 'SORT_MERGE' and _HAS_SMJ:
                return SortMergeJoinOperator(left, right, jt, on)

        # 大小驱动的默认选择
        if build < NANO_THRESHOLD and _HAS_NL:
            return NestedLoopJoinOperator(left, right, jt, on)
        if build > 10_000_000 and _HAS_GRACE:
            return GraceHashJoinOperator(left, right, jt, on)
        if build > STANDARD_THRESHOLD and _HAS_RADIX:
            return RadixJoinOperator(left, right, jt, on)
        return HashJoinOperator(left, right, jt, on)

    def _drain(self, op):
        schema = op.output_schema()
        cn = [n for n, _ in schema]; ct = [t for _, t in schema]
        op.open(); rows = []
        while True:
            b = op.next_batch()
            if b is None: break
            b = Operator._ensure_batch(b)
            if b is None: break
            rows.extend(b.to_rows())
        op.close()
        return ExecutionResult(columns=cn, column_types=ct, rows=rows, row_count=len(rows))

    def _drain_with_monitoring(self, op, tier):
        schema = op.output_schema()
        cn = [n for n, _ in schema]; ct = [t for _, t in schema]
        start = time.perf_counter()
        op.open(); rows = []
        while True:
            b = op.next_batch()
            if b is None: break
            b = Operator._ensure_batch(b)
            if b is None: break
            rows.extend(b.to_rows())
        op.close()
        elapsed = time.perf_counter() - start
        result = ExecutionResult(columns=cn, column_types=ct, rows=rows, row_count=len(rows))
        if self._rt_opt and tier in ('STD', 'TURBO'):
            rs = self._rt_opt.record(type(op).__name__, estimated_rows=1000,
                actual_rows=len(rows), elapsed_ms=elapsed * 1000)
            if rs.is_off and elapsed > 0.1:
                suggestion = self._rt_opt.suggest_strategy_change(
                    type(op).__name__, len(rows))
                if suggestion:
                    result.message = (
                        f"[{tier}] {elapsed:.3f}s ⚠️ 估算偏差 "
                        f"{rs.estimation_error:.1f}x, 建议: {suggestion}")
        return result

    def update_stats(self, table_name, stats):
        self._stats[table_name] = stats
        # 更新策略选择器的统计信息
        if self._strategy_selector and _HAS_STRATEGY:
            self._strategy_selector._stats = self._stats


class _PrecomputedOperator(Operator):
    def __init__(self, batches, columns, store):
        super().__init__()
        self._batches = batches; self._idx = 0
        self._columns = columns; self._store = store
    def output_schema(self):
        cm = {c.name: c.dtype for c in self._store.schema.columns}
        return [(n, cm[n]) for n in self._columns]
    def open(self): self._idx = 0
    def next_batch(self):
        if self._idx >= len(self._batches): return None
        b = self._batches[self._idx]; self._idx += 1; return b
    def close(self): pass


def _cmp(i, j, kc):
    for vals, d, np in kc:
        a, b = vals[i], vals[j]
        if a is None and b is None: continue
        if a is None:
            c = 1 if np == 'NULLS_LAST' else -1
            return -c if d == 'DESC' else c
        if b is None:
            c = -1 if np == 'NULLS_LAST' else 1
            return -c if d == 'DESC' else c
        if a < b: c = -1
        elif a > b: c = 1
        else: continue
        return -c if d == 'DESC' else c
    return 0
