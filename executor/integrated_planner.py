from __future__ import annotations
"""集成规划器 — 全部组件接入执行路径。
DPccp多表JOIN + MicroAdaptive策略 + Pipeline Fuser + Morsel并行 + RuntimeOptimizer。"""
import dataclasses
import functools
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from catalog.catalog import Catalog, ColumnSchema, TableSchema
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
except ImportError:
    _HAS_TOPN = False
try:
    from executor.operators.join.nested_loop_join import NestedLoopJoinOperator
    _HAS_NL = True
except ImportError:
    _HAS_NL = False
try:
    from executor.operators.join.radix_join import RadixJoinOperator
    _HAS_RADIX = True
except ImportError:
    _HAS_RADIX = False
try:
    from executor.operators.join.grace_join import GraceHashJoinOperator
    _HAS_GRACE = True
except ImportError:
    _HAS_GRACE = False
try:
    from executor.operators.scan.parallel_scan import ParallelScanOperator
    _HAS_PARALLEL = True
except ImportError:
    _HAS_PARALLEL = False
try:
    from executor.operators.scan.zone_map_scan import ZoneMapScanOperator
    _HAS_ZONEMAP_SCAN = True
except ImportError:
    _HAS_ZONEMAP_SCAN = False
try:
    from executor.codegen.compiler import ExprCompiler
    from executor.codegen.cache import CompileCache
    _HAS_JIT = True
except ImportError:
    _HAS_JIT = False
try:
    from planner.cost_model import CostModel, CostEstimate
    from planner.cardinality import CardinalityEstimator
    from planner.rules import PredicateReorder, PredicatePushdown, TopNPushdown
    _HAS_PLANNER = True
except ImportError:
    _HAS_PLANNER = False
try:
    from planner.join_reorder import JoinGraph, DPccp
    _HAS_DPCCP = True
except ImportError:
    _HAS_DPCCP = False
try:
    from executor.adaptive.micro_engine import MicroAdaptiveEngine
    from executor.adaptive.strategy import StrategySelector
    _HAS_ADAPTIVE = True
except ImportError:
    _HAS_ADAPTIVE = False
try:
    from executor.pipeline.fuser import try_fuse
    _HAS_FUSER = True
except ImportError:
    _HAS_FUSER = False
try:
    from planner.runtime_optimizer import RuntimeOptimizer
    _HAS_RUNTIME_OPT = True
except ImportError:
    _HAS_RUNTIME_OPT = False

from parser.ast import *
from storage.types import DataType, resolve_type_name
from metal.config import NANO_THRESHOLD, MICRO_THRESHOLD, STANDARD_THRESHOLD
from utils.errors import ExecutionError


class IntegratedPlanner:
    def __init__(self, function_registry: FunctionRegistry) -> None:
        self._registry = function_registry
        self._evaluator = ExpressionEvaluator(function_registry)
        self._simple = SimplePlanner(function_registry)
        self._compile_cache = CompileCache() if _HAS_JIT else None
        self._execution_count: Dict[str, int] = {}
        self._stats: Dict[str, Any] = {}
        self._rt_opt = RuntimeOptimizer() if _HAS_RUNTIME_OPT else None

    def execute(self, ast: Any, catalog: Catalog) -> ExecutionResult:
        if not isinstance(ast, SelectStmt):
            return self._simple.execute(ast, catalog)

        row_count = self._estimate_total_rows(ast, catalog)
        tier = self._select_tier(row_count)
        query_key = repr(ast)[:200]
        self._execution_count[query_key] = self._execution_count.get(query_key, 0) + 1
        exec_count = self._execution_count[query_key]

        # NANO路径
        if tier == 'NANO' and not self._has_complex_features(ast):
            return self._exec_nano(ast, catalog)

        # AST级优化
        if _HAS_PLANNER:
            ast = PredicatePushdown.apply(ast)
            ast = PredicateReorder.apply(ast)

        # MICRO路径：尝试Pipeline Fuser
        if tier == 'MICRO' and _HAS_FUSER:
            fused = self._try_fuse(ast, catalog)
            if fused is not None:
                return self._drain(fused)

        try:
            op = self._build_optimized_plan(ast, catalog, tier, exec_count)
            return self._drain_with_monitoring(op, tier)
        except Exception:
            return self._simple.execute(ast, catalog)

    # ══════════════════════════════════════════════════════════
    def _estimate_total_rows(self, ast: SelectStmt, catalog: Catalog) -> int:
        if ast.from_clause is None:
            return 1
        tname = ast.from_clause.table.name
        if catalog.table_exists(tname):
            return catalog.get_store(tname).row_count
        return 1000

    def _select_tier(self, row_count: int) -> str:
        if _HAS_ADAPTIVE:
            return MicroAdaptiveEngine.tier_name(row_count)
        if row_count < NANO_THRESHOLD: return 'NANO'
        if row_count < MICRO_THRESHOLD: return 'MICRO'
        if row_count < STANDARD_THRESHOLD: return 'STD'
        return 'TURBO'

    def _has_complex_features(self, ast: SelectStmt) -> bool:
        return (bool(ast.from_clause and ast.from_clause.joins)
                or any(self._simple._contains_agg(e) for e in ast.select_list)
                or any(self._simple._contains_window(e) for e in ast.select_list))

    # ══════════════════════════════════════════════════════════
    # NANO路径
    # ══════════════════════════════════════════════════════════
    def _exec_nano(self, ast, catalog):
        ast = self._simple._resolve_subqueries(ast, catalog)
        if ast.from_clause is None:
            return self._simple._exec_select(ast, catalog)
        store = catalog.get_store(ast.from_clause.table.name)
        schema = catalog.get_table(ast.from_clause.table.name)
        all_rows = store.read_all_rows()
        cn = schema.column_names
        ct = [c.dtype for c in schema.columns]

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

        ot = [ExpressionEvaluator.infer_type(e, dict(zip(cn, ct))) for e in pe] if rr else [DataType.VARCHAR] * len(pn)
        return ExecutionResult(columns=pn, column_types=ot, rows=rr, row_count=len(rr))

    # ══════════════════════════════════════════════════════════
    # Pipeline Fuser (MICRO路径)
    # ══════════════════════════════════════════════════════════
    def _try_fuse(self, ast, catalog):
        if not _HAS_FUSER: return None
        if ast.from_clause is None or ast.from_clause.joins: return None
        has_agg = any(self._simple._contains_agg(e) for e in ast.select_list)
        has_win = any(self._simple._contains_window(e) for e in ast.select_list)
        if has_agg or has_win: return None
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

    # ══════════════════════════════════════════════════════════
    # 优化计划构建
    # ══════════════════════════════════════════════════════════
    def _build_optimized_plan(self, ast, catalog, tier, exec_count):
        ast = self._simple._resolve_subqueries(ast, catalog)
        has_agg = any(self._simple._contains_agg(e) for e in ast.select_list)
        has_win = any(self._simple._contains_window(e) for e in ast.select_list)
        if has_agg or ast.group_by: return self._simple._plan_grouped(ast, catalog)
        if has_win: return self._simple._plan_windowed(ast, catalog)

        source = self._build_optimized_source(ast, catalog, tier)
        if ast.where: source = FilterOperator(source, ast.where)
        if ast.order_by:
            use_topn = ast.limit is not None and _HAS_TOPN and _HAS_PLANNER and TopNPushdown.should_use_top_n(ast)
            if use_topn:
                source = TopNOperator(source, [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by], TopNPushdown.get_limit_value(ast))
            else:
                source = SortOperator(source, [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by])
        source = ProjectOperator(source, self._simple._build_proj(ast))
        if ast.distinct: source = DistinctOperator(source)
        has_topn = ast.order_by and ast.limit is not None and _HAS_TOPN and _HAS_PLANNER and TopNPushdown.should_use_top_n(ast)
        if not has_topn and (ast.limit is not None or ast.offset is not None):
            source = LimitOperator(source, self._simple._eval_const(ast.limit), self._simple._eval_const(ast.offset) or 0)
        return source

    def _build_optimized_source(self, ast, catalog, tier):
        if ast.from_clause is None: return DualScan()
        fc = ast.from_clause; tref = fc.table
        if tref.subquery: return self._simple._plan_any(tref.subquery, catalog)
        if fc.joins: return self._build_optimized_joins(ast, catalog, tier)
        store = catalog.get_store(tref.name)
        needed = self._simple._collect_all_cols(ast)
        ordered = [c.name for c in store.schema.columns if c.name in needed]
        if not ordered: ordered = [store.schema.columns[0].name]
        rc = store.row_count

        # TURBO: Morsel并行扫描
        if tier == 'TURBO' and rc > STANDARD_THRESHOLD:
            try:
                from executor.pipeline.morsel import MorselDriver
                import os
                driver = MorselDriver(min(os.cpu_count() or 2, 4))
                total = store.get_chunk_count()
                cis = [ci for ci in range(total) if store.get_column_chunks(ordered[0])[ci].row_count > 0]
                def scan_chunk(ci):
                    cols = {}
                    for name in ordered:
                        cl = store.get_column_chunks(name)
                        if ci < len(cl): cols[name] = DataVector.from_column_chunk(cl[ci])
                    return VectorBatch(columns=cols, _column_order=list(ordered)) if cols else None
                results = driver.execute(cis, scan_chunk)
                batches = [r for r in results if r is not None]
                if batches: return _PrecomputedOperator(batches, ordered, store)
            except Exception:
                pass

        if _HAS_ZONEMAP_SCAN and ast.where and rc > MICRO_THRESHOLD:
            return ZoneMapScanOperator(tref.name, store, ordered, ast.where)
        if _HAS_PARALLEL and rc > STANDARD_THRESHOLD:
            import os
            return ParallelScanOperator(tref.name, store, ordered, min(os.cpu_count() or 2, 4))
        return SeqScan(tref.name, store, ordered)

    def _build_optimized_joins(self, ast, catalog, tier):
        fc = ast.from_clause
        num_tables = 1 + len(fc.joins)

        # DPccp: 3表以上时使用最优连接排序
        if num_tables >= 3 and _HAS_DPCCP:
            result = self._try_dpccp(ast, catalog)
            if result is not None: return result

        # 线性构建
        tref = fc.table; lq = tref.alias or tref.name
        store = catalog.get_store(tref.name)
        lc = [c.name for c in store.schema.columns]
        lr = store.row_count
        cur: Operator = ProjectOperator(SeqScan(tref.name, store, lc),
            [(f"{lq}.{c}", ColumnRef(table=None, column=c)) for c in lc])
        for jc in fc.joins:
            assert jc.table is not None
            rq = jc.table.alias or jc.table.name
            if jc.table.subquery:
                rop = self._simple._plan_any(jc.table.subquery, catalog)
                rs = rop.output_schema()
                rop = ProjectOperator(rop, [(f"{rq}.{n}", ColumnRef(table=None, column=n)) for n, _ in rs])
                rr = 1000
            else:
                rs = catalog.get_store(jc.table.name)
                rc = [c.name for c in rs.schema.columns]
                rop = ProjectOperator(SeqScan(jc.table.name, rs, rc),
                    [(f"{rq}.{c}", ColumnRef(table=None, column=c)) for c in rc])
                rr = rs.row_count
            if jc.join_type == 'CROSS': cur = CrossJoinOperator(cur, rop)
            else: cur = self._select_join_op(cur, rop, jc.join_type, jc.on, lr, rr, tier)
            lr = max(lr, rr)
        return cur

    def _try_dpccp(self, ast, catalog):
        """DPccp最优连接排序。"""
        fc = ast.from_clause
        graph = JoinGraph()
        base = fc.table.alias or fc.table.name
        graph.add_table(base, catalog.get_store(fc.table.name).row_count)
        for jc in fc.joins:
            if jc.table:
                t = jc.table.alias or jc.table.name
                if catalog.table_exists(jc.table.name):
                    graph.add_table(t, catalog.get_store(jc.table.name).row_count)
                else:
                    graph.add_table(t, 1000)
                if jc.on:
                    lk, rk = extract_equi_keys(jc.on)
                    if lk and rk:
                        lt = lk.split('.')[0] if '.' in lk else base
                        rt = rk.split('.')[0] if '.' in rk else t
                        graph.add_edge(lt, rt, jc.on)
        try:
            estimator = CardinalityEstimator(self._stats) if _HAS_PLANNER else None
            optimal = DPccp(graph, estimator).optimize()
            return self._dpccp_plan_to_operator(optimal.plan, ast, catalog)
        except Exception:
            return None

    def _dpccp_plan_to_operator(self, plan, ast, catalog):
        """将DPccp计划树转为算子树。"""
        if plan is None: return None
        if isinstance(plan, tuple):
            if plan[0] == 'SCAN':
                tname = plan[1]
                if not catalog.table_exists(tname): return None
                store = catalog.get_store(tname)
                cols = [c.name for c in store.schema.columns]
                scan = SeqScan(tname, store, cols)
                return ProjectOperator(scan, [(f"{tname}.{c}", ColumnRef(table=None, column=c)) for c in cols])
            algo = plan[0]
            left_op = self._dpccp_plan_to_operator(plan[1], ast, catalog)
            right_op = self._dpccp_plan_to_operator(plan[2], ast, catalog)
            if left_op is None or right_op is None: return None
            # 查找对应的ON条件
            on_expr = self._find_join_condition(ast, left_op, right_op)
            if algo == 'CROSS_JOIN': return CrossJoinOperator(left_op, right_op)
            return HashJoinOperator(left_op, right_op, 'INNER', on_expr)
        return None

    def _find_join_condition(self, ast, left_op, right_op):
        """从原始AST查找适用的JOIN条件。"""
        if ast.from_clause:
            for jc in ast.from_clause.joins:
                if jc.on: return jc.on
        return None

    def _select_join_op(self, left, right, jt, on, lr, rr, tier):
        build = min(lr, rr)
        if _HAS_PLANNER:
            lc = CostEstimate(rows=lr, width=100)
            rc = CostEstimate(rows=rr, width=100)
            algo, _ = CostModel.select_join_algorithm(lc, rc, 0.1)
            if algo == 'NESTED_LOOP' and _HAS_NL:
                return NestedLoopJoinOperator(left, right, jt, on)
        if build < NANO_THRESHOLD and _HAS_NL:
            return NestedLoopJoinOperator(left, right, jt, on)
        if build > 10_000_000 and _HAS_GRACE:
            return GraceHashJoinOperator(left, right, jt, on)
        if build > STANDARD_THRESHOLD and _HAS_RADIX:
            return RadixJoinOperator(left, right, jt, on)
        return HashJoinOperator(left, right, jt, on)

    # ══════════════════════════════════════════════════════════
    # 执行与监控
    # ══════════════════════════════════════════════════════════
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
        """执行并收集运行时统计。"""
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

        # 运行时监控反馈
        if self._rt_opt and tier in ('STD', 'TURBO'):
            self._rt_opt.record(type(op).__name__, estimated_rows=1000,
                               actual_rows=len(rows), elapsed_ms=elapsed * 1000)
            if self._rt_opt.should_reoptimize() and elapsed > 0.1:
                result.message = f"[{tier}] {elapsed:.3f}s ⚠️ cardinality off"
        return result

    def update_stats(self, table_name: str, stats: Any) -> None:
        self._stats[table_name] = stats


class _PrecomputedOperator(Operator):
    """包装预计算的batch列表。"""
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
        if a is None: c = 1 if np == 'NULLS_LAST' else -1; return -c if d == 'DESC' else c
        if b is None: c = -1 if np == 'NULLS_LAST' else 1; return -c if d == 'DESC' else c
        if a < b: c = -1
        elif a > b: c = 1
        else: continue
        return -c if d == 'DESC' else c
    return 0
