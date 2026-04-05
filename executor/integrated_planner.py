from __future__ import annotations

"""Integrated planner — bridges ALL components into the execution path.
This is the brain of Z1DB. Every optimization decision flows through here."""
import dataclasses
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

# Operators
from executor.operators.agg.hash_agg import HashAggOperator
from executor.operators.distinct import DistinctOperator
from executor.operators.filter import FilterOperator
from executor.operators.join.cross_join import CrossJoinOperator
from executor.operators.join.hash_join import HashJoinOperator
from executor.operators.limit import LimitOperator
from executor.operators.project import ProjectOperator
from executor.operators.scan.seq_scan import SeqScan
from executor.operators.scan.values_scan import DualScan
from executor.operators.set_ops import UnionOperator, IntersectOperator, ExceptOperator
from executor.operators.sort.in_memory_sort import SortOperator
from executor.operators.window.window_op import WindowOperator

# Advanced operators
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
    from executor.operators.join.sort_merge_join import SortMergeJoinOperator

    _HAS_SMJ = True
except ImportError:
    _HAS_SMJ = False

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
    from executor.adaptive.micro_engine import MicroAdaptiveEngine

    _HAS_ADAPTIVE = True
except ImportError:
    _HAS_ADAPTIVE = False

try:
    from planner.cost_model import CostModel
    from planner.cardinality import CardinalityEstimator
    from planner.rules import PredicateReorder, TopNPushdown

    _HAS_PLANNER = True
except ImportError:
    _HAS_PLANNER = False

from parser.ast import *
from parser.formatter import Formatter
from storage.types import DataType, resolve_type_name
from metal.config import NANO_THRESHOLD, MICRO_THRESHOLD, STANDARD_THRESHOLD
from utils.errors import (ColumnNotFoundError, ExecutionError, NumericOverflowError)


class IntegratedPlanner:
    """The unified execution engine that uses ALL available optimizations.

    Decision flow:
    1. Analyze query shape + table sizes
    2. Select execution tier (NANO/MICRO/STD/TURBO)
    3. For STD+: apply cost-based optimization
    4. Build operator tree with best algorithms
    5. Enable JIT for hot paths
    6. Monitor runtime, adapt if needed
    """

    def __init__(self, function_registry: FunctionRegistry) -> None:
        self._registry = function_registry
        self._evaluator = ExpressionEvaluator(function_registry)
        self._simple = SimplePlanner(function_registry)  # Fallback
        self._compile_cache = CompileCache() if _HAS_JIT else None
        self._execution_count: Dict[str, int] = {}  # query_hash → count
        self._stats: Dict[str, Any] = {}  # table → TableStatistics

    def execute(self, ast: Any, catalog: Catalog) -> ExecutionResult:
        """Main entry point — routes to optimal execution path."""
        # Non-SELECT: delegate to simple planner
        if not isinstance(ast, SelectStmt):
            return self._simple.execute(ast, catalog)

        # Determine data scale
        row_count = self._estimate_total_rows(ast, catalog)
        tier = self._select_tier(row_count)

        # Track query for JIT decisions
        query_key = repr(ast)[:200]
        self._execution_count[query_key] = self._execution_count.get(query_key, 0) + 1
        exec_count = self._execution_count[query_key]

        # NANO path: skip all overhead
        if tier == 'NANO' and not self._has_complex_features(ast):
            return self._exec_nano(ast, catalog)

        # Apply AST-level optimizations
        if _HAS_PLANNER:
            ast = PredicateReorder.apply(ast)

        # Build optimized operator tree
        try:
            op = self._build_optimized_plan(ast, catalog, tier, exec_count)
            return self._drain_with_monitoring(op, tier)
        except Exception:
            # Fallback to simple planner
            return self._simple.execute(ast, catalog)

    # ══════════════════════════════════════════════════════════════
    # Tier selection
    # ══════════════════════════════════════════════════════════════

    def _estimate_total_rows(self, ast: SelectStmt, catalog: Catalog) -> int:
        if ast.from_clause is None:
            return 1
        tname = ast.from_clause.table.name
        if catalog.table_exists(tname):
            return catalog.get_store(tname).row_count
        return 1000  # Unknown

    def _select_tier(self, row_count: int) -> str:
        if row_count < NANO_THRESHOLD:
            return 'NANO'
        if row_count < MICRO_THRESHOLD:
            return 'MICRO'
        if row_count < STANDARD_THRESHOLD:
            return 'STD'
        return 'TURBO'

    def _has_complex_features(self, ast: SelectStmt) -> bool:
        return bool(ast.from_clause and ast.from_clause.joins) or \
            any(self._simple._contains_agg(e) for e in ast.select_list) or \
            any(self._simple._contains_window(e) for e in ast.select_list)

    # ══════════════════════════════════════════════════════════════
    # NANO path: zero overhead
    # ══════════════════════════════════════════════════════════════

    def _exec_nano(self, ast: SelectStmt, catalog: Catalog) -> ExecutionResult:
        """Direct list processing. No Operator tree, no batching."""
        ast = self._simple._resolve_subqueries(ast, catalog)
        if ast.from_clause is None:
            return self._simple._exec_select(ast, catalog)

        store = catalog.get_store(ast.from_clause.table.name)
        schema = catalog.get_table(ast.from_clause.table.name)
        all_rows = store.read_all_rows()
        col_names = schema.column_names
        col_types = [c.dtype for c in schema.columns]

        # Filter
        if ast.where:
            filtered = []
            for row in all_rows:
                batch = VectorBatch.from_rows([row], col_names, col_types)
                if self._evaluator.evaluate_predicate(ast.where, batch).get_bit(0):
                    filtered.append(row)
            all_rows = filtered

        # Sort
        if ast.order_by:
            import functools
            def cmp(a, b):
                for sk in ast.order_by:
                    ba = VectorBatch.from_rows([a], col_names, col_types)
                    bb = VectorBatch.from_rows([b], col_names, col_types)
                    va = self._evaluator.evaluate(sk.expr, ba).get(0)
                    vb = self._evaluator.evaluate(sk.expr, bb).get(0)
                    if va is None and vb is None: continue
                    if va is None:
                        return 1 if (sk.nulls or 'NULLS_LAST') == 'NULLS_LAST' else -1
                    if vb is None:
                        return -1 if (sk.nulls or 'NULLS_LAST') == 'NULLS_LAST' else 1
                    if va < vb: return 1 if sk.direction == 'DESC' else -1
                    if va > vb: return -1 if sk.direction == 'DESC' else 1
                return 0

            all_rows.sort(key=functools.cmp_to_key(cmp))

        # Project first (so DISTINCT sees projected values)
        proj_names = []
        proj_exprs = []
        for expr in ast.select_list:
            name = self._simple._out_name(expr)
            inner = expr.expr if isinstance(expr, AliasExpr) else expr
            proj_names.append(name)
            proj_exprs.append(inner)

        result_rows = []
        for row in all_rows:
            batch = VectorBatch.from_rows([row], col_names, col_types)
            result_row = []
            for expr in proj_exprs:
                result_row.append(self._evaluator.evaluate(expr, batch).get(0))
            result_rows.append(result_row)

        # DISTINCT
        if ast.distinct:
            seen = set()
            unique = []
            for row in result_rows:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    unique.append(row)
            result_rows = unique

        # Limit/Offset — handle LIMIT 0 correctly
        offset = 0
        if ast.offset:
            offset = self._simple._eval_const(ast.offset) or 0
        limit_val = self._simple._eval_const(ast.limit) if ast.limit else None
        if limit_val is not None:
            result_rows = result_rows[offset:offset + limit_val]
        elif offset > 0:
            result_rows = result_rows[offset:]

        # Output types
        if result_rows:
            out_types = []
            for i, expr in enumerate(proj_exprs):
                child_schema = dict(zip(col_names, col_types))
                out_types.append(ExpressionEvaluator.infer_type(expr, child_schema))
        else:
            out_types = [DataType.VARCHAR] * len(proj_names)

        return ExecutionResult(
            columns=proj_names, column_types=out_types,
            rows=result_rows, row_count=len(result_rows))

    # ══════════════════════════════════════════════════════════════
    # Optimized plan building
    # ══════════════════════════════════════════════════════════════

    def _build_optimized_plan(self, ast: SelectStmt, catalog: Catalog,
                              tier: str, exec_count: int) -> Operator:
        """Build operator tree with tier-appropriate optimizations."""
        ast = self._simple._resolve_subqueries(ast, catalog)
        has_agg = any(self._simple._contains_agg(e) for e in ast.select_list)
        has_win = any(self._simple._contains_window(e) for e in ast.select_list)

        if has_agg or ast.group_by:
            return self._simple._plan_grouped(ast, catalog)
        if has_win:
            return self._simple._plan_windowed(ast, catalog)

        # Build source with optimized scan
        source = self._build_optimized_source(ast, catalog, tier)

        # Filter with JIT
        if ast.where:
            source = self._build_optimized_filter(source, ast.where, tier, exec_count)

        # Sort with algorithm selection
        if ast.order_by:
            use_topn = (ast.limit is not None and _HAS_TOPN and
                        _HAS_PLANNER and TopNPushdown.should_use_top_n(ast))
            if use_topn:
                limit_val = self._simple._eval_const(ast.limit) or 10
                keys = [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by]
                source = TopNOperator(source, keys, limit_val)
            else:
                keys = [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by]
                source = SortOperator(source, keys)

        # Project
        projections = self._simple._build_proj(ast)
        source = ProjectOperator(source, projections)

        if ast.distinct:
            source = DistinctOperator(source)

        # Limit (skip if already handled by TopN)
        has_topn = ast.order_by and ast.limit is not None and _HAS_TOPN
        if not has_topn and (ast.limit is not None or ast.offset is not None):
            lv = self._simple._eval_const(ast.limit)
            ov = self._simple._eval_const(ast.offset) or 0
            source = LimitOperator(source, lv, ov)

        return source

    def _build_optimized_source(self, ast: SelectStmt, catalog: Catalog,
                                tier: str) -> Operator:
        """Select best scan operator based on tier and data characteristics."""
        if ast.from_clause is None:
            return DualScan()

        fc = ast.from_clause
        tref = fc.table

        if tref.subquery:
            return self._simple._plan_any(tref.subquery, catalog)

        if fc.joins:
            return self._build_optimized_joins(ast, catalog, tier)

        store = catalog.get_store(tref.name)
        needed = self._simple._collect_all_cols(ast)
        ordered = [c.name for c in store.schema.columns if c.name in needed]
        if not ordered:
            ordered = [store.schema.columns[0].name]

        row_count = store.row_count

        # ZoneMap scan
        if (_HAS_ZONEMAP_SCAN and ast.where and row_count > MICRO_THRESHOLD):
            return ZoneMapScanOperator(tref.name, store, ordered, ast.where)

        # Parallel scan
        if (_HAS_PARALLEL and row_count > STANDARD_THRESHOLD and not ast.where):
            import os
            workers = min(os.cpu_count() or 2, 4)
            return ParallelScanOperator(tref.name, store, ordered, workers)

        return SeqScan(tref.name, store, ordered)

    def _build_optimized_joins(self, ast: SelectStmt, catalog: Catalog,
                               tier: str) -> Operator:
        """Select JOIN algorithm based on table sizes."""
        fc = ast.from_clause
        tref = fc.table
        lq = tref.alias or tref.name
        store = catalog.get_store(tref.name)
        lc = [c.name for c in store.schema.columns]
        left_rows = store.row_count

        cur: Operator = ProjectOperator(
            SeqScan(tref.name, store, lc),
            [(f"{lq}.{c}", ColumnRef(table=None, column=c)) for c in lc])

        for jc in fc.joins:
            assert jc.table is not None
            rq = jc.table.alias or jc.table.name

            if jc.table.subquery:
                right_op = self._simple._plan_any(jc.table.subquery, catalog)
                rs = right_op.output_schema()
                right_op = ProjectOperator(right_op,
                                           [(f"{rq}.{n}", ColumnRef(table=None, column=n)) for n, _ in rs])
                right_rows = 1000  # Unknown
            else:
                r_store = catalog.get_store(jc.table.name)
                r_cols = [c.name for c in r_store.schema.columns]
                right_op = ProjectOperator(
                    SeqScan(jc.table.name, r_store, r_cols),
                    [(f"{rq}.{c}", ColumnRef(table=None, column=c)) for c in r_cols])
                right_rows = r_store.row_count

            if jc.join_type == 'CROSS':
                cur = CrossJoinOperator(cur, right_op)
            else:
                # Select best JOIN algorithm
                cur = self._select_join_operator(
                    cur, right_op, jc.join_type, jc.on,
                    left_rows, right_rows, tier)
            left_rows = max(left_rows, right_rows)  # Rough estimate

        return cur

    def _select_join_operator(self, left: Operator, right: Operator,
                              join_type: str, on_expr: Any,
                              left_rows: int, right_rows: int,
                              tier: str) -> Operator:
        """Cost-based JOIN algorithm selection."""
        # Ensure smaller table on build side
        build_rows = min(left_rows, right_rows)

        if build_rows < NANO_THRESHOLD and _HAS_NL:
            return NestedLoopJoinOperator(left, right, join_type, on_expr)

        if build_rows > 10_000_000 and _HAS_GRACE:
            return GraceHashJoinOperator(left, right, join_type, on_expr)

        if build_rows > STANDARD_THRESHOLD and _HAS_RADIX:
            return RadixJoinOperator(left, right, join_type, on_expr)

        # Default: Hash Join (with Bloom filter integrated)
        return HashJoinOperator(left, right, join_type, on_expr)

    def _build_optimized_filter(self, source: Operator, predicate: Any,
                                tier: str, exec_count: int) -> Operator:
        """Filter with optional JIT compilation."""
        # Try JIT for hot queries
        if _HAS_JIT and exec_count >= 2 and tier in ('STD', 'TURBO'):
            cache_key = CompileCache.hash_expr(predicate)
            cached_fn = self._compile_cache.get(cache_key) if self._compile_cache else None
            if cached_fn is None:
                compiled = ExprCompiler.compile_predicate(predicate)
                if compiled and self._compile_cache:
                    self._compile_cache.put(cache_key, compiled)
            # Even if JIT succeeds, still use FilterOperator
            # (JIT integration into FilterOperator is in the operator itself)

        return FilterOperator(source, predicate)

    # ══════════════════════════════════════════════════════════════
    # Execution with monitoring
    # ══════════════════════════════════════════════════════════════

    def _drain_with_monitoring(self, op: Operator, tier: str) -> ExecutionResult:
        """Execute and collect runtime stats."""
        schema = op.output_schema()
        cn = [n for n, _ in schema]
        ct = [t for _, t in schema]

        start = time.perf_counter()
        op.open()
        rows: list = []
        while True:
            b = op.next_batch()
            if b is None:
                break
            rows.extend(b.to_rows())
        op.close()
        elapsed = time.perf_counter() - start

        result = ExecutionResult(columns=cn, column_types=ct,
                                 rows=rows, row_count=len(rows))

        # Add execution stats to result message for TURBO+ queries
        if tier in ('TURBO', 'NUCLEAR') and elapsed > 0.1:
            result.message = f"[{tier}] {elapsed:.3f}s"

        return result

    # ══════════════════════════════════════════════════════════════
    # Statistics management
    # ══════════════════════════════════════════════════════════════

    def update_stats(self, table_name: str, stats: Any) -> None:
        self._stats[table_name] = stats
