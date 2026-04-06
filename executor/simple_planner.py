from __future__ import annotations
"""SELECT 规划器 + 路由门面。
DDL 委托到 ddl_executor，DML 委托到 dml_executor。"""
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple
from catalog.catalog import Catalog
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.result import ExecutionResult
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.functions.registry import FunctionRegistry
from executor.operators.agg.hash_agg import HashAggOperator
from executor.operators.distinct import DistinctOperator
from executor.operators.filter import FilterOperator
from executor.operators.join.cross_join import CrossJoinOperator
from executor.operators.join.hash_join import HashJoinOperator
from executor.operators.limit import LimitOperator
from executor.operators.project import ProjectOperator
from executor.operators.scan.seq_scan import SeqScan
from executor.operators.scan.values_scan import DualScan
from executor.operators.set_ops import (
    UnionOperator, IntersectOperator, ExceptOperator)
from executor.operators.sort.in_memory_sort import SortOperator
from executor.operators.window.window_op import WindowOperator
from executor.ddl_executor import DDLExecutor
from executor.dml_executor import DMLExecutor
from parser.ast import *
from parser.ast_utils import contains_agg as _contains_agg_util
from parser.ast_utils import contains_window as _contains_window_util
from parser.formatter import Formatter
from storage.types import DataType, resolve_type_name
from utils.errors import (ColumnNotFoundError, ExecutionError)


class _ScalarAggOperator(Operator):
    def __init__(self, batch: VectorBatch) -> None:
        super().__init__()
        self._batch = batch; self._emitted = False
    def output_schema(self):
        return [(n, self._batch.columns[n].dtype)
                for n in self._batch.column_names]
    def open(self): self._emitted = False
    def next_batch(self):
        if self._emitted: return None
        self._emitted = True; return self._batch
    def close(self): pass


class SimplePlanner:
    def __init__(self, function_registry: FunctionRegistry,
                 budget: Optional[Any] = None) -> None:
        self._registry = function_registry
        self._evaluator = ExpressionEvaluator(function_registry)
        self._budget = budget
        self._ddl = DDLExecutor()
        self._dml = DMLExecutor(self._evaluator, planner=self)

    def execute(self, ast: Any, catalog: Catalog) -> ExecutionResult:
        if isinstance(ast, ExplainStmt):
            return self._exec_explain(ast, catalog)
        if isinstance(ast, AlterTableStmt):
            return self._ddl.exec_alter(ast, catalog)
        if isinstance(ast, SetOperationStmt):
            return self._exec_set_op(ast, catalog)
        if isinstance(ast, CopyStmt):
            return self._dml.exec_copy(ast, catalog)
        if isinstance(ast, CreateIndexStmt):
            return self._ddl.exec_create_index(ast, catalog)
        if isinstance(ast, DropIndexStmt):
            return self._ddl.exec_drop_index(ast, catalog)
        if isinstance(ast, CreateTableStmt):
            return self._ddl.exec_create(ast, catalog)
        if isinstance(ast, DropTableStmt):
            return self._ddl.exec_drop(ast, catalog)
        if isinstance(ast, InsertStmt):
            return self._dml.exec_insert(ast, catalog)
        if isinstance(ast, UpdateStmt):
            return self._dml.exec_update(ast, catalog)
        if isinstance(ast, DeleteStmt):
            return self._dml.exec_delete(ast, catalog)
        if isinstance(ast, SelectStmt):
            return self._exec_select(ast, catalog)
        raise ExecutionError(f"不支持: {type(ast).__name__}")

    # ═══ EXPLAIN ═══

    def _exec_explain(self, ast, catalog):
        inner = ast.statement; op = None
        if isinstance(inner, SelectStmt):
            inner = self._resolve_subqueries(inner, catalog)
            ha = any(self._contains_agg(e) for e in inner.select_list)
            hw = any(self._contains_window(e) for e in inner.select_list)
            if ha or inner.group_by: op = self._plan_grouped(inner, catalog)
            elif hw: op = self._plan_windowed(inner, catalog)
            else: op = self._plan_select(inner, catalog)
        elif isinstance(inner, SetOperationStmt):
            op = self._plan_any(inner, catalog)
        if op is None:
            return ExecutionResult(
                message=f"EXPLAIN 不支持 {type(inner).__name__}")
        text = op.explain()
        try:
            from planner.physical_plan import PhysicalPlanner
            from planner.cardinality import CardinalityEstimator
            if isinstance(inner, SelectStmt):
                table_rows = {}
                if inner.from_clause:
                    tname = inner.from_clause.table.name
                    if catalog.table_exists(tname):
                        table_rows[tname] = catalog.get_store(tname).row_count
                    for jc in inner.from_clause.joins:
                        if jc.table and catalog.table_exists(jc.table.name):
                            table_rows[jc.table.name] = catalog.get_store(jc.table.name).row_count
                pp = PhysicalPlanner(CardinalityEstimator())
                plan = pp.plan(inner, table_rows)
                text += '\n\n--- 代价估算 ---\n' + plan.explain()
        except Exception: pass
        rows = [[line] for line in text.split('\n') if line.strip()]
        return ExecutionResult(columns=['Plan'], column_types=[DataType.VARCHAR],
                               rows=rows, row_count=len(rows))

    # ═══ SET OPS ═══

    def _exec_set_op(self, ast, catalog):
        lo = self._plan_any(ast.left, catalog)
        ro = self._plan_any(ast.right, catalog)
        n = ast.op.upper()
        if n == 'UNION': op = UnionOperator(lo, ro, ast.all)
        elif n == 'INTERSECT': op = IntersectOperator(lo, ro, ast.all)
        elif n == 'EXCEPT': op = ExceptOperator(lo, ro, ast.all)
        else: raise ExecutionError(f"未知集合操作: {n}")
        return self._drain(op)

    def _plan_any(self, ast, catalog):
        if isinstance(ast, SetOperationStmt):
            l = self._plan_any(ast.left, catalog)
            r = self._plan_any(ast.right, catalog)
            if ast.op.upper() == 'UNION': return UnionOperator(l, r, ast.all)
            if ast.op.upper() == 'INTERSECT': return IntersectOperator(l, r, ast.all)
            return ExceptOperator(l, r, ast.all)
        if isinstance(ast, SelectStmt):
            ast = self._resolve_subqueries(ast, catalog)
            ha = any(self._contains_agg(e) for e in ast.select_list)
            hw = any(self._contains_window(e) for e in ast.select_list)
            if ha or ast.group_by: return self._plan_grouped(ast, catalog)
            if hw: return self._plan_windowed(ast, catalog)
            return self._plan_select(ast, catalog)
        raise ExecutionError(f"不支持: {type(ast).__name__}")

    # ═══ SELECT ═══

    def _exec_select(self, ast, catalog):
        ast = self._resolve_subqueries(ast, catalog)
        ha = any(self._contains_agg(e) for e in ast.select_list)
        hw = any(self._contains_window(e) for e in ast.select_list)
        if ha or ast.group_by: return self._drain(self._plan_grouped(ast, catalog))
        if hw: return self._drain(self._plan_windowed(ast, catalog))
        return self._drain(self._plan_select(ast, catalog))

    def _resolve_subqueries(self, ast, catalog):
        ch = {}
        if ast.where: ch['where'] = self._resolve_sq(ast.where, catalog)
        ch['select_list'] = [self._resolve_sq(e, catalog) for e in ast.select_list]
        if ast.having: ch['having'] = self._resolve_sq(ast.having, catalog)
        return dataclasses.replace(ast, **ch)

    def _resolve_sq(self, node, catalog):
        if node is None: return None
        if isinstance(node, SubqueryExpr):
            r = self.execute(node.query, catalog)
            if r.rows and r.columns:
                return Literal(value=r.rows[0][0],
                               inferred_type=r.column_types[0] if r.column_types else DataType.INT)
            return Literal(value=None, inferred_type=DataType.UNKNOWN)
        if isinstance(node, InExpr):
            nv = []
            for v in node.values:
                if isinstance(v, SubqueryExpr):
                    r = self.execute(v.query, catalog)
                    for row in r.rows:
                        nv.append(Literal(value=row[0],
                                          inferred_type=r.column_types[0] if r.column_types else DataType.INT))
                else: nv.append(v)
            return dataclasses.replace(node, values=nv,
                                        expr=self._resolve_sq(node.expr, catalog))
        if isinstance(node, ExistsExpr):
            r = self.execute(node.query, catalog)
            e = len(r.rows) > 0
            return Literal(value=not e if node.negated else e,
                           inferred_type=DataType.BOOLEAN)
        if isinstance(node, tuple):
            return tuple(self._resolve_sq(i, catalog) for i in node)
        if not dataclasses.is_dataclass(node) or isinstance(node, type): return node
        ch = {}
        for f in dataclasses.fields(node):
            c = getattr(node, f.name)
            if isinstance(c, list): ch[f.name] = [self._resolve_sq(i, catalog) for i in c]
            elif isinstance(c, tuple): ch[f.name] = tuple(self._resolve_sq(i, catalog) for i in c)
            elif dataclasses.is_dataclass(c) and not isinstance(c, type): ch[f.name] = self._resolve_sq(c, catalog)
        return dataclasses.replace(node, **ch) if ch else node

    # ═══ 计划构建 ═══

    def _plan_select(self, ast, catalog):
        c = self._build_source(ast, catalog)
        if ast.where: c = FilterOperator(c, ast.where)
        if ast.order_by:
            c = SortOperator(c, [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by])
        c = ProjectOperator(c, self._build_proj(ast))
        if ast.distinct: c = DistinctOperator(c)
        if ast.limit is not None or ast.offset is not None:
            c = LimitOperator(c, self._eval_const(ast.limit),
                              self._eval_const(ast.offset) or 0)
        return c

    def _plan_windowed(self, ast, catalog):
        c = self._build_source(ast, catalog)
        if ast.where: c = FilterOperator(c, ast.where)
        wm: Dict[str, WindowCall] = {}
        ns = [self._extract_windows(e, wm) for e in ast.select_list]
        if wm: c = WindowOperator(c, list(wm.items()))
        proj = [(self._out_name(o), (s.expr if isinstance(s, AliasExpr) else s))
                for o, s in zip(ast.select_list, ns)]
        c = ProjectOperator(c, proj)
        if ast.order_by:
            c = SortOperator(c, [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by])
        if ast.distinct: c = DistinctOperator(c)
        if ast.limit is not None or ast.offset is not None:
            c = LimitOperator(c, self._eval_const(ast.limit),
                              self._eval_const(ast.offset) or 0)
        return c

    def _plan_grouped(self, ast, catalog):
        src = self._build_source(ast, catalog)
        if ast.where: src = FilterOperator(src, ast.where)
        ss = dict(src.output_schema())
        ge = [(self._col_name(k), k) for k in ast.group_by.keys] if ast.group_by else []
        am: Dict[str, AggregateCall] = {}
        sub = [self._extract_aggs(e, am) for e in ast.select_list]
        hs = self._extract_aggs(ast.having, am) if ast.having else None
        ae = list(am.items())
        if not ge and not ae: return self._plan_select(ast, catalog)
        if not ge:
            batch = self._compute_scalar_agg(src, ss, am, sub, ast)
            return _ScalarAggOperator(batch)
        fae = []
        for t, ac in ae:
            if ac.distinct and ac.name.upper() == 'COUNT':
                fae.append((t, AggregateCall(name='COUNT_DISTINCT', args=ac.args)))
            elif ac.distinct and ac.name.upper() == 'SUM':
                fae.append((t, AggregateCall(name='SUM_DISTINCT', args=ac.args)))
            elif ac.distinct and ac.name.upper() == 'AVG':
                fae.append((t, AggregateCall(name='AVG_DISTINCT', args=ac.args)))
            else: fae.append((t, ac))
        c: Operator = HashAggOperator(src, ge, fae, self._registry, budget=self._budget)
        if hs is not None: c = FilterOperator(c, hs)
        if ast.order_by:
            c = SortOperator(c, [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by])
        fp = [(self._out_name(o), (s.expr if isinstance(s, AliasExpr) else s))
              for o, s in zip(ast.select_list, sub)]
        c = ProjectOperator(c, fp)
        if ast.distinct: c = DistinctOperator(c)
        if ast.limit is not None or ast.offset is not None:
            c = LimitOperator(c, self._eval_const(ast.limit),
                              self._eval_const(ast.offset) or 0)
        return c

    def _compute_scalar_agg(self, source, ss, am, sub, ast):
        st = {}
        for t, ac in am.items():
            n = ac.name.upper()
            if ac.distinct and n == 'COUNT': n = 'COUNT_DISTINCT'
            elif ac.distinct and n == 'SUM': n = 'SUM_DISTINCT'
            elif ac.distinct and n == 'AVG': n = 'AVG_DISTINCT'
            fn = self._registry.get_aggregate(n)
            if n == 'STRING_AGG' and len(ac.args) >= 2:
                try:
                    sv = self._evaluator.evaluate(
                        ac.args[1], VectorBatch.single_row_no_columns()).get(0)
                    st[t] = (fn, fn.init_with_sep(str(sv) if sv else ','))
                except Exception: st[t] = (fn, fn.init())
            else: st[t] = (fn, fn.init())
        source.open()
        while True:
            b = source.next_batch()
            if b is None: break
            b = Operator._ensure_batch(b)
            if b is None: break
            for t, ac in am.items():
                fn, s = st[t]
                if ac.args and isinstance(ac.args[0], StarExpr):
                    s = fn.update(s, None, b.row_count)
                else:
                    s = fn.update(s, self._evaluator.evaluate(ac.args[0], b), b.row_count)
                st[t] = (fn, s)
        source.close()
        sd = {}
        for t, (fn, s) in st.items():
            v = fn.finalize(s); ac = am[t]
            it = [ExpressionEvaluator.infer_type(ac.args[0], ss)] if ac.args and not isinstance(ac.args[0], StarExpr) else []
            sd[t] = DataVector.from_scalar(v, fn.return_type(it))
        sb = VectorBatch(columns=sd, _row_count=1)
        rc = {}; on = []
        for o, s in zip(ast.select_list, sub):
            nm = self._out_name(o)
            rc[nm] = self._evaluator.evaluate(s, sb)
            on.append(nm)
        return VectorBatch(columns=rc, _column_order=on, _row_count=1)

    # ═══ 数据源 ═══

    def _build_source(self, ast, catalog):
        if ast.from_clause is None: return DualScan()
        fc = ast.from_clause; tref = fc.table
        if tref.subquery: return self._plan_any(tref.subquery, catalog)
        if tref.func_args is not None or tref.name.lower() == 'generate_series':
            return self._build_generate_series(ast)
        has_join = bool(fc.joins)
        if has_join:
            need_resolve = any(getattr(jc, 'natural', False) or getattr(jc, 'using', None)
                               for jc in fc.joins)
            resolved = self._resolve_natural_using(fc, catalog) if need_resolve else fc.joins
            lq = tref.alias or tref.name; st = catalog.get_store(tref.name)
            lc = [c.name for c in st.schema.columns]
            cur: Operator = ProjectOperator(SeqScan(tref.name, st, lc),
                [(f"{lq}.{c}", ColumnRef(table=None, column=c)) for c in lc])
            for jc in resolved:
                assert jc.table is not None
                if jc.table.subquery:
                    rop = self._plan_any(jc.table.subquery, catalog)
                    rq = jc.table.alias or '__sub'
                    rs = rop.output_schema()
                    rop = ProjectOperator(rop,
                        [(f"{rq}.{n}", ColumnRef(table=None, column=n)) for n, _ in rs])
                else:
                    rq = jc.table.alias or jc.table.name
                    if jc.table.name.lower() == 'generate_series' or jc.table.func_args is not None:
                        rop = self._build_generate_series_from_ref(jc.table)
                        if rop:
                            if jc.join_type == 'CROSS': cur = CrossJoinOperator(cur, rop)
                            else: cur = HashJoinOperator(cur, rop, jc.join_type, jc.on)
                            continue
                    rs = catalog.get_store(jc.table.name)
                    rc = [c.name for c in rs.schema.columns]
                    rop = ProjectOperator(SeqScan(jc.table.name, rs, rc),
                        [(f"{rq}.{c}", ColumnRef(table=None, column=c)) for c in rc])
                if jc.join_type == 'CROSS': cur = CrossJoinOperator(cur, rop)
                else: cur = HashJoinOperator(cur, rop, jc.join_type, jc.on)
            return cur
        else:
            st = catalog.get_store(tref.name); needed = self._collect_all_cols(ast)
            ordered = [c.name for c in st.schema.columns if c.name in needed]
            if not ordered: ordered = [st.schema.columns[0].name]
            return SeqScan(tref.name, st, ordered)

    def _build_generate_series(self, ast):
        try:
            from executor.operators.scan.values_scan import GenerateSeriesOperator
            tref = ast.from_clause.table; alias = tref.alias or 'generate_series'
            if tref.func_args and len(tref.func_args) >= 2:
                parsed = GenerateSeriesOperator.parse_args(tref.func_args, self._evaluator)
                if parsed: return GenerateSeriesOperator(*parsed, col_name=alias)
            return GenerateSeriesOperator(1, 10, 1, col_name=alias)
        except ImportError: return DualScan()

    def _build_generate_series_from_ref(self, tref):
        try:
            from executor.operators.scan.values_scan import GenerateSeriesOperator
            alias = tref.alias or 'generate_series'
            if tref.func_args and len(tref.func_args) >= 2:
                parsed = GenerateSeriesOperator.parse_args(tref.func_args, self._evaluator)
                if parsed: return GenerateSeriesOperator(*parsed, col_name=alias)
            return GenerateSeriesOperator(1, 10, 1, col_name=alias)
        except ImportError: return None

    def _resolve_natural_using(self, fc, catalog):
        bn = fc.table.name; ba = fc.table.alias or bn
        bc = set(catalog.get_table_columns(bn)) if catalog.table_exists(bn) else set()
        resolved = []
        for jc in fc.joins:
            if jc.table is None: resolved.append(jc); continue
            rn = jc.table.name; ra = jc.table.alias or rn
            if getattr(jc, 'natural', False) and jc.table.subquery is None:
                if catalog.table_exists(rn):
                    rc = set(catalog.get_table_columns(rn)); common = bc & rc
                    if common:
                        resolved.append(JoinClause(join_type=jc.join_type, table=jc.table,
                            on=self._build_equi_on(sorted(common), ba, ra)))
                    else: resolved.append(JoinClause(join_type='CROSS', table=jc.table))
                else: resolved.append(jc)
            elif getattr(jc, 'using', None):
                resolved.append(JoinClause(join_type=jc.join_type, table=jc.table,
                    on=self._build_equi_on(jc.using, ba, ra)))
            else: resolved.append(jc)
            if catalog.table_exists(rn): bc |= set(catalog.get_table_columns(rn))
        return resolved

    @staticmethod
    def _build_equi_on(columns, la, ra):
        conds = [BinaryExpr(op='=',
                             left=ColumnRef(table=la, column=c),
                             right=ColumnRef(table=ra, column=c)) for c in columns]
        r = conds[0]
        for c in conds[1:]: r = BinaryExpr(op='AND', left=r, right=c)
        return r

    # ═══ 提取 ═══

    def _extract_aggs(self, expr, am):
        if expr is None: return None
        if isinstance(expr, AggregateCall):
            t = f'__agg_{len(am)}'; am[t] = expr
            return ColumnRef(table=None, column=t)
        if isinstance(expr, AliasExpr):
            return AliasExpr(expr=self._extract_aggs(expr.expr, am), alias=expr.alias)
        return self._rec_extract(expr, am)

    def _rec_extract(self, node, am):
        if node is None: return None
        if isinstance(node, AggregateCall): return self._extract_aggs(node, am)
        if isinstance(node, tuple): return tuple(self._rec_extract(i, am) for i in node)
        if not dataclasses.is_dataclass(node) or isinstance(node, type): return node
        ch = {}
        for f in dataclasses.fields(node):
            c = getattr(node, f.name)
            if isinstance(c, AggregateCall): ch[f.name] = self._extract_aggs(c, am)
            elif isinstance(c, list):
                nl = []
                for item in c:
                    if isinstance(item, AggregateCall): nl.append(self._extract_aggs(item, am))
                    elif isinstance(item, tuple): nl.append(tuple(self._rec_extract(x, am) for x in item))
                    elif dataclasses.is_dataclass(item) and not isinstance(item, type): nl.append(self._rec_extract(item, am))
                    else: nl.append(item)
                ch[f.name] = nl
            elif isinstance(c, tuple): ch[f.name] = tuple(self._rec_extract(i, am) for i in c)
            elif dataclasses.is_dataclass(c) and not isinstance(c, type): ch[f.name] = self._rec_extract(c, am)
        return dataclasses.replace(node, **ch) if ch else node

    def _extract_windows(self, expr, wm):
        if isinstance(expr, WindowCall):
            t = f'__win_{len(wm)}'; wm[t] = expr
            return ColumnRef(table=None, column=t)
        if isinstance(expr, AliasExpr):
            return AliasExpr(expr=self._extract_windows(expr.expr, wm), alias=expr.alias)
        if not dataclasses.is_dataclass(expr) or isinstance(expr, type): return expr
        ch = {}
        for f in dataclasses.fields(expr):
            c = getattr(expr, f.name)
            if isinstance(c, WindowCall): ch[f.name] = self._extract_windows(c, wm)
            elif isinstance(c, list): ch[f.name] = [self._extract_windows(i, wm) for i in c]
            elif dataclasses.is_dataclass(c) and not isinstance(c, type): ch[f.name] = self._extract_windows(c, wm)
        return dataclasses.replace(expr, **ch) if ch else expr

    # ═══ 委托到 ast_utils ═══

    def _contains_agg(self, n): return _contains_agg_util(n)
    def _contains_window(self, n): return _contains_window_util(n)

    # ═══ 工具方法 ═══

    def _collect_all_cols(self, ast):
        r: Set[str] = set()
        for e in ast.select_list: r |= self._cc(e)
        if ast.where: r |= self._cc(ast.where)
        for sk in (ast.order_by or []): r |= self._cc(sk.expr)
        if ast.group_by:
            for k in ast.group_by.keys: r |= self._cc(k)
        if ast.having: r |= self._cc(ast.having)
        return r

    def _cc(self, n):
        if n is None: return set()
        if isinstance(n, ColumnRef):
            r = {n.column}
            if n.table: r.add(f"{n.table}.{n.column}")
            return r
        if isinstance(n, tuple): return set().union(*(self._cc(i) for i in n))
        if not dataclasses.is_dataclass(n) or isinstance(n, type): return set()
        r: Set[str] = set()
        for f in dataclasses.fields(n):
            c = getattr(n, f.name)
            if isinstance(c, list):
                for i in c: r |= self._cc(i)
            else: r |= self._cc(c)
        return r

    def _build_proj(self, ast):
        proj = []
        for expr in ast.select_list:
            nm = self._out_name(expr)
            inner = expr.expr if isinstance(expr, AliasExpr) else expr
            if isinstance(inner, StarExpr):
                raise ExecutionError("内部错误: StarExpr 未展开")
            proj.append((nm, inner))
        return proj

    def _out_name(self, expr):
        if isinstance(expr, AliasExpr): return expr.alias
        if isinstance(expr, ColumnRef): return expr.column
        return Formatter.expr_to_sql(expr)

    def _col_name(self, expr):
        if isinstance(expr, ColumnRef): return expr.column
        return Formatter.expr_to_sql(expr)

    def _eval_const(self, expr):
        if expr is None: return None
        dummy = VectorBatch.single_row_no_columns()
        v = self._evaluator.evaluate(expr, dummy).get(0)
        if v is None: return None
        v = int(v)
        if v < 0: raise ExecutionError("LIMIT/OFFSET 必须非负")
        return v

    def _drain(self, op):
        s = op.output_schema(); cn = [n for n, _ in s]; ct = [t for _, t in s]
        op.open(); rows = []
        while True:
            b = op.next_batch()
            if b is None: break
            b = Operator._ensure_batch(b)
            if b is None: break
            rows.extend(b.to_rows())
        op.close()
        return ExecutionResult(columns=cn, column_types=ct, rows=rows, row_count=len(rows))
