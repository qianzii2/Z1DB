from __future__ import annotations
"""Simple planner — Phase 3: windows, set ops, subqueries, COUNT DISTINCT."""
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple
from catalog.catalog import Catalog, ColumnSchema, TableSchema
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
from executor.operators.set_ops import UnionOperator, IntersectOperator, ExceptOperator
from executor.operators.sort.in_memory_sort import SortOperator
from executor.operators.window.window_op import WindowOperator
from parser.ast import *
from parser.formatter import Formatter
from storage.types import DataType, resolve_type_name
from utils.errors import (ColumnNotFoundError, ExecutionError, NumericOverflowError)


# ======================================================================
# ScalarAggOperator: wraps scalar aggregate as a proper Operator
# ======================================================================
class _ScalarAggOperator(Operator):
    """Wraps a scalar aggregate result as an Operator producing one batch."""

    def __init__(self, batch: VectorBatch) -> None:
        super().__init__()
        self._batch = batch
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return [(n, self._batch.columns[n].dtype) for n in self._batch.column_names]

    def open(self) -> None:
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        return self._batch

    def close(self) -> None:
        pass


class SimplePlanner:
    def __init__(self, function_registry: FunctionRegistry) -> None:
        self._registry = function_registry
        self._evaluator = ExpressionEvaluator(function_registry)

    def execute(self, ast: Any, catalog: Catalog) -> ExecutionResult:
        # EXPLAIN
        if isinstance(ast, ExplainStmt):
            return self._exec_explain(ast, catalog)
        # ALTER TABLE
        if isinstance(ast, AlterTableStmt):
            return self._exec_alter(ast, catalog)
        if isinstance(ast, SetOperationStmt):
            return self._exec_set_op(ast, catalog)
        if isinstance(ast, CreateTableStmt):
            return self._exec_create(ast, catalog)
        if isinstance(ast, DropTableStmt):
            return self._exec_drop(ast, catalog)
        if isinstance(ast, InsertStmt):
            return self._exec_insert(ast, catalog)
        if isinstance(ast, UpdateStmt):
            return self._exec_update(ast, catalog)
        if isinstance(ast, DeleteStmt):
            return self._exec_delete(ast, catalog)
        if isinstance(ast, SelectStmt):
            return self._exec_select(ast, catalog)
        raise ExecutionError(f"unsupported: {type(ast).__name__}")

    def _exec_explain(self, ast: ExplainStmt, catalog: Catalog) -> ExecutionResult:
        """Build the plan tree and return its textual representation."""
        inner = ast.statement
        if isinstance(inner, SelectStmt):
            inner = self._resolve_subqueries(inner, catalog)
            has_agg = any(self._contains_agg(e) for e in inner.select_list)
            has_win = any(self._contains_window(e) for e in inner.select_list)
            if has_agg or inner.group_by:
                op = self._plan_grouped(inner, catalog)
            elif has_win:
                op = self._plan_windowed(inner, catalog)
            else:
                op = self._plan_select(inner, catalog)
        elif isinstance(inner, SetOperationStmt):
            op = self._plan_any(inner, catalog)
        else:
            return ExecutionResult(message=f"EXPLAIN not supported for {type(inner).__name__}")
        text = op.explain()
        rows = [[line] for line in text.split('\n') if line.strip()]
        return ExecutionResult(columns=['Plan'], column_types=[DataType.VARCHAR],
                               rows=rows, row_count=len(rows))

    def _exec_alter(self, ast: AlterTableStmt, catalog: Catalog) -> ExecutionResult:
        if ast.action == 'ADD_COLUMN':
            assert ast.column_def is not None
            from storage.types import resolve_type_name
            dt, ml = resolve_type_name(ast.column_def.type_name.name, ast.column_def.type_name.params)
            col = ColumnSchema(name=ast.column_def.name, dtype=dt,
                               nullable=ast.column_def.nullable,
                               primary_key=ast.column_def.primary_key, max_length=ml)
            catalog.alter_add_column(ast.table, col)
            return ExecutionResult(message='OK')
        if ast.action == 'DROP_COLUMN':
            catalog.alter_drop_column(ast.table, ast.column_name)
            return ExecutionResult(message='OK')
        if ast.action == 'RENAME_COLUMN':
            catalog.alter_rename_column(ast.table, ast.column_name, ast.new_name)
            return ExecutionResult(message='OK')
        raise ExecutionError(f"unsupported ALTER action: {ast.action}")

    # ═══ Set operations ═══
    def _exec_set_op(self, ast: SetOperationStmt, catalog: Catalog) -> ExecutionResult:
        left_op = self._plan_any(ast.left, catalog)
        right_op = self._plan_any(ast.right, catalog)
        op_name = ast.op.upper()
        if op_name == 'UNION':
            op: Operator = UnionOperator(left_op, right_op, ast.all)
        elif op_name == 'INTERSECT':
            op = IntersectOperator(left_op, right_op, ast.all)
        elif op_name == 'EXCEPT':
            op = ExceptOperator(left_op, right_op, ast.all)
        else:
            raise ExecutionError(f"unknown set op: {op_name}")
        return self._drain(op)

    def _plan_any(self, ast: Any, catalog: Catalog) -> Operator:
        """Plan any statement subtree, always returning an Operator."""
        if isinstance(ast, SetOperationStmt):
            left = self._plan_any(ast.left, catalog)
            right = self._plan_any(ast.right, catalog)
            if ast.op.upper() == 'UNION':
                return UnionOperator(left, right, ast.all)
            if ast.op.upper() == 'INTERSECT':
                return IntersectOperator(left, right, ast.all)
            return ExceptOperator(left, right, ast.all)
        if isinstance(ast, SelectStmt):
            # Pre-process subqueries
            ast = self._resolve_subqueries(ast, catalog)
            has_agg = any(self._contains_agg(e) for e in ast.select_list)
            has_win = any(self._contains_window(e) for e in ast.select_list)
            if has_agg or ast.group_by:
                return self._plan_grouped(ast, catalog)
            if has_win:
                return self._plan_windowed(ast, catalog)
            return self._plan_select(ast, catalog)
        raise ExecutionError(f"unsupported in set op: {type(ast).__name__}")

    # ═══ DDL ═══
    def _exec_create(self, ast: CreateTableStmt, catalog: Catalog) -> ExecutionResult:
        cols = []
        for cd in ast.columns:
            dt, ml = resolve_type_name(cd.type_name.name, cd.type_name.params)
            cols.append(ColumnSchema(name=cd.name, dtype=dt, nullable=cd.nullable,
                                     primary_key=cd.primary_key, max_length=ml))
        catalog.create_table(TableSchema(name=ast.table, columns=cols), ast.if_not_exists)
        return ExecutionResult(message='OK')

    def _exec_drop(self, ast: DropTableStmt, catalog: Catalog) -> ExecutionResult:
        catalog.drop_table(ast.table, ast.if_exists)
        return ExecutionResult(message='OK')

    # ═══ DML ═══
    def _exec_insert(self, ast: InsertStmt, catalog: Catalog) -> ExecutionResult:
        schema = catalog.get_table(ast.table)
        store = catalog.get_store(ast.table)
        if ast.columns:
            seen: set = set()
            sc = {c.name for c in schema.columns}
            for c in ast.columns:
                if c in seen:
                    raise ExecutionError(f"duplicate column: '{c}'")
                if c not in sc:
                    raise ColumnNotFoundError(c)
                seen.add(c)
        dummy = VectorBatch.single_row_no_columns()
        count = 0
        for ve in ast.values:
            pv = [self._evaluator.evaluate(e, dummy).get(0) for e in ve]
            if ast.columns:
                full = self._reorder(pv, ast.columns, schema)
            else:
                if len(pv) != len(schema.columns):
                    raise ExecutionError(f"expected {len(schema.columns)} values, got {len(pv)}")
                full = pv
            store.append_row(self._validate_row(full, schema))
            count += 1
        w = 'row' if count == 1 else 'rows'
        return ExecutionResult(affected_rows=count, message=f"Inserted {count} {w}")

    def _reorder(self, vals: list, cols: List[str], schema: TableSchema) -> list:
        cm = dict(zip(cols, vals))
        result = []
        for c in schema.columns:
            if c.name in cm:
                result.append(cm[c.name])
            elif c.nullable:
                result.append(None)
            else:
                raise ExecutionError(f"column '{c.name}' is NOT NULL")
        return result

    def _validate_row(self, row: list, schema: TableSchema) -> list:
        if len(row) != len(schema.columns):
            raise ExecutionError("column count mismatch")
        result = []
        for val, col in zip(row, schema.columns):
            if val is None:
                if not col.nullable:
                    raise ExecutionError(f"NULL for NOT NULL '{col.name}'")
                result.append(None)
                continue
            dt = col.dtype
            try:
                if dt == DataType.INT:
                    v = int(val)
                    if not (-2_147_483_648 <= v <= 2_147_483_647):
                        raise NumericOverflowError(f"overflow: {v}")
                    result.append(v)
                elif dt == DataType.BIGINT:
                    result.append(int(val))
                elif dt in (DataType.FLOAT, DataType.DOUBLE):
                    result.append(float(val))
                elif dt == DataType.BOOLEAN:
                    result.append(val if isinstance(val, bool) else bool(val))
                elif dt in (DataType.VARCHAR, DataType.TEXT):
                    s = str(val)
                    if col.max_length and len(s) > col.max_length:
                        s = s[:col.max_length]
                    result.append(s)
                else:
                    result.append(val)
            except (ValueError, TypeError) as e:
                raise ExecutionError(f"cannot convert {val!r} to {dt.name}: {e}")
        return result

    def _exec_update(self, ast: UpdateStmt, catalog: Catalog) -> ExecutionResult:
        schema = catalog.get_table(ast.table)
        store = catalog.get_store(ast.table)
        all_rows = store.read_all_rows()
        cn = schema.column_names
        ct = [c.dtype for c in schema.columns]
        count = 0
        for ri, row in enumerate(all_rows):
            if ast.where:
                b = VectorBatch.from_rows([row], cn, ct)
                if not self._evaluator.evaluate_predicate(ast.where, b).get_bit(0):
                    continue
            for a in ast.assignments:
                ci = next((i for i, c in enumerate(schema.columns) if c.name == a.column), None)
                if ci is None:
                    raise ColumnNotFoundError(a.column)
                b = VectorBatch.from_rows([row], cn, ct)
                all_rows[ri][ci] = self._evaluator.evaluate(a.value, b).get(0)
            count += 1
        store.truncate()
        for r in all_rows:
            store.append_row(r)
        w = 'row' if count == 1 else 'rows'
        return ExecutionResult(affected_rows=count, message=f"Updated {count} {w}")

    def _exec_delete(self, ast: DeleteStmt, catalog: Catalog) -> ExecutionResult:
        schema = catalog.get_table(ast.table)
        store = catalog.get_store(ast.table)
        if ast.where is None:
            c = store.row_count
            store.truncate()
            return ExecutionResult(affected_rows=c,
                                   message=f"Deleted {c} {'row' if c == 1 else 'rows'}")
        ar = store.read_all_rows()
        cn = schema.column_names
        ct = [c.dtype for c in schema.columns]
        keep = []
        count = 0
        for row in ar:
            b = VectorBatch.from_rows([row], cn, ct)
            if self._evaluator.evaluate_predicate(ast.where, b).get_bit(0):
                count += 1
            else:
                keep.append(row)
        store.truncate()
        for r in keep:
            store.append_row(r)
        return ExecutionResult(affected_rows=count,
                               message=f"Deleted {count} {'row' if count == 1 else 'rows'}")

    # ═══ SELECT ═══
    def _exec_select(self, ast: SelectStmt, catalog: Catalog) -> ExecutionResult:
        ast = self._resolve_subqueries(ast, catalog)
        has_agg = any(self._contains_agg(e) for e in ast.select_list)
        has_win = any(self._contains_window(e) for e in ast.select_list)
        if has_agg or ast.group_by:
            return self._drain(self._plan_grouped(ast, catalog))
        if has_win:
            return self._drain(self._plan_windowed(ast, catalog))
        return self._drain(self._plan_select(ast, catalog))

    def _resolve_subqueries(self, ast: SelectStmt, catalog: Catalog) -> SelectStmt:
        if ast.where:
            ast = dataclasses.replace(ast, where=self._resolve_sq_node(ast.where, catalog))
        new_sl = [self._resolve_sq_node(e, catalog) for e in ast.select_list]
        return dataclasses.replace(ast, select_list=new_sl)

    def _resolve_sq_node(self, node: Any, catalog: Catalog) -> Any:
        if node is None:
            return None
        if isinstance(node, SubqueryExpr):
            result = self.execute(node.query, catalog)
            if result.rows and result.columns:
                return Literal(value=result.rows[0][0],
                               inferred_type=result.column_types[0] if result.column_types else DataType.INT)
            return Literal(value=None, inferred_type=DataType.UNKNOWN)
        if isinstance(node, InExpr):
            new_vals = []
            for v in node.values:
                if isinstance(v, SubqueryExpr):
                    result = self.execute(v.query, catalog)
                    for row in result.rows:
                        val = row[0]
                        dt = result.column_types[0] if result.column_types else DataType.INT
                        new_vals.append(Literal(value=val, inferred_type=dt))
                else:
                    new_vals.append(v)
            return dataclasses.replace(node, values=new_vals,
                                       expr=self._resolve_sq_node(node.expr, catalog))
        if isinstance(node, ExistsExpr):
            result = self.execute(node.query, catalog)
            exists = len(result.rows) > 0
            val = not exists if node.negated else exists
            return Literal(value=val, inferred_type=DataType.BOOLEAN)
        if isinstance(node, tuple):
            return tuple(self._resolve_sq_node(i, catalog) for i in node)
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        changes: dict = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                changes[f.name] = [self._resolve_sq_node(i, catalog) for i in child]
            elif isinstance(child, tuple):
                changes[f.name] = tuple(self._resolve_sq_node(i, catalog) for i in child)
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._resolve_sq_node(child, catalog)
        return dataclasses.replace(node, **changes) if changes else node

    # ═══ Plan builders ═══
    def _plan_select(self, ast: SelectStmt, catalog: Catalog) -> Operator:
        c = self._build_source(ast, catalog)
        if ast.where:
            c = FilterOperator(c, ast.where)
        if ast.order_by:
            c = SortOperator(c, [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by])
        c = ProjectOperator(c, self._build_proj(ast))
        if ast.distinct:
            c = DistinctOperator(c)
        if ast.limit is not None or ast.offset is not None:
            c = LimitOperator(c, self._eval_const(ast.limit),
                              self._eval_const(ast.offset) or 0)
        return c

    def _plan_windowed(self, ast: SelectStmt, catalog: Catalog) -> Operator:
        c = self._build_source(ast, catalog)
        if ast.where:
            c = FilterOperator(c, ast.where)
        win_map: Dict[str, WindowCall] = {}
        new_sl = [self._extract_windows(e, win_map) for e in ast.select_list]
        if win_map:
            c = WindowOperator(c, list(win_map.items()))
        proj = []
        for orig, subst in zip(ast.select_list, new_sl):
            name = self._out_name(orig)
            inner = subst.expr if isinstance(subst, AliasExpr) else subst
            proj.append((name, inner))
        if ast.order_by:
            c = SortOperator(c, [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by])
        c = ProjectOperator(c, proj)
        if ast.distinct:
            c = DistinctOperator(c)
        if ast.limit is not None or ast.offset is not None:
            c = LimitOperator(c, self._eval_const(ast.limit),
                              self._eval_const(ast.offset) or 0)
        return c

    def _plan_grouped(self, ast: SelectStmt, catalog: Catalog) -> Operator:
        """Always returns an Operator (never ExecutionResult)."""
        source = self._build_source(ast, catalog)
        if ast.where:
            source = FilterOperator(source, ast.where)
        scan_schema = dict(source.output_schema())

        ge: List[Tuple[str, Any]] = []
        if ast.group_by:
            for k in ast.group_by.keys:
                ge.append((self._col_name(k), k))

        agg_map: Dict[str, AggregateCall] = {}
        substituted = [self._extract_aggs(e, agg_map) for e in ast.select_list]
        having_subst = self._extract_aggs(ast.having, agg_map) if ast.having else None

        ae = [(t, ac) for t, ac in agg_map.items()]
        if not ge and not ae:
            return self._plan_select(ast, catalog)

        # Scalar aggregate (no GROUP BY) — wrap as Operator
        if not ge:
            batch = self._compute_scalar_agg(source, scan_schema, agg_map, substituted, ast)
            c: Operator = _ScalarAggOperator(batch)
            return c

        # Handle COUNT(DISTINCT)
        final_ae = []
        for temp, ac in ae:
            if ac.distinct and ac.name.upper() == 'COUNT':
                final_ae.append((temp, AggregateCall(name='COUNT_DISTINCT', args=ac.args)))
            else:
                final_ae.append((temp, ac))

        c = HashAggOperator(source, ge, final_ae, self._registry)
        if having_subst is not None:
            c = FilterOperator(c, having_subst)
        if ast.order_by:
            c = SortOperator(c, [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by])
        fp = [(self._out_name(o), (s.expr if isinstance(s, AliasExpr) else s))
              for o, s in zip(ast.select_list, substituted)]
        c = ProjectOperator(c, fp)
        if ast.distinct:
            c = DistinctOperator(c)
        if ast.limit is not None or ast.offset is not None:
            c = LimitOperator(c, self._eval_const(ast.limit),
                              self._eval_const(ast.offset) or 0)
        return c

    def _compute_scalar_agg(self, source: Operator, scan_schema: Dict[str, DataType],
                            agg_map: Dict[str, AggregateCall], substituted: list,
                            ast: Any) -> VectorBatch:
        """Run scalar aggregate and return a single-row VectorBatch."""
        agg_states: Dict[str, tuple] = {}
        for temp, ac in agg_map.items():
            name = ac.name.upper()
            if ac.distinct and name == 'COUNT':
                name = 'COUNT_DISTINCT'
            func = self._registry.get_aggregate(name)
            agg_states[temp] = (func, func.init())

        source.open()
        while True:
            batch = source.next_batch()
            if batch is None:
                break
            for temp, ac in agg_map.items():
                func, state = agg_states[temp]
                if ac.args and isinstance(ac.args[0], StarExpr):
                    state = func.update(state, None, batch.row_count)
                else:
                    state = func.update(state, self._evaluator.evaluate(ac.args[0], batch),
                                        batch.row_count)
                agg_states[temp] = (func, state)
        source.close()

        sd: Dict[str, DataVector] = {}
        for temp, (func, state) in agg_states.items():
            val = func.finalize(state)
            ac = agg_map[temp]
            if ac.args and not isinstance(ac.args[0], StarExpr):
                it = [ExpressionEvaluator.infer_type(ac.args[0], scan_schema)]
            else:
                it = []
            sd[temp] = DataVector.from_scalar(val, func.return_type(it))
        sb = VectorBatch(columns=sd, _row_count=1)

        rc: Dict[str, DataVector] = {}
        on: List[str] = []
        for orig, subst in zip(ast.select_list, substituted):
            name = self._out_name(orig)
            rc[name] = self._evaluator.evaluate(subst, sb)
            on.append(name)

        return VectorBatch(columns=rc, _column_order=on, _row_count=1)

    # ═══ Source ═══
    def _build_source(self, ast: SelectStmt, catalog: Catalog) -> Operator:
        if ast.from_clause is None:
            return DualScan()
        fc = ast.from_clause
        tref = fc.table
        if tref.subquery:
            return self._plan_any(tref.subquery, catalog)
        has_join = bool(fc.joins)
        if has_join:
            lq = tref.alias or tref.name
            st = catalog.get_store(tref.name)
            lc = [c.name for c in st.schema.columns]
            cur: Operator = ProjectOperator(
                SeqScan(tref.name, st, lc),
                [(f"{lq}.{c}", ColumnRef(table=None, column=c)) for c in lc])
            for jc in fc.joins:
                assert jc.table is not None
                if jc.table.subquery:
                    right_op = self._plan_any(jc.table.subquery, catalog)
                    rq = jc.table.alias or '__sub'
                    rs = right_op.output_schema()
                    right_proj = [(f"{rq}.{n}", ColumnRef(table=None, column=n))
                                  for n, _ in rs]
                    right_op = ProjectOperator(right_op, right_proj)
                else:
                    rq = jc.table.alias or jc.table.name
                    r_store = catalog.get_store(jc.table.name)
                    r_cols = [c.name for c in r_store.schema.columns]
                    right_op = ProjectOperator(
                        SeqScan(jc.table.name, r_store, r_cols),
                        [(f"{rq}.{c}", ColumnRef(table=None, column=c)) for c in r_cols])
                if jc.join_type == 'CROSS':
                    cur = CrossJoinOperator(cur, right_op)
                else:
                    cur = HashJoinOperator(cur, right_op, jc.join_type, jc.on)
            return cur
        else:
            st = catalog.get_store(tref.name)
            needed = self._collect_all_cols(ast)
            ordered = [c.name for c in st.schema.columns if c.name in needed]
            if not ordered:
                ordered = [st.schema.columns[0].name]
            return SeqScan(tref.name, st, ordered)

    # ═══ Extraction helpers ═══
    def _extract_aggs(self, expr: Any, agg_map: Dict[str, AggregateCall]) -> Any:
        if expr is None:
            return None
        if isinstance(expr, AggregateCall):
            t = f'__agg_{len(agg_map)}'
            agg_map[t] = expr
            return ColumnRef(table=None, column=t)
        if isinstance(expr, AliasExpr):
            return AliasExpr(expr=self._extract_aggs(expr.expr, agg_map), alias=expr.alias)
        return self._rec_extract(expr, agg_map)

    def _rec_extract(self, node: Any, agg_map: Dict) -> Any:
        if node is None:
            return None
        if isinstance(node, AggregateCall):
            return self._extract_aggs(node, agg_map)
        if isinstance(node, tuple):
            return tuple(self._rec_extract(i, agg_map) for i in node)
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        ch: dict = {}
        for f in dataclasses.fields(node):
            c = getattr(node, f.name)
            if isinstance(c, AggregateCall):
                ch[f.name] = self._extract_aggs(c, agg_map)
            elif isinstance(c, list):
                new_list = []
                for item in c:
                    if isinstance(item, AggregateCall):
                        new_list.append(self._extract_aggs(item, agg_map))
                    elif isinstance(item, tuple):
                        new_list.append(tuple(self._rec_extract(x, agg_map) for x in item))
                    elif dataclasses.is_dataclass(item) and not isinstance(item, type):
                        new_list.append(self._rec_extract(item, agg_map))
                    else:
                        new_list.append(item)
                ch[f.name] = new_list
            elif isinstance(c, tuple):
                ch[f.name] = tuple(self._rec_extract(i, agg_map) for i in c)
            elif dataclasses.is_dataclass(c) and not isinstance(c, type):
                ch[f.name] = self._rec_extract(c, agg_map)
        return dataclasses.replace(node, **ch) if ch else node

    def _extract_windows(self, expr: Any, win_map: Dict[str, WindowCall]) -> Any:
        if isinstance(expr, WindowCall):
            t = f'__win_{len(win_map)}'
            win_map[t] = expr
            return ColumnRef(table=None, column=t)
        if isinstance(expr, AliasExpr):
            return AliasExpr(expr=self._extract_windows(expr.expr, win_map), alias=expr.alias)
        if not dataclasses.is_dataclass(expr) or isinstance(expr, type):
            return expr
        ch: dict = {}
        for f in dataclasses.fields(expr):
            c = getattr(expr, f.name)
            if isinstance(c, WindowCall):
                ch[f.name] = self._extract_windows(c, win_map)
            elif isinstance(c, list):
                ch[f.name] = [self._extract_windows(i, win_map) for i in c]
            elif dataclasses.is_dataclass(c) and not isinstance(c, type):
                ch[f.name] = self._extract_windows(c, win_map)
        return dataclasses.replace(expr, **ch) if ch else expr

    def _contains_agg(self, n: Any) -> bool:
        if isinstance(n, AggregateCall):
            return True
        if isinstance(n, WindowCall):
            return False
        if isinstance(n, AliasExpr):
            return self._contains_agg(n.expr)
        if isinstance(n, tuple):
            return any(self._contains_agg(i) for i in n)
        if n is None or not dataclasses.is_dataclass(n) or isinstance(n, type):
            return False
        for f in dataclasses.fields(n):
            c = getattr(n, f.name)
            if isinstance(c, list):
                if any(self._contains_agg(i) for i in c):
                    return True
            elif self._contains_agg(c):
                return True
        return False

    def _contains_window(self, n: Any) -> bool:
        if isinstance(n, WindowCall):
            return True
        if isinstance(n, AliasExpr):
            return self._contains_window(n.expr)
        if n is None or not dataclasses.is_dataclass(n) or isinstance(n, type):
            return False
        for f in dataclasses.fields(n):
            c = getattr(n, f.name)
            if isinstance(c, list):
                if any(self._contains_window(i) for i in c):
                    return True
            elif self._contains_window(c):
                return True
        return False

    def _collect_all_cols(self, ast: SelectStmt) -> Set[str]:
        r: Set[str] = set()
        for e in ast.select_list:
            r |= self._cc(e)
        if ast.where:
            r |= self._cc(ast.where)
        for sk in (ast.order_by or []):
            r |= self._cc(sk.expr)
        if ast.group_by:
            for k in ast.group_by.keys:
                r |= self._cc(k)
        if ast.having:
            r |= self._cc(ast.having)
        return r

    def _cc(self, n: Any) -> Set[str]:
        if n is None:
            return set()
        if isinstance(n, ColumnRef):
            r = {n.column}
            if n.table:
                r.add(f"{n.table}.{n.column}")
            return r
        if isinstance(n, tuple):
            result: Set[str] = set()
            for item in n:
                result |= self._cc(item)
            return result
        if not dataclasses.is_dataclass(n) or isinstance(n, type):
            return set()
        r: Set[str] = set()
        for f in dataclasses.fields(n):
            c = getattr(n, f.name)
            if isinstance(c, list):
                for item in c:
                    r |= self._cc(item)
            else:
                r |= self._cc(c)
        return r

    def _build_proj(self, ast: SelectStmt) -> List[tuple]:
        proj: List[tuple] = []
        for expr in ast.select_list:
            name = self._out_name(expr)
            inner = expr.expr if isinstance(expr, AliasExpr) else expr
            if isinstance(inner, StarExpr):
                raise ExecutionError("internal: StarExpr not resolved")
            proj.append((name, inner))
        return proj

    def _out_name(self, expr: Any) -> str:
        if isinstance(expr, AliasExpr):
            return expr.alias
        if isinstance(expr, ColumnRef):
            return expr.column
        return Formatter.expr_to_sql(expr)

    def _col_name(self, expr: Any) -> str:
        if isinstance(expr, ColumnRef):
            return expr.column
        return Formatter.expr_to_sql(expr)

    def _eval_const(self, expr: Any) -> Optional[int]:
        if expr is None:
            return None
        dummy = VectorBatch.single_row_no_columns()
        vec = self._evaluator.evaluate(expr, dummy)
        val = vec.get(0)
        if val is None:
            return None
        val = int(val)
        if val < 0:
            raise ExecutionError("LIMIT/OFFSET must be non-negative")
        return val

    def _drain(self, op: Operator) -> ExecutionResult:
        schema = op.output_schema()
        cn = [n for n, _ in schema]
        ct = [t for _, t in schema]
        op.open()
        rows: list = []
        while True:
            b = op.next_batch()
            if b is None:
                break
            rows.extend(b.to_rows())
        op.close()
        return ExecutionResult(columns=cn, column_types=ct, rows=rows, row_count=len(rows))
