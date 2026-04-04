from __future__ import annotations
"""Simple planner — Phase 2: joins, grouping, distinct, DML."""
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
from executor.operators.sort.in_memory_sort import SortOperator
from parser.ast import (
    AggregateCall, AliasExpr, Assignment, BinaryExpr, CaseExpr, CastExpr,
    ColumnRef, CreateTableStmt, DeleteStmt, DropTableStmt, FromClause,
    InsertStmt, IsNullExpr, JoinClause, Literal, SelectStmt, SortKey,
    StarExpr, UnaryExpr, UpdateStmt,
)
from parser.formatter import Formatter
from storage.types import DataType, resolve_type_name, is_numeric
from utils.errors import (ColumnNotFoundError, ExecutionError, NumericOverflowError)


class SimplePlanner:
    def __init__(self, function_registry: FunctionRegistry) -> None:
        self._registry = function_registry
        self._evaluator = ExpressionEvaluator(function_registry)

    # ==================================================================
    def execute(self, ast: Any, catalog: Catalog) -> ExecutionResult:
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

    # ==================================================================
    # DDL
    # ==================================================================
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

    # ==================================================================
    # INSERT
    # ==================================================================
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
            raise ExecutionError(f"column count mismatch")
        result = []
        for val, col in zip(row, schema.columns):
            if val is None:
                if not col.nullable:
                    raise ExecutionError(f"NULL for NOT NULL column '{col.name}'")
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
                    result.append(bool(val) if not isinstance(val, bool) else val)
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

    # ==================================================================
    # UPDATE
    # ==================================================================
    def _exec_update(self, ast: UpdateStmt, catalog: Catalog) -> ExecutionResult:
        schema = catalog.get_table(ast.table)
        store = catalog.get_store(ast.table)
        all_rows = store.read_all_rows()
        col_names = schema.column_names
        col_types = [c.dtype for c in schema.columns]
        count = 0
        for ri, row in enumerate(all_rows):
            if ast.where:
                batch = VectorBatch.from_rows([row], col_names, col_types)
                mask = self._evaluator.evaluate_predicate(ast.where, batch)
                if not mask.get_bit(0):
                    continue
            for assign in ast.assignments:
                ci = None
                for idx, c in enumerate(schema.columns):
                    if c.name == assign.column:
                        ci = idx
                        break
                if ci is None:
                    raise ColumnNotFoundError(assign.column)
                batch = VectorBatch.from_rows([row], col_names, col_types)
                val = self._evaluator.evaluate(assign.value, batch).get(0)
                all_rows[ri][ci] = val
            count += 1
        store.truncate()
        for r in all_rows:
            store.append_row(r)
        w = 'row' if count == 1 else 'rows'
        return ExecutionResult(affected_rows=count, message=f"Updated {count} {w}")

    # ==================================================================
    # DELETE
    # ==================================================================
    def _exec_delete(self, ast: DeleteStmt, catalog: Catalog) -> ExecutionResult:
        schema = catalog.get_table(ast.table)
        store = catalog.get_store(ast.table)
        if ast.where is None:
            count = store.row_count
            store.truncate()
        else:
            all_rows = store.read_all_rows()
            col_names = schema.column_names
            col_types = [c.dtype for c in schema.columns]
            to_keep = []
            count = 0
            for row in all_rows:
                batch = VectorBatch.from_rows([row], col_names, col_types)
                mask = self._evaluator.evaluate_predicate(ast.where, batch)
                if mask.get_bit(0):
                    count += 1
                else:
                    to_keep.append(row)
            store.truncate()
            for r in to_keep:
                store.append_row(r)
        w = 'row' if count == 1 else 'rows'
        return ExecutionResult(affected_rows=count, message=f"Deleted {count} {w}")

    # ==================================================================
    # SELECT
    # ==================================================================
    def _exec_select(self, ast: SelectStmt, catalog: Catalog) -> ExecutionResult:
        has_agg = any(self._contains_agg(e) for e in ast.select_list)
        has_group = ast.group_by is not None
        if has_agg or has_group:
            return self._exec_grouped(ast, catalog)
        op = self._plan_select(ast, catalog)
        return self._drain(op)

    def _plan_select(self, ast: SelectStmt, catalog: Catalog) -> Operator:
        current = self._build_source(ast, catalog)
        if ast.where:
            current = FilterOperator(current, ast.where)
        if ast.order_by:
            keys = [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by]
            current = SortOperator(current, keys)
        projections = self._build_projections(ast)
        current = ProjectOperator(current, projections)
        if ast.distinct:
            current = DistinctOperator(current)
        if ast.limit is not None or ast.offset is not None:
            lv = self._eval_const(ast.limit)
            ov = self._eval_const(ast.offset) or 0
            current = LimitOperator(current, lv, ov)
        return current

    # ==================================================================
    # Source (FROM + JOINs)
    # ==================================================================
    def _build_source(self, ast: SelectStmt, catalog: Catalog) -> Operator:
        if ast.from_clause is None:
            return DualScan()

        fc = ast.from_clause
        tref = fc.table
        has_join = bool(fc.joins)

        if has_join:
            left_qual = tref.alias or tref.name
            store = catalog.get_store(tref.name)
            left_cols = [c.name for c in store.schema.columns]
            left_scan: Operator = SeqScan(tref.name, store, left_cols)
            # Rename to qualified: left_qual.col
            left_proj = [(f"{left_qual}.{c}", ColumnRef(table=None, column=c))
                         for c in left_cols]
            current: Operator = ProjectOperator(left_scan, left_proj)

            for jc in fc.joins:
                assert jc.table is not None
                right_qual = jc.table.alias or jc.table.name
                r_store = catalog.get_store(jc.table.name)
                r_cols = [c.name for c in r_store.schema.columns]
                right_scan: Operator = SeqScan(jc.table.name, r_store, r_cols)
                right_proj = [(f"{right_qual}.{c}", ColumnRef(table=None, column=c))
                              for c in r_cols]
                right_op: Operator = ProjectOperator(right_scan, right_proj)

                if jc.join_type == 'CROSS':
                    current = CrossJoinOperator(current, right_op)
                else:
                    current = HashJoinOperator(current, right_op, jc.join_type, jc.on)
            return current
        else:
            store = catalog.get_store(tref.name)
            needed = self._collect_all_cols(ast)
            ordered = [c.name for c in store.schema.columns if c.name in needed]
            if not ordered:
                ordered = [store.schema.columns[0].name]
            return SeqScan(tref.name, store, ordered)

    # ==================================================================
    # GROUP BY / Aggregate
    # ==================================================================
    def _exec_grouped(self, ast: SelectStmt, catalog: Catalog) -> ExecutionResult:
        source = self._build_source(ast, catalog)
        if ast.where:
            source = FilterOperator(source, ast.where)

        scan_schema = dict(source.output_schema())

        # Group key expressions
        group_exprs: List[Tuple[str, Any]] = []
        if ast.group_by:
            for key in ast.group_by.keys:
                name = self._col_name(key)
                group_exprs.append((name, key))

        # Extract ALL aggregates from select_list AND having
        agg_map: Dict[str, AggregateCall] = {}
        substituted = [self._extract_aggs(e, agg_map) for e in ast.select_list]
        having_subst = self._extract_aggs(ast.having, agg_map) if ast.having else None

        agg_exprs: List[Tuple[str, AggregateCall]] = []
        for temp, ac in agg_map.items():
            agg_exprs.append((temp, ac))

        if not group_exprs and not agg_exprs:
            op = self._plan_select(ast, catalog)
            return self._drain(op)

        # Scalar aggregate (no GROUP BY)
        if not group_exprs:
            return self._exec_scalar_agg(ast, source, scan_schema, agg_map, substituted)

        # Hash aggregation
        agg_op: Operator = HashAggOperator(source, group_exprs, agg_exprs, self._registry)
        current: Operator = agg_op

        # HAVING — use substituted version (aggregates replaced with column refs)
        if having_subst is not None:
            current = FilterOperator(current, having_subst)

        if ast.order_by:
            keys = [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by]
            current = SortOperator(current, keys)

        # Final projection
        final_proj: List[Tuple[str, Any]] = []
        for orig, subst in zip(ast.select_list, substituted):
            name = self._output_name(orig)
            inner = subst.expr if isinstance(subst, AliasExpr) else subst
            final_proj.append((name, inner))
        current = ProjectOperator(current, final_proj)

        if ast.distinct:
            current = DistinctOperator(current)
        if ast.limit is not None or ast.offset is not None:
            lv = self._eval_const(ast.limit)
            ov = self._eval_const(ast.offset) or 0
            current = LimitOperator(current, lv, ov)
        return self._drain(current)

    def _exec_scalar_agg(self, ast: SelectStmt, source: Operator,
                         scan_schema: Dict[str, DataType],
                         agg_map: Dict[str, AggregateCall],
                         substituted: list) -> ExecutionResult:
        agg_states: Dict[str, tuple] = {}
        for temp, ac in agg_map.items():
            func = self._registry.get_aggregate(ac.name)
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
                    av = self._evaluator.evaluate(ac.args[0], batch)
                    state = func.update(state, av, batch.row_count)
                agg_states[temp] = (func, state)
        source.close()

        single_data: Dict[str, DataVector] = {}
        for temp, (func, state) in agg_states.items():
            val = func.finalize(state)
            ac = agg_map[temp]
            if ac.args and isinstance(ac.args[0], StarExpr):
                it: list = []
            else:
                it = [ExpressionEvaluator.infer_type(ac.args[0], scan_schema)]
            rt = func.return_type(it)
            single_data[temp] = DataVector.from_scalar(val, rt)
        single_batch = VectorBatch(columns=single_data, _row_count=1)

        result_cols: Dict[str, DataVector] = {}
        ordered: List[str] = []
        for orig, subst in zip(ast.select_list, substituted):
            name = self._output_name(orig)
            vec = self._evaluator.evaluate(subst, single_batch)
            result_cols[name] = vec
            ordered.append(name)

        final = VectorBatch(columns=result_cols, _column_order=ordered, _row_count=1)
        return ExecutionResult.from_batch(final)

    # ==================================================================
    # Helpers
    # ==================================================================
    def _extract_aggs(self, expr: Any, agg_map: Dict[str, AggregateCall]) -> Any:
        if expr is None:
            return None
        if isinstance(expr, AggregateCall):
            temp = f'__agg_{len(agg_map)}'
            agg_map[temp] = expr
            return ColumnRef(table=None, column=temp)
        if isinstance(expr, AliasExpr):
            return AliasExpr(expr=self._extract_aggs(expr.expr, agg_map), alias=expr.alias)
        return self._recursive_extract(expr, agg_map)

    def _recursive_extract(self, node: Any, agg_map: Dict) -> Any:
        if node is None:
            return None
        if isinstance(node, AggregateCall):
            return self._extract_aggs(node, agg_map)
        if isinstance(node, tuple):
            return tuple(self._recursive_extract(item, agg_map) for item in node)
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        changes: dict = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, AggregateCall):
                changes[f.name] = self._extract_aggs(child, agg_map)
            elif isinstance(child, list):
                new_list = []
                for item in child:
                    if isinstance(item, tuple):
                        new_list.append(tuple(self._recursive_extract(x, agg_map) for x in item))
                    elif isinstance(item, AggregateCall):
                        new_list.append(self._extract_aggs(item, agg_map))
                    elif dataclasses.is_dataclass(item) and not isinstance(item, type):
                        new_list.append(self._recursive_extract(item, agg_map))
                    else:
                        new_list.append(item)
                changes[f.name] = new_list
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._recursive_extract(child, agg_map)
        return dataclasses.replace(node, **changes) if changes else node

    def _contains_agg(self, node: Any) -> bool:
        if isinstance(node, AggregateCall):
            return True
        if isinstance(node, AliasExpr):
            return self._contains_agg(node.expr)
        if isinstance(node, tuple):
            return any(self._contains_agg(item) for item in node)
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return False
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                if any(self._contains_agg(c) for c in child):
                    return True
            elif self._contains_agg(child):
                return True
        return False

    def _collect_all_cols(self, ast: SelectStmt) -> Set[str]:
        result: Set[str] = set()
        for e in ast.select_list:
            result |= self._collect_cols(e)
        if ast.where:
            result |= self._collect_cols(ast.where)
        for sk in (ast.order_by or []):
            result |= self._collect_cols(sk.expr)
        if ast.group_by:
            for k in ast.group_by.keys:
                result |= self._collect_cols(k)
        if ast.having:
            result |= self._collect_cols(ast.having)
        return result

    def _collect_cols(self, node: Any) -> Set[str]:
        if node is None:
            return set()
        if isinstance(node, ColumnRef):
            result = {node.column}
            if node.table:
                result.add(f"{node.table}.{node.column}")
            return result
        if isinstance(node, tuple):
            result: Set[str] = set()
            for item in node:
                result |= self._collect_cols(item)
            return result
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return set()
        result = set()
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for item in child:
                    result |= self._collect_cols(item)
            else:
                result |= self._collect_cols(child)
        return result

    def _build_projections(self, ast: SelectStmt) -> List[tuple]:
        proj: List[tuple] = []
        for expr in ast.select_list:
            name = self._output_name(expr)
            inner = expr.expr if isinstance(expr, AliasExpr) else expr
            if isinstance(inner, StarExpr):
                raise ExecutionError("internal: StarExpr not resolved")
            proj.append((name, inner))
        return proj

    def _output_name(self, expr: Any) -> str:
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
