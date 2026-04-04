from __future__ import annotations
"""Simple planner — builds and executes operator trees for Phase 1."""

import dataclasses
from typing import Any, Dict, List, Optional, Set

from catalog.catalog import Catalog, ColumnSchema, TableSchema
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.result import ExecutionResult
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.functions.registry import FunctionRegistry
from executor.operators.filter import FilterOperator
from executor.operators.limit import LimitOperator
from executor.operators.project import ProjectOperator
from executor.operators.scan.seq_scan import SeqScan
from executor.operators.scan.values_scan import DualScan
from executor.operators.sort.in_memory_sort import SortOperator
from parser.ast import (
    AggregateCall, AliasExpr, BinaryExpr, ColumnRef, CreateTableStmt,
    DropTableStmt, InsertStmt, IsNullExpr, Literal, SelectStmt,
    SortKey, StarExpr, UnaryExpr,
)
from parser.formatter import Formatter
from storage.types import DataType, resolve_type_name, is_numeric
from utils.errors import (
    ColumnNotFoundError, DivisionByZeroError, ExecutionError,
    NumericOverflowError,
)


class SimplePlanner:
    """One-shot planner: parse → plan → execute."""

    def __init__(self, function_registry: FunctionRegistry) -> None:
        self._registry = function_registry
        self._evaluator = ExpressionEvaluator(function_registry)

    # ==================================================================
    # Public entry
    # ==================================================================
    def execute(self, ast: Any, catalog: Catalog) -> ExecutionResult:
        if isinstance(ast, CreateTableStmt):
            return self._execute_create(ast, catalog)
        if isinstance(ast, DropTableStmt):
            return self._execute_drop(ast, catalog)
        if isinstance(ast, InsertStmt):
            return self._execute_insert(ast, catalog)
        if isinstance(ast, SelectStmt):
            self._check_unsupported(ast)
            if self._is_scalar_aggregate(ast):
                return self._execute_scalar_aggregate(ast, catalog)
            op = self.plan(ast, catalog)
            return self._drain_operator(op)
        raise ExecutionError(f"unsupported statement: {type(ast).__name__}")

    # ==================================================================
    # Unsupported feature gate
    # ==================================================================
    def _check_unsupported(self, ast: SelectStmt) -> None:
        if ast.distinct:
            raise ExecutionError("DISTINCT is not yet supported")
        if ast.from_clause and ast.from_clause.joins:
            raise ExecutionError("JOIN is not yet supported")
        if ast.group_by:
            raise ExecutionError("GROUP BY is not yet supported")
        if ast.having:
            raise ExecutionError("HAVING is not yet supported")
        if ast.ctes:
            raise ExecutionError("CTE is not yet supported")

    # ==================================================================
    # DDL
    # ==================================================================
    def _execute_create(self, ast: CreateTableStmt, catalog: Catalog) -> ExecutionResult:
        columns: List[ColumnSchema] = []
        for cd in ast.columns:
            dtype, max_len = resolve_type_name(cd.type_name.name, cd.type_name.params)
            columns.append(ColumnSchema(
                name=cd.name, dtype=dtype,
                nullable=cd.nullable, primary_key=cd.primary_key,
                max_length=max_len,
            ))
        schema = TableSchema(name=ast.table, columns=columns)
        catalog.create_table(schema, ast.if_not_exists)
        return ExecutionResult(message='OK')

    def _execute_drop(self, ast: DropTableStmt, catalog: Catalog) -> ExecutionResult:
        catalog.drop_table(ast.table, ast.if_exists)
        return ExecutionResult(message='OK')

    # ==================================================================
    # INSERT
    # ==================================================================
    def _execute_insert(self, ast: InsertStmt, catalog: Catalog) -> ExecutionResult:
        schema = catalog.get_table(ast.table)
        store = catalog.get_store(ast.table)

        # Column list validation
        if ast.columns:
            seen: set = set()
            schema_cols = {c.name for c in schema.columns}
            for c in ast.columns:
                if c in seen:
                    raise ExecutionError(f"duplicate column in INSERT: '{c}'")
                if c not in schema_cols:
                    raise ColumnNotFoundError(c)
                seen.add(c)

        dummy = VectorBatch.single_row_no_columns()
        count = 0

        for value_exprs in ast.values:
            py_vals = [self._evaluator.evaluate(expr, dummy).get(0) for expr in value_exprs]

            if ast.columns:
                full_row = self._reorder_to_schema(py_vals, ast.columns, schema)
            else:
                if len(py_vals) != len(schema.columns):
                    raise ExecutionError(
                        f"expected {len(schema.columns)} values, got {len(py_vals)}")
                full_row = py_vals

            validated = self._validate_and_cast_row(full_row, schema)
            store.append_row(validated)
            count += 1

        word = 'row' if count == 1 else 'rows'
        return ExecutionResult(affected_rows=count, message=f"Inserted {count} {word}")

    def _reorder_to_schema(self, values: list, col_names: List[str],
                           schema: TableSchema) -> list:
        col_map = dict(zip(col_names, values))
        result = []
        for col in schema.columns:
            if col.name in col_map:
                result.append(col_map[col.name])
            elif col.nullable:
                result.append(None)
            else:
                raise ExecutionError(f"column '{col.name}' is NOT NULL and has no default")
        return result

    def _validate_and_cast_row(self, row: list, schema: TableSchema) -> list:
        if len(row) != len(schema.columns):
            raise ExecutionError(
                f"column count mismatch: expected {len(schema.columns)}, got {len(row)}")
        result = []
        for val, col in zip(row, schema.columns):
            if val is None:
                if not col.nullable:
                    raise ExecutionError(f"NULL value for NOT NULL column '{col.name}'")
                result.append(None)
                continue
            dt = col.dtype
            try:
                if dt == DataType.INT:
                    v = int(val)
                    if not (-2_147_483_648 <= v <= 2_147_483_647):
                        raise NumericOverflowError(f"integer overflow for column '{col.name}': {v}")
                    result.append(v)
                elif dt == DataType.BIGINT:
                    result.append(int(val))
                elif dt in (DataType.FLOAT, DataType.DOUBLE):
                    result.append(float(val))
                elif dt == DataType.BOOLEAN:
                    if isinstance(val, bool):
                        result.append(val)
                    else:
                        result.append(bool(val))
                elif dt in (DataType.VARCHAR, DataType.TEXT):
                    s = str(val)
                    if col.max_length and len(s) > col.max_length:
                        s = s[:col.max_length]
                    result.append(s)
                elif dt == DataType.DATE:
                    result.append(int(val))
                elif dt == DataType.TIMESTAMP:
                    result.append(int(val))
                else:
                    result.append(val)
            except (ValueError, TypeError) as e:
                raise ExecutionError(
                    f"cannot convert value {val!r} to {dt.name} for column '{col.name}': {e}")
        return result

    # ==================================================================
    # Scalar aggregate
    # ==================================================================
    def _is_scalar_aggregate(self, ast: SelectStmt) -> bool:
        return (not ast.group_by and
                any(self._contains_aggregate(e) for e in ast.select_list))

    def _contains_aggregate(self, node: Any) -> bool:
        if node is None:
            return False
        if isinstance(node, AggregateCall):
            return True
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return False
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                if any(self._contains_aggregate(c) for c in child):
                    return True
            elif self._contains_aggregate(child):
                return True
        return False

    def _execute_scalar_aggregate(self, ast: SelectStmt,
                                  catalog: Catalog) -> ExecutionResult:
        # Phase A: extract AggregateCall → temp column refs
        agg_map: Dict[str, AggregateCall] = {}
        substituted = [self._extract_and_substitute(e, agg_map) for e in ast.select_list]

        # Build scan + filter
        op = self._build_scan_filter(ast, catalog)
        scan_schema = dict(op.output_schema())

        # Phase B: init states
        agg_states: Dict[str, tuple] = {}
        for temp, ac in agg_map.items():
            func = self._registry.get_aggregate(ac.name)
            agg_states[temp] = (func, func.init())

        # Phase C: consume data
        op.open()
        while True:
            batch = op.next_batch()
            if batch is None:
                break
            for temp, ac in agg_map.items():
                func, state = agg_states[temp]
                if len(ac.args) == 1 and isinstance(ac.args[0], StarExpr):
                    state = func.update(state, None, batch.row_count)
                else:
                    arg_vec = self._evaluator.evaluate(ac.args[0], batch)
                    state = func.update(state, arg_vec, batch.row_count)
                agg_states[temp] = (func, state)
        op.close()

        # Phase D: finalize → single-row batch
        single_data: Dict[str, DataVector] = {}
        for temp, (func, state) in agg_states.items():
            val = func.finalize(state)
            ac = agg_map[temp]
            if len(ac.args) == 1 and isinstance(ac.args[0], StarExpr):
                input_types: List[DataType] = []
            else:
                input_types = [ExpressionEvaluator.infer_type(ac.args[0], scan_schema)]
            ret_type = func.return_type(input_types)
            single_data[temp] = DataVector.from_scalar(val, ret_type)
        single_batch = VectorBatch(columns=single_data, _row_count=1)

        # Phase E: evaluate substituted select list
        result_columns: Dict[str, DataVector] = {}
        ordered_names: List[str] = []
        for orig, subst in zip(ast.select_list, substituted):
            name = self._output_column_name(orig)
            vec = self._evaluator.evaluate(subst, single_batch)
            result_columns[name] = vec
            ordered_names.append(name)

        final = VectorBatch(columns=result_columns, _column_order=ordered_names, _row_count=1)
        return ExecutionResult.from_batch(final)

    def _extract_and_substitute(self, expr: Any, agg_map: Dict[str, AggregateCall]) -> Any:
        if isinstance(expr, AggregateCall):
            temp = f'__agg_{len(agg_map)}'
            agg_map[temp] = expr
            return ColumnRef(table=None, column=temp)
        if isinstance(expr, AliasExpr):
            return AliasExpr(
                expr=self._extract_and_substitute(expr.expr, agg_map),
                alias=expr.alias,
            )
        return self._recursive_substitute(expr, agg_map)

    def _recursive_substitute(self, node: Any, agg_map: Dict[str, AggregateCall]) -> Any:
        if node is None:
            return None
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        if isinstance(node, AggregateCall):
            return self._extract_and_substitute(node, agg_map)
        changes: dict = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, AggregateCall):
                changes[f.name] = self._extract_and_substitute(child, agg_map)
            elif isinstance(child, list):
                new_list = []
                for item in child:
                    if isinstance(item, AggregateCall) or (
                            dataclasses.is_dataclass(item) and not isinstance(item, type)):
                        new_list.append(self._extract_and_substitute(item, agg_map))
                    else:
                        new_list.append(item)
                changes[f.name] = new_list
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._recursive_substitute(child, agg_map)
        return dataclasses.replace(node, **changes) if changes else node

    def _build_scan_filter(self, ast: SelectStmt, catalog: Catalog) -> Operator:
        if ast.from_clause is None:
            return DualScan()
        table = ast.from_clause.table.name
        store = catalog.get_store(table)
        needed = self._collect_all_columns_set(ast)
        ordered = [c.name for c in store.schema.columns if c.name in needed]
        if not ordered:
            # Need at least something for row counting
            ordered = [store.schema.columns[0].name]
        scan: Operator = SeqScan(table, store, ordered)
        if ast.where:
            scan = FilterOperator(scan, ast.where)
        return scan

    # ==================================================================
    # Normal SELECT plan
    # ==================================================================
    def plan(self, ast: SelectStmt, catalog: Catalog) -> Operator:
        all_cols = self._collect_all_columns_set(ast)

        if ast.from_clause is None:
            scan: Operator = DualScan()
        else:
            table = ast.from_clause.table.name
            store = catalog.get_store(table)
            ordered = [c.name for c in store.schema.columns if c.name in all_cols]
            if not ordered and store.schema.columns:
                ordered = [store.schema.columns[0].name]
            scan = SeqScan(table, store, ordered)

        current: Operator = scan

        if ast.where:
            current = FilterOperator(current, ast.where)

        if ast.order_by:
            keys = [(sk.expr, sk.direction, sk.nulls) for sk in ast.order_by]
            current = SortOperator(current, keys)

        projections = self._build_projections(ast)
        current = ProjectOperator(current, projections)

        if ast.limit is not None or ast.offset is not None:
            limit_val = self._eval_const(ast.limit)
            offset_val = self._eval_const(ast.offset) or 0
            current = LimitOperator(current, limit_val, offset_val)

        return current

    # ==================================================================
    # Column collection
    # ==================================================================
    def _collect_all_columns_set(self, ast: SelectStmt) -> Set[str]:
        result: Set[str] = set()
        for e in ast.select_list:
            result |= self._collect_columns(e)
        if ast.where:
            result |= self._collect_columns(ast.where)
        for sk in (ast.order_by or []):
            result |= self._collect_columns(sk.expr)
        return result

    def _collect_columns(self, node: Any) -> Set[str]:
        if node is None:
            return set()
        if isinstance(node, ColumnRef):
            return {node.column}
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return set()
        result: Set[str] = set()
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for item in child:
                    result |= self._collect_columns(item)
            else:
                result |= self._collect_columns(child)
        return result

    # ==================================================================
    # Projections
    # ==================================================================
    def _build_projections(self, ast: SelectStmt) -> List[tuple]:
        projections: List[tuple] = []
        for expr in ast.select_list:
            name = self._output_column_name(expr)
            inner = expr.expr if isinstance(expr, AliasExpr) else expr
            if isinstance(inner, StarExpr):
                raise ExecutionError("internal: StarExpr not resolved")
            projections.append((name, inner))
        return projections

    def _output_column_name(self, expr: Any) -> str:
        if isinstance(expr, AliasExpr):
            return expr.alias
        if isinstance(expr, ColumnRef):
            return expr.column
        return Formatter.expr_to_sql(expr)

    # ==================================================================
    # Helpers
    # ==================================================================
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

    def _drain_operator(self, op: Operator) -> ExecutionResult:
        schema = op.output_schema()
        col_names = [n for n, _ in schema]
        col_types = [t for _, t in schema]
        op.open()
        rows: list = []
        while True:
            batch = op.next_batch()
            if batch is None:
                break
            rows.extend(batch.to_rows())
        op.close()
        return ExecutionResult(
            columns=col_names, column_types=col_types,
            rows=rows, row_count=len(rows),
        )
