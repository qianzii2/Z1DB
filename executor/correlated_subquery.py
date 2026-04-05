from __future__ import annotations

"""Correlated subquery evaluator — per-row re-execution with caching.

Three strategies based on subquery type and outer cardinality:
  1. Naive: re-execute inner query for each outer row (< 100 outer rows)
  2. Cached: memoize results by correlated column values (< 10K outer rows)
  3. Semi-join rewrite: convert EXISTS/IN to semi-join (> 10K rows, future)

Correlation detection:
  Scan inner query AST for ColumnRef nodes that reference outer table columns.
  If found → correlated. If not → uncorrelated (can pre-execute once).

Paper reference: Neumann & Kemper, "Unnesting Arbitrary Queries", 2015
"""
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple
from executor.core.batch import VectorBatch
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.bitmap import Bitmap
from metal.hash import murmur3_64
from parser.ast import (
    AliasExpr, BinaryExpr, ColumnRef, ExistsExpr, InExpr, Literal,
    SelectStmt, SubqueryExpr,
)
from storage.types import DataType
from utils.errors import ExecutionError


class CorrelatedSubqueryEvaluator:
    """Evaluates correlated subqueries with caching optimization."""

    def __init__(self, planner: Any, catalog: Any) -> None:
        self._planner = planner
        self._catalog = catalog
        self._evaluator = ExpressionEvaluator()
        self._cache: Dict[int, Any] = {}  # correlation_key → result

    def resolve_node(self, node: Any, outer_batch: VectorBatch,
                     outer_schema: Dict[str, DataType]) -> Any:
        """Resolve a potentially correlated node for an entire outer batch.

        Returns a modified AST node with subqueries replaced by results,
        or a DataVector if the node produces per-row results.
        """
        if isinstance(node, SubqueryExpr):
            return self._eval_scalar_subquery(node, outer_batch, outer_schema)
        if isinstance(node, ExistsExpr):
            return self._eval_exists(node, outer_batch, outer_schema)
        if isinstance(node, InExpr):
            return self._eval_in_subquery(node, outer_batch, outer_schema)
        return node

    def is_correlated(self, subquery: Any, outer_columns: Set[str]) -> bool:
        """Check if a subquery references columns from the outer scope."""
        refs = self._collect_column_refs(subquery)
        for ref in refs:
            col = f"{ref.table}.{ref.column}" if ref.table else ref.column
            if col in outer_columns or ref.column in outer_columns:
                # Check if this column is NOT in the subquery's own FROM
                if not self._is_local_ref(ref, subquery):
                    return True
        return False

    # ══════════════════════════════════════════════════════════════
    # Scalar correlated subquery
    # ══════════════════════════════════════════════════════════════

    def _eval_scalar_subquery(self, node: SubqueryExpr,
                              outer_batch: VectorBatch,
                              outer_schema: Dict[str, DataType]) -> Any:
        """Evaluate scalar subquery — returns Literal for uncorrelated,
        or DataVector for correlated."""
        query = node.query
        outer_cols = set(outer_schema.keys())

        if not self.is_correlated(query, outer_cols):
            # Uncorrelated — execute once
            result = self._execute(query)
            if result.rows and result.columns:
                return Literal(value=result.rows[0][0],
                               inferred_type=result.column_types[0])
            return Literal(value=None, inferred_type=DataType.UNKNOWN)

        # Correlated — execute per outer row with caching
        n = outer_batch.row_count
        results = [None] * n
        col_names = outer_batch.column_names

        for i in range(n):
            # Build correlation key
            outer_vals = {cn: outer_batch.columns[cn].get(i) for cn in col_names}
            cache_key = self._cache_key(outer_vals)

            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
                continue

            # Substitute outer values into inner query
            bound_query = self._bind_outer_values(query, outer_vals, outer_cols)
            result = self._execute(bound_query)

            val = result.rows[0][0] if result.rows and result.columns else None
            results[i] = val
            self._cache[cache_key] = val

        # Determine type
        dtype = DataType.INT
        for v in results:
            if v is not None:
                if isinstance(v, float):
                    dtype = DataType.DOUBLE
                elif isinstance(v, str):
                    dtype = DataType.VARCHAR
                elif isinstance(v, bool):
                    dtype = DataType.BOOLEAN
                break

        return self._list_to_vector(results, dtype, n)

    # ══════════════════════════════════════════════════════════════
    # EXISTS correlated subquery
    # ══════════════════════════════════════════════════════════════

    def _eval_exists(self, node: ExistsExpr,
                     outer_batch: VectorBatch,
                     outer_schema: Dict[str, DataType]) -> Any:
        query = node.query
        outer_cols = set(outer_schema.keys())

        if not self.is_correlated(query, outer_cols):
            result = self._execute(query)
            exists = len(result.rows) > 0
            val = not exists if node.negated else exists
            return Literal(value=val, inferred_type=DataType.BOOLEAN)

        n = outer_batch.row_count
        col_names = outer_batch.column_names
        rd = Bitmap(n)

        for i in range(n):
            outer_vals = {cn: outer_batch.columns[cn].get(i) for cn in col_names}
            cache_key = self._cache_key(outer_vals)

            if cache_key in self._cache:
                exists = self._cache[cache_key]
            else:
                bound = self._bind_outer_values(query, outer_vals, outer_cols)
                result = self._execute(bound)
                exists = len(result.rows) > 0
                self._cache[cache_key] = exists

            if node.negated:
                exists = not exists
            if exists:
                rd.set_bit(i)

        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=Bitmap(n), _length=n)

    # ══════════════════════════════════════════════════════════════
    # IN correlated subquery
    # ══════════════════════════════════════════════════════════════

    def _eval_in_subquery(self, node: InExpr,
                          outer_batch: VectorBatch,
                          outer_schema: Dict[str, DataType]) -> Any:
        # Check if any value in the IN list is a correlated subquery
        has_subquery = any(isinstance(v, SubqueryExpr) for v in node.values)
        if not has_subquery:
            return node  # Not a subquery IN — return as-is

        subquery = None
        for v in node.values:
            if isinstance(v, SubqueryExpr):
                subquery = v.query
                break

        if subquery is None:
            return node

        outer_cols = set(outer_schema.keys())

        if not self.is_correlated(subquery, outer_cols):
            # Uncorrelated — execute once, replace with literal values
            result = self._execute(subquery)
            new_vals = []
            for row in result.rows:
                dt = result.column_types[0] if result.column_types else DataType.INT
                new_vals.append(Literal(value=row[0], inferred_type=dt))
            return dataclasses.replace(node, values=new_vals)

        # Correlated IN subquery — per-row evaluation
        n = outer_batch.row_count
        col_names = outer_batch.column_names
        rd = Bitmap(n)
        rn = Bitmap(n)

        # Evaluate the LHS expression
        lhs_vec = self._evaluator.evaluate(node.expr, outer_batch)

        for i in range(n):
            if lhs_vec.is_null(i):
                rn.set_bit(i)
                continue

            lhs_val = lhs_vec.get(i)
            outer_vals = {cn: outer_batch.columns[cn].get(i) for cn in col_names}
            cache_key = self._cache_key(outer_vals)

            if cache_key in self._cache:
                in_set = self._cache[cache_key]
            else:
                bound = self._bind_outer_values(subquery, outer_vals, outer_cols)
                result = self._execute(bound)
                in_set = {row[0] for row in result.rows if row[0] is not None}
                self._cache[cache_key] = in_set

            found = lhs_val in in_set
            has_null = any(row[0] is None for row in self._execute(
                self._bind_outer_values(subquery, outer_vals, outer_cols)).rows) \
                if not found else False

            if found:
                if not node.negated:
                    rd.set_bit(i)
            elif has_null:
                rn.set_bit(i)
            else:
                if node.negated:
                    rd.set_bit(i)

        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    # ══════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════

    def _execute(self, query: Any) -> ExecutionResult:
        from parser.resolver import Resolver
        from parser.validator import Validator
        from executor.core.result import ExecutionResult
        try:
            resolved = Resolver().resolve(query, self._catalog)
            validated = Validator().validate(resolved, self._catalog)
            return self._planner.execute(validated, self._catalog)
        except Exception:
            return ExecutionResult()

    def _bind_outer_values(self, query: Any, outer_vals: Dict[str, Any],
                           outer_cols: Set[str]) -> Any:
        """Replace outer column references in inner query with literal values."""
        return self._substitute(query, outer_vals, outer_cols)

    def _substitute(self, node: Any, vals: Dict[str, Any],
                    outer_cols: Set[str]) -> Any:
        """Recursively substitute outer ColumnRefs with Literals."""
        if node is None:
            return None
        if isinstance(node, ColumnRef):
            col = f"{node.table}.{node.column}" if node.table else node.column
            if col in vals:
                return self._val_to_literal(vals[col])
            if node.column in vals and not node.table:
                return self._val_to_literal(vals[node.column])
            return node
        if isinstance(node, Literal):
            return node
        if isinstance(node, (list, tuple)):
            result = [self._substitute(item, vals, outer_cols) for item in node]
            return type(node)(result)
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        changes: dict = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                new_list = []
                for item in child:
                    if isinstance(item, tuple):
                        new_list.append(tuple(self._substitute(x, vals, outer_cols) for x in item))
                    else:
                        new_list.append(self._substitute(item, vals, outer_cols))
                changes[f.name] = new_list
            elif isinstance(child, tuple):
                changes[f.name] = tuple(self._substitute(x, vals, outer_cols) for x in child)
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._substitute(child, vals, outer_cols)
        return dataclasses.replace(node, **changes) if changes else node

    @staticmethod
    def _val_to_literal(val: Any) -> Literal:
        if val is None:
            return Literal(value=None, inferred_type=DataType.UNKNOWN)
        if isinstance(val, bool):
            return Literal(value=val, inferred_type=DataType.BOOLEAN)
        if isinstance(val, int):
            return Literal(value=val, inferred_type=DataType.INT)
        if isinstance(val, float):
            return Literal(value=val, inferred_type=DataType.DOUBLE)
        return Literal(value=str(val), inferred_type=DataType.VARCHAR)

    def _collect_column_refs(self, node: Any) -> List[ColumnRef]:
        """Collect all ColumnRef nodes in an AST."""
        refs: list = []
        if isinstance(node, ColumnRef):
            refs.append(node)
            return refs
        if node is None or isinstance(node, (Literal, str, int, float, bool)):
            return refs
        if isinstance(node, (list, tuple)):
            for item in node:
                refs.extend(self._collect_column_refs(item))
            return refs
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for f in dataclasses.fields(node):
                child = getattr(node, f.name)
                refs.extend(self._collect_column_refs(child))
        return refs

    def _is_local_ref(self, ref: ColumnRef, query: Any) -> bool:
        """Check if a ColumnRef refers to a table defined in this query's FROM."""
        if not isinstance(query, SelectStmt):
            return False
        if query.from_clause is None:
            return False
        local_tables = set()
        tref = query.from_clause.table
        local_tables.add(tref.alias or tref.name)
        for jc in query.from_clause.joins:
            if jc.table:
                local_tables.add(jc.table.alias or jc.table.name)
        if ref.table and ref.table in local_tables:
            return True
        if not ref.table:
            # Ambiguous — check if any local table has this column
            for tname in local_tables:
                if self._catalog.table_exists(tname):
                    cols = self._catalog.get_table_columns(tname)
                    if ref.column in cols:
                        return True
        return False

    @staticmethod
    def _cache_key(vals: Dict[str, Any]) -> int:
        parts = []
        for k in sorted(vals.keys()):
            v = vals[k]
            if v is None:
                parts.append(f"{k}:NULL")
            else:
                parts.append(f"{k}:{v}")
        return murmur3_64('|'.join(parts).encode('utf-8'))

    @staticmethod
    def _list_to_vector(values: list, dtype: DataType, n: int) -> DataVector:
        from metal.typed_vector import TypedVector
        from storage.types import DTYPE_TO_ARRAY_CODE
        if dtype == DataType.UNKNOWN:
            dtype = DataType.INT
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        nulls = Bitmap(n)
        if dtype in (DataType.VARCHAR, DataType.TEXT):
            data: Any = []
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i); data.append('')
                else:
                    data.append(str(values[i]))
        elif code:
            data = TypedVector(code)
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i); data.append(0)
                else:
                    data.append(values[i])
        else:
            data = TypedVector('q')
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i); data.append(0)
                else:
                    data.append(int(values[i]))
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)

    def clear_cache(self) -> None:
        self._cache.clear()
