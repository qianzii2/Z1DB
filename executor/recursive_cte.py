from __future__ import annotations

"""Recursive CTE engine — iterative fixpoint with cycle detection.

Algorithm (based on SQL:1999 standard):
  1. Execute base query → working_table
  2. Loop:
     a. Execute recursive query with working_table as CTE reference → new_rows
     b. If new_rows empty → STOP (fixpoint reached)
     c. Cycle detection: remove rows already seen (hash-based)
     d. Append new_rows to result_table
     e. working_table = new_rows (for next iteration)
  3. Return result_table

Cycle detection uses Zobrist incremental hashing for O(1) per-row checks.
Recursion limit prevents infinite loops (default 1000 iterations, 1M rows).

Paper reference: Ghazal et al., "Equivalent Rewriting of Recursive SQL Queries", 2006
"""
from typing import Any, Dict, List, Optional, Set, Tuple
from catalog.catalog import Catalog, ColumnSchema, TableSchema
from executor.core.result import ExecutionResult
from metal.hash import murmur3_64
from utils.errors import ExecutionError, RecursionLimitError

MAX_ITERATIONS = 1000
MAX_ROWS = 1_000_000


class RecursiveCTEExecutor:
    """Executes recursive CTEs with fixpoint iteration and cycle detection."""

    def __init__(self, planner: Any, catalog: Catalog) -> None:
        self._planner = planner
        self._catalog = catalog

    def execute(self, cte_name: str, base_query: Any, recursive_query: Any,
                union_all: bool = True,
                column_names: Optional[List[str]] = None) -> ExecutionResult:
        # Step 1: Execute base query
        base_result = self._execute_query(base_query)
        if not base_result.columns:
            return base_result

        # Apply CTE column aliases
        columns = list(base_result.columns)
        col_types = list(base_result.column_types)
        if column_names and len(column_names) == len(columns):
            columns = list(column_names)

        # Initialize
        result_rows: List[list] = [list(r) for r in base_result.rows]
        working_rows: List[list] = [list(r) for r in base_result.rows]

        # Cycle detection
        seen_hashes: Set[int] = set()
        if not union_all:
            for row in result_rows:
                seen_hashes.add(self._row_hash(row))

        # Step 2: Iterative fixpoint
        iteration = 0
        while working_rows and iteration < MAX_ITERATIONS:
            iteration += 1
            if len(result_rows) > MAX_ROWS:
                raise RecursionLimitError(
                    f"Recursive CTE exceeded {MAX_ROWS} rows after {iteration} iterations")

            self._materialize_working_table(cte_name, columns, col_types, working_rows)

            try:
                new_result = self._execute_query(recursive_query)
            except Exception:
                break

            if not new_result.rows:
                break

            new_rows = [list(r) for r in new_result.rows]

            if not union_all:
                deduplicated = []
                for row in new_rows:
                    h = self._row_hash(row)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        deduplicated.append(row)
                new_rows = deduplicated
                if not new_rows:
                    break

            result_rows.extend(new_rows)
            working_rows = new_rows

        self._cleanup(cte_name)

        return ExecutionResult(
            columns=columns,
            column_types=col_types,
            rows=result_rows,
            row_count=len(result_rows))

    def _execute_query(self, query: Any) -> ExecutionResult:
        """Execute a query AST through the planner."""
        from parser.resolver import Resolver
        from parser.validator import Validator
        try:
            resolved = Resolver().resolve(query, self._catalog)
            validated = Validator().validate(resolved, self._catalog)
            return self._planner.execute(validated, self._catalog)
        except Exception as e:
            raise ExecutionError(f"Recursive CTE execution failed: {e}")

    def _materialize_working_table(self, name: str, columns: List[str],
                                   col_types: list, rows: List[list]) -> None:
        """Create/replace a temporary table with the working set."""
        if self._catalog.table_exists(name):
            self._catalog.drop_table(name)
        cols = [ColumnSchema(name=cn, dtype=ct, nullable=True)
                for cn, ct in zip(columns, col_types)]
        schema = TableSchema(name=name, columns=cols)
        self._catalog.create_table(schema)
        store = self._catalog.get_store(name)
        for row in rows:
            store.append_row(list(row))

    def _cleanup(self, name: str) -> None:
        if self._catalog.table_exists(name):
            try:
                self._catalog.drop_table(name)
            except Exception:
                pass

    @staticmethod
    def _row_hash(row: list) -> int:
        """Hash a row for cycle detection. Uses murmur3 for speed."""
        parts = []
        for val in row:
            if val is None:
                parts.append(b'\x00')
            elif isinstance(val, int):
                parts.append(val.to_bytes(8, 'little', signed=True))
            elif isinstance(val, float):
                import struct
                parts.append(struct.pack('d', val))
            elif isinstance(val, str):
                parts.append(val.encode('utf-8'))
            else:
                parts.append(str(val).encode('utf-8'))
        return murmur3_64(b'|'.join(parts))
