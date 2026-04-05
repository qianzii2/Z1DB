from __future__ import annotations
"""Z1DB engine — top-level façade with all optimizations."""
from typing import Optional
import dataclasses

from catalog.catalog import Catalog, ColumnSchema, TableSchema
from catalog.statistics import TableStatistics
from executor.core.result import ExecutionResult
from executor.functions.registry import FunctionRegistry
from executor.memory_budget import MemoryBudget
from executor.optimizer import QueryOptimizer
from executor.result_cache import ResultCache
from executor.simple_planner import SimplePlanner
from metal.hash import murmur3_64
from parser.ast import SelectStmt, ExplainStmt, SetOperationStmt
from parser.lexer import Lexer
from parser.parser import Parser
from parser.resolver import Resolver
from parser.validator import Validator
from storage.types import DataType
from utils.timer import Timer

_CTE_PREFIX = '__cte_'


class Engine:
    def __init__(self, data_dir: str = ':memory:') -> None:
        self._catalog = Catalog(data_dir)
        self._registry = FunctionRegistry()
        self._registry.register_defaults()
        self._planner = SimplePlanner(self._registry)
        self._optimizer = QueryOptimizer()
        self._stats: dict[str, TableStatistics] = {}
        self._suppress_persist = False
        self._budget = MemoryBudget()
        self._result_cache = ResultCache()
        self._table_versions: dict[str, int] = {}
        try:
            from executor.integrated_planner import IntegratedPlanner
            self._integrated = IntegratedPlanner(self._registry)
        except ImportError:
            self._integrated = None

    def execute(self, sql: str) -> ExecutionResult:
        # ── 改进 6: 查询结果缓存 ──
        sql_hash = ResultCache.hash_sql(sql)
        cached = self._result_cache.get(sql_hash, self._table_versions)
        if cached is not None:
            return cached

        with Timer() as t:
            tokens = Lexer(sql).tokenize()
            ast = Parser(tokens).parse()
            cte_tables = self._materialize_ctes(ast)
            try:
                if cte_tables:
                    ast = self._strip_ctes(ast)
                ast = Resolver().resolve(ast, self._catalog)
                ast = Validator().validate(ast, self._catalog)
                ast = self._optimize_ast(ast)
                if self._integrated and isinstance(ast, SelectStmt):
                    result = self._integrated.execute(ast, self._catalog)
                else:
                    result = self._planner.execute(ast, self._catalog)
            finally:
                self._suppress_persist = True
                for tname in cte_tables:
                    try:
                        self._catalog.drop_table(tname)
                    except Exception:
                        pass
                self._suppress_persist = False

        result.timing = t.elapsed

        # Persist + invalidate cache for mutations
        is_mutation = result.affected_rows > 0 or result.message == 'OK'
        if self._catalog.is_persistent and not self._suppress_persist and is_mutation:
            self._catalog.persist()

        # Invalidate cache for DML/DDL
        if is_mutation:
            self._invalidate_modified_tables(ast)
        elif isinstance(ast, SelectStmt):
            # Cache SELECT results
            tables = self._extract_tables(ast)
            versions = {t: self._table_versions.get(t, 0) for t in tables}
            self._result_cache.put(sql_hash, result, versions)

        return result

    def _invalidate_modified_tables(self, ast: object) -> None:
        """Bump version and invalidate cache for modified tables."""
        from parser.ast import (InsertStmt, UpdateStmt, DeleteStmt,
                                CreateTableStmt, DropTableStmt, AlterTableStmt)
        table = None
        if isinstance(ast, InsertStmt): table = ast.table
        elif isinstance(ast, UpdateStmt): table = ast.table
        elif isinstance(ast, DeleteStmt): table = ast.table
        elif isinstance(ast, CreateTableStmt): table = ast.table
        elif isinstance(ast, DropTableStmt): table = ast.table
        elif isinstance(ast, AlterTableStmt): table = ast.table
        if table:
            self._table_versions[table] = self._table_versions.get(table, 0) + 1
            self._result_cache.invalidate_table(table)

    def _extract_tables(self, ast: SelectStmt) -> set:
        """Extract table names referenced in a SELECT."""
        tables = set()
        if ast.from_clause and ast.from_clause.table:
            tables.add(ast.from_clause.table.name)
            for jc in ast.from_clause.joins:
                if jc.table:
                    tables.add(jc.table.name)
        return tables

    def _materialize_ctes(self, ast: object) -> list[str]:
        created: list[str] = []
        if isinstance(ast, SelectStmt) and ast.ctes:
            for cte_entry in ast.ctes:
                # Support 2-tuple, 3-tuple, and 4-tuple formats
                cte_name = cte_entry[0]
                cte_query = cte_entry[1]
                is_recursive = cte_entry[2] if len(cte_entry) > 2 else False
                cte_columns = cte_entry[3] if len(cte_entry) > 3 else None

                nested = self._materialize_ctes(cte_query)
                created.extend(nested)

                if is_recursive:
                    cte_result = self._execute_recursive_cte(cte_name, cte_query, cte_columns)
                else:
                    try:
                        resolved = Resolver().resolve(cte_query, self._catalog)
                        validated = Validator().validate(resolved, self._catalog)
                        optimized = self._optimize_ast(validated)
                        cte_result = self._planner.execute(optimized, self._catalog)
                    except Exception:
                        continue

                if cte_result.columns:
                    # Apply column aliases if provided
                    col_names = list(cte_result.columns)
                    if cte_columns and len(cte_columns) == len(col_names):
                        col_names = list(cte_columns)

                    cols = [ColumnSchema(name=cn, dtype=ct, nullable=True)
                            for cn, ct in zip(col_names, cte_result.column_types)]
                    internal_name = f'{_CTE_PREFIX}{cte_name}'
                    self._suppress_persist = True
                    try:
                        for name in (internal_name, cte_name):
                            if not self._catalog.table_exists(name):
                                schema = TableSchema(name=name, columns=cols)
                                self._catalog.create_table(schema)
                                store = self._catalog.get_store(name)
                                for row in cte_result.rows:
                                    store.append_row(list(row))
                                created.append(name)
                    finally:
                        self._suppress_persist = False
        elif isinstance(ast, ExplainStmt):
            created.extend(self._materialize_ctes(ast.statement))
        elif isinstance(ast, SetOperationStmt):
            created.extend(self._materialize_ctes(ast.left))
            created.extend(self._materialize_ctes(ast.right))
        return created

    def _execute_recursive_cte(self, cte_name: str, query: Any,
                               column_names: Optional[list] = None) -> ExecutionResult:
        from executor.recursive_cte import RecursiveCTEExecutor
        from executor.core.result import ExecutionResult

        if isinstance(query, SetOperationStmt) and query.op.upper() == 'UNION':
            base_query = query.left
            recursive_query = query.right
            union_all = query.all
            executor = RecursiveCTEExecutor(self._planner, self._catalog)
            return executor.execute(cte_name, base_query, recursive_query,
                                    union_all, column_names)

        try:
            resolved = Resolver().resolve(query, self._catalog)
            validated = Validator().validate(resolved, self._catalog)
            optimized = self._optimize_ast(validated)
            return self._planner.execute(optimized, self._catalog)
        except Exception:
            return ExecutionResult()

    def _strip_ctes(self, ast: object) -> object:
        if isinstance(ast, SelectStmt) and ast.ctes:
            return dataclasses.replace(ast, ctes=[])
        if isinstance(ast, ExplainStmt):
            return dataclasses.replace(ast, statement=self._strip_ctes(ast.statement))
        return ast

    def _optimize_ast(self, ast: object) -> object:
        if isinstance(ast, SelectStmt):
            return self._optimizer.optimize(ast)
        if isinstance(ast, ExplainStmt):
            if isinstance(ast.statement, SelectStmt):
                return dataclasses.replace(ast, statement=self._optimizer.optimize(ast.statement))
        if isinstance(ast, SetOperationStmt):
            left = self._optimize_ast(ast.left)
            right = self._optimize_ast(ast.right)
            return dataclasses.replace(ast, left=left, right=right)
        return ast

    def analyze_table(self, table_name: str) -> TableStatistics:
        schema = self._catalog.get_table(table_name)
        store = self._catalog.get_store(table_name)
        stats = TableStatistics.compute(table_name, store, schema)
        self._stats[table_name] = stats
        return stats

    def get_table_stats(self, table_name: str) -> Optional[TableStatistics]:
        return self._stats.get(table_name)

    @property
    def data_dir(self) -> str:
        return self._catalog.data_dir

    def get_table_names(self) -> list[str]:
        return [t for t in self._catalog.list_tables() if not t.startswith(_CTE_PREFIX)]

    def get_table_schema(self, name: str) -> TableSchema:
        return self._catalog.get_table(name)

    def get_table_row_count(self, name: str) -> int:
        return self._catalog.get_store(name).row_count
