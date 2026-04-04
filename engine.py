from __future__ import annotations
"""Z1DB engine — top-level façade with optimizer and CTE handling."""
from typing import Optional
import dataclasses

from catalog.catalog import Catalog, ColumnSchema, TableSchema
from catalog.statistics import TableStatistics
from executor.core.result import ExecutionResult
from executor.functions.registry import FunctionRegistry
from executor.optimizer import QueryOptimizer
from executor.simple_planner import SimplePlanner
from parser.ast import SelectStmt, ExplainStmt, SetOperationStmt
from parser.lexer import Lexer
from parser.parser import Parser
from parser.resolver import Resolver
from parser.validator import Validator
from storage.types import DataType
from utils.timer import Timer


class Engine:
    def __init__(self, data_dir: str = ':memory:') -> None:
        self._catalog = Catalog(data_dir)
        self._registry = FunctionRegistry()
        self._registry.register_defaults()
        self._planner = SimplePlanner(self._registry)
        self._optimizer = QueryOptimizer()
        self._stats: dict[str, TableStatistics] = {}

    def execute(self, sql: str) -> ExecutionResult:
        with Timer() as t:
            tokens = Lexer(sql).tokenize()
            ast = Parser(tokens).parse()

            # Materialize CTEs BEFORE resolve/validate
            cte_tables = self._materialize_ctes(ast)
            try:
                if cte_tables:
                    ast = self._strip_ctes(ast)
                ast = Resolver().resolve(ast, self._catalog)
                ast = Validator().validate(ast, self._catalog)
                ast = self._optimize_ast(ast)
                result = self._planner.execute(ast, self._catalog)
            finally:
                for tname in cte_tables:
                    try:
                        self._catalog.drop_table(tname)
                    except Exception:
                        pass

        result.timing = t.elapsed
        if self._catalog.is_persistent:
            if result.affected_rows > 0 or result.message == 'OK':
                self._catalog.persist()
        return result

    def _materialize_ctes(self, ast: object) -> list[str]:
        """Find and materialize all CTEs. Returns list of temp table names."""
        created: list[str] = []
        if isinstance(ast, SelectStmt) and ast.ctes:
            for cte_name, cte_query in ast.ctes:
                # Materialize nested CTEs within this CTE query first
                nested = self._materialize_ctes(cte_query)
                created.extend(nested)
                # Execute CTE query directly through planner
                try:
                    resolved = Resolver().resolve(cte_query, self._catalog)
                    validated = Validator().validate(resolved, self._catalog)
                    optimized = self._optimize_ast(validated)
                    cte_result = self._planner.execute(optimized, self._catalog)
                except Exception:
                    continue
                if cte_result.columns:
                    cols = [ColumnSchema(name=cn, dtype=ct, nullable=True)
                            for cn, ct in zip(cte_result.columns, cte_result.column_types)]
                    if not self._catalog.table_exists(cte_name):
                        schema = TableSchema(name=cte_name, columns=cols)
                        self._catalog.create_table(schema)
                        store = self._catalog.get_store(cte_name)
                        for row in cte_result.rows:
                            store.append_row(row)
                        created.append(cte_name)
        elif isinstance(ast, ExplainStmt):
            created.extend(self._materialize_ctes(ast.statement))
        elif isinstance(ast, SetOperationStmt):
            created.extend(self._materialize_ctes(ast.left))
            created.extend(self._materialize_ctes(ast.right))
        return created

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
        return self._catalog.list_tables()

    def get_table_schema(self, name: str) -> TableSchema:
        return self._catalog.get_table(name)

    def get_table_row_count(self, name: str) -> int:
        return self._catalog.get_store(name).row_count
