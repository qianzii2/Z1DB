from __future__ import annotations
"""Z1DB engine — top-level façade with optimizer."""
from catalog.catalog import Catalog, TableSchema
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
            ast = Resolver().resolve(ast, self._catalog)
            ast = Validator().validate(ast, self._catalog)
            # Apply optimizer
            ast = self._optimize_ast(ast)
            result = self._planner.execute(ast, self._catalog)
        result.timing = t.elapsed
        if self._catalog.is_persistent:
            if result.affected_rows > 0 or result.message == 'OK':
                self._catalog.persist()
        return result

    def _optimize_ast(self, ast: object) -> object:
        """Apply query optimizer to SELECT statements."""
        if isinstance(ast, SelectStmt):
            return self._optimizer.optimize(ast)
        if isinstance(ast, ExplainStmt):
            if isinstance(ast.statement, SelectStmt):
                import dataclasses
                optimized = self._optimizer.optimize(ast.statement)
                return dataclasses.replace(ast, statement=optimized)
        if isinstance(ast, SetOperationStmt):
            import dataclasses
            left = self._optimize_ast(ast.left)
            right = self._optimize_ast(ast.right)
            return dataclasses.replace(ast, left=left, right=right)
        return ast

    def analyze_table(self, table_name: str) -> TableStatistics:
        """Compute and cache statistics for a table."""
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


# Fix missing import
from typing import Optional
