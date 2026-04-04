from __future__ import annotations
"""Z1DB engine — top-level façade."""
from catalog.catalog import Catalog, TableSchema
from executor.core.result import ExecutionResult
from executor.functions.registry import FunctionRegistry
from executor.simple_planner import SimplePlanner
from parser.lexer import Lexer
from parser.parser import Parser
from parser.resolver import Resolver
from parser.validator import Validator
from utils.timer import Timer


class Engine:
    def __init__(self, data_dir: str = ':memory:') -> None:
        self._catalog = Catalog(data_dir)
        self._registry = FunctionRegistry()
        self._registry.register_defaults()
        self._planner = SimplePlanner(self._registry)

    def execute(self, sql: str) -> ExecutionResult:
        with Timer() as t:
            tokens = Lexer(sql).tokenize()
            ast = Parser(tokens).parse()
            ast = Resolver().resolve(ast, self._catalog)
            ast = Validator().validate(ast, self._catalog)
            result = self._planner.execute(ast, self._catalog)
        result.timing = t.elapsed
        # Persist after every mutation
        if self._catalog.is_persistent:
            if result.affected_rows > 0 or result.message == 'OK':
                self._catalog.persist()
        return result

    @property
    def data_dir(self) -> str:
        return self._catalog.data_dir

    def get_table_names(self) -> list[str]:
        return self._catalog.list_tables()

    def get_table_schema(self, name: str) -> TableSchema:
        return self._catalog.get_table(name)

    def get_table_row_count(self, name: str) -> int:
        return self._catalog.get_store(name).row_count
