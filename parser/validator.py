from __future__ import annotations
"""Semantic validation of parsed AST nodes."""

from typing import Any, Protocol

from parser.ast import (
    AggregateCall, AliasExpr, BinaryExpr, ColumnRef, CreateTableStmt,
    InsertStmt, IsNullExpr, SelectStmt, StarExpr, UnaryExpr,
)
from utils.errors import ColumnNotFoundError, SemanticError, TableNotFoundError


class CatalogInfo(Protocol):
    def get_table_columns(self, table: str) -> list[str]: ...
    def table_exists(self, table: str) -> bool: ...


class Validator:
    """Performs semantic checks that do not require execution."""

    def validate(self, ast: Any, catalog_info: CatalogInfo) -> Any:
        if isinstance(ast, SelectStmt):
            self._validate_select(ast, catalog_info)
        elif isinstance(ast, InsertStmt):
            self._validate_insert(ast, catalog_info)
        elif isinstance(ast, CreateTableStmt):
            self._validate_create(ast, catalog_info)
        return ast

    # ------------------------------------------------------------------
    def _validate_select(self, ast: SelectStmt, cat: CatalogInfo) -> None:
        # Table existence
        if ast.from_clause is not None:
            tname = ast.from_clause.table.name
            if not cat.table_exists(tname):
                raise TableNotFoundError(tname)
            known_columns = set(cat.get_table_columns(tname))
        else:
            known_columns = set()

        # Column existence in select, where, order
        if ast.from_clause is not None:
            for expr in ast.select_list:
                self._check_columns(expr, known_columns)
            if ast.where:
                self._check_columns(ast.where, known_columns)
            for sk in ast.order_by:
                self._check_columns(sk.expr, known_columns)

        # Mixed aggregate check: aggregate + bare column without GROUP BY
        has_agg = any(self._contains_aggregate(e) for e in ast.select_list)
        if has_agg and not ast.group_by:
            for expr in ast.select_list:
                self._check_bare_column_in_aggregate(expr)

        # Nested aggregate check
        for expr in ast.select_list:
            self._check_nested_aggregate(expr, depth=0)

    def _validate_insert(self, ast: InsertStmt, cat: CatalogInfo) -> None:
        if not cat.table_exists(ast.table):
            raise TableNotFoundError(ast.table)
        if ast.columns is not None:
            known = set(cat.get_table_columns(ast.table))
            for c in ast.columns:
                if c not in known:
                    raise ColumnNotFoundError(c)

    def _validate_create(self, ast: CreateTableStmt, cat: CatalogInfo) -> None:
        if not ast.columns:
            raise SemanticError("CREATE TABLE requires at least one column")
        seen: set[str] = set()
        for cd in ast.columns:
            if cd.name in seen:
                raise SemanticError(f"duplicate column name: '{cd.name}'")
            seen.add(cd.name)

    # -- helpers -------------------------------------------------------
    def _check_columns(self, node: Any, known: set[str]) -> None:
        if node is None:
            return
        if isinstance(node, ColumnRef):
            if node.column not in known:
                raise ColumnNotFoundError(node.column)
            return
        if isinstance(node, AliasExpr):
            self._check_columns(node.expr, known)
            return
        if isinstance(node, StarExpr):
            return
        import dataclasses
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for f in dataclasses.fields(node):
                child = getattr(node, f.name)
                if isinstance(child, list):
                    for item in child:
                        self._check_columns(item, known)
                elif child is not None:
                    self._check_columns(child, known)

    def _contains_aggregate(self, node: Any) -> bool:
        if isinstance(node, AggregateCall):
            return True
        if isinstance(node, AliasExpr):
            return self._contains_aggregate(node.expr)
        import dataclasses
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return False
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                if any(self._contains_aggregate(c) for c in child):
                    return True
            elif self._contains_aggregate(child):
                return True
        return False

    def _check_bare_column_in_aggregate(self, node: Any) -> None:
        """In a scalar aggregate context (no GROUP BY), bare ColumnRef is illegal."""
        if isinstance(node, AggregateCall):
            return  # aggregates are fine
        if isinstance(node, ColumnRef):
            raise SemanticError(
                f"column '{node.column}' must appear in GROUP BY or aggregate function")
        if isinstance(node, AliasExpr):
            self._check_bare_column_in_aggregate(node.expr)
            return
        import dataclasses
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for item in child:
                    self._check_bare_column_in_aggregate(item)
            elif child is not None:
                self._check_bare_column_in_aggregate(child)

    def _check_nested_aggregate(self, node: Any, depth: int) -> None:
        """Reject nested aggregates like SUM(COUNT(*))."""
        if isinstance(node, AggregateCall):
            if depth > 0:
                raise SemanticError(f"nested aggregate calls are not allowed: {node.name}")
            for arg in node.args:
                self._check_nested_aggregate(arg, depth + 1)
            return
        if isinstance(node, AliasExpr):
            self._check_nested_aggregate(node.expr, depth)
            return
        import dataclasses
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for item in child:
                    self._check_nested_aggregate(item, depth)
            elif child is not None:
                self._check_nested_aggregate(child, depth)
