from __future__ import annotations
"""Name resolution: star expansion, alias replacement, column normalisation."""

import dataclasses
from typing import Any, Dict, List, Optional, Protocol

from parser.ast import (
    AliasExpr, ColumnRef, GroupByClause, Literal, SelectStmt, SortKey, StarExpr,
)
from storage.types import DataType


class CatalogInfo(Protocol):
    def get_table_columns(self, table: str) -> list[str]: ...
    def table_exists(self, table: str) -> bool: ...


class Resolver:
    """Resolve names against schema information."""

    def resolve(self, ast: Any, catalog_info: CatalogInfo) -> Any:
        if isinstance(ast, SelectStmt):
            return self._resolve_select(ast, catalog_info)
        return ast

    # ------------------------------------------------------------------
    def _resolve_select(self, ast: SelectStmt, cat: CatalogInfo) -> SelectStmt:
        # 1. Build scope
        scope: Dict[Optional[str], List[str]] = {}
        if ast.from_clause is not None:
            tref = ast.from_clause.table
            if cat.table_exists(tref.name):
                cols = cat.get_table_columns(tref.name)
            else:
                cols = []
            # Always register under None for unqualified access
            scope[None] = cols
            if tref.alias:
                scope[tref.alias] = cols

        # 2. Expand StarExpr
        new_select: list = []
        for item in ast.select_list:
            inner = item.expr if isinstance(item, AliasExpr) else item
            if isinstance(inner, StarExpr):
                if inner.table is not None:
                    if inner.table not in scope:
                        from utils.errors import SemanticError
                        raise SemanticError(f"unknown table alias: {inner.table}")
                    for col in scope[inner.table]:
                        new_select.append(ColumnRef(table=None, column=col))
                else:
                    # Unqualified * — expand all scope entries under None
                    if None in scope:
                        for col in scope[None]:
                            new_select.append(ColumnRef(table=None, column=col))
            else:
                new_select.append(item)

        # 3. Build alias map from select list
        alias_map: Dict[str, Any] = {}
        for item in new_select:
            if isinstance(item, AliasExpr):
                alias_map[item.alias] = item.expr

        # 4. ORDER BY: resolve aliases and ordinal references
        new_order: list = []
        for sk in ast.order_by:
            resolved_expr = self._resolve_order_expr(sk.expr, alias_map, new_select)
            new_order.append(dataclasses.replace(sk, expr=resolved_expr))

        # 5. GROUP BY: resolve ordinal references
        new_group_by = ast.group_by
        if ast.group_by is not None:
            new_keys = []
            for key in ast.group_by.keys:
                new_keys.append(self._resolve_order_expr(key, alias_map, new_select))
            new_group_by = GroupByClause(keys=new_keys)

        # 6. Normalise qualified ColumnRef for single-table queries
        if ast.from_clause is not None and ast.from_clause.table.alias:
            alias = ast.from_clause.table.alias
            new_select = [self._normalise_column_refs(e, alias, scope) for e in new_select]
            new_order = [
                dataclasses.replace(sk, expr=self._normalise_column_refs(sk.expr, alias, scope))
                for sk in new_order
            ]
            where = self._normalise_column_refs(ast.where, alias, scope) if ast.where else None
        else:
            where = ast.where

        return dataclasses.replace(
            ast,
            select_list=new_select,
            order_by=new_order,
            group_by=new_group_by,
            where=where,
        )

    # ------------------------------------------------------------------
    def _resolve_order_expr(self, expr: Any, alias_map: Dict[str, Any],
                            select_list: list) -> Any:
        """Replace alias refs and ordinal literals in ORDER BY / GROUP BY."""
        if isinstance(expr, ColumnRef) and expr.table is None and expr.column in alias_map:
            return alias_map[expr.column]
        if isinstance(expr, Literal) and isinstance(expr.value, int) and expr.inferred_type in (
                DataType.INT, DataType.BIGINT):
            ordinal = expr.value
            if 1 <= ordinal <= len(select_list):
                target = select_list[ordinal - 1]
                if isinstance(target, AliasExpr):
                    return target.expr
                return target
        return expr

    # ------------------------------------------------------------------
    def _normalise_column_refs(self, node: Any, alias: str,
                               scope: Dict[Optional[str], List[str]]) -> Any:
        """Strip table qualifier when it matches the single-table alias."""
        if node is None:
            return None
        if isinstance(node, ColumnRef):
            if node.table == alias:
                return ColumnRef(table=None, column=node.column)
            return node
        if isinstance(node, AliasExpr):
            return dataclasses.replace(
                node, expr=self._normalise_column_refs(node.expr, alias, scope))
        if not dataclasses.is_dataclass(node):
            return node
        changes: dict = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                new_list = [self._normalise_column_refs(item, alias, scope) for item in child]
                changes[f.name] = new_list
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._normalise_column_refs(child, alias, scope)
        return dataclasses.replace(node, **changes) if changes else node
