from __future__ import annotations
"""Name resolution: star expansion, alias replacement, join scope."""
import dataclasses
from typing import Any, Dict, List, Optional, Protocol
from parser.ast import (AliasExpr, ColumnRef, GroupByClause, JoinClause,
                         Literal, SelectStmt, SortKey, StarExpr)
from storage.types import DataType

class CatalogInfo(Protocol):
    def get_table_columns(self, table: str) -> list[str]: ...
    def table_exists(self, table: str) -> bool: ...

class Resolver:
    def resolve(self, ast: Any, catalog_info: CatalogInfo) -> Any:
        if isinstance(ast, SelectStmt): return self._resolve_select(ast, catalog_info)
        return ast

    def _resolve_select(self, ast: SelectStmt, cat: CatalogInfo) -> SelectStmt:
        # Build scope: qualifier → [columns]
        scope: Dict[Optional[str], List[str]] = {}
        all_cols_ordered: List[tuple[Optional[str], str]] = []
        if ast.from_clause is not None:
            tref = ast.from_clause.table
            self._add_table_to_scope(tref.name, tref.alias, cat, scope, all_cols_ordered)
            for jc in ast.from_clause.joins:
                if jc.table:
                    self._add_table_to_scope(jc.table.name, jc.table.alias, cat, scope, all_cols_ordered)

        has_join = bool(ast.from_clause and ast.from_clause.joins)

        # Expand StarExpr
        new_select: list = []
        for item in ast.select_list:
            inner = item.expr if isinstance(item, AliasExpr) else item
            if isinstance(inner, StarExpr):
                if inner.table is not None:
                    if inner.table in scope:
                        for col in scope[inner.table]:
                            new_select.append(ColumnRef(table=inner.table if has_join else None, column=col))
                    else:
                        from utils.errors import SemanticError
                        raise SemanticError(f"unknown table alias: {inner.table}")
                else:
                    for qual, col in all_cols_ordered:
                        new_select.append(ColumnRef(table=qual if has_join else None, column=col))
            else:
                new_select.append(item)

        # Build alias map
        alias_map: Dict[str, Any] = {}
        for item in new_select:
            if isinstance(item, AliasExpr): alias_map[item.alias] = item.expr

        # ORDER BY resolve
        new_order = [dataclasses.replace(sk, expr=self._resolve_ref(sk.expr, alias_map, new_select))
                     for sk in ast.order_by]

        # GROUP BY resolve
        new_gb = ast.group_by
        if ast.group_by:
            new_gb = GroupByClause(keys=[self._resolve_ref(k, alias_map, new_select) for k in ast.group_by.keys])

        # HAVING resolve aliases
        new_having = ast.having
        if ast.having:
            new_having = self._resolve_having(ast.having, alias_map)

        # Single-table normalisation (strip qualifier)
        where = ast.where
        if not has_join and ast.from_clause and ast.from_clause.table.alias:
            alias = ast.from_clause.table.alias
            new_select = [self._norm(e, alias) for e in new_select]
            new_order = [dataclasses.replace(sk, expr=self._norm(sk.expr, alias)) for sk in new_order]
            where = self._norm(where, alias) if where else None
            if new_having: new_having = self._norm(new_having, alias)

        return dataclasses.replace(ast, select_list=new_select, order_by=new_order,
                                   group_by=new_gb, having=new_having, where=where)

    def _add_table_to_scope(self, name: str, alias: Optional[str], cat: CatalogInfo,
                            scope: Dict, all_cols: list) -> None:
        if cat.table_exists(name):
            cols = cat.get_table_columns(name)
        else:
            cols = []
        qualifier = alias or name
        scope[None] = scope.get(None, []) + cols  # unqualified access
        scope[qualifier] = cols
        for c in cols:
            all_cols.append((qualifier, c))

    def _resolve_ref(self, expr: Any, alias_map: Dict, select_list: list) -> Any:
        if isinstance(expr, ColumnRef) and expr.table is None and expr.column in alias_map:
            return alias_map[expr.column]
        if isinstance(expr, Literal) and isinstance(expr.value, int) and expr.inferred_type in (DataType.INT, DataType.BIGINT):
            o = expr.value
            if 1 <= o <= len(select_list):
                t = select_list[o - 1]
                return t.expr if isinstance(t, AliasExpr) else t
        return expr

    def _resolve_having(self, node: Any, alias_map: Dict) -> Any:
        if isinstance(node, ColumnRef) and node.table is None and node.column in alias_map:
            return alias_map[node.column]
        if not dataclasses.is_dataclass(node) or isinstance(node, type): return node
        changes = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                changes[f.name] = [self._resolve_having(i, alias_map) for i in child]
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._resolve_having(child, alias_map)
        return dataclasses.replace(node, **changes) if changes else node

    def _norm(self, node: Any, alias: str) -> Any:
        if node is None: return None
        if isinstance(node, ColumnRef):
            if node.table == alias: return ColumnRef(table=None, column=node.column)
            return node
        if isinstance(node, AliasExpr):
            return dataclasses.replace(node, expr=self._norm(node.expr, alias))
        if not dataclasses.is_dataclass(node) or isinstance(node, type): return node
        changes = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                changes[f.name] = [self._norm(i, alias) for i in child]
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._norm(child, alias)
        return dataclasses.replace(node, **changes) if changes else node
