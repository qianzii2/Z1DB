from __future__ import annotations
"""名称解析：星号展开、别名替换、JOIN 作用域。
[P09] GROUP BY 和 HAVING 中的歧义列也被检测。"""
import dataclasses
from typing import Any, Dict, List, Optional, Protocol, Set
from parser.ast import (AliasExpr, ColumnRef, GroupByClause,
                         JoinClause, Literal, SelectStmt, SortKey,
                         StarExpr)
from storage.types import DataType

try:
    from parser.ast import (SetOperationStmt, ExplainStmt, AlterTableStmt)
except ImportError:
    SetOperationStmt = None
    ExplainStmt = None
    AlterTableStmt = None


class CatalogInfo(Protocol):
    def get_table_columns(self, table: str) -> list[str]: ...
    def table_exists(self, table: str) -> bool: ...


class Resolver:
    def resolve(self, ast: Any, catalog_info: CatalogInfo) -> Any:
        if isinstance(ast, SelectStmt):
            return self._resolve_select(ast, catalog_info)
        if ExplainStmt is not None and isinstance(ast, ExplainStmt):
            return dataclasses.replace(
                ast, statement=self.resolve(ast.statement, catalog_info))
        if SetOperationStmt is not None and isinstance(ast, SetOperationStmt):
            return dataclasses.replace(
                ast,
                left=self.resolve(ast.left, catalog_info),
                right=self.resolve(ast.right, catalog_info))
        return ast

    def _resolve_select(self, ast: SelectStmt,
                        cat: CatalogInfo) -> SelectStmt:
        scope: Dict[Optional[str], List[str]] = {}
        all_cols_ordered: List[tuple[Optional[str], str]] = []
        if ast.from_clause is not None:
            tref = ast.from_clause.table
            if tref.subquery is None:
                self._add_table(tref.name, tref.alias, cat,
                                scope, all_cols_ordered)
            else:
                qualifier = tref.alias or tref.name
                sq_cols = self._infer_subquery_columns(tref.subquery)
                scope[None] = scope.get(None, []) + sq_cols
                scope[qualifier] = sq_cols
                for c in sq_cols:
                    all_cols_ordered.append((qualifier, c))
            for jc in ast.from_clause.joins:
                if jc.table and jc.table.subquery is None:
                    self._add_table(jc.table.name, jc.table.alias,
                                    cat, scope, all_cols_ordered)
                elif jc.table and jc.table.subquery is not None:
                    jq = jc.table.alias or jc.table.name
                    jcols = self._infer_subquery_columns(jc.table.subquery)
                    scope[None] = scope.get(None, []) + jcols
                    scope[jq] = jcols
                    for c in jcols:
                        all_cols_ordered.append((jq, c))

        has_join = bool(ast.from_clause and ast.from_clause.joins)
        ambiguous_cols = self._find_ambiguous_columns(scope) if has_join else set()

        # 展开 StarExpr
        new_select: list = []
        for item in ast.select_list:
            inner = item.expr if isinstance(item, AliasExpr) else item
            if isinstance(inner, StarExpr):
                if inner.table is not None:
                    if inner.table in scope:
                        for col in scope[inner.table]:
                            new_select.append(ColumnRef(
                                table=inner.table if has_join else None,
                                column=col))
                    else:
                        from utils.errors import SemanticError
                        raise SemanticError(f"未知表别名: {inner.table}")
                else:
                    if all_cols_ordered:
                        for qual, col in all_cols_ordered:
                            new_select.append(ColumnRef(
                                table=qual if has_join else None,
                                column=col))
                    elif None in scope:
                        for col in scope[None]:
                            new_select.append(ColumnRef(table=None, column=col))
                    else:
                        new_select.append(item)
            else:
                new_select.append(item)

        # [P09] 歧义检测：SELECT + WHERE + ORDER BY + GROUP BY + HAVING
        if ambiguous_cols:
            for item in new_select:
                self._check_ambiguous(item, ambiguous_cols)
            if ast.where:
                self._check_ambiguous(ast.where, ambiguous_cols)
            for sk in ast.order_by:
                self._check_ambiguous(sk.expr, ambiguous_cols)
            if ast.group_by:
                for k in ast.group_by.keys:
                    self._check_ambiguous(k, ambiguous_cols)
            if ast.having:
                self._check_ambiguous(ast.having, ambiguous_cols)

        alias_map: Dict[str, Any] = {}
        for item in new_select:
            if isinstance(item, AliasExpr):
                alias_map[item.alias] = item.expr

        new_order = [
            dataclasses.replace(
                sk, expr=self._resolve_ref(sk.expr, alias_map, new_select))
            for sk in ast.order_by]

        new_gb = ast.group_by
        if ast.group_by:
            new_gb = GroupByClause(keys=[
                self._resolve_ref(k, alias_map, new_select)
                for k in ast.group_by.keys])

        new_having = ast.having
        if ast.having:
            new_having = self._resolve_aliases(ast.having, alias_map)

        where = ast.where
        if not has_join and ast.from_clause and ast.from_clause.table.alias:
            alias = ast.from_clause.table.alias
            new_select = [self._norm(e, alias) for e in new_select]
            new_order = [
                dataclasses.replace(sk, expr=self._norm(sk.expr, alias))
                for sk in new_order]
            where = self._norm(where, alias) if where else None
            if new_having:
                new_having = self._norm(new_having, alias)

        return dataclasses.replace(
            ast, select_list=new_select, order_by=new_order,
            group_by=new_gb, having=new_having, where=where)

    def _add_table(self, name, alias, cat, scope, all_cols):
        if cat.table_exists(name):
            cols = cat.get_table_columns(name)
        else:
            cols = []
        qualifier = alias or name
        scope[None] = scope.get(None, []) + cols
        scope[qualifier] = cols
        for c in cols:
            all_cols.append((qualifier, c))

    @staticmethod
    def _infer_subquery_columns(subquery):
        if not isinstance(subquery, SelectStmt):
            return []
        cols = []
        for item in subquery.select_list:
            if isinstance(item, AliasExpr):
                cols.append(item.alias)
            elif isinstance(item, ColumnRef):
                cols.append(item.column)
            elif isinstance(item, StarExpr):
                return []
            else:
                cols.append(f'__expr_{len(cols)}')
        return cols

    @staticmethod
    def _find_ambiguous_columns(scope):
        col_sources: Dict[str, int] = {}
        for qualifier, cols in scope.items():
            if qualifier is None:
                continue
            for c in cols:
                col_sources[c] = col_sources.get(c, 0) + 1
        return {c for c, cnt in col_sources.items() if cnt > 1}

    def _check_ambiguous(self, node, ambiguous):
        if node is None:
            return
        if isinstance(node, ColumnRef):
            if node.table is None and node.column in ambiguous:
                from utils.errors import SemanticError
                raise SemanticError(
                    f"列 '{node.column}' 引用不明确，请使用 表名.列名 形式")
            return
        if isinstance(node, AliasExpr):
            self._check_ambiguous(node.expr, ambiguous)
            return
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for item in child:
                    self._check_ambiguous(item, ambiguous)
            elif child is not None and dataclasses.is_dataclass(child) and not isinstance(child, type):
                self._check_ambiguous(child, ambiguous)

    def _resolve_ref(self, expr, alias_map, select_list):
        if (isinstance(expr, ColumnRef) and expr.table is None
                and expr.column in alias_map):
            resolved = alias_map[expr.column]
            try:
                from parser.ast import WindowCall
                if isinstance(resolved, WindowCall):
                    return expr
            except ImportError:
                pass
            return resolved
        if (isinstance(expr, Literal) and isinstance(expr.value, int)
                and expr.inferred_type in (DataType.INT, DataType.BIGINT)):
            o = expr.value
            if 1 <= o <= len(select_list):
                t = select_list[o - 1]
                result = t.expr if isinstance(t, AliasExpr) else t
                try:
                    from parser.ast import WindowCall
                    if isinstance(result, WindowCall):
                        return expr
                except ImportError:
                    pass
                return result
        return expr

    def _resolve_aliases(self, node, alias_map):
        if isinstance(node, ColumnRef) and node.table is None and node.column in alias_map:
            return alias_map[node.column]
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        changes = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                changes[f.name] = [self._resolve_aliases(i, alias_map) for i in child]
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._resolve_aliases(child, alias_map)
        return dataclasses.replace(node, **changes) if changes else node

    def _norm(self, node, alias):
        if node is None:
            return None
        if isinstance(node, ColumnRef):
            return ColumnRef(table=None, column=node.column) if node.table == alias else node
        if isinstance(node, AliasExpr):
            return dataclasses.replace(node, expr=self._norm(node.expr, alias))
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return node
        changes = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                changes[f.name] = [self._norm(i, alias) for i in child]
            elif dataclasses.is_dataclass(child) and not isinstance(child, type):
                changes[f.name] = self._norm(child, alias)
        return dataclasses.replace(node, **changes) if changes else node
