from __future__ import annotations
"""名称解析：星号展开、别名替换、JOIN 作用域。
处理 GROUP BY / HAVING / ORDER BY 中的别名和序号引用。
检测 JOIN 中的歧义列引用。"""
import dataclasses
from typing import Any, Dict, List, Optional, Set
from parser.ast import (
    AliasExpr, ColumnRef, GroupByClause, JoinClause,
    Literal, SelectStmt, SortKey, StarExpr)
from storage.types import DataType

try:
    from parser.ast import SetOperationStmt, ExplainStmt
except ImportError:
    SetOperationStmt = None
    ExplainStmt = None


class CatalogInfo:
    """Resolver 依赖的 Catalog 接口（Protocol 方式解耦）。"""
    def get_table_columns(self, table: str) -> list[str]: ...
    def table_exists(self, table: str) -> bool: ...


class Resolver:
    """SQL 名称解析器。"""

    def resolve(self, ast: Any, catalog: CatalogInfo) -> Any:
        """主入口。根据 AST 类型分发。"""
        if isinstance(ast, SelectStmt):
            return self._resolve_select(ast, catalog)
        if ExplainStmt and isinstance(ast, ExplainStmt):
            inner = self.resolve(ast.statement, catalog)
            return dataclasses.replace(ast, statement=inner)
        if SetOperationStmt and isinstance(ast, SetOperationStmt):
            left = self.resolve(ast.left, catalog)
            right = self.resolve(ast.right, catalog)
            return dataclasses.replace(
                ast, left=left, right=right)
        return ast

    def _resolve_select(self, ast, cat):
        """SELECT 语句的完整名称解析。"""
        # 构建作用域
        scope: Dict[Optional[str], List[str]] = {}
        all_cols_ordered: List[tuple] = []
        has_subquery_source = False

        if ast.from_clause is not None:
            tref = ast.from_clause.table
            if tref.subquery is None:
                if cat.table_exists(tref.name):
                    self._add_table(
                        tref.name, tref.alias, cat,
                        scope, all_cols_ordered)
                else:
                    # Could be a CTE or generate_series - treat as subquery source
                    has_subquery_source = True
            else:
                has_subquery_source = True
                qualifier = tref.alias or tref.name
                sq_cols = self._infer_subquery_columns(
                    tref.subquery)
                scope[None] = scope.get(None, []) + sq_cols
                scope[qualifier] = sq_cols
                for c in sq_cols:
                    all_cols_ordered.append((qualifier, c))
            for jc in ast.from_clause.joins:
                if jc.table and jc.table.subquery is None:
                    if cat.table_exists(jc.table.name):
                        self._add_table(
                            jc.table.name, jc.table.alias,
                            cat, scope, all_cols_ordered)
                    else:
                        has_subquery_source = True
                elif jc.table and jc.table.subquery is not None:
                    has_subquery_source = True
                    jq = jc.table.alias or jc.table.name
                    jcols = self._infer_subquery_columns(
                        jc.table.subquery)
                    scope[None] = scope.get(None, []) + jcols
                    scope[jq] = jcols
                    for c in jcols:
                        all_cols_ordered.append((jq, c))

        has_join = bool(
            ast.from_clause and ast.from_clause.joins)
        ambiguous_cols = (
            self._find_ambiguous_columns(scope)
            if has_join else set())

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
                        raise SemanticError(
                            f"未知表别名: {inner.table}")
                else:
                    if all_cols_ordered:
                        for qual, col in all_cols_ordered:
                            new_select.append(ColumnRef(
                                table=qual if has_join else None,
                                column=col))
                    elif None in scope:
                        for col in scope[None]:
                            new_select.append(
                                ColumnRef(table=None, column=col))
                    else:
                        # Can't expand * - leave as is for later handling
                        new_select.append(item)
            else:
                new_select.append(item)

        # 歧义检测
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

        # 别名映射
        alias_map: Dict[str, Any] = {}
        for item in new_select:
            if isinstance(item, AliasExpr):
                alias_map[item.alias] = item.expr

        # ORDER BY 别名/序号解析
        new_order = [
            dataclasses.replace(
                sk, expr=self._resolve_ref(
                    sk.expr, alias_map, new_select))
            for sk in ast.order_by]

        # GROUP BY 别名/序号解析
        new_gb = ast.group_by
        if ast.group_by:
            new_gb = GroupByClause(keys=[
                self._resolve_ref(k, alias_map, new_select)
                for k in ast.group_by.keys])

        # HAVING 别名解析
        new_having = ast.having
        if ast.having:
            new_having = self._resolve_aliases(
                ast.having, alias_map)

        # 单表别名规范化（去掉多余的表限定符）
        where = ast.where
        if (not has_join and ast.from_clause
                and ast.from_clause.table.alias):
            alias = ast.from_clause.table.alias
            new_select = [self._norm(e, alias)
                          for e in new_select]
            new_order = [
                dataclasses.replace(
                    sk, expr=self._norm(sk.expr, alias))
                for sk in new_order]
            where = self._norm(where, alias) if where else None
            if new_having:
                new_having = self._norm(new_having, alias)

        return dataclasses.replace(
            ast, select_list=new_select,
            order_by=new_order, group_by=new_gb,
            having=new_having, where=where)

    # ═══ 辅助方法 ═══

    def _add_table(self, name, alias, cat, scope, all_cols):
        """将表的列加入作用域。"""
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
        """从子查询 AST 推断输出列名。"""
        if isinstance(subquery, SelectStmt):
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
        # Handle SetOperationStmt: infer from left side
        try:
            from parser.ast import SetOperationStmt
            if isinstance(subquery, SetOperationStmt):
                return Resolver._infer_subquery_columns(subquery.left)
        except ImportError:
            pass
        return []

    @staticmethod
    def _find_ambiguous_columns(scope):
        """找出在多个表中出现的列名。"""
        col_sources: Dict[str, int] = {}
        for qualifier, cols in scope.items():
            if qualifier is None:
                continue
            for c in cols:
                col_sources[c] = col_sources.get(c, 0) + 1
        return {c for c, cnt in col_sources.items() if cnt > 1}

    def _check_ambiguous(self, node, ambiguous):
        """检测未限定的歧义列引用。"""
        if node is None:
            return
        if isinstance(node, ColumnRef):
            if node.table is None and node.column in ambiguous:
                from utils.errors import SemanticError
                raise SemanticError(
                    f"列 '{node.column}' 引用不明确，"
                    f"请使用 表名.列名 形式")
            return
        if isinstance(node, AliasExpr):
            self._check_ambiguous(node.expr, ambiguous)
            return
        if (not dataclasses.is_dataclass(node)
                or isinstance(node, type)):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for item in child:
                    self._check_ambiguous(item, ambiguous)
            elif (child is not None
                  and dataclasses.is_dataclass(child)
                  and not isinstance(child, type)):
                self._check_ambiguous(child, ambiguous)

    def _resolve_ref(self, expr, alias_map, select_list):
        """解析 ORDER BY / GROUP BY 中的别名和序号引用。"""
        # 别名引用
        if (isinstance(expr, ColumnRef)
                and expr.table is None
                and expr.column in alias_map):
            resolved = alias_map[expr.column]
            try:
                from parser.ast import WindowCall
                if isinstance(resolved, WindowCall):
                    return expr  # 窗口函数不展开
            except ImportError:
                pass
            return resolved
        # 序号引用（ORDER BY 1, GROUP BY 2）
        if (isinstance(expr, Literal)
                and isinstance(expr.value, int)
                and expr.inferred_type in (
                    DataType.INT, DataType.BIGINT)):
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
        """递归替换 HAVING 中的别名引用。"""
        if (isinstance(node, ColumnRef)
                and node.table is None
                and node.column in alias_map):
            return alias_map[node.column]
        if (not dataclasses.is_dataclass(node)
                or isinstance(node, type)):
            return node
        changes = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                changes[f.name] = [
                    self._resolve_aliases(i, alias_map)
                    for i in child]
            elif (dataclasses.is_dataclass(child)
                  and not isinstance(child, type)):
                changes[f.name] = self._resolve_aliases(
                    child, alias_map)
        return (dataclasses.replace(node, **changes)
                if changes else node)

    def _norm(self, node, alias):
        """单表模式：去掉与表别名匹配的限定符。"""
        if node is None:
            return None
        if isinstance(node, ColumnRef):
            if node.table == alias:
                return ColumnRef(table=None, column=node.column)
            return node
        if isinstance(node, AliasExpr):
            return dataclasses.replace(
                node, expr=self._norm(node.expr, alias))
        if (not dataclasses.is_dataclass(node)
                or isinstance(node, type)):
            return node
        changes = {}
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                changes[f.name] = [
                    self._norm(i, alias) for i in child]
            elif (dataclasses.is_dataclass(child)
                  and not isinstance(child, type)):
                changes[f.name] = self._norm(child, alias)
        return (dataclasses.replace(node, **changes)
                if changes else node)

