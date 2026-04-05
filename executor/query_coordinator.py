from __future__ import annotations
"""查询协调器 — 串联correlated_subquery和subquery_unnest。
先尝试unnest重写为JOIN，失败时回退到逐行关联执行。"""
from typing import Any, Optional
from parser.ast import SelectStmt, ExistsExpr, InExpr, SubqueryExpr, BinaryExpr


class QueryCoordinator:
    """协调子查询的解关联化策略。"""

    def __init__(self, planner: Any, catalog: Any) -> None:
        self._planner = planner
        self._catalog = catalog

    def optimize_subqueries(self, ast: SelectStmt) -> SelectStmt:
        """主入口：尝试子查询优化。
        1. 先尝试unnest重写为JOIN
        2. 失败时保留原样（SimplePlanner的_resolve_sq会逐行执行）
        """
        if ast.where is None:
            return ast

        # 阶段1：尝试unnest
        if self._has_subquery_in_where(ast.where):
            try:
                from executor.subquery_unnest import SubqueryUnnester
                unnested = SubqueryUnnester().unnest(ast)
                if unnested is not ast:
                    return unnested
            except Exception:
                pass

        # 阶段2：对关联子查询，标记外层列以便后续逐行执行
        if self._has_correlated_subquery(ast):
            try:
                from executor.correlated_subquery import CorrelatedResolver
                resolver = CorrelatedResolver(self._planner, self._catalog)
                outer_cols = self._collect_outer_columns(ast)
                new_where = resolver.resolve(
                    ast.where, outer_row=None, outer_cols=outer_cols)
                if new_where is not ast.where:
                    import dataclasses
                    return dataclasses.replace(ast, where=new_where)
            except Exception:
                pass

        return ast

    def _has_subquery_in_where(self, node: Any) -> bool:
        """检查WHERE中是否有子查询。"""
        if isinstance(node, (SubqueryExpr, ExistsExpr)):
            return True
        if isinstance(node, InExpr):
            return any(isinstance(v, SubqueryExpr) for v in node.values)
        if isinstance(node, BinaryExpr):
            return (self._has_subquery_in_where(node.left)
                    or self._has_subquery_in_where(node.right))
        return False

    def _has_correlated_subquery(self, ast: SelectStmt) -> bool:
        """检查是否有关联子查询（引用外层表的列）。"""
        if ast.from_clause is None:
            return False
        outer_tables = {ast.from_clause.table.alias
                        or ast.from_clause.table.name}
        for jc in ast.from_clause.joins:
            if jc.table:
                outer_tables.add(jc.table.alias or jc.table.name)
        return self._check_correlated(ast.where, outer_tables)

    def _check_correlated(self, node: Any, outer_tables: set) -> bool:
        if node is None:
            return False
        if isinstance(node, SubqueryExpr):
            return self._refs_outer(node.query, outer_tables)
        if isinstance(node, ExistsExpr):
            return self._refs_outer(node.query, outer_tables)
        if isinstance(node, BinaryExpr):
            return (self._check_correlated(node.left, outer_tables)
                    or self._check_correlated(node.right, outer_tables))
        return False

    def _refs_outer(self, query: Any, outer_tables: set) -> bool:
        """检查子查询是否引用了外层表。"""
        if not isinstance(query, SelectStmt):
            return False
        from parser.ast import ColumnRef
        import dataclasses
        refs = self._collect_all_refs(query.where)
        for ref in refs:
            if ref.table and ref.table in outer_tables:
                return True
        return False

    def _collect_all_refs(self, node: Any) -> list:
        from parser.ast import ColumnRef
        if isinstance(node, ColumnRef):
            return [node]
        if node is None:
            return []
        import dataclasses
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return []
        result = []
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for item in child:
                    result.extend(self._collect_all_refs(item))
            elif child is not None:
                result.extend(self._collect_all_refs(child))
        return result

    def _collect_outer_columns(self, ast: SelectStmt) -> set:
        """收集外层表的所有列名（供关联子查询替换使用）。"""
        cols: set = set()
        if ast.from_clause is None:
            return cols
        tref = ast.from_clause.table
        tname = tref.name
        talias = tref.alias or tname
        if self._catalog.table_exists(tname):
            for c in self._catalog.get_table_columns(tname):
                cols.add(c)
                cols.add(f"{talias}.{c}")
        for jc in ast.from_clause.joins:
            if jc.table and jc.table.subquery is None:
                jname = jc.table.name
                jalias = jc.table.alias or jname
                if self._catalog.table_exists(jname):
                    for c in self._catalog.get_table_columns(jname):
                        cols.add(c)
                        cols.add(f"{jalias}.{c}")
        return cols
