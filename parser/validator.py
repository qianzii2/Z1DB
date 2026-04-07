from __future__ import annotations
"""语义验证 — GROUP BY 检查、嵌套聚合检测、表/列存在性验证。
[M08] CHECK 约束验证桩（预留）。"""
import dataclasses
from typing import Any, Protocol, Set
from parser.ast import (
    AggregateCall, AliasExpr, ColumnRef,
    CreateTableStmt, DeleteStmt, InsertStmt,
    SelectStmt, StarExpr, UpdateStmt,
    FunctionCall, BinaryExpr, UnaryExpr,
    CaseExpr, CastExpr, IsNullExpr, InExpr,
    BetweenExpr, LikeExpr)
from parser.ast_utils import contains_agg
from utils.errors import (
    ColumnNotFoundError, SemanticError,
    TableNotFoundError)

try:
    from parser.ast import (
        WindowCall, ExplainStmt, SetOperationStmt,
        AlterTableStmt, SubqueryExpr, ExistsExpr)
except ImportError:
    WindowCall = None
    ExplainStmt = None
    SetOperationStmt = None
    AlterTableStmt = None
    SubqueryExpr = None
    ExistsExpr = None

_CTE_PREFIX = '__cte_'

class CatalogInfo(Protocol):
    def get_table_columns(self, table: str) -> list[str]: ...
    def table_exists(self, table: str) -> bool: ...


class Validator:
    """语义验证器。检查 AST 的语义正确性。"""

    def validate(self, ast: Any,
                 catalog_info: CatalogInfo,
                 cte_names: set = None) -> Any:
        self._cte_names = cte_names or set()
        # Collect CTE names from the AST itself
        if isinstance(ast, SelectStmt) and ast.ctes:
            for entry in ast.ctes:
                self._cte_names.add(entry[0])
        if isinstance(ast, SelectStmt):
            self._validate_select(ast, catalog_info)
        elif isinstance(ast, InsertStmt):
            self._validate_insert(ast, catalog_info)
        elif isinstance(ast, UpdateStmt):
            self._validate_update(ast, catalog_info)
        elif isinstance(ast, DeleteStmt):
            self._validate_delete(ast, catalog_info)
        elif isinstance(ast, CreateTableStmt):
            self._validate_create(ast, catalog_info)
        elif (ExplainStmt is not None
              and isinstance(ast, ExplainStmt)):
            self.validate(ast.statement, catalog_info, self._cte_names)
        elif (SetOperationStmt is not None
              and isinstance(ast, SetOperationStmt)):
            self.validate(ast.left, catalog_info, self._cte_names)
            self.validate(ast.right, catalog_info, self._cte_names)
        return ast

    def _is_cte_table(self, name: str) -> bool:
        """Check if a table name is a CTE reference."""
        return (name in self._cte_names
                or name.startswith(_CTE_PREFIX)
                or f'{_CTE_PREFIX}{name}' in self._cte_names)

    def _validate_select(self, ast, cat):
        """验证 SELECT：表/列存在性、GROUP BY 一致性、嵌套聚合。"""
        known: Set[str] = set()
        known_qualified: Set[str] = set()
        has_subquery_source = False

        if ast.from_clause:
            tref = ast.from_clause.table
            if tref.subquery is None and tref.func_args is None:
                if cat.table_exists(tref.name):
                    self._add_known(
                        tref.name, tref.alias, cat,
                        known, known_qualified)
                elif self._is_cte_table(tref.name):
                    has_subquery_source = True
                    # Try to load from materialized CTE table
                    cte_table = f'{_CTE_PREFIX}{tref.name}'
                    if cat.table_exists(cte_table):
                        self._add_known(
                            cte_table, tref.alias or tref.name, cat,
                            known, known_qualified)
                else:
                    raise TableNotFoundError(tref.name)
            else:
                has_subquery_source = True
            for jc in ast.from_clause.joins:
                if jc.table and jc.table.subquery is None and jc.table.func_args is None:
                    if cat.table_exists(jc.table.name):
                        self._add_known(
                            jc.table.name, jc.table.alias,
                            cat, known, known_qualified)
                    elif self._is_cte_table(jc.table.name):
                        has_subquery_source = True
                        cte_table = f'{_CTE_PREFIX}{jc.table.name}'
                        if cat.table_exists(cte_table):
                            self._add_known(
                                cte_table, jc.table.alias or jc.table.name,
                                cat, known, known_qualified)
                    else:
                        raise TableNotFoundError(jc.table.name)
                elif jc.table and jc.table.subquery is not None:
                    has_subquery_source = True

        # 列存在性检查（有子查询源时跳过）
        if known and not has_subquery_source:
            for expr in ast.select_list:
                self._check_columns_exist(
                    expr, known, known_qualified)
            if ast.where:
                self._check_columns_exist(
                    ast.where, known, known_qualified)
            if ast.order_by:
                for sk in ast.order_by:
                    self._check_columns_exist(
                        sk.expr, known, known_qualified)
            if ast.having:
                self._check_columns_exist(
                    ast.having, known, known_qualified)

        # GROUP BY 一致性检查
        has_agg = any(
            contains_agg(e) for e in ast.select_list)
        # Check if there are scalar subqueries - if all aggs are inside subqueries, skip check
        if has_agg and not self._has_only_subquery_aggs(ast.select_list):
            gb_cols: Set[str] = set()
            gb_exprs: list = []
            if ast.group_by:
                for k in ast.group_by.keys:
                    self._collect_column_names(k, gb_cols)
                    gb_exprs.append(k)
            for expr in ast.select_list:
                self._check_bare_col(
                    expr, gb_cols, gb_exprs)

        # 嵌套聚合检测
        for expr in ast.select_list:
            self._check_nested_agg(expr, 0)

    def _has_only_subquery_aggs(self, select_list) -> bool:
        """Check if all aggregates are inside subqueries (scalar subquery in SELECT)."""
        for item in select_list:
            if self._has_direct_agg(item):
                return False
        return True

    def _has_direct_agg(self, node) -> bool:
        """Check if node contains a direct (non-subquery) aggregate."""
        if isinstance(node, AggregateCall):
            return True
        if SubqueryExpr and isinstance(node, SubqueryExpr):
            return False  # Don't look inside subqueries
        if ExistsExpr and isinstance(node, ExistsExpr):
            return False
        if isinstance(node, AliasExpr):
            return self._has_direct_agg(node.expr)
        if (node is None
                or not dataclasses.is_dataclass(node)
                or isinstance(node, type)):
            return False
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    if self._has_direct_agg(i):
                        return True
            elif child is not None:
                if self._has_direct_agg(child):
                    return True
        return False

    def _validate_create(self, ast, cat):
        """[M08] CHECK 约束验证桩。当前仅验证列名唯一性。"""
        if not ast.columns:
            raise SemanticError("CREATE TABLE 至少需要一列")
        seen: set = set()
        for cd in ast.columns:
            if cd.name in seen:
                raise SemanticError(f"列名重复: '{cd.name}'")
            seen.add(cd.name)

    def _validate_insert(self, ast, cat):
        if not cat.table_exists(ast.table):
            raise TableNotFoundError(ast.table)
        if ast.columns:
            table_cols = set(cat.get_table_columns(ast.table))
            for c in ast.columns:
                if c not in table_cols:
                    raise ColumnNotFoundError(c)

    def _validate_update(self, ast, cat):
        if not cat.table_exists(ast.table):
            raise TableNotFoundError(ast.table)

    def _validate_delete(self, ast, cat):
        if not cat.table_exists(ast.table):
            raise TableNotFoundError(ast.table)

    def _has_outer_agg(self, node):
        """Check if node contains aggregates at the outer level (not inside subqueries)."""
        if isinstance(node, AggregateCall):
            return True
        if SubqueryExpr and isinstance(node, SubqueryExpr):
            return False
        if ExistsExpr and isinstance(node, ExistsExpr):
            return False
        if WindowCall and isinstance(node, WindowCall):
            return False
        if isinstance(node, AliasExpr):
            return self._has_outer_agg(node.expr)
        if isinstance(node, tuple):
            return any(self._has_outer_agg(i) for i in node)
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return False
        for f in dataclasses.fields(node):
            c = getattr(node, f.name)
            if isinstance(c, list):
                if any(self._has_outer_agg(i) for i in c):
                    return True
            elif self._has_outer_agg(c):
                return True
        return False

    def _add_known(self, name, alias, cat,
                   known, known_qualified):
        if not cat.table_exists(name):
            if not self._is_cte_table(name):
                raise TableNotFoundError(name)
            return
        cols = cat.get_table_columns(name)
        qualifier = alias or name
        for c in cols:
            known.add(c)
            known_qualified.add(f"{qualifier}.{c}")

    def _check_columns_exist(self, node, known,
                             known_qualified):
        """宽松检查列存在性。子查询和窗口函数内不检查。"""
        if node is None:
            return
        if SubqueryExpr and isinstance(node, SubqueryExpr):
            return
        if ExistsExpr and isinstance(node, ExistsExpr):
            return
        if isinstance(node, ColumnRef):
            return  # 宽松模式：别名和内部列名不强制检查
        if isinstance(node, AliasExpr):
            self._check_columns_exist(
                node.expr, known, known_qualified)
            return
        if isinstance(node, AggregateCall):
            for a in node.args:
                if not isinstance(a, StarExpr):
                    self._check_columns_exist(
                        a, known, known_qualified)
            return
        if WindowCall and isinstance(node, WindowCall):
            return
        if isinstance(node, StarExpr):
            return
        if (not dataclasses.is_dataclass(node)
                or isinstance(node, type)):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    self._check_columns_exist(
                        i, known, known_qualified)
            elif child is not None:
                self._check_columns_exist(
                    child, known, known_qualified)

    def _collect_column_names(self, node, cols):
        """收集表达式中的列名（GROUP BY 键提取）。"""
        if isinstance(node, ColumnRef):
            cols.add(node.column)
            if node.table:
                cols.add(f"{node.table}.{node.column}")
            return
        if isinstance(node, AliasExpr):
            self._collect_column_names(node.expr, cols)
            return
        if (node is None
                or not dataclasses.is_dataclass(node)
                or isinstance(node, type)):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    self._collect_column_names(i, cols)
            elif child is not None:
                self._collect_column_names(child, cols)

    def _check_bare_col(self, node, gb_cols, gb_exprs):
        """检查 SELECT 中的裸列引用是否在 GROUP BY 中。
        聚合内部和窗口函数内部不检查。"""
        if isinstance(node, AggregateCall):
            return  # 聚合内部不检查
        if WindowCall and isinstance(node, WindowCall):
            return
        if SubqueryExpr and isinstance(node, SubqueryExpr):
            return  # 标量子查询不需要 GROUP BY
        if ExistsExpr and isinstance(node, ExistsExpr):
            return
        if self._matches_group_by(node, gb_exprs):
            return
        if isinstance(node, ColumnRef):
            if node.column in gb_cols:
                return
            if (node.table
                    and f"{node.table}.{node.column}" in gb_cols):
                return
            raise SemanticError(
                f"列 '{node.column}' 必须在 "
                f"GROUP BY 或聚合函数中")
        if isinstance(node, AliasExpr):
            self._check_bare_col(
                node.expr, gb_cols, gb_exprs)
            return
        if (node is None
                or not dataclasses.is_dataclass(node)
                or isinstance(node, type)):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    self._check_bare_col(
                        i, gb_cols, gb_exprs)
            elif child is not None:
                self._check_bare_col(
                    child, gb_cols, gb_exprs)

    @staticmethod
    def _matches_group_by(node, gb_exprs):
        for gb in gb_exprs:
            if node == gb:
                return True
        return False

    def _check_nested_agg(self, node, depth):
        """检测嵌套聚合（如 SUM(COUNT(*))）。"""
        if isinstance(node, AggregateCall):
            if depth > 0:
                raise SemanticError(
                    f"嵌套聚合: {node.name}")
            for a in node.args:
                self._check_nested_agg(a, depth + 1)
            return
        if WindowCall and isinstance(node, WindowCall):
            return
        if isinstance(node, AliasExpr):
            self._check_nested_agg(node.expr, depth)
            return
        if (node is None
                or not dataclasses.is_dataclass(node)
                or isinstance(node, type)):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    self._check_nested_agg(i, depth)
            elif child is not None:
                self._check_nested_agg(child, depth)
