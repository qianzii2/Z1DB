from __future__ import annotations
"""语义验证 — 委托 contains_agg 到 ast_utils。"""
import dataclasses
from typing import Any, Protocol, Set
from parser.ast import (AggregateCall, AliasExpr, ColumnRef,
                         CreateTableStmt, DeleteStmt, InsertStmt,
                         SelectStmt, StarExpr, UpdateStmt,
                         FunctionCall, BinaryExpr, UnaryExpr,
                         CaseExpr, CastExpr, IsNullExpr, InExpr,
                         BetweenExpr, LikeExpr)
from parser.ast_utils import contains_agg
from utils.errors import (ColumnNotFoundError, SemanticError,
                           TableNotFoundError)

try:
    from parser.ast import (WindowCall, ExplainStmt,
                             SetOperationStmt, AlterTableStmt,
                             SubqueryExpr, ExistsExpr)
except ImportError:
    WindowCall = None
    ExplainStmt = None
    SetOperationStmt = None
    AlterTableStmt = None
    SubqueryExpr = None
    ExistsExpr = None


class CatalogInfo(Protocol):
    def get_table_columns(self, table: str) -> list[str]: ...
    def table_exists(self, table: str) -> bool: ...


class Validator:
    def validate(self, ast: Any, catalog_info: CatalogInfo) -> Any:
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
        elif ExplainStmt is not None and isinstance(ast, ExplainStmt):
            self.validate(ast.statement, catalog_info)
        elif SetOperationStmt is not None and isinstance(ast, SetOperationStmt):
            self.validate(ast.left, catalog_info)
            self.validate(ast.right, catalog_info)
        return ast

    def _validate_select(self, ast: SelectStmt, cat: CatalogInfo) -> None:
        known: Set[str] = set()
        known_qualified: Set[str] = set()
        has_subquery_source = False

        if ast.from_clause:
            tref = ast.from_clause.table
            if tref.subquery is None:
                self._add_known(tref.name, tref.alias, cat, known, known_qualified)
            else:
                has_subquery_source = True
            for jc in ast.from_clause.joins:
                if jc.table and jc.table.subquery is None:
                    self._add_known(jc.table.name, jc.table.alias, cat, known, known_qualified)
                elif jc.table and jc.table.subquery is not None:
                    has_subquery_source = True

        if known and not has_subquery_source:
            for expr in ast.select_list:
                self._check_columns_exist(expr, known, known_qualified)
            if ast.where:
                self._check_columns_exist(ast.where, known, known_qualified)
            if ast.order_by:
                for sk in ast.order_by:
                    self._check_columns_exist(sk.expr, known, known_qualified)
            if ast.having:
                self._check_columns_exist(ast.having, known, known_qualified)

        has_agg = any(contains_agg(e) for e in ast.select_list)
        if has_agg:
            gb_cols: Set[str] = set()
            gb_exprs: list = []
            if ast.group_by:
                for k in ast.group_by.keys:
                    self._collect_column_names(k, gb_cols)
                    gb_exprs.append(k)
            for expr in ast.select_list:
                self._check_bare_col(expr, gb_cols, gb_exprs)

        for expr in ast.select_list:
            self._check_nested_agg(expr, 0)

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

    def _validate_create(self, ast, cat):
        if not ast.columns:
            raise SemanticError("CREATE TABLE 至少需要一列")
        seen: set = set()
        for cd in ast.columns:
            if cd.name in seen:
                raise SemanticError(f"列名重复: '{cd.name}'")
            seen.add(cd.name)

    def _add_known(self, name, alias, cat, known, known_qualified):
        if not cat.table_exists(name):
            raise TableNotFoundError(name)
        cols = cat.get_table_columns(name)
        qualifier = alias or name
        for c in cols:
            known.add(c)
            known_qualified.add(f"{qualifier}.{c}")

    def _check_columns_exist(self, node, known, known_qualified):
        if node is None:
            return
        if SubqueryExpr and isinstance(node, SubqueryExpr):
            return
        if ExistsExpr and isinstance(node, ExistsExpr):
            return
        if isinstance(node, ColumnRef):
            # 宽松模式：别名和内部列名不强制检查
            return
        if isinstance(node, AliasExpr):
            self._check_columns_exist(node.expr, known, known_qualified)
            return
        if isinstance(node, AggregateCall):
            for a in node.args:
                if not isinstance(a, StarExpr):
                    self._check_columns_exist(a, known, known_qualified)
            return
        if WindowCall and isinstance(node, WindowCall):
            return
        if isinstance(node, StarExpr):
            return
        if not dataclasses.is_dataclass(node) or isinstance(node, type):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    self._check_columns_exist(i, known, known_qualified)
            elif child is not None:
                self._check_columns_exist(child, known, known_qualified)

    def _collect_column_names(self, node, cols):
        if isinstance(node, ColumnRef):
            cols.add(node.column)
            if node.table:
                cols.add(f"{node.table}.{node.column}")
            return
        if isinstance(node, AliasExpr):
            self._collect_column_names(node.expr, cols)
            return
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    self._collect_column_names(i, cols)
            elif child is not None:
                self._collect_column_names(child, cols)

    def _check_bare_col(self, node, gb_cols, gb_exprs):
        if isinstance(node, AggregateCall):
            return
        if WindowCall and isinstance(node, WindowCall):
            return
        if self._matches_group_by(node, gb_exprs):
            return
        if isinstance(node, ColumnRef):
            if node.column in gb_cols:
                return
            if node.table and f"{node.table}.{node.column}" in gb_cols:
                return
            raise SemanticError(
                f"列 '{node.column}' 必须在 GROUP BY 或聚合函数中")
        if isinstance(node, AliasExpr):
            self._check_bare_col(node.expr, gb_cols, gb_exprs)
            return
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    self._check_bare_col(i, gb_cols, gb_exprs)
            elif child is not None:
                self._check_bare_col(child, gb_cols, gb_exprs)

    @staticmethod
    def _matches_group_by(node, gb_exprs):
        for gb in gb_exprs:
            if node == gb:
                return True
        return False

    def _check_nested_agg(self, node, depth):
        if isinstance(node, AggregateCall):
            if depth > 0:
                raise SemanticError(f"嵌套聚合: {node.name}")
            for a in node.args:
                self._check_nested_agg(a, depth + 1)
            return
        if WindowCall and isinstance(node, WindowCall):
            return
        if isinstance(node, AliasExpr):
            self._check_nested_agg(node.expr, depth)
            return
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    self._check_nested_agg(i, depth)
            elif child is not None:
                self._check_nested_agg(child, depth)
