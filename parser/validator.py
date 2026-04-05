from __future__ import annotations
"""语义验证 — GROUP BY豁免、嵌套聚合检测、表/列存在性检查。"""
import dataclasses
from typing import Any, Protocol, Set
from parser.ast import (AggregateCall, AliasExpr, ColumnRef, CreateTableStmt,
                         DeleteStmt, InsertStmt, SelectStmt, StarExpr, UpdateStmt)
from utils.errors import ColumnNotFoundError, SemanticError, TableNotFoundError

try:
    from parser.ast import WindowCall, ExplainStmt, SetOperationStmt, AlterTableStmt
except ImportError:
    WindowCall = None  # type: ignore
    ExplainStmt = None  # type: ignore
    SetOperationStmt = None  # type: ignore
    AlterTableStmt = None  # type: ignore


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
        if ast.from_clause:
            tref = ast.from_clause.table
            if tref.subquery is None:
                self._add_known(tref.name, cat, known)
            for jc in ast.from_clause.joins:
                if jc.table and jc.table.subquery is None:
                    self._add_known(jc.table.name, cat, known)

        has_agg = any(self._contains_agg(e) for e in ast.select_list)

        if has_agg:
            # 收集GROUP BY中的列名用于豁免
            gb_cols: Set[str] = set()
            if ast.group_by:
                for k in ast.group_by.keys:
                    self._collect_column_names(k, gb_cols)
            # 检查SELECT列表中的裸列（必须在GROUP BY中或聚合函数内）
            for expr in ast.select_list:
                self._check_bare_col(expr, gb_cols)

        # 检查嵌套聚合
        for expr in ast.select_list:
            self._check_nested_agg(expr, 0)

    def _validate_insert(self, ast: InsertStmt, cat: CatalogInfo) -> None:
        if not cat.table_exists(ast.table):
            raise TableNotFoundError(ast.table)
        if ast.columns:
            known = set(cat.get_table_columns(ast.table))
            for c in ast.columns:
                if c not in known:
                    raise ColumnNotFoundError(c)

    def _validate_update(self, ast: UpdateStmt, cat: CatalogInfo) -> None:
        if not cat.table_exists(ast.table):
            raise TableNotFoundError(ast.table)

    def _validate_delete(self, ast: DeleteStmt, cat: CatalogInfo) -> None:
        if not cat.table_exists(ast.table):
            raise TableNotFoundError(ast.table)

    def _validate_create(self, ast: CreateTableStmt,
                         cat: CatalogInfo) -> None:
        if not ast.columns:
            raise SemanticError("CREATE TABLE requires at least one column")
        seen: set = set()
        for cd in ast.columns:
            if cd.name in seen:
                raise SemanticError(f"duplicate column: '{cd.name}'")
            seen.add(cd.name)

    def _add_known(self, name: str, cat: CatalogInfo,
                   known: Set[str]) -> None:
        if not cat.table_exists(name):
            raise TableNotFoundError(name)
        known.update(cat.get_table_columns(name))

    def _collect_column_names(self, node: Any, cols: Set[str]) -> None:
        """收集表达式中的列名（用于GROUP BY豁免判断）。"""
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

    def _contains_agg(self, node: Any) -> bool:
        if isinstance(node, AggregateCall):
            return True
        if WindowCall is not None and isinstance(node, WindowCall):
            return False
        if isinstance(node, AliasExpr):
            return self._contains_agg(node.expr)
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return False
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                if any(self._contains_agg(c) for c in child):
                    return True
            elif self._contains_agg(child):
                return True
        return False

    def _check_bare_col(self, node: Any, gb_cols: Set[str]) -> None:
        """检查裸列引用是否在GROUP BY中或聚合函数内。"""
        if isinstance(node, AggregateCall):
            return  # 聚合函数内的列不需要检查
        if WindowCall is not None and isinstance(node, WindowCall):
            return
        if isinstance(node, ColumnRef):
            # 检查是否在GROUP BY列中
            if node.column in gb_cols:
                return
            if node.table and f"{node.table}.{node.column}" in gb_cols:
                return
            raise SemanticError(
                f"column '{node.column}' must appear in GROUP BY or aggregate")
        if isinstance(node, AliasExpr):
            self._check_bare_col(node.expr, gb_cols)
            return
        if node is None or not dataclasses.is_dataclass(node) or isinstance(node, type):
            return
        for f in dataclasses.fields(node):
            child = getattr(node, f.name)
            if isinstance(child, list):
                for i in child:
                    self._check_bare_col(i, gb_cols)
            elif child is not None:
                self._check_bare_col(child, gb_cols)

    def _check_nested_agg(self, node: Any, depth: int) -> None:
        if isinstance(node, AggregateCall):
            if depth > 0:
                raise SemanticError(f"nested aggregate: {node.name}")
            for a in node.args:
                self._check_nested_agg(a, depth + 1)
            return
        if WindowCall is not None and isinstance(node, WindowCall):
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
