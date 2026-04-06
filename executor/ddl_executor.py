from __future__ import annotations
"""DDL 执行器 — CREATE/DROP/ALTER TABLE + CREATE/DROP INDEX。"""
from typing import Any
from catalog.catalog import Catalog, ColumnSchema, TableSchema
from executor.core.result import ExecutionResult
from storage.types import resolve_type_name
from utils.errors import ExecutionError


class DDLExecutor:
    """无状态 DDL 执行器。所有方法为静态式（仅依赖 catalog）。"""

    @staticmethod
    def exec_create(ast: Any, catalog: Catalog) -> ExecutionResult:
        cols = []
        for cd in ast.columns:
            dt, ml = resolve_type_name(cd.type_name.name, cd.type_name.params)
            cols.append(ColumnSchema(
                name=cd.name, dtype=dt, nullable=cd.nullable,
                primary_key=cd.primary_key, max_length=ml))
        catalog.create_table(
            TableSchema(name=ast.table, columns=cols), ast.if_not_exists)
        return ExecutionResult(message='OK')

    @staticmethod
    def exec_drop(ast: Any, catalog: Catalog) -> ExecutionResult:
        catalog.drop_table(ast.table, ast.if_exists)
        return ExecutionResult(message='OK')

    @staticmethod
    def exec_alter(ast: Any, catalog: Catalog) -> ExecutionResult:
        if ast.action == 'ADD_COLUMN':
            dt, ml = resolve_type_name(
                ast.column_def.type_name.name,
                ast.column_def.type_name.params)
            catalog.alter_add_column(ast.table, ColumnSchema(
                name=ast.column_def.name, dtype=dt,
                nullable=ast.column_def.nullable,
                primary_key=ast.column_def.primary_key,
                max_length=ml))
            return ExecutionResult(message='OK')
        if ast.action == 'DROP_COLUMN':
            catalog.alter_drop_column(ast.table, ast.column_name)
            return ExecutionResult(message='OK')
        if ast.action == 'RENAME_COLUMN':
            catalog.alter_rename_column(
                ast.table, ast.column_name, ast.new_name)
            return ExecutionResult(message='OK')
        raise ExecutionError(f"不支持的 ALTER 操作: {ast.action}")

    @staticmethod
    def exec_create_index(ast: Any, catalog: Catalog) -> ExecutionResult:
        mgr = catalog.index_manager
        if mgr is None:
            raise ExecutionError("索引管理器不可用")
        try:
            mgr.create_index(
                ast.index_name, ast.table, ast.columns,
                ast.unique, ast.if_not_exists)
            store = catalog.get_store(ast.table)
            count = mgr.build_index(ast.index_name, store, catalog)
            return ExecutionResult(
                message=f'CREATE INDEX {ast.index_name} ({count} 行已索引)')
        except Exception as e:
            raise ExecutionError(str(e))

    @staticmethod
    def exec_drop_index(ast: Any, catalog: Catalog) -> ExecutionResult:
        mgr = catalog.index_manager
        if mgr is None:
            raise ExecutionError("索引管理器不可用")
        try:
            mgr.drop_index(ast.index_name, ast.if_exists)
            return ExecutionResult(message='OK')
        except Exception as e:
            raise ExecutionError(str(e))
