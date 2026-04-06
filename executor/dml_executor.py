from __future__ import annotations
"""DML 执行器 — INSERT/UPDATE/DELETE/COPY。
批量 WHERE 评估，不再逐行创建 VectorBatch。"""
from typing import Any, Dict, List, Optional, Set
from catalog.catalog import Catalog
from executor.core.batch import VectorBatch
from executor.core.result import ExecutionResult
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType
from utils.errors import (ColumnNotFoundError, ExecutionError,
                           NumericOverflowError)


def _safe_cast_bool(val: Any) -> bool:
    if isinstance(val, bool): return val
    if isinstance(val, (int, float)): return val != 0
    if isinstance(val, str):
        l = val.strip().lower()
        if l in ('true', '1', 't', 'yes', 'on'): return True
        if l in ('false', '0', 'f', 'no', 'off'): return False
        raise ExecutionError(f"无法转换 '{val}' 为 BOOLEAN")
    return bool(val)


class DMLExecutor:
    """DML 执行。需要 evaluator 做表达式求值。"""

    def __init__(self, evaluator: ExpressionEvaluator,
                 planner: Any = None) -> None:
        self._evaluator = evaluator
        self._planner = planner  # 用于 INSERT...SELECT

    def exec_insert(self, ast: Any, catalog: Catalog) -> ExecutionResult:
        schema = catalog.get_table(ast.table)
        store = catalog.get_store(ast.table)
        if ast.query is not None:
            # INSERT...SELECT
            result = self._planner.execute(ast.query, catalog)
            count = 0
            for row in result.rows:
                pv = list(row)
                full = self._reorder(pv, ast.columns, schema) if ast.columns else pv
                if not ast.columns and len(pv) != len(schema.columns):
                    raise ExecutionError(
                        f"期望 {len(schema.columns)} 个值，实际 {len(pv)} 个")
                store.append_row(self._validate_row(full, schema))
                count += 1
            w = 'row' if count == 1 else 'rows'
            return ExecutionResult(affected_rows=count,
                                   message=f"Inserted {count} {w}")
        # INSERT...VALUES
        if ast.columns:
            seen = set()
            sc = {c.name for c in schema.columns}
            for c in ast.columns:
                if c in seen:
                    raise ExecutionError(f"重复列: '{c}'")
                if c not in sc:
                    raise ColumnNotFoundError(c)
                seen.add(c)
        dummy = VectorBatch.single_row_no_columns()
        count = 0
        for ve in ast.values:
            pv = [self._evaluator.evaluate(e, dummy).get(0) for e in ve]
            full = self._reorder(pv, ast.columns, schema) if ast.columns else pv
            if not ast.columns and len(pv) != len(schema.columns):
                raise ExecutionError(
                    f"期望 {len(schema.columns)} 个值，实际 {len(pv)} 个")
            store.append_row(self._validate_row(full, schema))
            count += 1
        w = 'row' if count == 1 else 'rows'
        return ExecutionResult(affected_rows=count,
                               message=f"Inserted {count} {w}")

    def exec_update(self, ast: Any, catalog: Catalog) -> ExecutionResult:
        """批量 WHERE 评估 + 批量值计算。"""
        schema = catalog.get_table(ast.table)
        store = catalog.get_store(ast.table)
        ar = store.read_all_rows()
        cn = schema.column_names
        ct = [c.dtype for c in schema.columns]
        if not ar:
            return ExecutionResult(affected_rows=0, message="Updated 0 rows")
        # 批量 WHERE
        if ast.where:
            batch = VectorBatch.from_rows(ar, cn, ct)
            mask = self._evaluator.evaluate_predicate(ast.where, batch)
            matching = set(mask.to_indices())
        else:
            matching = set(range(len(ar)))
        if not matching:
            return ExecutionResult(affected_rows=0, message="Updated 0 rows")
        # 批量计算新值
        batch = VectorBatch.from_rows(ar, cn, ct)
        updates_by_col: Dict[int, Dict[int, Any]] = {}
        for a in ast.assignments:
            ci = next((i for i, c in enumerate(schema.columns)
                       if c.name == a.column), None)
            if ci is None:
                raise ColumnNotFoundError(a.column)
            val_vec = self._evaluator.evaluate(a.value, batch)
            updates_by_col[ci] = {ri: val_vec.get(ri) for ri in matching}
        count = len(matching)
        if hasattr(store, 'update_rows') and count > 0:
            for ci, new_vals in updates_by_col.items():
                store.update_rows(matching, ci, new_vals)
        elif count > 0:
            for ri in range(len(ar)):
                for ci, new_vals in updates_by_col.items():
                    if ri in new_vals:
                        ar[ri][ci] = new_vals[ri]
            store.truncate()
            for r in ar:
                store.append_row(r)
        w = 'row' if count == 1 else 'rows'
        return ExecutionResult(affected_rows=count,
                               message=f"Updated {count} {w}")

    def exec_delete(self, ast: Any, catalog: Catalog) -> ExecutionResult:
        """批量 WHERE 评估。"""
        schema = catalog.get_table(ast.table)
        store = catalog.get_store(ast.table)
        if ast.where is None:
            c = store.row_count
            store.truncate()
            return ExecutionResult(
                affected_rows=c,
                message=f"Deleted {c} {'row' if c == 1 else 'rows'}")
        ar = store.read_all_rows()
        if not ar:
            return ExecutionResult(affected_rows=0, message="Deleted 0 rows")
        cn = schema.column_names
        ct = [c.dtype for c in schema.columns]
        batch = VectorBatch.from_rows(ar, cn, ct)
        mask = self._evaluator.evaluate_predicate(ast.where, batch)
        to_delete = set(mask.to_indices())
        count = 0
        if to_delete:
            if hasattr(store, 'delete_rows'):
                count = store.delete_rows(to_delete)
            else:
                keep = [r for i, r in enumerate(ar) if i not in to_delete]
                count = len(ar) - len(keep)
                store.truncate()
                for r in keep:
                    store.append_row(r)
        return ExecutionResult(
            affected_rows=count,
            message=f"Deleted {count} {'row' if count == 1 else 'rows'}")

    def exec_copy(self, ast: Any, catalog: Catalog) -> ExecutionResult:
        try:
            from executor.copy_executor import CopyExecutor
            ce = CopyExecutor(catalog)
            if ast.direction == 'FROM':
                return ce.copy_from(ast.table, ast.file_path,
                                     has_header=ast.has_header,
                                     delimiter=ast.delimiter)
            elif ast.direction == 'TO':
                return ce.copy_to(ast.table, ast.file_path,
                                   has_header=ast.has_header,
                                   delimiter=ast.delimiter)
            else:
                raise ExecutionError(f"COPY 方向不支持: {ast.direction}")
        except ImportError:
            raise ExecutionError("COPY 不可用")

    # ═══ 辅助 ═══

    @staticmethod
    def _reorder(vals: list, cols: list, schema) -> list:
        cm = dict(zip(cols, vals))
        result = []
        for c in schema.columns:
            if c.name in cm:
                result.append(cm[c.name])
            elif c.nullable:
                result.append(None)
            else:
                raise ExecutionError(f"列 '{c.name}' 不允许 NULL")
        return result

    @staticmethod
    def _validate_row(row: list, schema) -> list:
        if len(row) != len(schema.columns):
            raise ExecutionError("列数不匹配")
        result = []
        for val, col in zip(row, schema.columns):
            if val is None:
                if not col.nullable:
                    raise ExecutionError(f"NULL 写入 NOT NULL 列 '{col.name}'")
                result.append(None)
                continue
            dt = col.dtype
            try:
                if dt == DataType.INT:
                    v = int(val)
                    if not (-2_147_483_648 <= v <= 2_147_483_647):
                        raise NumericOverflowError(f"溢出: {v}")
                    result.append(v)
                elif dt == DataType.BIGINT:
                    result.append(int(val))
                elif dt in (DataType.FLOAT, DataType.DOUBLE):
                    result.append(float(val))
                elif dt == DataType.BOOLEAN:
                    result.append(_safe_cast_bool(val))
                elif dt in (DataType.VARCHAR, DataType.TEXT):
                    s = str(val)
                    if col.max_length and len(s) > col.max_length:
                        s = s[:col.max_length]
                    result.append(s)
                else:
                    result.append(val)
            except (ValueError, TypeError) as e:
                raise ExecutionError(
                    f"无法转换 {val!r} 为 {dt.name}: {e}")
        return result
