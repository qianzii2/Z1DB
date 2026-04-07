from __future__ import annotations
"""递归CTE引擎 — 不动点迭代+环检测。
用truncate+重填替代drop+create，避免每次迭代重建表。"""
from typing import Any, Dict, List, Optional, Set, Tuple
from catalog.catalog import Catalog, ColumnSchema, TableSchema
from executor.core.result import ExecutionResult
from metal.hash import murmur3_64
from utils.errors import ExecutionError, RecursionLimitError

MAX_ITERATIONS = 1000
MAX_ROWS = 1_000_000


class RecursiveCTEExecutor:
    """执行递归CTE，支持不动点迭代和环检测。"""

    def __init__(self, planner: Any, catalog: Catalog) -> None:
        self._planner = planner
        self._catalog = catalog

    def execute(self, cte_name: str, base_query: Any, recursive_query: Any,
                union_all: bool = True,
                column_names: Optional[List[str]] = None) -> ExecutionResult:
        # 步骤1：执行基础查询
        base_result = self._execute_query(base_query)
        if not base_result.columns:
            return base_result

        # 应用CTE列别名
        columns = list(base_result.columns)
        col_types = list(base_result.column_types)
        if column_names and len(column_names) == len(columns):
            columns = list(column_names)

        # 初始化
        result_rows: List[list] = [list(r) for r in base_result.rows]
        working_rows: List[list] = [list(r) for r in base_result.rows]

        # 环检测
        seen_hashes: Set[int] = set()
        if not union_all:
            for row in result_rows:
                seen_hashes.add(self._row_hash(row))

        # 步骤2：不动点迭代
        iteration = 0
        table_created = False
        while working_rows and iteration < MAX_ITERATIONS:
            iteration += 1
            if len(result_rows) > MAX_ROWS:
                raise RecursionLimitError(
                    f"Recursive CTE exceeded {MAX_ROWS} rows "
                    f"after {iteration} iterations")

            # 物化工作表（首次create，后续truncate+重填）
            self._materialize_working_table(
                cte_name, columns, col_types, working_rows, table_created)
            table_created = True

            try:
                new_result = self._execute_query(recursive_query)
            except Exception:
                break

            if not new_result.rows:
                break

            new_rows = [list(r) for r in new_result.rows]

            if not union_all:
                deduplicated = []
                for row in new_rows:
                    h = self._row_hash(row)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        deduplicated.append(row)
                new_rows = deduplicated
                if not new_rows:
                    break

            result_rows.extend(new_rows)
            working_rows = new_rows

        result = ExecutionResult(
            columns=columns,
            column_types=col_types,
            rows=result_rows,
            row_count=len(result_rows))

        self._cleanup(cte_name)

        return result

    def _execute_query(self, query: Any) -> ExecutionResult:
        """通过planner执行查询AST。"""
        from parser.resolver import Resolver
        from parser.validator import Validator
        try:
            resolved = Resolver().resolve(query, self._catalog)
            validated = Validator().validate(resolved, self._catalog)
            return self._planner.execute(validated, self._catalog)
        except Exception as e:
            raise ExecutionError(f"Recursive CTE execution failed: {e}")

    def _materialize_working_table(self, name: str, columns: List[str],
                                   col_types: list, rows: List[list],
                                   already_exists: bool) -> None:
        """物化工作表。已存在时truncate重填，不存在时create。"""
        if already_exists and self._catalog.table_exists(name):
            # 直接清空重填，避免drop+create开销
            store = self._catalog.get_store(name)
            store.truncate()
        else:
            # 首次创建
            if self._catalog.table_exists(name):
                self._catalog.drop_table(name)
            cols = [ColumnSchema(name=cn, dtype=ct, nullable=True)
                    for cn, ct in zip(columns, col_types)]
            schema = TableSchema(name=name, columns=cols)
            self._catalog.create_table(schema)
            store = self._catalog.get_store(name)
        # 填充数据
        for row in rows:
            store.append_row(list(row))

    def _cleanup(self, name: str) -> None:
        if self._catalog.table_exists(name):
            try:
                self._catalog.drop_table(name)
            except Exception:
                pass

    @staticmethod
    def _row_hash(row: list) -> int:
        """行哈希，用于环检测。使用长度前缀编码避免分隔符碰撞。"""
        import struct
        parts = []
        for val in row:
            if val is None:
                parts.append(b'\x00\x00\x00\x00')  # 4字节NULL标记
            elif isinstance(val, int):
                parts.append(b'\x01' + val.to_bytes(8, 'little', signed=True))
            elif isinstance(val, float):
                parts.append(b'\x02' + struct.pack('d', val))
            elif isinstance(val, str):
                encoded = val.encode('utf-8')
                parts.append(b'\x03' + len(encoded).to_bytes(4, 'little')
                             + encoded)
            else:
                encoded = str(val).encode('utf-8')
                parts.append(b'\x04' + len(encoded).to_bytes(4, 'little')
                             + encoded)
        return murmur3_64(b''.join(parts))
