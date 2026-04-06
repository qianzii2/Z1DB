from __future__ import annotations
"""Pipeline 融合 — 合并 Scan+Filter+Project 为单循环。
消除 2/3 的逐行 Python 函数调用。"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.codegen.compiler import ExprCompiler
from storage.types import DataType


class FusedScanFilterProject(Operator):
    """融合 Scan → Filter → Project 为单趟扫描。

    改进：逐 chunk 读取而非 read_all_rows()，支持 LIMIT 提前终止。"""

    def __init__(self, store: Any, columns: List[str],
                 filter_fn: Optional[Callable],
                 project_fn: Optional[Callable],
                 output_names: List[str],
                 output_types: List[DataType],
                 limit: Optional[int] = None) -> None:
        super().__init__()
        self._store = store
        self._columns = columns
        self._filter_fn = filter_fn
        self._project_fn = project_fn
        self._output_names = output_names
        self._output_types = output_types
        self._limit = limit
        self._result_rows: list = []
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return list(zip(self._output_names, self._output_types))

    def open(self) -> None:
        self._result_rows = []
        self._emitted = False
        col_names = [c.name for c in self._store.schema.columns]
        count = 0
        done = False

        # [FIX-C03] 安全检查 store 是否支持 chunk 级读取
        if not hasattr(self._store, 'get_chunk_count'):
            # 回退到 read_all_rows
            self._fallback_open(col_names)
            return

        chunk_count = self._store.get_chunk_count()
        for ci in range(chunk_count):
            if done:
                break
            first_chunks = self._store.get_column_chunks(col_names[0])
            if ci >= len(first_chunks) or first_chunks[ci].row_count == 0:
                continue

            n_rows = first_chunks[ci].row_count
            chunk_cols = {}
            for cname in col_names:
                cl = self._store.get_column_chunks(cname)
                if ci < len(cl):
                    chunk_cols[cname] = cl[ci]

            for ri in range(n_rows):
                if self._limit is not None and count >= self._limit:
                    done = True
                    break

                # [FIX-C03] 使用 ColumnChunk.get() — 已在 column_chunk.py 中验证存在
                row_dict = {}
                for cname in col_names:
                    if cname in chunk_cols:
                        row_dict[cname] = chunk_cols[cname].get(ri)
                    else:
                        row_dict[cname] = None

                if self._filter_fn is not None:
                    if not self._filter_fn(row_dict):
                        continue

                if self._project_fn is not None:
                    projected = self._project_fn(row_dict)
                else:
                    projected = [row_dict.get(n)
                                 for n in self._output_names]

                self._result_rows.append(projected)
                count += 1

    def _fallback_open(self, col_names: list) -> None:
        """[FIX-C03] 回退路径：使用 read_all_rows。"""
        all_rows = self._store.read_all_rows()
        count = 0
        for row in all_rows:
            if self._limit is not None and count >= self._limit:
                break
            row_dict = {col_names[i]: row[i]
                        for i in range(min(len(col_names), len(row)))}
            if self._filter_fn is not None:
                if not self._filter_fn(row_dict):
                    continue
            if self._project_fn is not None:
                projected = self._project_fn(row_dict)
            else:
                projected = [row_dict.get(n)
                             for n in self._output_names]
            self._result_rows.append(projected)
            count += 1

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._output_names,
                                     self._output_types)
        return VectorBatch.from_rows(self._result_rows,
                                     self._output_names,
                                     self._output_types)

    def close(self) -> None:
        pass


def try_fuse(scan_op: Any, filter_expr: Any,
             project_exprs: List[Tuple[str, Any]],
             limit: Optional[int], store: Any,
             output_types: List[DataType]) -> Optional[Operator]:
    """尝试融合 scan+filter+project。无法编译时返回 None。"""
    # 编译过滤谓词
    filter_fn = None
    if filter_expr is not None:
        filter_fn = ExprCompiler.compile_predicate(filter_expr)
        if filter_fn is None:
            return None  # 无法编译 → 不融合

    # 编译投影列表
    project_fn = None
    proj_exprs_only = [expr for _, expr in project_exprs]
    proj_names = [name for name, _ in project_exprs]
    if proj_exprs_only:
        project_fn = ExprCompiler.compile_projection(
            proj_exprs_only, proj_names)
        if project_fn is None:
            return None

    columns = [c.name for c in store.schema.columns]
    return FusedScanFilterProject(
        store=store,
        columns=columns,
        filter_fn=filter_fn,
        project_fn=project_fn,
        output_names=proj_names,
        output_types=output_types,
        limit=limit,
    )
