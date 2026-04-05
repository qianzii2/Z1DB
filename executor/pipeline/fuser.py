from __future__ import annotations
"""Pipeline 融合 — 合并 Scan+Filter+Project 为单循环。
修复：使用列存 chunk 扫描替代 read_all_rows()。"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.codegen.compiler import ExprCompiler
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from storage.types import DTYPE_TO_ARRAY_CODE, DataType


class FusedScanFilterProject(Operator):
    """融合 Scan → Filter → Project 为单次遍历。"""

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

        # 修复：使用列存 chunk 逐 chunk 扫描
        chunk_count = self._store.get_chunk_count()
        for ci in range(chunk_count):
            if self._limit is not None and count >= self._limit:
                break

            # 读取当前 chunk 的行数
            first_col = self._store.schema.columns[0].name
            chunks = self._store.get_column_chunks(first_col)
            if ci >= len(chunks) or chunks[ci].row_count == 0:
                continue

            n_rows = chunks[ci].row_count
            for ri in range(n_rows):
                if self._limit is not None and count >= self._limit:
                    break

                # 构建行字典（从各列 chunk 读取）
                row_dict = {}
                for cn in col_names:
                    col_chunks = self._store.get_column_chunks(cn)
                    if ci < len(col_chunks):
                        row_dict[cn] = col_chunks[ci].get(ri)
                    else:
                        row_dict[cn] = None

                # 过滤
                if self._filter_fn is not None:
                    if not self._filter_fn(row_dict):
                        continue

                # 投影
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
            return VectorBatch.empty(self._output_names, self._output_types)
        return VectorBatch.from_rows(self._result_rows,
                                     self._output_names, self._output_types)

    def close(self) -> None:
        pass


def try_fuse(scan_op: Any, filter_expr: Any,
             project_exprs: List[Tuple[str, Any]],
             limit: Optional[int], store: Any,
             output_types: List[DataType]) -> Optional[Operator]:
    """尝试融合 scan+filter+project。无法编译则返回 None。"""
    filter_fn = None
    if filter_expr is not None:
        filter_fn = ExprCompiler.compile_predicate(filter_expr)
        if filter_fn is None:
            return None

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
