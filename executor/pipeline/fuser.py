from __future__ import annotations

"""Pipeline fusion — merge Scan+Filter+Project into a single loop.
Eliminates 2/3 of per-row Python function calls."""
from typing import Any, Callable, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.codegen.compiler import ExprCompiler
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from storage.types import DTYPE_TO_ARRAY_CODE, DataType


class FusedScanFilterProject(Operator):
    """Fused Scan → Filter → Project in a single pass.

    Instead of:
      scan.next_batch() → filter.evaluate_predicate() → project.evaluate()
    We do:
      for each row: if filter(row): emit(project(row))

    3 Python loops → 1 Python loop.
    """

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
        all_rows = self._store.read_all_rows()
        col_names = [c.name for c in self._store.schema.columns]
        count = 0

        for row in all_rows:
            if self._limit is not None and count >= self._limit:
                break
            # Build row dict
            row_dict = {col_names[i]: row[i] for i in range(len(col_names))}

            # Apply filter
            if self._filter_fn is not None:
                if not self._filter_fn(row_dict):
                    continue

            # Apply projection
            if self._project_fn is not None:
                projected = self._project_fn(row_dict)
            else:
                projected = [row_dict.get(n) for n in self._output_names]

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


def try_fuse(scan_op: Any, filter_expr: Any, project_exprs: List[Tuple[str, Any]],
             limit: Optional[int], store: Any, output_types: List[DataType]) -> Optional[Operator]:
    """Attempt to fuse scan+filter+project. Returns None if fusion not possible."""
    # Try to compile filter
    filter_fn = None
    if filter_expr is not None:
        filter_fn = ExprCompiler.compile_predicate(filter_expr)
        if filter_fn is None:
            return None  # Can't compile → can't fuse

    # Try to compile projection
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
