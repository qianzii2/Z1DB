from __future__ import annotations
"""Volcano 执行模型基类。[RF3] 新增共享 drain_operator 函数。"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from executor.core.batch import VectorBatch
from executor.core.result import ExecutionResult
from storage.types import DataType
from executor.core.pool import BatchPool

class Operator(ABC):
    def __init__(self) -> None:
        self.children: List[Operator] = []

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def next_batch(self) -> Optional[VectorBatch]: ...

    @abstractmethod
    def close(self) -> None: ...

    def output_schema(self) -> List[tuple[str, DataType]]:
        return []

    def explain(self, indent: int = 0) -> str:
        name = type(self).__name__
        prefix = '  ' * indent
        lines = [f"{prefix}{name}"]
        for child in self.children:
            lines.append(child.explain(indent + 1))
        return '\n'.join(lines)

    @staticmethod
    def _ensure_batch(batch: Any) -> Optional[VectorBatch]:
        """确保拿到具体 VectorBatch。LazyBatch 会被物化。"""
        if batch is None:
            return None
        try:
            from executor.core.lazy_batch import LazyBatch
            if isinstance(batch, LazyBatch):
                return batch.materialize()
        except ImportError:
            pass
        return batch


def drain_operator(op: Operator) -> ExecutionResult:
    """[RF3] 排空算子并返回 ExecutionResult。
    消除 SimplePlanner._drain / IntegratedPlanner._drain / set_ops._drain 重复。"""
    schema = op.output_schema()
    cn = [n for n, _ in schema]
    ct = [t for _, t in schema]
    op.open()
    rows = []
    while True:
        b = op.next_batch()
        if b is None:
            break
        b = Operator._ensure_batch(b)
        if b is None:
            break
        rows.extend(b.to_rows())
    op.close()
    return ExecutionResult(
        columns=cn, column_types=ct,
        rows=rows, row_count=len(rows))


class PooledOperator(Operator):
    """支持 BatchPool 的算子基类。
    open() 创建池，close() 释放池。"""

    def __init__(self) -> None:
        super().__init__()
        self._pool: Optional[BatchPool] = None

    def open(self) -> None:
        self._pool = BatchPool()

    def close(self) -> None:
        if self._pool:
            self._pool.reset()
            self._pool = None

    @property
    def pool(self) -> Optional[BatchPool]:
        return self._pool