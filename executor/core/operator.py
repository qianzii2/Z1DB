from __future__ import annotations
"""Volcano执行模型基类。所有算子继承此类。"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from executor.core.batch import VectorBatch
from storage.types import DataType


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
        """确保拿到具体VectorBatch。LazyBatch会被物化。
        所有算子在消费child.next_batch()结果时应调用此方法。"""
        if batch is None:
            return None
        try:
            from executor.core.lazy_batch import LazyBatch
            if isinstance(batch, LazyBatch):
                return batch.materialize()
        except ImportError:
            pass
        return batch
